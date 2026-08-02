#!/usr/bin/env python3
"""Benchmark 1NF-10NF with RF, DNN, CNN, and a mamba6mA-style model."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_knf_mamba6ma_21nt import DynamicFusion, HybridConfig, RMSNorm, ResidualBlock


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "21nt_newData"
DEFAULT_OUT = DATA / "knf_1to10_four_models"
MODEL_NAMES = ("RF", "DNN", "CNN", "mamba6mA")
COLORS = {"RF": "#2A6F97", "DNN": "#2F855A", "CNN": "#C6533D", "mamba6mA": "#7656A5"}
MARKERS = {"RF": "o", "DNN": "^", "CNN": "s", "mamba6mA": "D"}
BASE_INDEX = {base: index for index, base in enumerate("ACGU")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--k_values", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def clean_sequence(value: object, window_len: int = 21) -> str:
    sequence = str(value).strip().upper().replace("T", "U")
    sequence = "".join(base if base in BASE_INDEX else "N" for base in sequence)
    if len(sequence) != window_len:
        raise ValueError(f"Expected {window_len} nt, got {len(sequence)}: {sequence}")
    return sequence


def load_split(split: str) -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(DATA / f"{split}.csv")
    missing = {"_site_sequence", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{split}.csv is missing columns: {sorted(missing)}")
    sequences = [clean_sequence(value) for value in frame["_site_sequence"]]
    labels = frame["label"].to_numpy(dtype=np.int64)
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError(f"Unexpected labels in {split}.csv")
    return sequences, labels


def sparse_kmer_frequency(sequences: list[str], k: int) -> sparse.csr_matrix:
    """Encode exact kNF, ignoring N-containing windows and normalizing valid windows."""
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row, sequence in enumerate(sequences):
        counts: dict[int, int] = {}
        total = 0
        for start in range(len(sequence) - k + 1):
            index = 0
            for base in sequence[start : start + k]:
                digit = BASE_INDEX.get(base)
                if digit is None:
                    break
                index = index * 4 + digit
            else:
                counts[index] = counts.get(index, 0) + 1
                total += 1
        if total:
            for index, count in counts.items():
                rows.append(row)
                columns.append(index)
                values.append(count / total)
    return sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, columns)),
        shape=(len(sequences), 4**k),
        dtype=np.float32,
    )


def restrict_to_seen(matrix: sparse.csr_matrix, seen: np.ndarray) -> sparse.csr_matrix:
    """Remap theoretical k-mer columns to columns observed in the training split."""
    coo = matrix.tocoo()
    positions = np.searchsorted(seen, coo.col)
    clipped = np.minimum(positions, len(seen) - 1)
    keep = (positions < len(seen)) & (seen[clipped] == coo.col)
    return sparse.csr_matrix(
        (coo.data[keep], (coo.row[keep], positions[keep])),
        shape=(matrix.shape[0], len(seen)),
        dtype=np.float32,
    )


def matrix_to_tokens(matrix: sparse.csr_matrix, width: int, padding_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.full((matrix.shape[0], width), padding_index, dtype=np.int64)
    frequencies = np.zeros((matrix.shape[0], width), dtype=np.float32)
    mask = np.zeros((matrix.shape[0], width), dtype=np.float32)
    for row in range(matrix.shape[0]):
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        count = min(stop - start, width)
        if count:
            ids[row, :count] = matrix.indices[start : start + count]
            frequencies[row, :count] = matrix.data[start : start + count]
            mask[row, :count] = 1.0
    return ids, frequencies, mask


def make_loader(
    tokens: tuple[np.ndarray, np.ndarray, np.ndarray], labels: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    ids, frequencies, mask = tokens
    dataset = TensorDataset(
        torch.from_numpy(ids),
        torch.from_numpy(frequencies),
        torch.from_numpy(mask),
        torch.from_numpy(labels.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


class SparseKnfBase(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.padding_index = vocab_size
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=self.padding_index)
        self.input_norm = nn.LayerNorm(d_model)

    def token_vectors(self, ids: torch.Tensor, frequencies: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        vectors = self.input_norm(self.embedding(ids))
        valid_count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        scale = frequencies * valid_count
        return vectors * scale.unsqueeze(-1) * mask.unsqueeze(-1)


class SparseKnfDNN(SparseKnfBase):
    def __init__(self, vocab_size: int, d_model: int, dropout: float) -> None:
        super().__init__(vocab_size, d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, ids: torch.Tensor, frequencies: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        vectors = self.token_vectors(ids, frequencies, mask)
        valid_count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = vectors.sum(dim=1) / valid_count
        return self.classifier(pooled).squeeze(-1)


class SparseKnfCNN(SparseKnfBase):
    def __init__(self, vocab_size: int, d_model: int, dropout: float) -> None:
        super().__init__(vocab_size, d_model)
        self.branches = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=kernel // 2, bias=False) for kernel in (3, 5)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, ids: torch.Tensor, frequencies: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        vectors = self.token_vectors(ids, frequencies, mask).transpose(1, 2)
        expanded_mask = mask.unsqueeze(1)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = []
        for convolution in self.branches:
            hidden = F.gelu(convolution(vectors)) * expanded_mask
            pooled.append(hidden.sum(dim=2) / denominator)
        return self.classifier(torch.cat(pooled, dim=1)).squeeze(-1)


class SparseKnfMamba6MA(SparseKnfBase):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__(vocab_size, d_model)
        config = HybridConfig(
            input_dim=d_model,
            seq_len=seq_len,
            d_model=d_model,
            n_layer=1,
            expand=2,
            d_state=8,
            dt_rank=max(1, math.ceil(d_model / 16)),
            dropout=dropout,
            classifier_hidden=[64],
            input_norm=True,
            projection_hidden=0,
        )
        self.config = config
        self.layers3 = nn.ModuleList([ResidualBlock(config, 3)])
        self.layers5 = nn.ModuleList([ResidualBlock(config, 5)])
        self.layers7 = nn.ModuleList([ResidualBlock(config, 7)])
        self.norm_f = RMSNorm(d_model)
        self.fusion = DynamicFusion(d_model)
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def run_branch(self, values: torch.Tensor, layers: nn.ModuleList) -> torch.Tensor:
        for layer in layers:
            values = layer(values)
        return self.norm_f(values)

    def forward(self, ids: torch.Tensor, frequencies: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        vectors = self.token_vectors(ids, frequencies, mask)
        x3 = self.run_branch(vectors, self.layers3)
        x5 = self.run_branch(vectors, self.layers5)
        x7 = self.run_branch(vectors, self.layers7)
        fused = self.fusion(x3, x5, x7) * mask.unsqueeze(-1)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = fused.sum(dim=1) / denominator
        return self.classifier(pooled).squeeze(-1)


def make_deep_model(name: str, vocab_size: int, seq_len: int, args: argparse.Namespace) -> nn.Module:
    if name == "DNN":
        return SparseKnfDNN(vocab_size, args.d_model, args.dropout)
    if name == "CNN":
        return SparseKnfCNN(vocab_size, args.d_model, args.dropout)
    if name == "mamba6mA":
        return SparseKnfMamba6MA(vocab_size, seq_len, args.d_model, args.dropout)
    raise ValueError(name)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, labels = [], []
    for ids, frequencies, mask, y in loader:
        logits = model(ids.to(device), frequencies.to(device), mask.to(device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probabilities), np.concatenate(labels).astype(int)


def train_deep_model(
    name: str,
    vocab_size: int,
    seq_len: int,
    train_tokens: tuple[np.ndarray, np.ndarray, np.ndarray],
    valid_tokens: tuple[np.ndarray, np.ndarray, np.ndarray],
    test_tokens: tuple[np.ndarray, np.ndarray, np.ndarray],
    labels: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, torch.Tensor]]:
    set_seed(args.seed)
    model = make_deep_model(name, vocab_size, seq_len, args).to(device)
    loaders = {
        "train": make_loader(train_tokens, labels["train"], args.batch_size, True),
        "train_eval": make_loader(train_tokens, labels["train"], args.batch_size, False),
        "valid": make_loader(valid_tokens, labels["valid"], args.batch_size, False),
        "test": make_loader(test_tokens, labels["test"], args.batch_size, False),
    }
    positives = int(labels["train"].sum())
    negatives = len(labels["train"]) - positives
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / max(positives, 1), device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_auc, best_epoch, best_state = float("-inf"), 0, None
    patience = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for ids, frequencies, mask, y in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids.to(device), frequencies.to(device), mask.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        valid_probability, valid_y = predict(model, loaders["valid"], device)
        valid_auc = float(roc_auc_score(valid_y, valid_probability))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_AUROC": valid_auc})
        if valid_auc > best_auc + 1e-6:
            best_auc = valid_auc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if epoch == 1 or epoch % 5 == 0 or patience >= args.patience:
            print(f"  {name} epoch={epoch:02d} loss={np.mean(losses):.4f} valid_auc={valid_auc:.4f} patience={patience}/{args.patience}", flush=True)
        if patience >= args.patience:
            break
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {name}")
    model.load_state_dict(best_state)
    predictions = {}
    result: dict[str, object] = {
        "best_epoch": best_epoch,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
    }
    for split, loader_name in (("train", "train_eval"), ("valid", "valid"), ("test", "test")):
        probability, y = predict(model, loaders[loader_name], device)
        predictions[split] = probability
        result[f"{split}_AUROC"] = float(roc_auc_score(y, probability))
        result[f"{split}_AUPRC"] = float(average_precision_score(y, probability))
    return result, predictions, best_state


def train_rf(
    matrices: dict[str, sparse.csr_matrix], labels: dict[str, np.ndarray], k: int, args: argparse.Namespace
) -> tuple[dict[str, object], dict[str, np.ndarray], RandomForestClassifier]:
    params = {
        "n_estimators": args.n_estimators,
        "max_features": max(1, 2 * k),
        "min_samples_leaf": 1,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "random_state": args.seed,
    }
    model = RandomForestClassifier(**params).fit(matrices["train"], labels["train"])
    predictions = {split: model.predict_proba(matrix)[:, 1] for split, matrix in matrices.items()}
    result: dict[str, object] = {"best_epoch": None, "parameter_count": None, "rf_params": params}
    for split, probability in predictions.items():
        result[f"{split}_AUROC"] = float(roc_auc_score(labels[split], probability))
        result[f"{split}_AUPRC"] = float(average_precision_score(labels[split], probability))
    return result, predictions, model


def completed_metrics(out_dir: Path) -> pd.DataFrame:
    paths = sorted(out_dir.glob("*/k*/metrics.json"))
    rows = []
    for path in paths:
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def write_summary_and_plot(out_dir: Path) -> None:
    summary = completed_metrics(out_dir)
    if summary.empty:
        return
    summary = summary.sort_values(["model", "k"])
    summary.to_csv(out_dir / "knf_1to10_four_models_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=True)
    for axis, split, title in zip(axes, ("valid", "test"), ("Validation AUROC", "Test AUROC")):
        for model_name in MODEL_NAMES:
            part = summary[summary["model"].eq(model_name)].sort_values("k")
            if part.empty:
                continue
            axis.plot(part["k"], part[f"{split}_AUROC"], label=model_name, color=COLORS[model_name], marker=MARKERS[model_name], linewidth=2, markersize=5.5)
        axis.set_xticks(range(1, 11))
        axis.set_xlabel("k in kNF")
        axis.set_title(title)
        axis.grid(axis="y", color="#D9DEE3", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("AUROC")
    axes[0].legend(frameon=False)
    values = summary[["valid_AUROC", "test_AUROC"]].to_numpy(dtype=float)
    lower = max(0.45, math.floor((np.nanmin(values) - 0.03) * 20) / 20)
    upper = min(1.0, math.ceil((np.nanmax(values) + 0.03) * 20) / 20)
    axes[0].set_ylim(lower, upper)
    fig.suptitle("1NF-10NF performance across four classifiers (seed=42)")
    fig.tight_layout()
    fig.savefig(out_dir / "knf_1to10_four_models_auroc_curve.png", dpi=300)
    fig.savefig(out_dir / "knf_1to10_four_models_auroc_curve.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    k_values = [int(value) for value in args.k_values.split(",") if value.strip()]
    model_names = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = set(model_names) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    set_seed(args.seed)
    device = resolve_device(args.device)
    torch.set_num_threads(min(16, max(1, torch.get_num_threads())))
    split_data = {split: load_split(split) for split in ("train", "valid", "test")}
    labels = {split: values[1] for split, values in split_data.items()}
    sequences = {split: values[0] for split, values in split_data.items()}
    print(f"Device={device}; k={k_values}; models={model_names}", flush=True)

    manifest = {
        "encoding": "one kNF block per experiment; frequencies use valid A/C/G/U windows only",
        "high_order_policy": "remove k-mers never observed in training; do not renormalize after removal",
        "token_order": "nonzero k-mers sorted by canonical base-4 A/C/G/U feature index",
        "selection": "deep models early-stopped by validation AUROC; RF uses a fixed predeclared configuration",
        "seed": args.seed,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for k in k_values:
        print(f"\n=== {k}NF ===", flush=True)
        theoretical = {split: sparse_kmer_frequency(values, k) for split, values in sequences.items()}
        seen = np.unique(theoretical["train"].indices)
        matrices = {split: restrict_to_seen(matrix, seen) for split, matrix in theoretical.items()}
        vocab_size = len(seen)
        seq_len = 21 - k + 1
        tokens = {split: matrix_to_tokens(matrix, seq_len, vocab_size) for split, matrix in matrices.items()}
        print(f"theoretical_features={4**k:,}; train_seen={vocab_size:,}; max_tokens={seq_len}", flush=True)

        for model_name in model_names:
            run_dir = args.out_dir / model_name / f"k{k}"
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists() and not args.force:
                print(f"  {model_name}: already complete", flush=True)
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"  training {model_name}", flush=True)
            if model_name == "RF":
                result, predictions, model = train_rf(matrices, labels, k, args)
                import joblib

                joblib.dump(model, run_dir / "best_model.joblib", compress=3)
            else:
                result, predictions, state = train_deep_model(
                    model_name, vocab_size, seq_len, tokens["train"], tokens["valid"], tokens["test"], labels, args, device
                )
                torch.save(state, run_dir / "best_model.pt")
                pd.DataFrame(result.pop("history")).to_csv(run_dir / "training_history.csv", index=False)
            for split, probability in predictions.items():
                pd.DataFrame({"label": labels[split], "probability": probability}).to_csv(run_dir / f"{split}_predictions.csv", index=False)
            record = {
                "model": model_name,
                "k": k,
                "encoding": f"{k}NF",
                "theoretical_features": 4**k,
                "train_seen_features": vocab_size,
                "seed": args.seed,
                **result,
            }
            metrics_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"  {model_name}: valid={record['valid_AUROC']:.4f} test={record['test_AUROC']:.4f}", flush=True)
            write_summary_and_plot(args.out_dir)
    write_summary_and_plot(args.out_dir)


if __name__ == "__main__":
    main()
