#!/usr/bin/env python3
"""Train a mamba6mA-style classifier on KNF 2/3/4-mer encoded 21 nt windows."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "latest_data" / "cdhit90_n8_augmented_21nt_split" / "model_inputs_21nt"
OUT_DIR = PROJECT_ROOT / "latest_data" / "cdhit90_n8_augmented_21nt_split" / "model_runs" / "knf_mamba6ma_21nt_k234_grouped"
BASES = "ACGU"
FIXED_THRESHOLD = 0.5


@dataclass
class HybridConfig:
    input_dim: int
    seq_len: int
    d_model: int
    n_layer: int
    expand: int
    d_state: int
    dt_rank: int
    dropout: float
    classifier_hidden: list[int]
    input_norm: bool
    projection_hidden: int

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class DynamicPSConvNet(nn.Module):
    def __init__(self, window_size: int, inner_dim: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.window_size = window_size
        self.inner_dim = inner_dim
        self.seq_len = seq_len
        self.pad_len = window_size // 2
        self.dense_layer_net = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(window_size * inner_dim, inner_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(seq_len)
            ]
        )

    def forward(self, input_mat: torch.Tensor) -> torch.Tensor:
        h_d = F.pad(input_mat, (self.pad_len, self.pad_len), "constant", 0)
        h_d = torch.transpose(h_d, 1, 2)
        pieces = []
        for index in range(self.pad_len, self.seq_len + self.pad_len):
            segment = h_d[:, index - self.pad_len : index + self.pad_len + 1, :]
            segment_flat = segment.reshape(-1, self.window_size * self.inner_dim)
            pieces.append(self.dense_layer_net[index - self.pad_len](segment_flat))
        return torch.stack(pieces, dim=1)


class DynamicMambaBlock(nn.Module):
    def __init__(self, config: HybridConfig, window_size: int) -> None:
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.d_model, config.d_inner * 2, bias=False)
        self.conv = DynamicPSConvNet(window_size, config.d_inner, config.seq_len, config.dropout)
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + config.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        a = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split([self.config.d_inner, self.config.d_inner], dim=-1)
        x = x.transpose(1, 2)
        x = F.silu(self.conv(x))
        y = self.ssm(x)
        y = y * F.silu(res)
        return self.out_proj(y)

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        _, n_state = self.A_log.shape
        a = -torch.exp(self.A_log.float())
        d = self.D.float()
        x_dbl = self.x_proj(x)
        delta, b, c = x_dbl.split([self.config.dt_rank, n_state, n_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        return self.selective_scan(x, delta, a, b, c, d)

    @staticmethod
    def selective_scan(
        u: torch.Tensor,
        delta: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, d_in = u.shape
        n_state = a.shape[1]
        delta_a = torch.exp(torch.einsum("bld,dn->bldn", delta, a))
        delta_b_u = torch.einsum("bld,bln,bld->bldn", delta, b, u)
        state = torch.zeros((batch_size, d_in, n_state), device=u.device, dtype=u.dtype)
        outputs = []
        for index in range(seq_len):
            state = delta_a[:, index] * state + delta_b_u[:, index]
            outputs.append(torch.einsum("bdn,bn->bd", state, c[:, index, :]))
        y = torch.stack(outputs, dim=1)
        return y + u * d


class ResidualBlock(nn.Module):
    def __init__(self, config: HybridConfig, window_size: int) -> None:
        super().__init__()
        self.mixer = DynamicMambaBlock(config, window_size)
        self.norm = RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(self.norm(x)) + x


class DynamicFusion(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.wa = nn.Parameter(torch.randn(1, 1, d_model))
        self.wb = nn.Parameter(torch.randn(1, 1, d_model))
        self.wc = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, y3: torch.Tensor, y5: torch.Tensor, y7: torch.Tensor) -> torch.Tensor:
        ya = torch.sigmoid(self.wa * y3)
        yb = torch.sigmoid(self.wb * y5)
        yc = torch.sigmoid(self.wc * y7)
        weights = F.softmax(torch.cat([ya, yb, yc], dim=-1), dim=-1)
        weights = weights.view(weights.size(0), weights.size(1), 3, -1)
        return weights[:, :, 0] * y3 + weights[:, :, 1] * y5 + weights[:, :, 2] * y7


class KnfMamba6MA(nn.Module):
    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        projection_layers: list[nn.Module] = []
        if config.input_norm:
            projection_layers.append(nn.LayerNorm(config.input_dim))
        if config.projection_hidden > 0:
            projection_layers.extend(
                [
                    nn.Linear(config.input_dim, config.projection_hidden),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.projection_hidden, config.d_model),
                ]
            )
        else:
            projection_layers.append(nn.Linear(config.input_dim, config.d_model))
        projection_layers.extend([nn.GELU(), nn.Dropout(config.dropout)])
        self.input_projection = nn.Sequential(*projection_layers)
        self.layers3 = nn.ModuleList([ResidualBlock(config, 3) for _ in range(config.n_layer)])
        self.layers5 = nn.ModuleList([ResidualBlock(config, 5) for _ in range(config.n_layer)])
        self.layers7 = nn.ModuleList([ResidualBlock(config, 7) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_model)
        self.fusion_module = DynamicFusion(config.d_model)

        classifier_layers: list[nn.Module] = []
        in_dim = config.seq_len * config.d_model
        for hidden_dim in config.classifier_hidden:
            classifier_layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(config.dropout)])
            in_dim = hidden_dim
        classifier_layers.append(nn.Linear(in_dim, 1))
        self.classifier = nn.Sequential(*classifier_layers)

    def _run_branch(self, x: torch.Tensor, layers: nn.ModuleList) -> torch.Tensor:
        for layer in layers:
            x = layer(x)
        return self.norm_f(x)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(features.float())
        x3 = self._run_branch(x, self.layers3)
        x5 = self._run_branch(x, self.layers5)
        x7 = self._run_branch(x, self.layers7)
        x = self.fusion_module(x3, x5, x7)
        return self.classifier(torch.flatten(x, start_dim=1)).squeeze(-1)


def parse_hidden_dims(value: str) -> list[int]:
    if not value.strip():
        return []
    dims = [int(item) for item in value.split(",") if item.strip()]
    if any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("hidden dimensions must be positive")
    return dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", type=Path, default=SPLIT_DIR / "train.csv")
    parser.add_argument("--val_csv", type=Path, default=SPLIT_DIR / "valid.csv")
    parser.add_argument("--test_csv", type=Path, default=SPLIT_DIR / "test.csv")
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--sequence_column", default="_site_sequence")
    parser.add_argument("--id_column", default="_site_id")
    parser.add_argument("--window_len", type=int, default=21)
    parser.add_argument("--k_values", default="2,3,4")
    parser.add_argument(
        "--tokenization",
        choices=["k_group", "feature"],
        default="k_group",
        help="k_group uses one padded vector token per k; feature uses one scalar token per k-mer.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d_state", type=int, default=8)
    parser.add_argument("--dt_rank", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--classifier_hidden", type=parse_hidden_dims, default=[128])
    parser.add_argument("--projection_hidden", type=int, default=0)
    parser.add_argument(
        "--input_norm",
        action="store_true",
        help="Apply LayerNorm over each token vector. Default is off because KNF tokens are scalar values.",
    )
    parser.add_argument("--pos_weight", choices=["none", "auto"], default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def clean_sequence(seq: str) -> str:
    seq = str(seq).strip().upper().replace("T", "U")
    return "".join(base if base in BASES else "N" for base in seq)


def read_split(path: Path, split_name: str, sequence_column: str, id_column: str, window_len: int) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = {"label", sequence_column} - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    out = data.copy()
    out["id"] = out[id_column].astype(str) if id_column in out.columns else [f"{split_name}_{i + 1:06d}" for i in range(len(out))]
    out["sequence"] = out[sequence_column].map(clean_sequence)
    out["label"] = out["label"].astype(int)
    bad = out["sequence"].str.len() != window_len
    if bad.any():
        raise ValueError(f"{path} has {int(bad.sum())} sequences not {window_len} nt")
    return out


def make_kmer_groups(k_values: list[int]) -> dict[int, list[str]]:
    groups = {}
    for k in k_values:
        groups[k] = ["".join(parts) for parts in itertools.product(BASES, repeat=k)]
    return groups


def knf_counts_for_k(seq: str, kmers: list[str]) -> np.ndarray:
    k = len(kmers[0])
    counts = dict.fromkeys(kmers, 0)
    total = 0
    for pos in range(0, len(seq) - k + 1):
        word = seq[pos : pos + k]
        if "N" in word:
            continue
        if word in counts:
            counts[word] += 1
            total += 1
    denom = max(total, 1)
    return np.asarray([counts[kmer] / denom for kmer in kmers], dtype=np.float32)


def encode_knf_feature_tokens(data: pd.DataFrame, groups: dict[int, list[str]]) -> torch.Tensor:
    rows = []
    for seq in data["sequence"]:
        rows.append(np.concatenate([knf_counts_for_k(seq, groups[k]) for k in groups], axis=0))
    vectors = np.vstack(rows).astype(np.float32)
    return torch.tensor(vectors[:, :, None], dtype=torch.float32)


def encode_knf_k_group_tokens(data: pd.DataFrame, groups: dict[int, list[str]]) -> torch.Tensor:
    max_dim = max(len(kmers) for kmers in groups.values())
    encoded = np.zeros((len(data), len(groups), max_dim), dtype=np.float32)
    for row_index, seq in enumerate(data["sequence"]):
        for token_index, k in enumerate(groups):
            counts = knf_counts_for_k(seq, groups[k])
            encoded[row_index, token_index, : len(counts)] = counts
    return torch.tensor(encoded, dtype=torch.float32)


def encode_knf(data: pd.DataFrame, groups: dict[int, list[str]], tokenization: str) -> torch.Tensor:
    if tokenization == "feature":
        return encode_knf_feature_tokens(data, groups)
    if tokenization == "k_group":
        return encode_knf_k_group_tokens(data, groups)
    raise ValueError(f"Unsupported tokenization: {tokenization}")


def standardize(train_x: torch.Tensor, *others: torch.Tensor) -> tuple[torch.Tensor, ...]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return tuple(((x - mean) / std).float() for x in (train_x, *others))


def metric_dict(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    predictions = (probabilities >= FIXED_THRESHOLD).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "AUROC": float(roc_auc_score(labels, probabilities)),
        "AUPRC": float(average_precision_score(labels, probabilities)),
        "Accuracy": float(accuracy_score(labels, predictions)),
        "Precision": float(precision_score(labels, predictions, zero_division=0)),
        "Recall": float(recall_score(labels, predictions, zero_division=0)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, predictions)),
        "Specificity": float(tn / max(tn + fp, 1)),
        "Sensitivity": float(tp / max(tp + fn, 1)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels_out = [], []
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        logits = model(features).view(-1)
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels_out.append(labels.numpy().astype(np.int64))
    return np.concatenate(probs), np.concatenate(labels_out)


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def make_loader(x: torch.Tensor, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(x, torch.tensor(y, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    weighted_loss = 0.0
    n_samples = 0
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1)
        logits = model(features).view(-1)
        loss = criterion(logits, labels)
        weighted_loss += float(loss.item()) * labels.numel()
        n_samples += labels.numel()
    return weighted_loss / max(n_samples, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict, list[dict[str, float | int]]]:
    model.to(device)
    positives = int(train_labels.sum())
    negatives = int(len(train_labels) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1)], dtype=torch.float32, device=device) if args.pos_weight == "auto" else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best = {"epoch": 0, "val_MCC": float("-inf"), "val_AUROC": float("-inf")}
    history = []
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))

        val_probs, val_labels = predict(model, val_loader, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        val_metrics = metric_dict(val_labels, val_probs)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        is_better = val_metrics["MCC"] > best["val_MCC"] or (
            val_metrics["MCC"] == best["val_MCC"] and val_metrics["AUROC"] > best["val_AUROC"]
        )
        if is_better:
            best_state = clone_state_dict(model)
            best = {"epoch": epoch, **{f"val_{key}": value for key, value in val_metrics.items()}}
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch == 1 or epoch % 5 == 0 or patience_counter >= args.patience:
            print(
                f"epoch={epoch:03d} loss={np.mean(losses):.4f} val_loss={val_loss:.4f} "
                f"val_auc={val_metrics['AUROC']:.4f} val_mcc={val_metrics['MCC']:.4f} "
                f"patience={patience_counter}/{args.patience}",
                flush=True,
            )
        if patience_counter >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("No best checkpoint selected")
    model.load_state_dict(best_state)
    return model, best, history


def save_predictions(path: Path, data: pd.DataFrame, probabilities: np.ndarray) -> None:
    out = data.copy()
    out["probability"] = probabilities
    out["predicted_label"] = (probabilities >= FIXED_THRESHOLD).astype(np.int64)
    out.to_csv(path, index=False)


def split_summary(data: pd.DataFrame) -> dict[str, int]:
    labels = data["label"].value_counts().to_dict()
    return {
        "size": int(len(data)),
        "positive": int(labels.get(1, 0)),
        "negative": int(labels.get(0, 0)),
        "contains_N": int(data["sequence"].str.contains("N", regex=False).sum()),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    k_values = [int(item) for item in args.k_values.split(",") if item.strip()]
    kmer_groups = make_kmer_groups(k_values)

    train = read_split(args.train_csv, "train", args.sequence_column, args.id_column, args.window_len)
    val = read_split(args.val_csv, "valid", args.sequence_column, args.id_column, args.window_len)
    test = read_split(args.test_csv, "test", args.sequence_column, args.id_column, args.window_len)
    y_train = train["label"].to_numpy(dtype=np.int64)
    y_val = val["label"].to_numpy(dtype=np.int64)
    y_test = test["label"].to_numpy(dtype=np.int64)

    x_train = encode_knf(train, kmer_groups, args.tokenization)
    x_val = encode_knf(val, kmer_groups, args.tokenization)
    x_test = encode_knf(test, kmer_groups, args.tokenization)
    x_train, x_val, x_test = standardize(x_train, x_val, x_test)

    config = HybridConfig(
        input_dim=int(x_train.shape[-1]),
        seq_len=int(x_train.shape[1]),
        d_model=args.d_model,
        n_layer=args.n_layer,
        expand=args.expand,
        d_state=args.d_state,
        dt_rank=args.dt_rank if args.dt_rank > 0 else math.ceil(args.d_model / 16),
        dropout=args.dropout,
        classifier_hidden=args.classifier_hidden,
        input_norm=args.input_norm,
        projection_hidden=args.projection_hidden,
    )
    model = KnfMamba6MA(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device}", flush=True)
    print(
        f"KNF k-values: {k_values}; tokenization={args.tokenization}; "
        f"input_shape={tuple(x_train.shape[1:])}",
        flush=True,
    )
    print(f"Train/Val/Test: {len(train)}/{len(val)}/{len(test)}", flush=True)
    print(f"Model config: {asdict(config)}", flush=True)
    print(f"Trainable parameters: {parameter_count:,}", flush=True)

    train_loader = make_loader(x_train, y_train, args.batch_size, True)
    train_eval_loader = make_loader(x_train, y_train, args.batch_size, False)
    val_loader = make_loader(x_val, y_val, args.batch_size, False)
    test_loader = make_loader(x_test, y_test, args.batch_size, False)
    model, best, history = train_model(model, train_loader, val_loader, y_train, args, device)

    train_probs, train_labels = predict(model, train_eval_loader, device)
    val_probs, val_labels = predict(model, val_loader, device)
    test_probs, test_labels = predict(model, test_loader, device)
    train_metrics = metric_dict(train_labels, train_probs)
    val_metrics = metric_dict(val_labels, val_probs)
    test_metrics = metric_dict(test_labels, test_probs)

    save_predictions(args.out_dir / "train_predictions.csv", train, train_probs)
    save_predictions(args.out_dir / "val_predictions.csv", val, val_probs)
    save_predictions(args.out_dir / "test_predictions.csv", test, test_probs)
    pd.DataFrame(history).to_csv(args.out_dir / "training_history.csv", index=False)
    order_lines = []
    for token_index, (k, kmers) in enumerate(kmer_groups.items()):
        if args.tokenization == "k_group":
            order_lines.append(f"token_{token_index}\tk={k}\t" + ",".join(kmers))
        else:
            order_lines.extend(kmers)
    (args.out_dir / "knf_feature_order.txt").write_text("\n".join(order_lines) + "\n", encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "best": best, "config": asdict(config), "args": vars(args)}, args.out_dir / "best_model.pt")

    results = {
        "model": "KNF k-mer frequency tokens + mamba6mA-style classifier",
        "dataset": "cdhit90_n8_augmented_21nt_split",
        "encoding": {
            "name": "KNF",
            "k_values": k_values,
            "tokenization": args.tokenization,
            "input_shape_per_sample": list(x_train.shape[1:]),
            "feature_count_total": int(sum(len(kmers) for kmers in kmer_groups.values())),
            "feature_order_file": str(args.out_dir / "knf_feature_order.txt"),
            "notes": (
                "k_group: one padded vector token per k-mer size. "
                "feature: each k-mer frequency is one scalar token, ordered by k then lexicographic A/C/G/U."
            ),
        },
        "threshold": FIXED_THRESHOLD,
        "selection_metric": "val_MCC_at_threshold_0.5",
        "split_summaries": {"train": split_summary(train), "val": split_summary(val), "test": split_summary(test)},
        "config": asdict(config),
        "trainable_parameters": int(parameter_count),
        "best_epoch": int(best["epoch"]),
        "best_validation": best,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "hyperparameters": {
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "pos_weight": args.pos_weight,
        },
    }
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (args.out_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", *train_metrics.keys()])
        writer.writeheader()
        writer.writerow({"split": "train", **train_metrics})
        writer.writerow({"split": "val", **val_metrics})
        writer.writerow({"split": "test", **test_metrics})
    print(json.dumps({"best_validation": best, "test_metrics": test_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
