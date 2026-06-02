#!/usr/bin/env python3
"""
RNA-FM + CNN for Glycosylation Site Prediction

Pipeline:
  sequence → RNA-FM (frozen) → token embeddings [batch, 51, hidden_dim]
  → CNN (kernel 3/5/7) → global max pooling → MLP → prediction

Compare against naive pooling methods (mean, max, cls, center_pm10)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RNAFM_ROOT = PROJECT_ROOT / "model" / "RNA-FM" / "RNA-FM-main"
RNAFM_WEIGHTS = PROJECT_ROOT / "model" / "RNA-FM" / "RNA-FM_pretrained.pth"
DATA_DIR = PROJECT_ROOT / "data" / "4974_pos_neg"
VALID_BASES = set("ACGUN")


@dataclass
class FastaRecords:
    ids: list[str]
    seqs: list[str]
    labels: np.ndarray


class RNADataset(Dataset):
    def __init__(self, records: FastaRecords) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records.seqs)

    def __getitem__(self, index: int) -> tuple[str, int, str]:
        return self.records.seqs[index], int(self.records.labels[index]), self.records.ids[index]


# ============================================================================
# CNN-based Classifiers
# ============================================================================

class CNNClassifier(nn.Module):
    """Single-scale CNN classifier"""
    def __init__(self, input_dim: int, kernel_size: int = 3, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_dim]
        x = x.transpose(1, 2)  # [batch, hidden_dim, seq_len]
        x = self.conv(x)  # [batch, hidden_dim, seq_len]
        x = self.bn(x)
        x = F.relu(x)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # global max pooling -> [batch, hidden_dim]
        x = self.dropout(x)
        x = self.fc(x)  # [batch, 1]
        return x.squeeze(-1)


class MultiScaleCNNClassifier(nn.Module):
    """Multi-scale CNN with parallel kernels"""
    def __init__(self, input_dim: int, kernel_sizes: list[int] = [3, 5, 7],
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * len(kernel_sizes), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_dim]
        x = x.transpose(1, 2)  # [batch, hidden_dim, seq_len]

        # Apply each conv + global max pooling
        conv_outputs = []
        for conv, bn in zip(self.convs, self.bns):
            conv_out = conv(x)  # [batch, hidden_dim, seq_len]
            conv_out = bn(conv_out)
            conv_out = F.relu(conv_out)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # [batch, hidden_dim]
            conv_outputs.append(pooled)

        # Concatenate all scales
        x = torch.cat(conv_outputs, dim=1)  # [batch, hidden_dim * num_kernels]
        x = self.dropout(x)
        x = self.fc(x)  # [batch, 1]
        return x.squeeze(-1)


class MLPClassifier(nn.Module):
    """Simple MLP for pooled embeddings"""
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ============================================================================
# RNA-FM Loading and Feature Extraction
# ============================================================================

def lazy_import_rnafm(rnafm_root: Path):
    root_str = str(rnafm_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    import fm
    return fm


def load_rnafm(weights_path: Path, rnafm_root: Path, device: torch.device):
    fm = lazy_import_rnafm(rnafm_root)
    model, alphabet = fm.pretrained.rna_fm_t12(str(weights_path))
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, alphabet


def clean_sequence(seq: str) -> str:
    seq = seq.strip().upper().replace("T", "U")
    return "".join(base if base in VALID_BASES else "N" for base in seq)


def read_fasta(path: Path) -> FastaRecords:
    ids: list[str] = []
    seqs: list[str] = []
    labels: list[int] = []
    header: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        nonlocal header, chunks
        if header is None:
            return
        seq = clean_sequence("".join(chunks))
        if not seq:
            return
        parts = header.split("|")
        seq_id = parts[0].lstrip(">")

        if seq_id.startswith("window_"):
            label = 1
        elif seq_id.startswith("neg_"):
            label = 0
        else:
            raise ValueError(f"Unknown ID prefix: {header}")

        ids.append(seq_id)
        seqs.append(seq)
        labels.append(label)
        header = None
        chunks = []

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line
                chunks = []
            else:
                chunks.append(line)
        flush()

    if not seqs:
        raise ValueError(f"No sequences in {path}")
    return FastaRecords(ids=ids, seqs=seqs, labels=np.asarray(labels, dtype=np.int64))


def collate_batch(batch, batch_converter):
    seqs, labels, ids = zip(*batch)
    raw_batch = list(zip(ids, seqs))
    _labels, _strs, tokens = batch_converter(raw_batch)
    return tokens, torch.tensor(labels, dtype=torch.float32), list(ids)


def valid_token_mask(tokens: torch.Tensor, alphabet) -> torch.Tensor:
    return (tokens != alphabet.padding_idx) & (tokens != alphabet.cls_idx) & (tokens != alphabet.eos_idx)


# ============================================================================
# Pooling Methods
# ============================================================================

def mean_pooling(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = embeddings * mask.unsqueeze(-1)
    return masked.sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)


def max_pooling(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = embeddings.masked_fill(~mask.unsqueeze(-1), -torch.inf)
    pooled = masked.max(dim=1).values
    return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


def cls_pooling(embeddings: torch.Tensor) -> torch.Tensor:
    return embeddings[:, 0, :]


def center_pm10_pooling(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.sum(dim=1)
    center_vectors = []
    for i, length in enumerate(lengths.tolist()):
        center_pos = 1 + length // 2
        start = max(1, center_pos - 10)
        end = min(1 + length, center_pos + 10 + 1)
        center_vectors.append(embeddings[i, start:end, :].mean(dim=0))
    return torch.stack(center_vectors, dim=0)


# ============================================================================
# Training and Evaluation
# ============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def extract_embeddings(records: FastaRecords, rnafm_model, alphabet, batch_size: int,
                       device: torch.device, repr_layer: int = 12) -> torch.Tensor:
    """Extract token-level embeddings from RNA-FM"""
    batch_converter = alphabet.get_batch_converter()
    loader = DataLoader(
        RNADataset(records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_batch(batch, batch_converter),
    )

    all_embeddings = []
    for tokens, _, _ in loader:
        tokens = tokens.to(device)
        outputs = rnafm_model(tokens, repr_layers=[repr_layer], return_contacts=False)
        embeddings = outputs["representations"][repr_layer]  # [batch, seq_len, hidden_dim]
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)  # [n_samples, seq_len, hidden_dim]


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for embeddings, labels in train_loader:
        embeddings, labels = embeddings.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        logits = model(embeddings)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    preds = (all_probs >= 0.5).astype(int)

    metrics = {
        "AUROC": float(roc_auc_score(all_labels, all_probs)),
        "AUPRC": float(average_precision_score(all_labels, all_probs)),
        "Accuracy": float(accuracy_score(all_labels, preds)),
        "Precision": float(precision_score(all_labels, preds, zero_division=0)),
        "Recall": float(recall_score(all_labels, preds, zero_division=0)),
        "F1": float(f1_score(all_labels, preds, zero_division=0)),
        "MCC": float(matthews_corrcoef(all_labels, preds)),
    }

    return metrics, all_probs, all_labels


def plot_curves(labels, probs, out_dir, method_name):
    """Plot ROC and PR curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auroc = roc_auc_score(labels, probs)
    ax1.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curve - {method_name}')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # PR curve
    precision, recall, _ = precision_recall_curve(labels, probs)
    auprc = average_precision_score(labels, probs)
    ax2.plot(recall, precision, label=f'AUPRC = {auprc:.4f}')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'PR Curve - {method_name}')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f'{method_name}_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_fasta", type=Path, default=DATA_DIR / "train.fa")
    parser.add_argument("--val_fasta", type=Path, default=DATA_DIR / "val.fa")
    parser.add_argument("--test_fasta", type=Path, default=DATA_DIR / "test.fa")
    parser.add_argument("--weights", type=Path, default=RNAFM_WEIGHTS)
    parser.add_argument("--rnafm_root", type=Path, default=RNAFM_ROOT)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--method", required=True,
                       choices=["cnn3", "cnn5", "cnn7", "multiscale_cnn",
                               "mean", "max", "cls", "center_pm10"])
    parser.add_argument("--embed_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--repr_layer", type=int, default=12)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Method: {args.method}")
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    train_records = read_fasta(args.train_fasta)
    val_records = read_fasta(args.val_fasta)
    test_records = read_fasta(args.test_fasta)

    print(f"Train: {len(train_records.seqs)}, Val: {len(val_records.seqs)}, Test: {len(test_records.seqs)}")

    # Load RNA-FM
    print("\nLoading RNA-FM...")
    rnafm_model, alphabet = load_rnafm(args.weights, args.rnafm_root, device)

    # Extract embeddings
    print("\nExtracting embeddings...")
    train_emb = extract_embeddings(train_records, rnafm_model, alphabet, args.embed_batch_size, device, args.repr_layer)
    val_emb = extract_embeddings(val_records, rnafm_model, alphabet, args.embed_batch_size, device, args.repr_layer)
    test_emb = extract_embeddings(test_records, rnafm_model, alphabet, args.embed_batch_size, device, args.repr_layer)

    print(f"Embedding shape: {train_emb.shape}")  # [n_samples, seq_len, hidden_dim]

    # Prepare data based on method
    if args.method.startswith("cnn") or args.method == "multiscale_cnn":
        # Use token-level embeddings directly for CNN
        train_x = train_emb
        val_x = val_emb
        test_x = test_emb
        input_dim = train_emb.shape[2]  # hidden_dim

        # Create model
        if args.method == "cnn3":
            model = CNNClassifier(input_dim, kernel_size=3, hidden_dim=args.hidden_dim, dropout=args.dropout)
        elif args.method == "cnn5":
            model = CNNClassifier(input_dim, kernel_size=5, hidden_dim=args.hidden_dim, dropout=args.dropout)
        elif args.method == "cnn7":
            model = CNNClassifier(input_dim, kernel_size=7, hidden_dim=args.hidden_dim, dropout=args.dropout)
        elif args.method == "multiscale_cnn":
            model = MultiScaleCNNClassifier(input_dim, kernel_sizes=[3, 5, 7],
                                           hidden_dim=args.hidden_dim//3, dropout=args.dropout)
    else:
        # Apply pooling first for baseline methods
        print(f"\nApplying {args.method} pooling...")
        # Need to create mask for pooling
        # For simplicity, assume all sequences are valid (no padding in middle)
        train_mask = torch.ones(train_emb.shape[0], train_emb.shape[1], dtype=torch.bool)
        val_mask = torch.ones(val_emb.shape[0], val_emb.shape[1], dtype=torch.bool)
        test_mask = torch.ones(test_emb.shape[0], test_emb.shape[1], dtype=torch.bool)

        if args.method == "mean":
            train_x = mean_pooling(train_emb, train_mask)
            val_x = mean_pooling(val_emb, val_mask)
            test_x = mean_pooling(test_emb, test_mask)
        elif args.method == "max":
            train_x = max_pooling(train_emb, train_mask)
            val_x = max_pooling(val_emb, val_mask)
            test_x = max_pooling(test_emb, test_mask)
        elif args.method == "cls":
            train_x = cls_pooling(train_emb)
            val_x = cls_pooling(val_emb)
            test_x = cls_pooling(test_emb)
        elif args.method == "center_pm10":
            train_x = center_pm10_pooling(train_emb, train_mask)
            val_x = center_pm10_pooling(val_emb, val_mask)
            test_x = center_pm10_pooling(test_emb, test_mask)

        input_dim = train_x.shape[1]
        model = MLPClassifier(input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)

    model = model.to(device)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_x, torch.tensor(train_records.labels, dtype=torch.float32))
    val_dataset = torch.utils.data.TensorDataset(val_x, torch.tensor(val_records.labels, dtype=torch.float32))
    test_dataset = torch.utils.data.TensorDataset(test_x, torch.tensor(test_records.labels, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.train_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False)

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auroc = 0
    patience_counter = 0

    print("\nTraining...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate(model, val_loader, device)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f} - Val AUROC: {val_metrics['AUROC']:.4f}")

        # Early stopping
        if val_metrics['AUROC'] > best_val_auroc:
            best_val_auroc = val_metrics['AUROC']
            torch.save(model.state_dict(), args.out_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model and evaluate
    model.load_state_dict(torch.load(args.out_dir / "best_model.pt"))

    print("\nEvaluating...")
    val_metrics, val_probs, val_labels = evaluate(model, val_loader, device)
    test_metrics, test_probs, test_labels = evaluate(model, test_loader, device)

    # Plot curves
    plot_curves(test_labels, test_probs, args.out_dir, args.method)

    # Save results
    results = {
        "method": args.method,
        "model": model.__class__.__name__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "config": vars(args),
    }

    with (args.out_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save predictions
    np.save(args.out_dir / "test_predictions.npy", test_probs)

    print("\n=== Results ===")
    print(f"Val AUROC: {val_metrics['AUROC']:.4f}")
    print(f"Test AUROC: {test_metrics['AUROC']:.4f}")
    print(f"Test AUPRC: {test_metrics['AUPRC']:.4f}")
    print(f"Test F1: {test_metrics['F1']:.4f}")
    print(f"Test MCC: {test_metrics['MCC']:.4f}")
    print(f"\nResults saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
