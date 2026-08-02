#!/usr/bin/env python3
"""Compare KNF orders with XGBoost, Random Forest, and SVM over repeated random splits."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "21nt_newData"
DEFAULT_OUT = DATA_DIR / "repeated_split_knf_classical_models"
BASE_INDEX = {base: index for index, base in enumerate("ACGU")}
ENCODINGS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("1-mer", (1,)),
    ("2-mer", (2,)),
    ("3-mer", (3,)),
    ("4-mer", (4,)),
    ("7-mer", (7,)),
    ("1-2mer", (1, 2)),
    ("1-3mer", (1, 2, 3)),
    ("1-4mer", (1, 2, 3, 4)),
    ("1-7mer", (1, 2, 3, 4, 5, 6, 7)),
)
MODELS = ("XGBoost", "Random Forest", "SVM")
COLORS = {"XGBoost": "#2F855A", "Random Forest": "#2A6F97", "SVM": "#C6533D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=16)
    parser.add_argument("--rf_trees", type=int, default=300)
    parser.add_argument("--xgb_trees", type=int, default=300)
    parser.add_argument("--svm_cache_mb", type=float, default=4096.0)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--encodings", default=",".join(name for name, _ in ENCODINGS))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def clean_sequence(value: object) -> str:
    sequence = str(value).strip().upper().replace("T", "U")
    sequence = "".join(base if base in BASE_INDEX else "N" for base in sequence)
    if len(sequence) != 21:
        raise ValueError(f"Expected 21 nt sequence, got {len(sequence)}: {sequence}")
    return sequence


def load_combined_data() -> pd.DataFrame:
    frames = []
    for split in ("train", "valid", "test"):
        path = DATA_DIR / f"{split}.csv"
        frame = pd.read_csv(path)
        missing = {"_site_sequence", "label"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame.copy()
        frame["original_split"] = split
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["sequence"] = combined["_site_sequence"].map(clean_sequence)
    combined["label"] = combined["label"].astype(np.int64)
    if set(combined["label"].unique()) - {0, 1}:
        raise ValueError("Labels must be binary")
    return combined


def sparse_kmer_frequency(sequences: list[str], k: int) -> sparse.csr_matrix:
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
    matrix = sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, columns)),
        shape=(len(sequences), 4**k),
        dtype=np.float32,
    )
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    good = np.isclose(row_sums, 1.0, atol=1e-6) | np.isclose(row_sums, 0.0, atol=1e-6)
    if not good.all():
        raise ValueError(f"{k}-mer frequency normalization failed")
    return matrix


def make_feature_matrices(sequences: list[str]) -> dict[str, sparse.csr_matrix]:
    blocks = {k: sparse_kmer_frequency(sequences, k) for k in range(1, 8)}
    matrices = {}
    for name, orders in ENCODINGS:
        matrices[name] = sparse.hstack([blocks[k] for k in orders], format="csr", dtype=np.float32)
    return matrices


def make_model(name: str, y_train: np.ndarray, split_seed: int, args: argparse.Namespace):
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=args.xgb_trees,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            scale_pos_weight=negatives / max(positives, 1),
            n_jobs=args.n_jobs,
            random_state=split_seed,
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=args.rf_trees,
            max_features="log2",
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            n_jobs=args.n_jobs,
            random_state=split_seed,
        )
    if name == "SVM":
        return SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            cache_size=args.svm_cache_mb,
        )
    raise ValueError(f"Unknown model: {name}")


def score_model(name: str, model, x_test: sparse.csr_matrix) -> np.ndarray:
    if name in {"XGBoost", "Random Forest"}:
        return model.predict_proba(x_test)[:, 1]
    return model.decision_function(x_test)


def append_result(path: Path, row: dict[str, object]) -> None:
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def summarize(metrics_path: Path, out_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path).sort_values(["encoding_order", "model_order", "repeat"])
    summary = (
        metrics.groupby(["encoding", "encoding_order", "model", "model_order"], as_index=False)
        .agg(
            mean_AUROC=("AUROC", "mean"),
            std_AUROC=("AUROC", "std"),
            median_AUROC=("AUROC", "median"),
            min_AUROC=("AUROC", "min"),
            max_AUROC=("AUROC", "max"),
            n=("AUROC", "size"),
            mean_fit_seconds=("fit_seconds", "mean"),
        )
        .sort_values(["encoding_order", "model_order"])
    )
    summary.to_csv(out_dir / "auroc_summary.csv", index=False)
    return metrics


def plot_boxplot(metrics: pd.DataFrame, out_dir: Path, n_splits: int, test_size: float) -> None:
    encoding_names = [name for name, _ in ENCODINGS if name in set(metrics["encoding"])]
    model_names = [name for name in MODELS if name in set(metrics["model"])]
    base_positions = np.arange(len(encoding_names), dtype=float)
    offsets = np.linspace(-0.26, 0.26, len(model_names)) if len(model_names) > 1 else np.zeros(1)
    width = min(0.22, 0.65 / max(len(model_names), 1))
    rng = np.random.default_rng(2026)

    fig, ax = plt.subplots(figsize=(14.2, 6.4))
    for model_index, model_name in enumerate(model_names):
        values = [
            metrics.loc[(metrics["encoding"] == encoding) & (metrics["model"] == model_name), "AUROC"].to_numpy()
            for encoding in encoding_names
        ]
        positions = base_positions + offsets[model_index]
        box = ax.boxplot(
            values,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#1F2933", "linewidth": 1.5},
            whiskerprops={"color": COLORS[model_name], "linewidth": 1.2},
            capprops={"color": COLORS[model_name], "linewidth": 1.2},
            boxprops={"edgecolor": COLORS[model_name], "linewidth": 1.3},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(COLORS[model_name])
            patch.set_alpha(0.28)
        for position, group in zip(positions, values):
            jitter = rng.uniform(-width * 0.22, width * 0.22, size=len(group))
            ax.scatter(
                np.full(len(group), position) + jitter,
                group,
                s=13,
                color=COLORS[model_name],
                alpha=0.68,
                edgecolors="none",
                zorder=3,
            )

    ax.set_xticks(base_positions, encoding_names)
    ax.set_xlim(-0.65, len(encoding_names) - 0.35)
    all_values = metrics["AUROC"].to_numpy()
    lower = max(0.45, np.floor((all_values.min() - 0.03) * 20) / 20)
    upper = min(1.0, np.ceil((all_values.max() + 0.03) * 20) / 20)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("KNF encoding")
    ax.set_ylabel("Test AUROC")
    ax.set_title(f"Classical classifiers across {n_splits} stratified random splits ({int((1-test_size)*100)}/{int(test_size*100)})")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[Patch(facecolor=COLORS[name], edgecolor=COLORS[name], alpha=0.35, label=name) for name in model_names],
        frameon=False,
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "knf_classical_models_10split_auroc_boxplot.png", dpi=300)
    fig.savefig(out_dir / "knf_classical_models_10split_auroc_boxplot.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    requested_models = [name.strip() for name in args.models.split(",") if name.strip()]
    requested_encodings = [name.strip() for name in args.encodings.split(",") if name.strip()]
    if set(requested_models) - set(MODELS):
        raise ValueError(f"Unknown models: {sorted(set(requested_models) - set(MODELS))}")
    known_encodings = {name for name, _ in ENCODINGS}
    if set(requested_encodings) - known_encodings:
        raise ValueError(f"Unknown encodings: {sorted(set(requested_encodings) - known_encodings)}")

    combined = load_combined_data()
    sequences = combined["sequence"].tolist()
    labels = combined["label"].to_numpy(dtype=np.int64)
    matrices = make_feature_matrices(sequences)
    metrics_path = args.out_dir / "repeated_split_auroc.csv"
    if args.force and metrics_path.exists():
        metrics_path.unlink()
    completed: set[tuple[int, str, str]] = set()
    if metrics_path.exists():
        old = pd.read_csv(metrics_path)
        completed = set(zip(old["repeat"].astype(int), old["encoding"], old["model"]))

    splitter = StratifiedShuffleSplit(
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    split_rows = []
    splits = []
    for repeat, (train_index, test_index) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        splits.append((train_index, test_index))
        split_rows.extend(
            itertools.chain(
                ({"repeat": repeat, "row_index": int(index), "subset": "train"} for index in train_index),
                ({"repeat": repeat, "row_index": int(index), "subset": "test"} for index in test_index),
            )
        )
    pd.DataFrame(split_rows).to_csv(args.out_dir / "random_split_assignments.csv", index=False)

    manifest = {
        "dataset_rows": len(combined),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "exact_duplicate_sequences": int(combined.duplicated("sequence", keep=False).sum()),
        "splitter": "StratifiedShuffleSplit",
        "n_splits": args.n_splits,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "encoding": "valid-window normalized k-mer frequencies; combination blocks are concatenated",
        "N_policy": "ignore N-containing k-mer windows; normalize each k block by its valid-window count",
        "unseen_policy": "remove columns not observed in that repeat's training subset without renormalizing",
        "model_selection": "fixed predeclared hyperparameters; random test subsets are not used for tuning",
        "models": requested_models,
        "encodings": requested_encodings,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("dataset_rows", "positive", "negative", "n_splits")}), flush=True)

    for repeat, (train_index, test_index) in enumerate(splits, start=1):
        y_train, y_test = labels[train_index], labels[test_index]
        split_seed = args.random_state + repeat - 1
        print(f"\n=== Repeat {repeat}/{args.n_splits}; seed={split_seed} ===", flush=True)
        for encoding_order, (encoding_name, orders) in enumerate(ENCODINGS):
            if encoding_name not in requested_encodings:
                continue
            full_matrix = matrices[encoding_name]
            train_full = full_matrix[train_index]
            test_full = full_matrix[test_index]
            seen_columns = np.unique(train_full.indices)
            x_train = train_full[:, seen_columns]
            x_test = test_full[:, seen_columns]
            print(
                f"  {encoding_name:<7} k={orders} theoretical={full_matrix.shape[1]:,} seen={len(seen_columns):,}",
                flush=True,
            )
            for model_order, model_name in enumerate(MODELS):
                if model_name not in requested_models or (repeat, encoding_name, model_name) in completed:
                    continue
                model = make_model(model_name, y_train, split_seed, args)
                start = time.perf_counter()
                model.fit(x_train, y_train)
                fit_seconds = time.perf_counter() - start
                probability = score_model(model_name, model, x_test)
                auroc = float(roc_auc_score(y_test, probability))
                row = {
                    "repeat": repeat,
                    "split_seed": split_seed,
                    "encoding": encoding_name,
                    "encoding_order": encoding_order,
                    "k_values": "-".join(map(str, orders)),
                    "theoretical_features": full_matrix.shape[1],
                    "train_seen_features": len(seen_columns),
                    "model": model_name,
                    "model_order": model_order,
                    "train_size": len(train_index),
                    "test_size": len(test_index),
                    "AUROC": auroc,
                    "fit_seconds": fit_seconds,
                }
                append_result(metrics_path, row)
                completed.add((repeat, encoding_name, model_name))
                print(f"    {model_name:<13} AUROC={auroc:.4f} fit={fit_seconds:.1f}s", flush=True)
                metrics = summarize(metrics_path, args.out_dir)
                plot_boxplot(metrics, args.out_dir, args.n_splits, args.test_size)

    metrics = summarize(metrics_path, args.out_dir)
    plot_boxplot(metrics, args.out_dir, args.n_splits, args.test_size)
    print("\nMean AUROC:", flush=True)
    print(
        pd.read_csv(args.out_dir / "auroc_summary.csv")
        .pivot(index="encoding", columns="model", values="mean_AUROC")
        .reindex([name for name, _ in ENCODINGS])
        .to_string(float_format=lambda value: f"{value:.4f}"),
        flush=True,
    )


if __name__ == "__main__":
    main()
