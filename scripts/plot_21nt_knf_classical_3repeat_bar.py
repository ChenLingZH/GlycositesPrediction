#!/usr/bin/env python3
"""Plot mean AUROC and variability for the first three repeated KNF experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "21nt_newData" / "repeated_split_knf_classical_models"
ENCODINGS = ("1-mer", "2-mer", "3-mer", "4-mer", "7-mer", "1-2mer", "1-3mer", "1-4mer", "1-7mer")
MODELS = ("XGBoost", "Random Forest", "SVM")
COLORS = {"XGBoost": "#2F855A", "Random Forest": "#2A6F97", "SVM": "#C6533D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--n_repeats", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.result_dir / "repeated_split_auroc.csv"
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics["repeat"].between(1, args.n_repeats)].copy()

    expected = args.n_repeats * len(ENCODINGS) * len(MODELS)
    keys = ["repeat", "encoding", "model"]
    if len(metrics) != expected or metrics.duplicated(keys).any():
        raise ValueError(f"Expected {expected} unique rows for repeats 1-{args.n_repeats}, got {len(metrics)}")
    observed = set(zip(metrics["encoding"], metrics["model"]))
    required = set((encoding, model) for encoding in ENCODINGS for model in MODELS)
    if observed != required:
        raise ValueError(f"Missing encoding/model combinations: {sorted(required - observed)}")

    summary = (
        metrics.groupby(["encoding", "model"], as_index=False)
        .agg(
            mean_AUROC=("AUROC", "mean"),
            variance_AUROC=("AUROC", "var"),
            std_AUROC=("AUROC", "std"),
            min_AUROC=("AUROC", "min"),
            max_AUROC=("AUROC", "max"),
            n=("AUROC", "size"),
        )
    )
    summary["encoding_order"] = summary["encoding"].map({name: index for index, name in enumerate(ENCODINGS)})
    summary["model_order"] = summary["model"].map({name: index for index, name in enumerate(MODELS)})
    summary = summary.sort_values(["encoding_order", "model_order"])
    summary.to_csv(args.result_dir / "three_repeat_auroc_mean_variance.csv", index=False)
    metrics.sort_values(keys).to_csv(args.result_dir / "three_repeat_auroc_values.csv", index=False)
    assignments_path = args.result_dir / "random_split_assignments.csv"
    assignments = pd.read_csv(assignments_path)
    assignments = assignments[assignments["repeat"].between(1, args.n_repeats)].copy()
    assignments.to_csv(args.result_dir / "three_repeat_split_assignments.csv", index=False)
    manifest = {
        "n_repeats": args.n_repeats,
        "split_method": "StratifiedShuffleSplit",
        "train_test_ratio": "80/20",
        "base_random_state": 42,
        "models": list(MODELS),
        "encodings": list(ENCODINGS),
        "bar_height": "mean test AUROC",
        "error_bar": "sample standard deviation of test AUROC",
        "variance_column": "sample variance with ddof=1",
        "source_metrics": "three_repeat_auroc_values.csv",
    }
    (args.result_dir / "three_repeat_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    x = np.arange(len(ENCODINGS), dtype=float)
    width = 0.24
    offsets = (-width, 0.0, width)
    fig, ax = plt.subplots(figsize=(14.4, 6.5))
    for model_index, model_name in enumerate(MODELS):
        part = summary[summary["model"].eq(model_name)].set_index("encoding").reindex(ENCODINGS)
        means = part["mean_AUROC"].to_numpy()
        standard_deviations = part["std_AUROC"].to_numpy()
        bars = ax.bar(
            x + offsets[model_index],
            means,
            width=width,
            yerr=standard_deviations,
            capsize=3.5,
            color=COLORS[model_name],
            alpha=0.84,
            edgecolor="white",
            linewidth=0.7,
            error_kw={"elinewidth": 1.15, "ecolor": "#29323A"},
            label=model_name,
            zorder=3,
        )
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + 0.012,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.4,
                rotation=90,
                color="#29323A",
            )

    ax.set_xticks(x, ENCODINGS)
    ax.set_xlim(-0.65, len(ENCODINGS) - 0.35)
    ax.set_ylim(0.45, 0.91)
    ax.set_xlabel("KNF encoding")
    ax.set_ylabel("Test AUROC (mean +/- SD)")
    ax.set_title("Classical classifiers across three stratified random 80/20 splits")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.result_dir / "knf_classical_models_3repeat_mean_sd_bar.png", dpi=300)
    fig.savefig(args.result_dir / "knf_classical_models_3repeat_mean_sd_bar.pdf")
    plt.close(fig)

    pivot = summary.pivot(index="encoding", columns="model", values="mean_AUROC").reindex(ENCODINGS)
    print("Mean AUROC (n=3):")
    print(pivot.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved summary: {args.result_dir / 'three_repeat_auroc_mean_variance.csv'}")


if __name__ == "__main__":
    main()
