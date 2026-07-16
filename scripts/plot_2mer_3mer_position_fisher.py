#!/usr/bin/env python3
"""Position-specific Fisher exact test heatmaps for 2/3/4/5-mer.

For each k-mer and each possible start position in the 21nt window, this script
tests whether the exact k-mer-at-position event is different between positive
and negative samples.

Main outputs:
  - TSV tables with counts, odds ratio, p-value and BH-FDR q-value.
  - -log10(q) heatmaps annotated with BH-FDR q-values.
  - Signed heatmaps: sign(log2 odds ratio) * -log10(q).
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "position_fisher_2mer_5mer"
BASES = ["A", "U", "C", "G"]


def clean_seq(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).upper().replace("T", "U")


def all_kmers(k: int) -> list[str]:
    return ["".join(x) for x in itertools.product(BASES, repeat=k)]


def infer_label(df: pd.DataFrame, label_col: str | None) -> pd.Series:
    if label_col:
        values = df[label_col]
        lowered = values.astype(str).str.lower()
        if lowered.isin(["positive", "negative"]).all():
            return lowered.eq("positive").astype(int)
        return values.astype(int)
    if "source_class" in df.columns:
        return (df["source_class"] == "positive").astype(int)
    if "label" in df.columns:
        values = df["label"]
        lowered = values.astype(str).str.lower()
        if lowered.isin(["positive", "negative"]).all():
            return lowered.eq("positive").astype(int)
        return values.astype(int)
    return df["representative_stage"].astype(str).str.contains("positive").astype(int)


def load_data(
    scope: str,
    input_tsv: Path | None,
    data_dir: Path,
    seq_col: str | None,
    label_col: str | None,
    window_len: int,
) -> pd.DataFrame:
    if input_tsv is not None:
        df = pd.read_csv(input_tsv, sep="\t", dtype=str, keep_default_na=False)
        if seq_col is None:
            for candidate in ["window_51nt", "sequence_51nt", "window_21nt_norm", "window_21nt", "seq"]:
                if candidate in df.columns:
                    seq_col = candidate
                    break
        if seq_col is None or seq_col not in df.columns:
            raise ValueError(f"Cannot infer sequence column in {input_tsv}; pass --seq-col")
        df["seq"] = df[seq_col].map(clean_seq)
        df["label"] = infer_label(df, label_col)
        df["split"] = scope
        return df[df["seq"].str.len() == window_len].copy()

    splits = ["train"] if scope == "train" else ["train", "valid", "test"]
    parts = []
    for split in splits:
        df = pd.read_csv(data_dir / f"{split}.tsv", sep="\t", dtype=str, keep_default_na=False)
        split_seq_col = "window_21nt_norm" if "window_21nt_norm" in df.columns else "window_21nt"
        df["seq"] = df[split_seq_col].map(clean_seq)
        df["label"] = infer_label(df, label_col)
        df["split"] = split
        parts.append(df)
    data = pd.concat(parts, ignore_index=True)
    data = data[data["seq"].str.len() == window_len].copy()
    return data


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    q_ranked = ranked * n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.minimum(q_ranked, 1.0)
    q = np.empty(n, dtype=float)
    q[order] = q_ranked
    return q


def compute_position_fisher(data: pd.DataFrame, k: int, window_len: int, center_index: float) -> pd.DataFrame:
    seqs = data["seq"].to_numpy()
    labels = data["label"].to_numpy()
    pos_total = int((labels == 1).sum())
    neg_total = int((labels == 0).sum())
    pos_seqs = seqs[labels == 1]
    neg_seqs = seqs[labels == 0]
    kmers = all_kmers(k)
    n_pos = window_len - k + 1
    rows = []
    for start in range(n_pos):
        pos_counts = pd.Series([seq[start : start + k] for seq in pos_seqs]).value_counts().to_dict()
        neg_counts = pd.Series([seq[start : start + k] for seq in neg_seqs]).value_counts().to_dict()
        for kmer in kmers:
            pos_hit = int(pos_counts.get(kmer, 0))
            neg_hit = int(neg_counts.get(kmer, 0))
            pos_nohit = pos_total - pos_hit
            neg_nohit = neg_total - neg_hit
            odds_ratio, pvalue = fisher_exact(
                [[pos_hit, pos_nohit], [neg_hit, neg_nohit]],
                alternative="two-sided",
            )
            log2_or = math.log2(((pos_hit + 0.5) * (neg_nohit + 0.5)) / ((neg_hit + 0.5) * (pos_nohit + 0.5)))
            rows.append(
                {
                    "k": k,
                    "kmer": kmer,
                    "start": start,
                    "relative_midpoint": start + (k - 1) / 2.0 - center_index,
                    "pos_hit": pos_hit,
                    "pos_nohit": pos_nohit,
                    "neg_hit": neg_hit,
                    "neg_nohit": neg_nohit,
                    "pos_rate": pos_hit / max(pos_total, 1),
                    "neg_rate": neg_hit / max(neg_total, 1),
                    "odds_ratio": odds_ratio,
                    "log2_or": log2_or,
                    "pvalue": pvalue,
                    "neg_log10_p": -math.log10(max(pvalue, 1e-300)),
                    "direction": "positive_enriched" if log2_or > 0 else "negative_enriched",
                }
            )
    out = pd.DataFrame(rows)
    out["qvalue_bh"] = bh_fdr(out["pvalue"].to_numpy())
    return out.sort_values(["pvalue", "kmer", "start"], ascending=[True, True, True])


def pvalue_label(p: float) -> str:
    if p < 1e-99:
        return "<1e-99"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


def select_rows(table: pd.DataFrame, k: int, mode: str, top_n: int) -> list[str]:
    if mode == "all":
        selected = all_kmers(k)
    else:
        best = table.groupby("kmer")["pvalue"].min().sort_values()
        selected = best.head(top_n).index.tolist()
    # Sort selected rows by their most significant position, then by k-mer.
    row_order = (
        table[table["kmer"].isin(selected)]
        .groupby("kmer")
        .agg(best_p=("pvalue", "min"), best_logor=("log2_or", lambda x: x.iloc[np.argmin(table.loc[x.index, "pvalue"].to_numpy())]))
        .sort_values(["best_p", "best_logor"], ascending=[True, False])
        .index.tolist()
    )
    return row_order


def draw_heatmap(
    table: pd.DataFrame,
    k: int,
    row_order: list[str],
    out_file: Path,
    title: str,
    signed: bool = False,
    cap: float = 10.0,
    annotate_q_threshold: float = 0.05,
    window_len: int = 21,
    cmap: str | None = None,
    show_q_labels: bool = True,
    center_index: float | None = None,
) -> None:
    n_pos = window_len - k + 1
    q_matrix = np.ones((len(row_order), n_pos), dtype=float)
    matrix = np.zeros((len(row_order), n_pos), dtype=float)
    lookup = table.set_index(["kmer", "start"])
    for i, kmer in enumerate(row_order):
        for start in range(n_pos):
            row = lookup.loc[(kmer, start)]
            q = float(row["qvalue_bh"])
            value = -math.log10(max(q, 1e-300))
            logor = float(row["log2_or"])
            matrix[i, start] = np.sign(logor) * min(value, cap) if signed else min(value, cap)
            q_matrix[i, start] = q

    fig_height = max(4.8, 0.34 * len(row_order))
    fig_width = max(10, 0.36 * n_pos + 3.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if signed:
        im = ax.imshow(matrix, aspect="auto", cmap=cmap or "RdBu_r", vmin=-cap, vmax=cap)
        cbar_label = f"signed -log10(q-value), capped at +/-{cap:g}"
    else:
        im = ax.imshow(matrix, aspect="auto", cmap=cmap or "magma_r", vmin=0, vmax=cap)
        cbar_label = f"-log10(q-value), capped at {cap:g}"
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=11, fontweight="bold")

    tick_step = 2 if n_pos <= 30 else 5
    xticks = np.arange(0, n_pos, tick_step)
    if n_pos - 1 not in xticks:
        xticks = np.r_[xticks, n_pos - 1]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x)) for x in xticks], fontsize=8)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=9)
    ax.set_xlabel(f"Start position in {window_len}nt window ({k}-mer)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{k}-mer", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")

    if center_index is not None:
        center_start = center_index - (k - 1) / 2.0
        if -0.5 <= center_start <= n_pos - 0.5:
            ax.axvline(center_start, color="#111827", linewidth=1.1, linestyle="--", alpha=0.75)

    if show_q_labels:
        for i in range(len(row_order)):
            for start in range(n_pos):
                q = q_matrix[i, start]
                if q < annotate_q_threshold:
                    ax.text(
                        start,
                        i,
                        pvalue_label(q),
                        ha="center",
                        va="center",
                        fontsize=5.8,
                        color="black",
                        fontweight="bold",
                    )

    ax.set_xlim(-0.5, n_pos - 0.5)
    ax.set_ylim(len(row_order) - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(out_file, dpi=240)
    plt.close(fig)


def plot_summary_bar(table: pd.DataFrame, out_file: Path, title: str, top_n: int = 25) -> None:
    best = table.sort_values("pvalue").head(top_n).copy()
    best["label"] = best.apply(lambda r: f"{r['kmer']} @ {int(r['start'])} ({r['relative_midpoint']:+.1f})", axis=1)
    best = best.sort_values("log2_or")
    colors = np.where(best["log2_or"] >= 0, "#2a9d8f", "#e76f51")
    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(best))))
    ax.barh(best["label"], best["log2_or"], color=colors, alpha=0.86)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("log2 odds ratio, positive vs negative", fontsize=11, fontweight="bold")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_file, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["train", "all"], default="train")
    parser.add_argument("--top-3mer", type=int, default=30)
    parser.add_argument("--top-4mer", type=int, default=30)
    parser.add_argument("--top-5mer", type=int, default=30)
    parser.add_argument("--cap", type=float, default=10.0)
    parser.add_argument("--annotate-q", type=float, default=0.05)
    parser.add_argument("--input-tsv", type=Path, help="Single TSV input. If omitted, read train/valid/test TSVs from --data-dir.")
    parser.add_argument("--data-dir", type=Path, default=ROOT, help="Directory containing train.tsv, valid.tsv and test.tsv.")
    parser.add_argument("--seq-col", help="Sequence column for --input-tsv, e.g. window_51nt.")
    parser.add_argument("--label-col", help="Optional label column. Accepts 0/1 or positive/negative.")
    parser.add_argument("--window-len", type=int, default=21)
    parser.add_argument("--center-index", type=float, help="0-based center index for relative_midpoint. Defaults to (window_len - 1) / 2.")
    parser.add_argument("--out-dir", type=Path, help="Output directory. Defaults to scripts/position_fisher_2mer_5mer/<scope>.")
    parser.add_argument("--ks", default="2,3,4,5", help="Comma-separated k values to test, e.g. 2,3.")
    parser.add_argument("--q-cmap", default="viridis_r", help="Matplotlib colormap for q-value heatmaps.")
    parser.add_argument("--signed-cmap", default="RdBu_r", help="Matplotlib colormap for signed heatmaps.")
    parser.add_argument("--hide-q-labels", action="store_true", help="Do not write q-value text in heatmap cells.")
    args = parser.parse_args()

    center_index = args.center_index if args.center_index is not None else (args.window_len - 1) / 2.0
    out_dir = args.out_dir if args.out_dir is not None else OUT_DIR / args.scope
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.scope, args.input_tsv, args.data_dir, args.seq_col, args.label_col, args.window_len)
    if data.empty:
        raise ValueError(f"No rows with sequence length {args.window_len}; check --input-tsv/--seq-col/--window-len")
    counts = data["label"].value_counts().to_dict()
    with open(out_dir / "sample_counts.txt", "w", encoding="utf-8") as f:
        f.write(f"scope\t{args.scope}\n")
        f.write(f"input_tsv\t{args.input_tsv or ''}\n")
        f.write(f"data_dir\t{args.data_dir}\n")
        f.write(f"window_len\t{args.window_len}\n")
        f.write(f"center_index\t{center_index}\n")
        f.write(f"n_total\t{len(data)}\n")
        f.write(f"n_positive\t{counts.get(1, 0)}\n")
        f.write(f"n_negative\t{counts.get(0, 0)}\n")

    top_n_by_k = {3: args.top_3mer, 4: args.top_4mer, 5: args.top_5mer}
    for k in [int(item) for item in args.ks.split(",") if item.strip()]:
        print(f"Computing position-specific Fisher tests for {k}-mer on {args.scope}...")
        table = compute_position_fisher(data, k, args.window_len, center_index)
        table.to_csv(out_dir / f"position_fisher_{k}mer.tsv", sep="\t", index=False)
        table[table["pvalue"] < 0.05].to_csv(out_dir / f"position_fisher_{k}mer_p_lt_0.05.tsv", sep="\t", index=False)
        table[table["qvalue_bh"] < 0.05].to_csv(out_dir / f"position_fisher_{k}mer_q_lt_0.05.tsv", sep="\t", index=False)
        plot_summary_bar(
            table,
            out_dir / f"top_position_{k}mer_log2or.png",
            f"Top position-specific {k}-mer enrichments ({args.scope})",
            top_n=30,
        )

        if k == 2:
            row_order = select_rows(table, k, mode="all", top_n=16)
            draw_heatmap(
                table,
                k,
                row_order,
                out_dir / "01_2mer_position_fisher_qvalue_heatmap.png",
                f"2-mer position-specific Fisher FDR significance ({args.scope})\nCell labels: BH-FDR q < {args.annotate_q:g}",
                signed=False,
                cap=args.cap,
                annotate_q_threshold=args.annotate_q,
                window_len=args.window_len,
                cmap=args.q_cmap,
                show_q_labels=not args.hide_q_labels,
                center_index=center_index,
            )
            draw_heatmap(
                table,
                k,
                row_order,
                out_dir / "02_2mer_position_fisher_signed_heatmap.png",
                f"2-mer signed position-specific Fisher signal ({args.scope})\nRed: positive enriched, Blue: negative enriched; labels: BH-FDR q < {args.annotate_q:g}",
                signed=True,
                cap=args.cap,
                annotate_q_threshold=args.annotate_q,
                window_len=args.window_len,
                cmap=args.signed_cmap,
                show_q_labels=not args.hide_q_labels,
                center_index=center_index,
            )
        elif k == 3:
            row_order_top = select_rows(table, k, mode="top", top_n=args.top_3mer)
            draw_heatmap(
                table,
                k,
                row_order_top,
                out_dir / f"03_3mer_top{args.top_3mer}_position_fisher_qvalue_heatmap.png",
                f"Top {args.top_3mer} 3-mer position-specific Fisher FDR significance ({args.scope})\nCell labels: BH-FDR q < {args.annotate_q:g}",
                signed=False,
                cap=args.cap,
                annotate_q_threshold=args.annotate_q,
                window_len=args.window_len,
                cmap=args.q_cmap,
                show_q_labels=not args.hide_q_labels,
                center_index=center_index,
            )
            draw_heatmap(
                table,
                k,
                row_order_top,
                out_dir / f"04_3mer_top{args.top_3mer}_position_fisher_signed_heatmap.png",
                f"Top {args.top_3mer} 3-mer signed position-specific Fisher signal ({args.scope})\nRed: positive enriched, Blue: negative enriched; labels: BH-FDR q < {args.annotate_q:g}",
                signed=True,
                cap=args.cap,
                annotate_q_threshold=args.annotate_q,
                window_len=args.window_len,
                cmap=args.signed_cmap,
                show_q_labels=not args.hide_q_labels,
                center_index=center_index,
            )
            row_order_all = select_rows(table, k, mode="all", top_n=64)
            draw_heatmap(
                table,
                k,
                row_order_all,
                out_dir / "05_3mer_all_position_fisher_qvalue_heatmap.png",
                f"All 3-mer position-specific Fisher FDR significance ({args.scope})\nCell labels: BH-FDR q < {args.annotate_q:g}",
                signed=False,
                cap=args.cap,
                annotate_q_threshold=min(args.annotate_q, 0.01),
                window_len=args.window_len,
                cmap=args.q_cmap,
                show_q_labels=not args.hide_q_labels,
                center_index=center_index,
            )
        else:
            top_n = top_n_by_k[k]
            prefix = {4: ("06", "07"), 5: ("08", "09")}[k]
            row_order_top = select_rows(table, k, mode="top", top_n=top_n)
            draw_heatmap(
                table,
                k,
                row_order_top,
                out_dir / f"{prefix[0]}_{k}mer_top{top_n}_position_fisher_qvalue_heatmap.png",
                f"Top {top_n} {k}-mer position-specific Fisher FDR significance ({args.scope})\nCell labels: BH-FDR q < {args.annotate_q:g}",
                signed=False,
                cap=args.cap,
                annotate_q_threshold=args.annotate_q,
                window_len=args.window_len,
                cmap=args.q_cmap,
                show_q_labels=not args.hide_q_labels,
                center_index=center_index,
            )
            draw_heatmap(
                table,
                k,
                row_order_top,
                out_dir / f"{prefix[1]}_{k}mer_top{top_n}_position_fisher_signed_heatmap.png",
                f"Top {top_n} {k}-mer signed position-specific Fisher signal ({args.scope})\nRed: positive enriched, Blue: negative enriched; labels: BH-FDR q < {args.annotate_q:g}",
                signed=True,
                cap=args.cap,
                annotate_q_threshold=args.annotate_q,
                window_len=args.window_len,
                cmap=args.signed_cmap,
                show_q_labels=not args.hide_q_labels,
                center_index=center_index,
            )

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
