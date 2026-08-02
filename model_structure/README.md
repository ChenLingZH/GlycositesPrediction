# Model Architecture Inventory

This directory contains source-only model architecture references. Trained
weights and generated predictions are intentionally excluded from Git.

## 1NF-10NF benchmark models

The executable benchmark is:

```text
scripts/run_21nt_knf_1to10_four_models.py
```

It defines the sparse KNF input adapter and the following classifiers:

| Model | Implementation | Input |
|---|---|---|
| RF | scikit-learn `RandomForestClassifier` | Exact sparse k-mer frequency matrix |
| DNN | `SparseKnfDNN` | Nonzero k-mer IDs and normalized frequencies |
| CNN | `SparseKnfCNN` | Canonically ordered nonzero k-mer tokens |
| mamba6mA | `SparseKnfMamba6MA` | Canonically ordered nonzero k-mer tokens |

The reusable mamba6mA components used by the benchmark are in:

```text
scripts/train_knf_mamba6ma_21nt.py
```

The KNF benchmark uses one mamba6mA residual layer per 3/5/7 dynamic branch
and applies `LayerNorm` to the k-mer embedding before the branches.

## Nucleotide-token mamba6mA reference

`model_structure/mamba6mA/` contains the nucleotide-token implementation with
the following components:

- nucleotide embedding with an explicit padding token;
- three position-specific convolution branches with windows 3, 5, and 7;
- Mamba selective state-space blocks;
- RMS normalization and residual connections;
- learned multi-representation fusion;
- binary classification head.

Its `ModelArgs` defaults (`seq_len=51`, `n_layer=6`) describe the standalone
reference configuration. They are not the settings used for the 21-nt KNF
benchmark. For 21-nt nucleotide-token experiments, set `seq_len=21`; for the
reported KNF benchmark, use the benchmark script and its one-layer model.

## Dependencies

The neural-network implementations require PyTorch. The standalone
nucleotide-token reference additionally imports `einops`, pandas,
scikit-learn, and tqdm in its training/evaluation entry points.
