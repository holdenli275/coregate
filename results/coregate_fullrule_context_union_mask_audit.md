# Mask-only label leakage audit (context_union)

Rows: 360; classes: 45; chance accuracy: 2.22%.

| Input | Accuracy | Macro-F1 |
|---|---:|---:|
| Mask statistics (real labels) | 93.70% ± 0.00 | 94.35% ± 0.00 |
| Mask statistics (100-label permutations) | 2.04% ± 0.06 | 1.57% ± 0.07 |

The audit uses only mask/node/edge counts and type/operation histograms under the existing four LOEO folds.
