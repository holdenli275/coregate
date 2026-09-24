# Mask-only label leakage audit (lineage)

Rows: 360; classes: 45; chance accuracy: 2.22%.

| Input | Accuracy | Macro-F1 |
|---|---:|---:|
| Mask statistics (real labels) | 38.70% ± 0.00 | 34.59% ± 0.00 |
| Mask statistics (100-label permutations) | 1.93% ± 0.07 | 0.96% ± 0.04 |

The audit uses only mask/node/edge counts and type/operation histograms under the existing four LOEO folds.
