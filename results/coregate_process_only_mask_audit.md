# Mask-only label leakage audit (process_only)

Rows: 360; classes: 45; chance accuracy: 2.22%.

| Input | Accuracy | Macro-F1 |
|---|---:|---:|
| Mask statistics (real labels) | 43.75% ± 0.00 | 45.73% ± 0.00 |
| Mask statistics (100-label permutations) | 1.79% ± 0.07 | 0.92% ± 0.06 |

The audit uses only mask/node/edge counts and type/operation histograms under the existing four LOEO folds.
