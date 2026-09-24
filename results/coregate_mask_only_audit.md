# Mask-only label leakage audit

Rows: 360; classes: 45; chance accuracy: 2.22%.

| Input | Accuracy | Macro-F1 |
|---|---:|---:|
| Mask statistics (real labels) | 91.41% ± 0.00 | 91.36% ± 0.00 |
| Mask statistics (100-label permutations) | 2.04% ± 0.05 | 1.53% ± 0.04 |

The audit uses only mask/node/edge counts and type/operation histograms under the existing four LOEO folds.
