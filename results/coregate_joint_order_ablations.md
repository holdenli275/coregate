# Joint environment-template and edge-rank ablations

Four LOEO folds, three seeds, 15 epochs, StandardScaler, matched ordinary TGN and label-blind CoreGate. Held-out masks are cleared.

| Protocol | Ordinary accuracy / F1 | Lineage accuracy / F1 | Paired gain accuracy / F1 | Positive pairs |
|---|---:|---:|---:|---:|
| ordered | 91.73% / 92.92% | 95.92% / 95.44% | +4.19 / +2.52 pp | 12/12 accuracy, 12/12 F1 |
| joint | 86.43% / 86.63% | 95.12% / 95.06% | +8.69 / +8.43 pp | 12/12 accuracy, 12/12 F1 |
| shuffled | 90.67% / 92.01% | 96.01% / 95.59% | +5.35 / +3.58 pp | 12/12 accuracy, 12/12 F1 |

## Difference from ordered LOEO

- joint ordinary: -5.30 pp accuracy, -6.29 pp macro-F1; negative in 10/12 and 10/12 paired folds.
- joint lineage: -0.80 pp accuracy, -0.38 pp macro-F1; negative in 6/12 and 6/12 paired folds.
- shuffled ordinary: -1.07 pp accuracy, -0.91 pp macro-F1; negative in 6/12 and 8/12 paired folds.
- shuffled lineage: +0.09 pp accuracy, +0.15 pp macro-F1; negative in 2/12 and 3/12 paired folds.

Joint holdout removes training graphs that share any template group with the held-out environment. Its training sets are substantially smaller and vary by fold. The shuffled condition keeps topology, edge operations, and each graph's normalized edge-rank values, but permutes their edge assignment; these are rank positions rather than raw timestamps.

All joint method pairs share train/test/support hashes; shuffled and ordered pairs also share those hashes. All held-out masks are cleared. The per-fold CSV and JSON contain exact scores, counts, and hashes.
