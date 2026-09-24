# Raw graph / mask / joint statistics audit

360 labeled graphs, 45 classes, four LOEO folds, three classifier seeds.

| Mask | Statistics | Accuracy | Macro-F1 | Features |
|---|---|---:|---:|---:|
| full | raw_only | 78.25% +/- 0.00 | 77.10% +/- 0.00 | 72 |
| full | mask_only | 89.57% +/- 0.00 | 88.84% +/- 0.00 | 72 |
| full | raw_plus_mask | 93.07% +/- 0.00 | 94.27% +/- 0.00 | 144 |
| lineage | raw_only | 78.25% +/- 0.00 | 77.10% +/- 0.00 | 72 |
| lineage | mask_only | 21.12% +/- 0.00 | 20.67% +/- 0.00 | 72 |
| lineage | raw_plus_mask | 79.44% +/- 0.00 | 78.34% +/- 0.00 | 144 |

## Paired descriptive differences versus raw-only

- full mask_only: +11.32 pp accuracy; +11.74 pp macro-F1.
- full raw_plus_mask: +14.82 pp accuracy; +17.16 pp macro-F1.
- lineage mask_only: -57.13 pp accuracy; -56.43 pp macro-F1.
- lineage raw_plus_mask: +1.19 pp accuracy; +1.24 pp macro-F1.

## Interpretation

Raw-only operation and node-type counts already carry substantial class information. FullRule mask statistics are stronger, and their concatenation with raw statistics is stronger still under this linear classifier. The label-blind lineage mask alone is much weaker; adding it to raw statistics gives only a small descriptive gain. This does not measure the value of lineage supervision for the learned TGN gate.

The earlier mask-only audit included full-graph counts and mask-to-graph ratios in its feature set. Its 91.41% FullRule and 38.70% lineage accuracies are therefore not directly comparable to the strict mask-only rows above.

The mask-only view contains only selected-node and induced-edge statistics; it excludes full-graph totals and ratios. Raw+mask concatenates the two disjoint feature blocks. All views use the same splits, train-fitted scaler, and multinomial logistic classifier. The three seeds give identical predictions, so the displayed zero standard deviation is not evidence of external stability.

Mask-only and raw+mask inspect held-out masks as an oracle diagnostic. They are not test-time inputs to CoreGate. The audit covers simple count histograms, not graph attributes or topology. See the JSON and per-fold CSV for exact results.
