# Label-blind CoreGate minimal submission experiment

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Ordinary | 91.73% +/- 0.67 | 92.92% +/- 0.53 |
| LearnedAttention | 87.40% +/- 1.17 | 88.95% +/- 1.51 |
| Gate-ShuffledLineage | 88.10% +/- 0.95 | 88.78% +/- 1.45 |
| CoreGate-Lineage | 95.92% +/- 0.16 | 95.44% +/- 0.19 |
| CoreGate-FullRule | 95.77% +/- 0.38 | 95.92% +/- 0.61 |

Lineage minus Ordinary: +4.19 / +2.52 points; positive in 12/12 Accuracy and 12/12 Macro-F1 pairs.
Lineage minus LearnedAttention: +8.52 / +6.49 points.
Lineage minus ShuffledLineage: +7.82 / +6.67 points.

## Label-invariance and mask-only checks

All 360 masks remain byte-identical after permuting technique labels; changed masks = 0.
Lineage mask-only statistics reach 38.70% Accuracy and 34.59% Macro-F1. This measures behavioral information, not construction-time label access.

## Template-disjoint

Ordinary: 89.93% / 85.83%; CoreGate-Lineage: 93.59% / 90.63%; FullRule: 93.73% / 91.33%.

## Interpretation

The gain persists when gate supervision is generated from collection-time scenario lineage without ATT&CK labels, technique rules, control.kind, or Core_Graph. The label-permutation hash audit rules out direct construction-time label access. The masks remain behaviorally discriminative, so the result should still be described as training-only privileged foreground supervision rather than causal subgraph recovery.
