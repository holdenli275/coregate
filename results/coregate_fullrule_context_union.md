# FullRule-ContextUnion sensitivity

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Ordinary | 91.73% +/- 0.67 | 92.92% +/- 0.53 |
| CoreGate-FullRule | 95.77% +/- 0.38 | 95.92% +/- 0.61 |
| CoreGate-Lineage | 95.92% +/- 0.16 | 95.44% +/- 0.19 |
| FullRule-ContextUnion | 94.77% +/- 1.09 | 94.89% +/- 0.68 |

## Mask-only audit

FullRule: 91.41% Accuracy / 91.36% Macro-F1.
ContextUnion: 93.70% Accuracy / 94.35% Macro-F1.
Change: +2.30 / +2.99 points.

## Paired model results

ContextUnion minus Ordinary: +3.04 Accuracy / +1.97 Macro-F1; positive in 10/12 and 10/12 pairs.
ContextUnion minus FullRule: -0.99 Accuracy / -1.03 Macro-F1.

## Decision

Reject ContextUnion as shortcut mitigation. It retains all FullRule nodes and adds label-blind process context, but the mask-only fingerprint becomes stronger rather than weaker. Its lower mean CoreGate result is therefore consistent with noisier gate supervision, not evidence that the rule shortcut was reduced.
