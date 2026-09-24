# Matched learned-pooling baselines

Four LOEO folds x three seeds, 15 epochs, StandardScaler, identical split/support selection.

| Method | Accuracy | Macro-F1 | Parameters |
|---|---:|---:|---:|
| Ordinary | 91.73% +/- 0.67 | 92.92% +/- 0.53 | 225,839 |
| LearnedAttention | 87.40% +/- 1.17 | 88.95% +/- 1.51 | 225,968 |
| SAGPool-50% | 87.48% +/- 2.04 | 88.38% +/- 2.28 | 226,097 |
| CoreGate-Lineage | 95.92% +/- 0.16 | 95.44% +/- 0.19 | 225,968 |
| CoreGate-FullRule | 95.77% +/- 0.38 | 95.92% +/- 0.61 | 225,968 |

## Paired results

SAGPool minus Ordinary: -4.25 Accuracy / -4.54 Macro-F1; positive in 1/12 and 2/12 pairs.
CoreGate-Lineage minus SAGPool: +8.44 Accuracy / +7.06 Macro-F1; positive in 12/12 and 12/12 pairs.

SAGPool uses the official PyG SAGPooling layer with a fixed 0.5 retention ratio and receives no evidence supervision. Its failure to match CoreGate-Lineage strengthens the conclusion that generic task-learned pooling alone does not explain the CoreGate gain.
