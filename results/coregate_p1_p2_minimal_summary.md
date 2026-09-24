# CoreGate P1/P2 minimal supplementary experiments

P1 uses StandardScaler and the same encoder checkpoint for embedding-only and embedding-plus-prototype inputs. CoreGate uses the supplementary shared node-representation forward path to avoid repeating the gate BCE encoder call; this path preserves the objective but changes dropout call order relative to the locked main run.

| Method | P1 embedding-only accuracy | P1 embedding+prototype accuracy | P1 embedding-only Macro-F1 |
|---|---:|---:|---:|
| ordinary | 91.64% ± 0.75 | 91.73% ± 0.67 | 92.76% ± 0.67 |
| coregate | 95.71% ± 0.38 | 95.77% ± 0.38 | 95.92% ± 0.61 |

P1 CoreGate − Ordinary: embedding-only accuracy 4.07 pp, Macro-F1 3.16 pp, both positive in 12/12 paired runs; embedding+prototype accuracy 4.03 pp and Macro-F1 3.00 pp.

P2 uses one fixed held-out template group per class; train/test template overlap is 0.

| Method | P2 accuracy | P2 Macro-F1 |
|---|---:|---:|
| ordinary | 89.93% ± 0.82 | 85.83% ± 0.83 |
| coregate | 93.73% ± 1.71 | 91.33% ± 2.05 |

P2 CoreGate − Ordinary: accuracy 3.79 pp, Macro-F1 5.50 pp; positive in 3/3 seeds for both metrics.
