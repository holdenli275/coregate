# Conservative process-only mask sensitivity

The training evidence target is the serialized evidence core intersected with process nodes only. File, socket, and pipe evidence nodes are excluded; held-out masks are cleared at inference.

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Ordinary | 91.73% +/- 0.67 | 92.92% +/- 0.53 |
| Process-only CoreGate | 95.31% +/- 0.75 | 95.52% +/- 1.02 |

Paired gain over Ordinary: +3.57 pp Accuracy (min 0.00, max 5.56; 11/12 positive) and +2.60 pp Macro-F1 (min -2.22, max 3.86; 11/12 positive).

The process-only mask statistics alone obtain 43.75% Accuracy and 45.73% Macro-F1, versus 91.41% / 91.36% for the full mask audit.

This is a sensitivity analysis, not a replacement for the locked full-evidence result. It supports using process-only evidence as a more conservative configuration while retaining a technique-specific weak-prior caveat.
