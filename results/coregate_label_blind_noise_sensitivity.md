# Label-blind mask-noise sensitivity

## Three-seed 50% positive-dropout result

Three matched model seeds and deterministic corruption seeds; four LOEO folds, 15 epochs.

| Condition | Accuracy | Macro-F1 |
|---|---:|---:|
| Ordinary | 91.73% +/- 0.67 | 92.92% +/- 0.53 |
| Lineage clean | 95.92% +/- 0.16 | 95.44% +/- 0.19 |
| Lineage drop 50% | 95.92% +/- 0.16 | 95.74% +/- 0.03 |

Drop 50% minus Ordinary: +4.19 Accuracy / +2.83 Macro-F1; positive in 12/12 and 12/12 paired runs.

## Single-seed noise curve

Single model seed and deterministic corruption seed; four LOEO folds, 15 epochs.

| Condition | Accuracy | Macro-F1 |
|---|---:|---:|
| Ordinary | 91.41% | 92.69% |
| Lineage clean | 96.11% | 95.56% |
| Lineage drop 10% | 96.11% | 95.82% |
| Lineage drop 30% | 95.82% | 95.57% |
| Lineage drop 50% | 96.11% | 95.76% |

## Paired gain over Ordinary

- Lineage clean: +4.70 Accuracy / +2.87 Macro-F1; positive folds 4/4 and 4/4.
- Lineage drop 10%: +4.70 Accuracy / +3.13 Macro-F1; positive folds 4/4 and 4/4.
- Lineage drop 30%: +4.42 Accuracy / +2.89 Macro-F1; positive folds 4/4 and 4/4.
- Lineage drop 50%: +4.70 Accuracy / +3.07 Macro-F1; positive folds 4/4 and 4/4.

## Interpretation

Deleting approximately 10%, 30%, or 50% of positive training-mask nodes does not remove the gain over Ordinary in the fixed-seed curve. The strict 50% condition is additionally confirmed over three matched model and corruption seeds. This supports tolerance to false-negative foreground annotations; it remains a supplementary robustness result rather than independent-host validation.
