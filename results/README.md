# `results/` — locked CoreGate outputs

Every number in the paper is recomputed from the per-seed run JSONs in this
directory by the summarizers listed in `docs/VERIFICATION.md` §1;
`docs/coregate_reproduction.md` lists the training commands that produced the
JSONs. The directory holds the artifacts the paper reads plus the
CSV/JSON/Markdown summaries the summarizers write. Every per-seed run JSON
records the full runner configuration for the released pipeline; the fields of
the earlier internal augmentation/contrastive protocols are not part of it and
were removed. The summarizers read
`data/experiment_results/leave_one_env_v1` when that directory exists (the
directory the runners write to) and otherwise read this one directly.

## Protocol

Catalog `reference_candidates_template_v2`: 374 entries, of which 360 are labeled
attack-emulation graphs (45 classes) and 14 are unlabeled and excluded. Four
environment groups (55/54/163/88 graphs) define leave-one-environment-out
(LOEO): every sample of one group is removed, the encoder and readout train on
the other three, and evaluation is on the held-out group. The split is fixed
before embedding, scaling, classifier fitting, and augmentation. One support
graph per class, `--augmentation none`, StandardScaler, shared gate forward, and
the held-out evidence mask cleared (`--test-core-mask none`) in every reported
run.

## Paper name -> artifact name

| Paper | Artifact |
|---|---|
| Ordinary | `coregate_p1_standard_ordinary_seed*` |
| LearnedAttention (no foreground loss) | `coregate_label_blind_noevidence_seed*` |
| Gate-ShuffledLineage | `coregate_label_blind_shuffled_seed*` |
| SAGPool-50% | `coregate_sagpool_seed*` |
| CoreGate-Lineage | `coregate_label_blind_lineage_seed*` |
| CoreGate-FullRule | `coregate_p1_standard_coregate_seed*` |
| Process-only | `coregate_process_only_seed*` |
| Lineage, drop 50% | `coregate_label_blind_noise_drop50_seed*` |
| FullRule-ContextUnion | `coregate_fullrule_context_union_seed*` |

## Main table (paper Table 1)

`coregate_label_blind_minimal_summary.{json,md}` carries every main row and
`coregate_label_blind_minimal_per_fold.csv` the 60 per-seed-fold records behind
the paired panel (each fold's Lineage - Ordinary difference for both metrics).
`coregate_label_blind_minimal_paper_text.md` is the ready-to-paste paragraph.

## Classifier-input and split diagnostics (paper Table 2)

`coregate_p1_standard_ordinary_seed*` and `coregate_p1_standard_coregate_seed*`
supply the embedding-only control and the shared LOEO rows;
`coregate_p1_per_fold.csv`, `coregate_p2_per_seed.csv`, and
`coregate_p1_p2_minimal_summary.{json,md}` aggregate them.
`coregate_label_blind_template_seed*` supplies the Lineage column of the fixed
template-disjoint split, and `template_disjoint_minimal_catalog.json` /
`template_disjoint_minimal_split.json` define that 105/255 split.

## Joint holdout and edge-rank control (paper Tables 2 and 3)

`coregate_joint_{ordinary,lineage}_seed*` and
`coregate_order_shuffled_{ordinary,lineage}_seed*` are the raw runs;
`coregate_joint_order_ablations.md`, `coregate_joint_order_ablations_per_fold.csv`, `coregate_joint_order_ablations_summary.json` hold the
per-fold records (train/test sizes, test classes, split hashes) and the
aggregates.

## Statistics audits (paper Table 4 and §5)

| Artifacts | Content |
|---|---|
| `coregate_three_view_statistics_audit.md`, `coregate_three_view_statistics_audit_per_fold.csv`, `coregate_three_view_statistics_audit_summary.json` | raw-only / masking-only / raw+mask count histograms on matched LOEO folds |
| `coregate_mask_only_audit.*`, `coregate_mask_feature_group_audit.json` | the earlier mixed count-and-ratio FullRule audit (91.41 / 91.36) and its feature groups |
| `coregate_fullrule_context_union_mask_audit.*` | the same audit for the rejected union (93.70 / 94.35) |
| `coregate_process_only_mask_audit.*` | mask audit for the process-only target (43.75 / 45.73) |
| `coregate_label_blind_lineage_mask_audit.*` | mask audit for the lineage target |

Mask-bearing views read held-out masks as oracle diagnostics. CoreGate inference
never receives them.

## Targets

| Artifacts | Content |
|---|---|
| `label_blind_lineage_masks.json` | the 360 lineage targets with positive counts and process counts |
| `label_blind_lineage_audit.json` | allowed/forbidden builder inputs, label-permutation invariance, min/median/max positives |
| `coregate_lineage_target_statistics.json` and a Markdown mirror | descriptive statistics quoted in the paper (mean 21.54, median 4, quartiles 2/5, range 1-879, seven maxima all T1553.004, 21.07-21.70% selected across folds) |
| `label_blind_lineage_drop{10,30,50}*.json` | deterministic positive-node dropout masks for the deletion curve |
| `fullrule_context_union_masks.json` | FullRule union label-blind context masks |

## Sensitivity and baseline studies

| Artifacts | Content |
|---|---|
| `coregate_process_only.md`, `coregate_process_only_per_fold.csv`, `coregate_process_only_summary.json` | conservative process-only target (95.31 / 95.52) |
| `coregate_fullrule_context_union.md`, `coregate_fullrule_context_union_per_fold.csv`, `coregate_fullrule_context_union_summary.json` | rejected union (94.77 / 94.89) |
| `coregate_label_blind_noise_sensitivity.md`, `coregate_label_blind_noise_sensitivity_per_fold.csv`, `coregate_label_blind_noise_sensitivity_summary.json`, `coregate_label_blind_noise_drop50_three_seed_per_fold.csv` | 10/30/50% deletion curve and the three-seed 50% condition |
| `coregate_learned_pooling_baselines.md`, `coregate_learned_pooling_baselines_per_fold.csv`, `coregate_learned_pooling_baselines_summary.json` | LearnedAttention and SAGPool-50% controls with parameter counts |

## Consistency checks

The summarizers assert their own leakage conditions — four folds per run,
cleared test masks, no joint template leakage, and shared train/test/support
hashes — and fail loudly if a packaged JSON does not satisfy them. They also
regenerate the `*_summary.*` and per-fold CSV files listed above, so a successful
run doubles as an internal-consistency check on the shipped artifacts.
