# CoreGate release inventory (v0.7.0)

This repository contains the reproduction code, locked experiment outputs,
dataset metadata, and citation material for *CoreGate: Execution-Lineage
Supervision for Attack Technique Recognition under Environment Shift*.

The inventory was re-cut for v0.7.0 from the internal ICASSP package v0.6: every
identifier was renamed to CoreGate, all references to other projects and to the
earlier feature-space augmentation line were removed, and `results/` was reduced
to the artifacts the paper reads. Where this file
and the repository disagree, the repository is authoritative.

## Contents

- `code/`: experiment runners, diagnostics, the eight result summarizers, a
  target-statistics audit, and tests.
- `coregate/`: graph-transformer encoder, gated readout, episodic training,
  lineage targets, and the graph-tensor cache helper.
- `results/`: locked runs, ordinary controls, ablations, sensitivity checks, and
  the audits described below.
  `results/README.md` lists the artifacts group by group, including the
  paper-to-artifact naming map.
- `dataset_metadata/`: catalog (374 entries, 360 labeled), data card, version,
  per-sample checksums, and audit summaries.
- `docs/`: reproduction commands, verification steps, claim boundaries, and
  data provenance.
- `paper/`: the artifact citation block, the paper's bibliography, and a note on
  how to cite. It holds no manuscript.

## Main reported result

Matched LOEO over 360 labeled attack-emulation graphs, 45 classes, four folds,
three seeds (20260921-20260923), 15 epochs, one support graph per class, held-out
masks cleared. Accuracy / macro-F1 in percent, mean +/- SD over seed means.

- Ordinary: 91.73 +/- 0.67 / 92.92 +/- 0.53;
- LearnedAttention (no foreground loss): 87.40 +/- 1.17 / 88.95 +/- 1.51;
- ShuffledLineage: 88.10 +/- 0.95 / 88.78 +/- 1.45;
- SAGPool-50%: 87.48 +/- 2.04 / 88.38 +/- 2.28;
- **CoreGate-Lineage: 95.92 +/- 0.16 / 95.44 +/- 0.19**;
- CoreGate-FullRule: 95.77 +/- 0.38 / 95.92 +/- 0.61;
- paired Lineage - Ordinary gain: +4.19 / +2.52 points, positive in 12/12
  seed-fold pairs (accuracy range 2.27-5.56 pp).

The scalar gate adds 129 parameters to the 225,839-parameter Ordinary model;
SAGPool-50% has 226,097.

## Artifact inventory

**Main table and gate controls.** `coregate_label_blind_lineage_seed*`,
`coregate_label_blind_noevidence_seed*` (LearnedAttention),
`coregate_label_blind_shuffled_seed*` (ShuffledLineage),
`coregate_label_blind_minimal_{per_fold.csv,summary.json,summary.md,paper_text.md}`,
and `coregate_label_blind_minimal_per_fold.csv`, which carries the 12 seed-fold
paired differences. `coregate_sagpool_seed*` and
`coregate_learned_pooling_baselines.{md,per_fold.csv,summary.json}` hold the
matched pooling baselines and the parameter counts.

**FullRule reference and the four-condition ablation.** `coregate_full_rule`
runs are `coregate_p1_standard_coregate_seed*`;
`coregate_minimal_ablation_{per_fold.csv,summary.json,summary.md,paper_text.md}`
contain 48 paired rows for Ordinary, LearnedAttention, Gate-Shuffled and
CoreGate-Real, with split/support hashes, parameter counts, sparsity checks, and
test-mask checks.

**Classifier-input and split diagnostics (paper Table 2).**
`coregate_p1_per_fold.csv`, `coregate_p2_per_seed.csv`, and
`coregate_p1_p2_minimal_summary.{json,md}` cover embedding-only versus
embedding+prototype inputs and the fixed 105/255 template-disjoint split;
`template_disjoint_minimal_{catalog,split}.json` define that split.
`coregate_p1_standard_*` and `coregate_p2_template_*` are the per-seed runs.

**Joint holdout and edge-rank control (paper Table 3).**
`coregate_joint_{ordinary,lineage}_seed*`,
`coregate_order_shuffled_{ordinary,lineage}_seed*`,
`results/coregate_joint_order_ablations.{md,per_fold.csv,summary.json}`. The CSV
holds 72 per-fold rows with train/test sizes, class counts, and split hashes.
Joint holdout gives 86.43 / 86.63 for Ordinary and 95.12 / 95.06 for Lineage;
shuffled edge ranks give 90.67 / 92.01 and 96.01 / 95.59; ordered
91.73 / 92.92 and 95.92 / 95.44. All paired train/test/support hashes match.

**Three-view statistics audit (paper Table 4).**
`results/coregate_three_view_statistics_audit.{md,per_fold.csv,summary.json}`
separates raw-only (78.25 / 77.10), strict mask-only (FullRule 89.57 / 88.84,
Lineage 21.12 / 20.67), and concatenated raw+mask counts (93.07 / 94.27 and
79.44 / 78.34) on matched LOEO folds. Mask-bearing views read held-out masks as
oracle diagnostics only.

**Legacy mask-only audit and mask files.**
`coregate_mask_only_audit.{md,per_fold.csv,permutation.csv,summary.json}` and
`coregate_mask_feature_group_audit.json` document the 91.41 / 91.36 FullRule
fingerprint (2.22% chance).

**Lineage target.** `label_blind_lineage_masks.json` (360 target masks with
positive counts and process counts), `label_blind_lineage_audit.json` (allowed and
forbidden inputs, label-permutation invariance, min/median/max positive counts),
and `coregate_label_blind_lineage_mask_audit*`. The builder reads only
`sample_id`, `annotation_path`, `events_path`, and `graph_path`; permuting all
technique labels changes 0 of 360 mask hashes.
`coregate_label_blind_minimal_summary.json` includes the fixed template-disjoint
pressure test: Ordinary 89.93 / 85.83, Lineage 93.59 / 90.63, FullRule
93.73 / 91.33.

**Target statistics.** `code/summarize_lineage_target_statistics.py` recomputes
the descriptive statistics quoted in the paper from
`label_blind_lineage_masks.json` and the catalog: mean 21.54 positive nodes per
graph, median 4, quartiles 2/4/5, range 1-879 with all seven maximum targets
belonging to one technique, and mean selected fractions of 21.07-21.70% across
folds. It writes `coregate_lineage_target_statistics.{json,md}`.

**Target-deletion sensitivity.** `coregate_label_blind_noise_sensitivity.md`,
its JSON summary and per-fold CSVs, the deterministic dropout mask files
`label_blind_lineage_drop{10,30,50}*`, and the 36-row three-seed strict-noise CSV
`coregate_label_blind_noise_drop50_three_seed_per_fold.csv`. The fixed-seed
10/30/50% curve ranges 95.82-96.11 accuracy and 95.57-95.82 macro-F1; the
three-seed 50% condition gives 95.92 / 95.74 with all 12 pairs improving.

**Sensitivity variants.** `coregate_process_only.{md,per_fold.csv,summary.json}`
and its mask audit give 95.31 / 95.52 for the process-only target, with a
43.75 / 45.73 mask-only audit. `coregate_fullrule_context_union.{md,per_fold.csv,summary.json}`
and its mask audit record the rejected union: it adds 498 nodes to 8,608 FullRule
positives, changes 218 of 360 graphs, raises mask-only accuracy from 91.41 / 91.36
to 93.70 / 94.35, and lowers the matched model result to 94.77 / 94.89.

**Other checks.** `template_disjoint_minimal_catalog.json` /
`template_disjoint_minimal_split.json` (the fixed 105/255 template-disjoint
split), `label_blind_lineage_drop{10,30,50}*.json` (the deterministic
positive-node dropout masks), `coregate_label_blind_lineage_mask_audit*`,
`coregate_process_only_mask_audit*`, and `coregate_fullrule_context_union_mask_audit*`
(per-variant mask audits), and the `test_core_mask` leakage assertions inside
every summarizer.

## Reproduction

Run from the repository root after `pip install -r requirements.txt` and
`pip install -e .`. The published numbers need neither GPU nor dataset:
`docs/VERIFICATION.md` §1 lists the eight summarizers and their expected output.
Training additionally needs `linux_graph_loader`, a dataset-to-tensor module that
is not part of this release (§3). The dataset archive is a separate deposit,
released upon acceptance of the paper.

## Claim boundary

Targets are automatically derived weak supervision, not independently validated
attack evidence: independent manual gold is zero. The gain is a within-catalog
recognition gain from training-only foreground supervision. It is not causal
subgraph recovery, not transfer to independent hosts or collectors, and not a
state-of-the-art claim. Mask-bearing statistics audits use held-out masks as
oracle diagnostics and never as CoreGate inputs.
