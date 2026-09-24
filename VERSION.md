# Version 0.7.0 (public release)

Public release accompanying *CoreGate: Execution-Lineage Supervision for Attack
Technique Recognition under Environment Shift*. Scientific content, results, and
the dataset are unchanged from internal package v0.6.

Changes in 0.7.0:

- repository identity rewritten around the paper: title, abstract, main table,
  citation metadata (`.zenodo.json`, `CITATION.cff`), inventory, and docs;
- every remaining reference to the earlier third-party dataset family removed;
  the dataset-to-tensor loader entry point is now `linux_graph_loader`, and the
  tensor-cache helper was renamed `coregate/graph_cache.py`;
- HAN/GCN/GAT static baselines and the feature-space augmentation artifacts
  removed: no table or sentence of the paper reads them;
- `results/` trimmed to the artifacts the paper reads (211 -> 107 files) plus one
  new derived audit, `coregate_lineage_target_statistics.{json,md}`; the run
  records no longer carry the earlier protocols' configuration fields;
- display labels aligned with the paper (`CoreGate-Lineage`, `LearnedAttention`);
  result keys and file names unchanged;
- the encoder module and class renamed to the paper's wording
  (`coregate/graph_encoder.py`, `GraphTransformerEncoder`), and the augmentation
  and contrastive branches plus their CLI flags removed from both runners;
- the eight summarizers fall back to `results/` when the runner output directory
  is absent, so the published numbers can be verified from a fresh checkout;
- the superseded v0.6 manuscripts were removed from `paper/`, which now holds
  the artifact citation block, the paper's bibliography in citation order, and a
  citation README.

---

# Version 0.6

CoreGate-TGN ICASSP submission package, generated 2026-09-23.

Version 0.6 adds:

- three-seed, four-fold joint environment/template holdout and edge-rank
  permutation controls, with 12 raw run JSON files, 72 fold records,
  split-hash checks, and reproduction scripts;
- revised manuscripts that distinguish normalized edge-list rank from real
  timestamps and static graph attention from recurrent TGN memory;
- all v0.5 contents unchanged apart from the updated manuscripts and runner.

Version 0.5 contains the complete v0.4 evidence bundle plus:

- a matched Raw Graph only / strict mask only / raw+mask statistics audit for
  both FullRule and LabelBlindLineage masks, with 72 per-fold records;
- updated Markdown and LaTeX manuscript discussion of the strong FullRule
  fingerprint and the oracle-only status of held-out-mask diagnostic views;
- a focused feature-partition test and a reproduction command.

Version 0.4 added:

- the final label-blind execution-lineage implementation and reproduction
  command;
- five-condition matched LOEO results over 4 folds x 3 seeds;
- label-permutation, mask-only, sparsity, parameter, split/support, and
  test-mask leakage checks;
- the three-seed fixed template-disjoint pressure test;
- updated Markdown and LaTeX manuscripts;
- consolidated reviewer-risk and conservative-claim guidance.
- a fixed-seed 10%/30%/50% positive-node dropout curve plus a three-seed 50%
  strict-noise confirmation, with complete masks, raw runs, paired fold
  results, and reproduction commands.
- a matched three-seed SAGPool-50% baseline and the existing scalar
  LearnedAttention control, both without evidence supervision.
- a complete but rejected FullRule-ContextUnion sensitivity, including its
  stronger mask-only fingerprint and lower model result.
