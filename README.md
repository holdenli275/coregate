# CoreGate: Execution-Lineage Supervision for Attack Technique Recognition under Environment Shift

Code, locked experiment outputs, dataset metadata, and citation material for the
paper *CoreGate: Execution-Lineage Supervision for Attack Technique Recognition
under Environment Shift*. Author names are withheld for double-blind review
and will be added on acceptance.

CoreGate turns process associations recorded in execution journals into
training-only foreground supervision. The targets need neither technique labels
nor technique-specific rules. A jointly trained encoder and gated readout keep
full-graph context, while inference needs only the observed graph: neither the
journal nor a stored mask is read at test time. The code is MIT-licensed; the
released result artifacts are CC BY 4.0.

## Main result

Matched leave-one-environment-out (LOEO) recognition over 360 attack-emulation
graphs and 45 classes, four folds x three seeds (20260921-20260923), 15 epochs,
one support graph per class, held-out evidence masks cleared. Accuracy / macro-F1
in percent, mean +/- SD over the three seed means.

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Ordinary | 91.73 +/- 0.67 | 92.92 +/- 0.53 |
| LearnedAttention (no foreground loss) | 87.40 +/- 1.17 | 88.95 +/- 1.51 |
| ShuffledLineage | 88.10 +/- 0.95 | 88.78 +/- 1.45 |
| SAGPool-50% | 87.48 +/- 2.04 | 88.38 +/- 2.28 |
| **CoreGate-Lineage** | **95.92 +/- 0.16** | **95.44 +/- 0.19** |
| CoreGate-FullRule | 95.77 +/- 0.38 | 95.92 +/- 0.61 |
| Lineage - Ordinary | **+4.19** | **+2.52** |

The scalar gate adds 129 parameters to the 225,839-parameter Ordinary model. The
gain is positive in all 12 seed-fold pairs (2.27-5.56 pp), persists under
embedding-only classification, a fixed template-disjoint split, joint
environment/template holdout, and edge-rank permutation, and survives deleting
half of the positive target nodes. Sensitivity variants (process-only target,
FullRule-ContextUnion) and the matched three-view statistics audit are in
`results/`.

## Claim boundaries

The boundaries below are the ones the released artifacts support; do not read
them more broadly.

- Targets are **automatically derived weak supervision**, not independently
  validated attack evidence, and not independent manual gold.
- The reported gain is a **within-catalog** gain from training-only foreground
  supervision. It is **not** evidence of causal subgraph recovery or of transfer
  to independent hosts and collectors.
- Simple count statistics still carry catalog-level class information (raw-only
  counts reach 78.25% accuracy under the same LOEO protocol, 2.22% chance), so
  the deployable claim rests on mask-free inference, not on the absence of every
  shortcut.
- Mask-bearing statistics audits read held-out masks as oracle diagnostics;
  CoreGate prediction never receives them.

## Repository layout

```
code/               runners, diagnostics, summarizers, and tests
coregate/           graph-transformer encoder, gated readout, episodic training
results/            locked runs, ablations, audits, per-fold records
dataset_metadata/   catalog, data card, checksums, audit summaries
docs/               reproduction commands, verification, claim positioning
paper/              artifact citation block, paper bibliography, citation notes
MANIFEST.md         release inventory
VERSION.md          release history
```

Naming: the paper's **CoreGate-Lineage** is the artifact key `coregate_lineage`
(`coregate_label_blind_*` files); **LearnedAttention** is `gate_no_evidence`,
**ShuffledLineage** is `gate_shuffled_lineage`, and **FullRule** is
`coregate_full_rule`. `results/README.md` lists artifacts group by group.

Two internal protocol names survive in file names and artifact fields, because
the locked runs record them verbatim and renaming them would break the recorded
commands: `episode_tgn` (the value of `training_protocol` in every run JSON) and
the `tgn` token in `code/run_coregate_tgn_*.py` and `coregate/tgn_episode.py`.
Both refer to the static pipeline in this repository - the graph-transformer
encoder in `coregate/graph_encoder.py` plus episodic prototypical training - not
to the recurrent temporal graph networks the paper contrasts with in Section 3.2.

## Dataset

The dataset is **not** stored in git, and per the paper's availability statement
the attack-emulation dataset is released upon acceptance. This repository ships
the catalog (`dataset_metadata/catalog.json`), the data card, and the per-sample
checksums.

`catalog.json` has 374 entries: 360 labeled attack-emulation graphs, 45 target
classes, four environment groups (55/54/163/88 graphs), 110 template groups of
which 55 cross environments, plus 14 unlabeled entries that are excluded from
evaluation. Samples were produced by executing Atomic Red Team tests in a
controlled Ubuntu 22.04 lab and capturing them with an eBPF collector; each
sample holds `graph.json`, `core_graph.json`, `events.jsonl`,
`annotations.jsonl`, `entity_identities.json`, and `subgraph_annotation.json`.
See `dataset_metadata/DATA_CARD.md` and `docs/DATA_PROVENANCE_AND_PRIVACY.md`.

If you already hold the dataset archive, place it at the repository root and
extract it:

```bash
tar -I zstd -xf coregate_dataset_template_v2.tar.zst
sha256sum -c data/linux_telemetry/reference_candidates_template_v2/checksums.sha256
# expected archive SHA-256:
# c0d52194277a5f4325735fb6471fbe50fb54d60df834efdeb6d93b3c87160617
```

The extracted tensor cache is optional and is regenerated by the runners.

## Installation

Python 3.8+ with PyTorch 2.0+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # makes the `coregate` package importable
# then install the PyG wheels matching your torch/CUDA build, e.g.
# pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu121.html
```

## Verifying the published numbers

Every number in the paper is recomputed from the per-seed run JSONs in
`results/` by pure-numpy summarizers. This path needs no GPU, no dataset, and no
training code:

```bash
# Reads data/experiment_results/leave_one_env_v1 when it exists (that is where
# the runners write) and falls back to the packaged results/ directory.
python3 code/summarize_label_blind_lineage_submission.py   # the main-table rows
python3 code/summarize_coregate_learned_pooling_baselines.py
python3 code/summarize_coregate_joint_order_ablations.py
python3 code/summarize_coregate_p1_p2.py
python3 code/summarize_coregate_process_only.py
python3 code/summarize_fullrule_context_union.py
python3 code/summarize_label_blind_noise_sensitivity.py
python3 code/summarize_lineage_target_statistics.py        # target-size statistics
```

`docs/VERIFICATION.md` lists the expected output of each. The summarizers
regenerate their own `*_summary.*` and per-fold CSV files, so a successful run
doubles as an internal-consistency check: the shipped `results/` files are
byte-identical to the regenerated ones.

## Re-running training: one module is not in this release

`code/graph_base.py` and `coregate/graph_cache.py` import `graph_tensor`
from `linux_graph_loader`, the dataset-to-tensor loader. That module is not part
of this release, so `code/run_coregate_tgn_leave_one_env.py` fails at import:

```
ModuleNotFoundError: No module named 'linux_graph_loader'
```

Add it (a JSON graph to PyG `Data` converter) before claiming end-to-end
reproducibility, and do not substitute a different feature layout: the reported
numbers depend on the exact one. `docs/coregate_reproduction.md` lists every
reported command; `docs/VERIFICATION.md` §3 records the dependency.

The locked primary command is:

```bash
python3 code/run_coregate_tgn_leave_one_env.py \
  --catalog data/linux_telemetry/reference_candidates_template_v2/catalog.json \
  --graph-field graph_path --epochs 15 --pooling learned_core \
  --core-gate-weight 0.1 --train-core-mask lineage --test-core-mask none \
  --classifier-scaler standard --share-gate-forward --augmentation none \
  --seed 20260921 \
  --out data/experiment_results/leave_one_env_v1/coregate_label_blind_lineage_seed20260921.json
```

`--test-core-mask none` is required for the mask-free claim and stays enabled in
every reported evaluation. Every reported result uses `--augmentation none`,
which is the only supported value; the earlier feature-space augmentation line is
not part of this release.

## Citation

GitHub renders `CITATION.cff` through the *Cite this repository* button.

```bibtex
@software{coregate2027artifact,
  title        = {{CoreGate}: Execution-Lineage Supervision for Attack Technique
                  Recognition under Environment Shift},
  author       = {{Anonymous Authors}},
  year         = {2027},
  version      = {0.7},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://github.com/holdenli275/coregate}
}
```

This repository is the anonymized double-blind copy: author names, the
repository owner, and the DOI are placeholders. A matching data-availability
paragraph and the dataset entry are provided in `paper/artifact_citation.tex`. The DOI placeholders must be replaced once the
release exists.

## License

- Code, scripts, and LaTeX sources: MIT, see `LICENSE`.
- Released result artifacts and the dataset: CC BY 4.0, see `DATA_LICENSE.md`.
- Upstream content (Atomic Red Team, MITRE ATT&CK) keeps its own license; see
  `docs/DATA_PROVENANCE_AND_PRIVACY.md`.

## Acknowledgements

The dataset was collected by executing MITRE Atomic Red Team tests in a
controlled lab. Built with PyTorch and PyTorch Geometric; technique identifiers
follow the MITRE ATT&CK framework.
