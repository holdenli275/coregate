# Verifying this release

## 1. Verify the published numbers without re-running training

Every number quoted in the paper is derived from the per-seed run JSONs in
`results/`. The summarizers are pure Python (numpy only) and need no GPU, no
dataset, and no training code.

```bash
pip install -r requirements.txt
pip install -e .            # makes the `coregate` package importable

# Each summarizer reads data/experiment_results/leave_one_env_v1, the directory
# the runners write to, and falls back to the packaged results/ when that
# directory does not exist. Both routes are equivalent: the released summaries
# in results/ are byte-identical to the ones these commands regenerate.

python3 code/summarize_coregate_joint_order_ablations.py
python3 code/summarize_label_blind_lineage_submission.py
python3 code/summarize_coregate_p1_p2.py
python3 code/summarize_coregate_learned_pooling_baselines.py
python3 code/summarize_coregate_process_only.py
python3 code/summarize_fullrule_context_union.py
python3 code/summarize_label_blind_noise_sensitivity.py
python3 code/summarize_lineage_target_statistics.py
```

Expected output, reproduced on 2026-09-24 against `results/` as packaged:

Accuracy / macro-F1 in percent.

| Summarizer | Result |
|---|---|
| `summarize_label_blind_lineage_submission.py` | CoreGate-Lineage 95.92 / 95.44 vs matched Ordinary 91.73 / 92.92, gain positive in 12/12 pairs; mask-only 38.70 / 34.59 |
| `summarize_coregate_joint_order_ablations.py` | ordered 91.73 / 92.92 vs 95.92 / 95.44; joint holdout 86.43 / 86.63 vs 95.12 / 95.06; shuffled ranks 90.67 / 92.01 vs 96.01 / 95.59 |
| `summarize_coregate_learned_pooling_baselines.py` | LearnedAttention 87.40 / 88.95, SAGPool-50% 87.48 / 88.38 vs CoreGate-Lineage 95.92 / 95.44 |
| `summarize_coregate_p1_p2.py` | P1 95.71 / 95.92 vs Ordinary 91.64 / 92.76; P2 template holdout 93.73 / 91.33 vs 89.93 / 85.83 |
| `summarize_coregate_process_only.py` | process-only mask 95.31 / 95.52 vs Ordinary 91.73 / 92.92; its mask-only audit 43.75 / 45.73 |
| `summarize_fullrule_context_union.py` | rejected ContextUnion 94.77 / 94.89 vs FullRule 95.77 / 95.92 while mask-only rises 91.41 / 91.36 → 93.70 / 94.35 |
| `summarize_label_blind_noise_sensitivity.py` | three-seed 50% dropout 95.92 / 95.74 vs Ordinary 91.73 / 92.92, 12/12 pairs positive; fixed-seed 10/30/50% curve 95.82–96.11 |
| `summarize_lineage_target_statistics.py` | 21.54 positive nodes per graph (median 4, quartiles 2/5, range 1–879, seven maxima all T1553.004); mean selected fraction 21.27%, 21.07–21.70% across folds |

Each summarizer asserts its own leakage conditions — four folds per run, cleared
test masks, no joint template leakage, and shared train/test/support hashes —
and fails loudly if the packaged JSONs do not satisfy them. Each one also
rewrites its own `*_summary.json`, `*_summary.md`, and per-fold CSV artifacts,
so a successful run doubles as an internal-consistency check on the shipped
files.

## 2. Verify the dataset

```bash
shasum -a 256 coregate_dataset_template_v2.tar.zst
# c0d52194277a5f4325735fb6471fbe50fb54d60df834efdeb6d93b3c87160617

tar -I zstd -xf coregate_dataset_template_v2.tar.zst
sha256sum -c data/linux_telemetry/reference_candidates_template_v2/checksums.sha256
```

The extracted tree is 374 sample directories (~3.24 GiB) plus catalog, data
card, audit, readiness, and checksum files.

## 3. Re-running the experiments: one module missing

`code/graph_base.py` and `coregate/graph_cache.py` both import `graph_tensor`
from `linux_graph_loader`, the dataset-to-tensor loader. That file was not part
of the archived package and is not in this repository, so
`code/run_coregate_tgn_leave_one_env.py` fails at import:

```
ModuleNotFoundError: No module named 'linux_graph_loader'
```

The loader returns one dict per sample with the keys the pipeline consumes:

```
x           (num_nodes, 27) float32
            the encoder reads 12 count channels: [:, 0:6] and [:, 21:27];
            the remaining columns are not used by the released feature layout
node_ids    list of str, one per row of x, in the same order
node_type   (num_nodes,) long, 0 = process, one-hot expanded to 4 indicators
edge_index  (2, num_edges) long
edge_type   (num_edges,) long, reduced mod 8 into the eight operation bins
```

`node_ids` is load-bearing: `_precomputed_mask_tensor` matches the released
`positive_node_ids` against it and raises if the lengths or counts disagree, so a
replacement loader that omits it cannot read the lineage masks. The edge order
is taken from the row order of `edge_index`, so the loader must not shuffle it.

Add the module and the full training path becomes importable.
`docs/coregate_reproduction.md` lists every reported command.

Do not invent a replacement loader: the reported numbers depend on the exact
feature layout produced by the original one.

### Test status

```
pytest code/tests  ->  1 passed
```

The test suite was reduced to the tests that still match the shipped modules.
The larger suites in the archived package covered modules that are not part of
this release.

### Coverage of `results/`

`results/` carries exactly the artifacts the paper reads. Everything quoted in
its tables and in the analysis text comes from the `coregate_p1_*`,
`coregate_p2_*`, `coregate_label_blind_*`, `coregate_sagpool_*`,
`coregate_joint_*`, `coregate_order_shuffled_*`,
`coregate_process_only_*`, `coregate_fullrule_context_union_*`,
`coregate_*_mask_audit*`, `coregate_three_view_statistics_audit*`,
`coregate_lineage_target_statistics*`, `label_blind_lineage_*`, and
`template_disjoint_*` files, all of which are present and consistent with the
summarizers in §1. The exploratory and feature-space-augmentation outputs of the
earlier internal package are not carried here.
