# Reproducing the CoreGate experiments

> **Path note.** The runners live in `code/`, not at the project root, and the
> `coregate` package must be importable (`pip install -e .` from the repository
> root). Run everything from the repository root: the output paths are relative.
>
> **Dependency note.** The runners import `linux_graph_loader`, the
> dataset-to-tensor loader, which is not part of this release. Add it before
> running anything below; see `docs/VERIFICATION.md` §3.
>
> **Augmentation note.** `--augmentation` accepts only `none`, which is what
> every reported result uses.
>
> **Verification note.** To check the published numbers without retraining, use
> `docs/VERIFICATION.md` §1.

The catalog is `data/linux_telemetry/reference_candidates_template_v2/catalog.json`
(extracted from the separate dataset archive). All reported runs share the
encoder architecture, splits, support selection, classifier, and training budget:
15 epochs, one graph per step, StandardScaler before a class-balanced ℓ2 logistic
regression (`C = 1`), and `--test-core-mask none`, which is required for the
mask-free claim and must stay enabled in every final evaluation.

## Main table

CoreGate-Lineage (journal-derived, technique-label-blind targets):

```bash
for seed in 20260921 20260922 20260923; do
  python3 code/run_coregate_tgn_leave_one_env.py \
    --catalog data/linux_telemetry/reference_candidates_template_v2/catalog.json \
    --graph-field graph_path --epochs 15 --pooling learned_core \
    --core-gate-weight 0.1 --train-core-mask lineage --test-core-mask none \
    --classifier-scaler standard --share-gate-forward --augmentation none \
    --seed "$seed" \
    --out data/experiment_results/leave_one_env_v1/coregate_label_blind_lineage_seed${seed}.json
done
```

The lineage targets themselves are built first, with:

```bash
bash code/run_label_blind_lineage_minimal_commands.sh
```

That script constructs the targets through an explicit input allowlist, checks
that label permutation changes 0 of 360 targets, and runs the matched Ordinary,
LearnedAttention and Gate-ShuffledLineage controls plus the fixed
template-disjoint pressure test. The CoreGate-FullRule row and the Ordinary row of
the main table come from `code/run_coregate_p1_p2_commands.sh`, which also writes
the embedding-only control and the template-disjoint split.

The matched pooling baselines (LearnedAttention, SAGPool-50%) are reproduced with:

```bash
bash code/run_coregate_sagpool_commands.sh
```

## Diagnostics

Joint environment/template holdout and edge-rank permutation:

```bash
bash code/run_coregate_joint_order_ablation_commands.sh
```

Target deletion (10/30/50% positive-node dropout, plus the three-seed strict 50%
condition):

```bash
bash code/run_label_blind_noise_sensitivity_commands.sh
```

Process-only target sensitivity:

```bash
for seed in 20260921 20260922 20260923; do
  python3 code/run_coregate_tgn_leave_one_env.py \
    --catalog data/linux_telemetry/reference_candidates_template_v2/catalog.json \
    --graph-field graph_path --pooling learned_core --core-gate-weight 0.1 \
    --train-core-mask process_only --test-core-mask none --augmentation none \
    --epochs 15 --classifier-scaler standard --share-gate-forward --seed "$seed" \
    --out data/experiment_results/leave_one_env_v1/coregate_process_only_seed${seed}.json
done
python3 code/summarize_coregate_process_only.py
python3 code/run_coregate_mask_only_audit.py --mask-variant process_only \
  --output-prefix coregate_process_only_mask_audit --permutations 100
```

Rejected FullRule-ContextUnion sensitivity:

```bash
bash code/run_fullrule_context_union_commands.sh
```

That command builds the union targets, runs their mask audit and the three
matched model seeds, and writes the `coregate_fullrule_context_union*` artifacts.

## Statistics audits

Three-view count audit (raw-only, strict masking-only, raw+mask) on matched LOEO
folds, with the lineage target file present:

```bash
python3 code/run_coregate_three_view_statistics_audit.py \
  --out-dir data/experiment_results/leave_one_env_v1
```

Legacy mixed count-and-ratio mask audit, including the FullRule feature groups:

```bash
python3 code/run_coregate_mask_only_audit.py --mask-variant evidence \
  --output-prefix coregate_mask_only_audit --permutations 100
python3 code/analyze_coregate_mask_feature_groups.py
```

Descriptive target statistics quoted in the paper:

```bash
python3 code/summarize_lineage_target_statistics.py
```

Graph tensors are cached beside the catalog after the first run, so later
repetitions avoid reparsing the graph files. The cache is keyed by sample id and
graph path only: if you replace `linux_graph_loader`, delete
`.<catalog-stem>_<graph-field>_tensor_cache.pt` beside the catalog first, or the
runners will read tensors built by the previous loader.
