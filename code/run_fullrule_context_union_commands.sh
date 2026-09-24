#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
catalog=data/linux_telemetry/reference_candidates_template_v2/catalog.json
outdir=data/experiment_results/leave_one_env_v1

python3 "$script_dir/build_fullrule_context_union_masks.py"
python3 "$script_dir/run_coregate_mask_only_audit.py" \
  --catalog "$catalog" --mask-variant context_union --permutations 100 \
  --context-union-mask-file "$outdir/fullrule_context_union_masks.json" \
  --out-dir "$outdir" \
  --output-prefix coregate_fullrule_context_union_mask_audit

for seed in 20260921 20260922 20260923; do
  python3 "$script_dir/run_coregate_tgn_leave_one_env.py" \
    --catalog "$catalog" --graph-field graph_path --pooling learned_core \
    --core-gate-weight 0.1 --train-core-mask context_union \
    --context-union-mask-file "$outdir/fullrule_context_union_masks.json" \
    --test-core-mask none --augmentation none --epochs 15 \
    --classifier-scaler standard --share-gate-forward --seed "$seed" \
    --out "$outdir/coregate_fullrule_context_union_seed${seed}.json"
done

python3 "$script_dir/summarize_fullrule_context_union.py"
