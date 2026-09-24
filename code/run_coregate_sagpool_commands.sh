#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
catalog=data/linux_telemetry/reference_candidates_template_v2/catalog.json
outdir=data/experiment_results/leave_one_env_v1

for seed in 20260921 20260922 20260923; do
  python3 "$script_dir/run_coregate_tgn_leave_one_env.py" \
    --catalog "$catalog" --graph-field graph_path --pooling sagpool \
    --train-core-mask none --test-core-mask none --augmentation none \
    --epochs 15 --classifier-scaler standard --seed "$seed" \
    --out "$outdir/coregate_sagpool_seed${seed}.json"
done

python3 "$script_dir/summarize_coregate_learned_pooling_baselines.py"
