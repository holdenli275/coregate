#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
outdir=data/experiment_results/leave_one_env_v1

for seed in 20260921 20260922 20260923; do
  python3 "$script_dir/run_coregate_tgn_leave_one_env.py" \
    --graph-field graph_path --pooling ordinary --test-core-mask none \
    --augmentation none --epochs 15 --classifier-scaler standard \
    --joint-template-disjoint --seed "$seed" \
    --out "$outdir/coregate_joint_ordinary_seed${seed}.json"
  python3 "$script_dir/run_coregate_tgn_leave_one_env.py" \
    --graph-field graph_path --pooling learned_core --core-gate-weight 0.1 \
    --train-core-mask lineage --test-core-mask none --augmentation none \
    --epochs 15 --classifier-scaler standard --share-gate-forward \
    --joint-template-disjoint --seed "$seed" \
    --out "$outdir/coregate_joint_lineage_seed${seed}.json"
  python3 "$script_dir/run_coregate_tgn_leave_one_env.py" \
    --graph-field graph_path --pooling ordinary --test-core-mask none \
    --augmentation none --epochs 15 --classifier-scaler standard \
    --edge-time-mode shuffled --seed "$seed" \
    --out "$outdir/coregate_order_shuffled_ordinary_seed${seed}.json"
  python3 "$script_dir/run_coregate_tgn_leave_one_env.py" \
    --graph-field graph_path --pooling learned_core --core-gate-weight 0.1 \
    --train-core-mask lineage --test-core-mask none --augmentation none \
    --epochs 15 --classifier-scaler standard --share-gate-forward \
    --edge-time-mode shuffled --seed "$seed" \
    --out "$outdir/coregate_order_shuffled_lineage_seed${seed}.json"
done

python3 "$script_dir/summarize_coregate_joint_order_ablations.py"
