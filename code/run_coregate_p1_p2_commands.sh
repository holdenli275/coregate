#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

catalog=data/linux_telemetry/reference_candidates_template_v2/catalog.json
split_catalog=data/experiment_results/leave_one_env_v1/template_disjoint_minimal_catalog.json
outdir=data/experiment_results/leave_one_env_v1

python3 "$script_dir/build_template_disjoint_minimal_catalog.py"

for seed in 20260921 20260922 20260923; do
  python3 "$script_dir/run_coregate_tgn_leave_one_env.py" \
    --catalog "$catalog" --graph-field graph_path --pooling ordinary \
    --test-core-mask none --augmentation none --epochs 15 \
    --classifier-scaler standard --seed "$seed" \
    --out "$outdir/coregate_p1_standard_ordinary_seed${seed}.json"
  python3 "$script_dir/run_coregate_tgn_leave_one_env.py" \
    --catalog "$catalog" --graph-field graph_path --pooling learned_core \
    --core-gate-weight 0.1 --test-core-mask none --augmentation none --epochs 15 \
    --classifier-scaler standard --share-gate-forward --seed "$seed" \
    --out "$outdir/coregate_p1_standard_coregate_seed${seed}.json"
  python3 "$script_dir/run_coregate_tgn_leave_one_env.py" \
    --catalog "$split_catalog" --graph-field graph_path \
    --holdout template_disjoint_test --pooling ordinary \
    --test-core-mask none --augmentation none --epochs 15 \
    --classifier-scaler standard --seed "$seed" \
    --out "$outdir/coregate_p2_template_ordinary_seed${seed}.json"
  python3 "$script_dir/run_coregate_tgn_leave_one_env.py" \
    --catalog "$split_catalog" --graph-field graph_path \
    --holdout template_disjoint_test --pooling learned_core \
    --core-gate-weight 0.1 --test-core-mask none --augmentation none --epochs 15 \
    --classifier-scaler standard --share-gate-forward --seed "$seed" \
    --out "$outdir/coregate_p2_template_coregate_seed${seed}.json"
done

python3 "$script_dir/summarize_coregate_p1_p2.py"
