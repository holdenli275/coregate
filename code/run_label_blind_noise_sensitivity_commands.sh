#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
catalog=data/linux_telemetry/reference_candidates_template_v2/catalog.json
outdir=data/experiment_results/leave_one_env_v1
source_mask="$outdir/label_blind_lineage_masks.json"
seed=20260921

for specification in "0.1 10" "0.3 30" "0.5 50"; do
  read -r rate label <<< "$specification"
  python3 "$script_dir/build_label_blind_lineage_noise_masks.py" \
    --source "$source_mask" --drop-rate "$rate" --noise-seed "$seed" \
    --out "$outdir/label_blind_lineage_drop${label}.json"
  python3 "$script_dir/run_coregate_tgn_leave_one_env.py" \
    --catalog "$catalog" --graph-field graph_path --pooling learned_core \
    --core-gate-weight 0.1 --train-core-mask lineage \
    --lineage-mask-file "$outdir/label_blind_lineage_drop${label}.json" \
    --test-core-mask none --augmentation none --epochs 15 \
    --classifier-scaler standard --share-gate-forward --seed "$seed" \
    --out "$outdir/coregate_label_blind_noise_drop${label}_seed${seed}.json"
done

for seed in 20260922 20260923; do
  python3 "$script_dir/build_label_blind_lineage_noise_masks.py" \
    --source "$source_mask" --drop-rate 0.5 --noise-seed "$seed" \
    --out "$outdir/label_blind_lineage_drop50_seed${seed}.json"
  python3 "$script_dir/run_coregate_tgn_leave_one_env.py" \
    --catalog "$catalog" --graph-field graph_path --pooling learned_core \
    --core-gate-weight 0.1 --train-core-mask lineage \
    --lineage-mask-file "$outdir/label_blind_lineage_drop50_seed${seed}.json" \
    --test-core-mask none --augmentation none --epochs 15 \
    --classifier-scaler standard --share-gate-forward --seed "$seed" \
    --out "$outdir/coregate_label_blind_noise_drop50_seed${seed}.json"
done

python3 "$script_dir/summarize_label_blind_noise_sensitivity.py"
