#!/usr/bin/env bash
set -euo pipefail

repo_dir=/home/user/hsoh/mh-denoise
run_dir="$repo_dir/data/fullpaper_acl_pipeline/production_accepted652_reuse_rejudge_full_v1_20261001"
wait_log="$run_dir/gpu_wait.log"
run_log="$run_dir/run.log"

cd "$repo_dir"
while true; do
  gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  if [[ "$gpu_used_mib" =~ ^[0-9]+$ ]] && (( gpu_used_mib <= 11000 )); then
    break
  fi
  date --iso-8601=seconds >> "$wait_log"
  echo "waiting_for_70000_MiB_free gpu_used_mib=$gpu_used_mib" >> "$wait_log"
  sleep 60
done

exec env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:. \
  .venv-fullpaper/bin/python -u scripts/rejudge_production_reuse_subset.py \
  --production-dir data/fullpaper_acl_pipeline/production_run_train1000_valid200_local_qwen35_27b_v3r1_20260910 \
  --prior-review-ledger data/fullpaper_acl_pipeline/production_run_train1000_valid200_local_qwen35_27b_v3r1_20260910_provisional_corrected_v2_20260926/review_ledger.jsonl \
  --codex-review-ledger data/fullpaper_acl_pipeline/codex_clean_candidate_review_batch20_v2_20260926/cumulative_ai_review_ledger.jsonl \
  --output-dir data/fullpaper_acl_pipeline/production_accepted652_reuse_rejudge_full_v1_20261001 \
  --judge-model-dir /mnt/ssd00/user-qwen35-27b-hf/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654 \
  --judge-max-new-tokens 3200 \
  --judge-timeout-seconds 1500 \
  --batch-size 8 \
  --seed-checkpoint data/fullpaper_acl_pipeline/production_accepted652_reuse_rejudge_full_v1_20261001/seed_rejudged_checkpoint.jsonl \
  --explicit-ids-file data/fullpaper_acl_pipeline/production_accepted652_reuse_rejudge_full_v1_20261001/explicit_ids.txt \
  >> "$run_log" 2>&1
