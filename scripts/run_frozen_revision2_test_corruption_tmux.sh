#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/user/hsoh/mh-denoise"
selection_dir="$repo_dir/data/fullpaper_acl_pipeline/frozen_revision2_test_selection_v1_20261008"
output_dir="$repo_dir/data/fullpaper_acl_pipeline/frozen_revision2_test_corruption_v1_20261008"
generator_dir="/home/user/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
judge_dir="/mnt/ssd00/user-qwen35-27b-hf/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
required_free_mib=70000
counselor_manifest="$repo_dir/outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/counselor_test_partial15_v1_20261008/manifest.json"
pause_state="$output_dir/pause_after_accepted.json"

mkdir -p "$output_dir"
wait_log="$output_dir/gpu_wait.log"
run_log="$output_dir/production.log"

cd "$repo_dir"
pause_args=()
if [[ ! -f "$counselor_manifest" ]] || \
   [[ "$(jq -r '.status // ""' "$counselor_manifest" 2>/dev/null || true)" != \
      "complete_counselor_blinded_15x3_export" ]]; then
  pause_args=(--pause-after-accepted 15 --pause-state-file "$pause_state")
fi
while true; do
  free_mib="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= required_free_mib )); then
    printf '%s gpu_ready free_mib=%s required_mib=%s\n' \
      "$(date --iso-8601=seconds)" "$free_mib" "$required_free_mib" >> "$wait_log"
    break
  fi
  printf '%s waiting_for_gpu free_mib=%s required_mib=%s\n' \
    "$(date --iso-8601=seconds)" "$free_mib" "$required_free_mib" >> "$wait_log"
  sleep 60
done

printf '%s\n' "$$" > "$output_dir/production.pid"
printf '%s production_launch pause_after_accepted=%s\n' \
  "$(date --iso-8601=seconds)" "$([[ ${#pause_args[@]} -gt 0 ]] && echo 15 || echo disabled)" \
  >> "$wait_log"
exec env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:. \
  .venv-fullpaper/bin/python -u scripts/run_fullpaper_corruption_production.py \
  --selection-file "$selection_dir/test_generation_selection.json" \
  --max-inputs 517 \
  --splits test \
  --known-holds-file "$repo_dir/data/fullpaper_acl_pipeline/development_training_shard_120_corrected_v2/held_out.jsonl" \
  --output-dir "$output_dir" \
  --worker-index 0 \
  --worker-count 1 \
  --selected-gpu 0 \
  --required-free-vram-mib "$required_free_mib" \
  --max-input-tokens 4096 \
  --generator-model-dir "$generator_dir" \
  --budget-ledger "$output_dir/no_paid_api_budget.sqlite3" \
  --max-api-requests 0 \
  --max-api-usd 0 \
  --input-usd-per-million-tokens 0 \
  --output-usd-per-million-tokens 0 \
  --budget-max-input-tokens-per-request 65536 \
  --judge-backend local-qwen35-27b \
  --local-judge-model-dir "$judge_dir" \
  --local-judge-max-new-tokens 3200 \
  --local-judge-timeout-seconds 300 \
  "${pause_args[@]}" \
  --automatic-clean-preflight >> "$run_log" 2>&1
