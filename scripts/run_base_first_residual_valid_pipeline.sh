#!/usr/bin/env bash
set -euo pipefail

repo=/home/user/hsoh/mh-denoise
root=/mnt/ssd00/hsoh/base_first_residual_valid_20261011_v1
python=/home/user/anaconda3/envs/mh-denoise/bin/python

mkdir -p "$root/logs" "$root/results" "$root/requests"
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1

run_logged() {
  local log_name=$1
  shift
  "$@" 2>&1 | tee -a "$root/logs/$log_name.log"
}

run_logged recovery "$python" scripts/run_base_first_residual_valid_analysis.py recover-extraction
run_logged prepare "$python" scripts/run_base_first_residual_valid_analysis.py prepare
run_logged pairwise_prepare "$python" scripts/run_base_first_residual_valid_analysis.py prepare-pairwise

if [[ -s "$root/requests/new_pointwise_requests.jsonl" ]]; then
  run_logged gpt41_pointwise "$python" scripts/run_gpt41_checkpoint_queue.py \
    --requests "$root/requests/new_pointwise_requests.jsonl" \
    --output "$root/results/new_pointwise_results.jsonl" \
    --queue-name base_first_residual_valid_pointwise \
    --initial-concurrency 4 --max-concurrency 8
else
  : > "$root/results/new_pointwise_results.jsonl"
fi

if [[ -s "$root/requests/new_pairwise_requests.jsonl" ]]; then
  run_logged gpt41_pairwise "$python" scripts/run_gpt41_checkpoint_queue.py \
    --requests "$root/requests/new_pairwise_requests.jsonl" \
    --output "$root/results/new_pairwise_results.jsonl" \
    --queue-name base_first_residual_valid_pairwise \
    --initial-concurrency 4 --max-concurrency 8
else
  : > "$root/results/new_pairwise_results.jsonl"
fi

run_logged risk "$python" scripts/run_base_first_residual_valid_analysis.py score-risk
run_logged aggregate "$python" scripts/run_base_first_residual_valid_analysis.py aggregate
run_logged pairwise_aggregate "$python" scripts/run_base_first_residual_valid_analysis.py aggregate-pairwise

date --iso-8601=seconds > "$root/PIPELINE_COMPLETE.txt"
