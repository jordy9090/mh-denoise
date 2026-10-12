#!/usr/bin/env bash
set -euo pipefail

repo=/home/user/hsoh/mh-denoise
root=/mnt/ssd00/hsoh/base_first_residual_valid_20261011_v1
python=/home/user/anaconda3/envs/mh-denoise/bin/python

mkdir -p "$root/logs" "$root/results" "$root/requests"
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1
export BASE_DRAFT_LLAMA_BATCH_SIZE=8

run_logged() {
  local log_name=$1
  shift
  "$@" 2>&1 | tee -a "$root/logs/$log_name.log"
}

run_logged second_pass_prepare "$python" scripts/run_valid43_base_draft_control_analysis.py prepare

if [[ -s "$root/requests/second_pass_new_pointwise.jsonl" ]]; then
  run_logged second_pass_gpt41_pointwise "$python" scripts/run_gpt41_checkpoint_queue.py \
    --requests "$root/requests/second_pass_new_pointwise.jsonl" \
    --output "$root/results/second_pass_new_pointwise.jsonl" \
    --queue-name base_draft_controls_valid_pointwise \
    --initial-concurrency 4 --max-concurrency 8
else
  : > "$root/results/second_pass_new_pointwise.jsonl"
fi

if [[ -s "$root/requests/second_pass_new_pairwise.jsonl" ]]; then
  run_logged second_pass_gpt41_pairwise "$python" scripts/run_gpt41_checkpoint_queue.py \
    --requests "$root/requests/second_pass_new_pairwise.jsonl" \
    --output "$root/results/second_pass_new_pairwise.jsonl" \
    --queue-name base_draft_controls_valid_pairwise \
    --initial-concurrency 4 --max-concurrency 8
else
  : > "$root/results/second_pass_new_pairwise.jsonl"
fi

run_logged second_pass_risk "$python" scripts/run_valid43_base_draft_control_analysis.py score-risk
run_logged second_pass_aggregate "$python" scripts/run_valid43_base_draft_control_analysis.py aggregate
run_logged second_pass_llama_prepare "$python" scripts/run_valid43_base_draft_control_llama.py prepare
run_logged second_pass_llama "$python" scripts/run_valid43_base_draft_control_llama.py run
run_logged second_pass_llama_aggregate "$python" scripts/run_valid43_base_draft_control_llama.py aggregate

date --iso-8601=seconds > "$root/SECOND_PASS_PIPELINE_COMPLETE.txt"
