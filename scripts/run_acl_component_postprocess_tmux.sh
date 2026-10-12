#!/usr/bin/env bash
set -euo pipefail

PY=/home/user/anaconda3/envs/mh-denoise/bin/python
REPO=/home/user/hsoh/mh-denoise
ROOT=/mnt/ssd00/hsoh/acl_component_ablation_matrix_20261009_v1
RUNNER=/home/user/hsoh/rubric_aligned_gpt41_rejudge_20261009_v1/code/run_gpt41_checkpoint_queue.py
LOG="$ROOT/logs/postprocess.log"
export ACL_LLAMA_BATCH_SIZE=16

mkdir -p "$ROOT/logs" "$ROOT/results" "$ROOT/requests"
exec > >(tee -a "$LOG") 2>&1
cd "$REPO"

run_api_queue() {
  local requests="$1"
  local output="$2"
  local queue="$3"
  local count
  count=$(wc -l < "$requests")
  if [[ "$count" -eq 0 ]]; then
    printf '{"queue":"%s","status":"no_new_requests"}\n' "$queue"
    : > "$output"
    return
  fi
  if ! "$PY" -u "$RUNNER" \
      --requests "$requests" \
      --output "$output" \
      --queue-name "$queue" \
      --initial-concurrency 4 \
      --max-concurrency 8 \
      --max-retries 5 \
      --timeout 120; then
    printf '{"queue":"%s","status":"completed_with_final_failures"}\n' "$queue"
  fi
}

"$PY" -u scripts/verify_acl_component_generation.py
"$PY" -u scripts/run_acl_component_selector.py prepare
"$PY" -u scripts/run_acl_adapter_pairwise.py prepare
run_api_queue \
  "$ROOT/requests/new_selector_requests.jsonl" \
  "$ROOT/results/new_selector_results.jsonl" \
  acl_component_selector
run_api_queue \
  "$ROOT/requests/adapter_pairwise_gpt41_new.jsonl" \
  "$ROOT/results/adapter_pairwise_gpt41_new_results.jsonl" \
  adapter_pairwise
"$PY" -u scripts/run_acl_component_selector.py aggregate

"$PY" -u scripts/run_acl_component_pointwise.py prepare
"$PY" -u scripts/run_acl_component_llama_pointwise.py prepare

run_api_queue \
  "$ROOT/requests/new_pointwise_requests.jsonl" \
  "$ROOT/results/new_pointwise_results.jsonl" \
  acl_component_pointwise &
API_PID=$!
printf '{"stage":"pointwise_api","pid":%s,"status":"running_in_parallel"}\n' "$API_PID"

"$PY" -u scripts/run_acl_component_pointwise.py score-risk
"$PY" -u scripts/run_acl_component_llama_pointwise.py run
"$PY" -u scripts/run_acl_component_llama_pointwise.py run
"$PY" -u scripts/run_acl_adapter_pairwise.py run-llama
"$PY" -u scripts/run_acl_adapter_pairwise.py run-llama

wait "$API_PID"
"$PY" -u scripts/run_acl_adapter_pairwise.py aggregate
"$PY" -u scripts/run_acl_component_pointwise.py aggregate
"$PY" -u scripts/run_acl_component_pointwise.py package
printf '{"status":"acl_component_pipeline_complete"}\n'
