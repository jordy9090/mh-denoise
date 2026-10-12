#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/ssd00/hsoh/original_prompt_sft_control_20261009_v1
WORKTREE=/home/user/hsoh/mh-denoise-original-prompt-sft-control-20261009
API_PY=/home/user/anaconda3/envs/mh-denoise/bin/python
LOCAL_PY=/home/user/anaconda3/envs/mh-denoise/bin/python
API_RUNNER=/home/user/hsoh/mh-denoise/scripts/run_gpt41_checkpoint_queue.py

cd "$WORKTREE"
mkdir -p "$ROOT/logs" "$ROOT/results"

run_api_queue() {
  local queue=$1
  local request_path="$ROOT/requests/${queue}.jsonl"
  local result_path="$ROOT/results/${queue}.jsonl"
  if [[ ! -s "$request_path" ]]; then
    printf '{"queue":"%s","status":"complete","requests":0,"completed":0,"remaining":0}\n' "$queue" > "$ROOT/results/${queue}.manifest.json"
    return
  fi
  "$API_PY" "$API_RUNNER" \
    --requests "$request_path" \
    --output "$result_path" \
    --queue-name "$queue" \
    --initial-concurrency 4 \
    --max-concurrency 8 \
    --max-retries 5 \
    --timeout 120
}

(
  run_api_queue control_pointwise_gpt41
  run_api_queue control_pairwise_gpt41
  run_api_queue control_selector_gpt41
) > "$ROOT/logs/gpt41_evaluation.log" 2>&1 &
api_pid=$!

cleanup() {
  if kill -0 "$api_pid" 2>/dev/null; then
    kill "$api_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"$LOCAL_PY" -u scripts/run_llama_original_prompt_sft_control.py \
  > "$ROOT/logs/llama_evaluation.log" 2>&1
wait "$api_pid"
trap - EXIT

printf '{"status":"complete"}\n' > "$ROOT/results/evaluation_queues_complete.json"
