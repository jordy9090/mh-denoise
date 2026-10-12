#!/usr/bin/env bash
set -euo pipefail

repo=/home/user/hsoh/mh-denoise
out=/mnt/ssd00/hsoh/gpt41_posthoc_selection_20261009_v1
python=/home/user/anaconda3/envs/mh-denoise/bin/python

mkdir -p "$out/results" "$out/logs"
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1

"$python" scripts/prepare_gpt41_posthoc_test.py 2>&1 | tee -a "$out/logs/test_prepare.log"
"$python" scripts/run_gpt41_checkpoint_queue.py \
  --requests "$out/requests/test_pointwise.jsonl" \
  --output "$out/results/test_pointwise_gpt41.jsonl" \
  --queue-name test_pointwise --initial-concurrency 4 --max-concurrency 8 \
  2>&1 | tee -a "$out/logs/test_pointwise.log"
"$python" scripts/run_gpt41_checkpoint_queue.py \
  --requests "$out/requests/test_selector.jsonl" \
  --output "$out/results/test_selector_gpt41.jsonl" \
  --queue-name test_selector --initial-concurrency 4 --max-concurrency 8 \
  2>&1 | tee -a "$out/logs/test_selector.log"
