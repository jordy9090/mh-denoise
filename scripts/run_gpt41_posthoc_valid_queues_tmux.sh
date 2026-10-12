#!/usr/bin/env bash
set -euo pipefail

repo=/home/user/hsoh/mh-denoise
out=/mnt/ssd00/hsoh/gpt41_posthoc_selection_20261009_v1
python=/home/user/anaconda3/envs/mh-denoise/bin/python

mkdir -p "$out/results" "$out/logs"
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"

"$python" scripts/prepare_gpt41_posthoc_valid.py 2>&1 | tee -a "$out/logs/valid_prepare.log"
for queue in valid_pointwise valid_selector valid_pairwise_replay63 valid_pairwise_new27; do
  "$python" scripts/run_gpt41_checkpoint_queue.py \
    --requests "$out/requests/${queue}.jsonl" \
    --output "$out/results/${queue}_gpt41.jsonl" \
    --queue-name "$queue" --initial-concurrency 4 --max-concurrency 8 \
    2>&1 | tee -a "$out/logs/${queue}.log"
done
