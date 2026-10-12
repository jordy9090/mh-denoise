#!/usr/bin/env bash
set -euo pipefail

worktree="/home/user/hsoh/mh-denoise-feedback-v1-valid-20261008"
output_dir="$worktree/outputs/valid_feedback_v1_20261008"
python_bin="/home/user/anaconda3/envs/mh-denoise/bin/python"

cd "$worktree"
mkdir -p "$output_dir"
exec 9>"$output_dir/.run.lock"
if ! flock -n 9; then
  echo "feedback_v1 is already running for $output_dir" >&2
  exit 73
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="scripts:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

"$python_bin" scripts/run_valid_feedback_v1.py --stage all --output-dir "$output_dir" \
  2>&1 | tee -a "$output_dir/run.log"
