#!/usr/bin/env bash
set -euo pipefail

worktree="/home/user/hsoh/mh-denoise-feedback-v1-valid-20261008"
output_dir="/mnt/ssd00/hsoh/feedback_v1_bc_20261009"
python_bin="/home/user/anaconda3/envs/mh-denoise/bin/python"

cd "$worktree"
mkdir -p "$output_dir"
exec 9>"$output_dir/.run.lock"
if ! flock -n 9; then
  echo "feedback_v1 B-C extension is already running" >&2
  exit 73
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/dev/shm
export TRITON_CACHE_DIR=/dev/shm/triton-feedback-v1-bc
export TORCH_EXTENSIONS_DIR=/dev/shm/torch-extensions-feedback-v1-bc

"$python_bin" scripts/run_valid_feedback_v1_bc.py --output-dir "$output_dir" \
  2>&1 | tee -a "$output_dir/run.log"
