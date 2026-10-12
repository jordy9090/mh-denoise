#!/usr/bin/env bash
set -euo pipefail

worktree=/home/user/hsoh/mh-denoise-feedback-v1-valid-20261008
output_dir=/mnt/ssd00/hsoh/test24_local_four_axis_20261009
python=/home/user/anaconda3/envs/mh-denoise/bin/python

mkdir -p "$output_dir" /dev/shm/triton-test24 /dev/shm/torchext-test24
exec 9>"$output_dir/run.lock"
flock -n 9 || {
  printf '%s\n' "TEST24 local evaluation is already running." >&2
  exit 1
}

cd "$worktree"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$worktree/scripts${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR=/dev/shm
export TRITON_CACHE_DIR=/dev/shm/triton-test24
export TORCH_EXTENSIONS_DIR=/dev/shm/torchext-test24

"$python" scripts/run_test24_local_four_axis_eval.py \
  --output-dir "$output_dir" \
  --judge-max-input-tokens 16384 \
  --judge-max-new-tokens 384 \
  2>&1 | tee -a "$output_dir/run.log"
