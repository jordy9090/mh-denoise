#!/usr/bin/env bash
set -euo pipefail

cd /home/user/hsoh/mh-denoise
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/dev/shm
export TRITON_CACHE_DIR=/dev/shm/triton-gpt41-posthoc
export TORCH_EXTENSIONS_DIR=/dev/shm/torchext-gpt41-posthoc
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
  /mnt/ssd00/hsoh/gpt41_posthoc_selection_20261009_v1/logs

/home/user/anaconda3/envs/mh-denoise/bin/python scripts/run_llama_posthoc_common.py \
  2>&1 | tee -a /mnt/ssd00/hsoh/gpt41_posthoc_selection_20261009_v1/logs/llama_posthoc_common.log
