#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/user/hsoh/mh-denoise"
gemma_config="/home/user/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2/config.json"
risk_config="/home/user/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594/config.json"
previous_state=""

cd "$repo_dir"
while true; do
  missing=()
  [[ -f "$gemma_config" ]] || missing+=("gemma_revision")
  [[ -f "$risk_config" ]] || missing+=("bert_base_revision")
  gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | paste -sd, -)"
  state="missing=${missing[*]:-none};gpu_pids=${gpu_pids:-none}"
  if [[ "$state" != "$previous_state" ]]; then
    printf '%s waiting %s\n' "$(date -Is)" "$state"
    previous_state="$state"
  fi
  if [[ ${#missing[@]} -eq 0 && -z "$gpu_pids" ]]; then
    printf '%s prerequisites satisfied; starting frozen ARR Revision 2 run\n' "$(date -Is)"
    exec bash scripts/run_arr_revision2_first_backbone.sh
  fi
  sleep 60
done
