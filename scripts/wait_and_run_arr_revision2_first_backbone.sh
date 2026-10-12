#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/user/hsoh/mh-denoise"
asset_verification="$repo_dir/data/fullpaper_acl_pipeline/arr_first_run_20261004/model_assets_verified.json"
previous_state=""

cd "$repo_dir"
while true; do
  missing=()
  if [[ ! -f "$asset_verification" ]]; then
    missing+=("verified_model_assets_and_length_contract")
  elif ! python - "$asset_verification" <<'PY' >/dev/null 2>&1
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload.get("status") == "verified_complete_pinned_assets_and_length_contract"
assert payload.get("all_context_checks_within_limit") is True
PY
  then
    missing+=("valid_model_asset_verification")
  fi
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
