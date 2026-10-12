#!/usr/bin/env bash
set -euo pipefail

repo="/home/user/hsoh/mh-denoise"
worktree="/home/user/hsoh/mh-denoise-surface-relaxed-v1-20261008"
handoff="/home/user/hsoh/revision2_test_live_handoff_20261008"
output="/home/user/hsoh/test60_rubric_aligned_final_selection_20261009_v1"
python_bin="/home/user/anaconda3/envs/mh-denoise/bin/python"
partial="$repo/data/fullpaper_acl_pipeline/frozen_revision2_test_partial60_surface_relaxed_v1_20261008"
staging="$repo/data/fullpaper_acl_pipeline/frozen_revision2_test_corruption_live_handoff_20261008"
selection="$repo/data/fullpaper_acl_pipeline/frozen_revision2_test_selection_v1_20261008/test_generation_selection.json"
canonical="$repo/data/fullpaper_acl_pipeline/canonical_clean_qa.jsonl"

mkdir -p "$output/logs" "$output/requests" "$output/results" "$output/inputs" "$staging"

stage() {
  printf '%s\n' "$1" > "$output/current_stage.txt"
  printf '%s stage=%s\n' "$(date --iso-8601=seconds)" "$1"
}

stage "preflight_partial60_and_contract"
latest_contract="$({ find "$handoff/snapshots" -maxdepth 1 -type f -name 'snapshot_accepted060_rows*.run_contract.json' -print; } | sort -V | tail -n 1)"
if [[ -z "$latest_contract" ]]; then
  echo "No accepted060 snapshot run contract found" >&2
  exit 1
fi
prefix="${latest_contract%.run_contract.json}"
checkpoint="${prefix}.results_checkpoint.part0.jsonl"
if [[ ! -f "$checkpoint" ]]; then
  echo "Snapshot checkpoint missing: $checkpoint" >&2
  exit 1
fi
ln -sfn "$latest_contract" "$staging/run_contract.json"
ln -sfn "$checkpoint" "$staging/results_checkpoint.part0.jsonl"
if [[ -f "${prefix}.worker_status.part0.json" ]]; then
  ln -sfn "${prefix}.worker_status.part0.json" "$staging/worker_status.part0.json"
fi
if [[ -f "${prefix}.pause_after_accepted.json" ]]; then
  ln -sfn "${prefix}.pause_after_accepted.json" "$staging/pause_after_accepted.json"
fi

if [[ ! -f "$partial/manifest.json" ]]; then
  env PYTHONPATH="$repo/scripts:$repo" "$python_bin" "$repo/scripts/build_frozen_revision2_test_partial_export.py" \
    --production-dir "$staging" \
    --selection-file "$selection" \
    --canonical-file "$canonical" \
    --output-dir "$partial" \
    --target-count 60
fi

"$python_bin" "$repo/scripts/run_test60_rubric_aligned_selection.py" prepare-contract

stage "missing_inference_sft_then_dpo_then_denoiser"
env H100_TEST_TARGET_MAX=60 H100_TEST_REQUIRED_FREE_MIB=16000 \
  bash "$worktree/scripts/run_h100_test_surface_relaxed_live_tmux.sh"

stage "freeze_and_reuse_or_enqueue_gpt_selector"
"$python_bin" "$repo/scripts/run_test60_rubric_aligned_selection.py" prepare-selector

new_requests="$output/requests/new_selector_requests.jsonl"
new_results="$output/results/new_selector_results.jsonl"
if [[ -s "$new_requests" ]]; then
  stage "gpt41_selector_new_missing_requests"
  set +e
  "$python_bin" "$repo/scripts/run_gpt41_checkpoint_queue.py" \
    --requests "$new_requests" \
    --output "$new_results" \
    --queue-name test60_new4_selector_missing \
    --initial-concurrency 4 \
    --max-concurrency 8 \
    --max-retries 5 \
    --timeout 120
  api_exit=$?
  set -e
  if [[ ! -e "$new_results" ]]; then
    : > "$new_results"
  fi
  printf '%s GPT selector runner exit=%s; finalizing with frozen fallback rules\n' "$(date --iso-8601=seconds)" "$api_exit"
else
  : > "$new_results"
  printf '%s no new GPT selector requests; exact frozen results fully reused\n' "$(date --iso-8601=seconds)"
fi

stage "finalize_test60"
"$python_bin" "$repo/scripts/run_test60_rubric_aligned_selection.py" finalize
stage "complete"
