#!/usr/bin/env bash
set -euo pipefail

worktree_dir="/workspace/mh-denoise"
source_repo_dir="/workspace/mh-denoise"
live_input_dir="/workspace/artifacts/revision2_test_live_handoff_20261008"

selection_file="$source_repo_dir/data/fullpaper_acl_pipeline/frozen_revision2_test_selection_v1_20261008/test_generation_selection.json"
canonical_file="$source_repo_dir/data/fullpaper_acl_pipeline/canonical_clean_qa.jsonl"
run_dir="$source_repo_dir/outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910"
generation_root="$run_dir/gemma/generation_1536_test_surface_relaxed_v1_20261008"
expert_root="$run_dir/expert_test_surface_relaxed_v1_20261008"
base_model="/workspace/cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
python_bin="/workspace/env/bin/python"
pipeline_log="$generation_root/h100_surface_relaxed_live_pipeline.log"
wait_log="$generation_root/h100_surface_relaxed_live_wait.log"
latest_status="$generation_root/latest_status.json"

required_free_mib="${H100_TEST_REQUIRED_FREE_MIB:-16000}"
poll_seconds="${H100_TEST_POLL_SECONDS:-60}"
target_max="${H100_TEST_TARGET_MAX:-15}"
threshold="0.7215308056776166"
pythonpath="$worktree_dir/scripts:$worktree_dir:$source_repo_dir/scripts:$source_repo_dir"

mkdir -p "$generation_root" "$expert_root"
cd "$worktree_dir"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$pipeline_log" >&2
}

run_logged() {
  log "run: $*"
  "$@" >> "$pipeline_log" 2>&1
}

choose_production_dir() {
  local staged
  if staged="$(materialize_live_snapshot)"; then
    printf '%s\n' "$staged"
    return 0
  fi
  local candidate
  for candidate in \
    "$live_input_dir" \
    "$live_input_dir/frozen_revision2_test_corruption_v1_20261008_a100" \
    "$live_input_dir/data/fullpaper_acl_pipeline/frozen_revision2_test_corruption_v1_20261008_a100" \
    "$source_repo_dir/data/fullpaper_acl_pipeline/frozen_revision2_test_corruption_v1_20261008_a100" \
    "$source_repo_dir/data/fullpaper_acl_pipeline/frozen_revision2_test_corruption_v1_20261008"; do
    if [[ -f "$candidate/run_contract.json" ]] && compgen -G "$candidate/results_checkpoint.part*.jsonl" > /dev/null; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

materialize_live_snapshot() {
  local latest_contract prefix checkpoint staging
  latest_contract="$(
    find "$live_input_dir/snapshots" -maxdepth 1 -type f \
      -name 'snapshot_accepted*_rows*.run_contract.json' -print 2>/dev/null \
      | sort -V | tail -n 1
  )"
  if [[ -z "$latest_contract" ]]; then
    return 1
  fi
  prefix="${latest_contract%.run_contract.json}"
  checkpoint="${prefix}.results_checkpoint.part0.jsonl"
  if [[ ! -f "$checkpoint" ]]; then
    return 1
  fi
  staging="$source_repo_dir/data/fullpaper_acl_pipeline/frozen_revision2_test_corruption_live_handoff_20261008"
  mkdir -p "$staging"
  ln -sfn "$latest_contract" "$staging/run_contract.json"
  ln -sfn "$checkpoint" "$staging/results_checkpoint.part0.jsonl"
  if [[ -f "${prefix}.worker_status.part0.json" ]]; then
    ln -sfn "${prefix}.worker_status.part0.json" "$staging/worker_status.part0.json"
  fi
  if [[ -f "${prefix}.pause_after_accepted.json" ]]; then
    ln -sfn "${prefix}.pause_after_accepted.json" "$staging/pause_after_accepted.json"
  fi
  printf '%s\n' "$staging"
}

wait_for_gpu() {
  local free_mib
  while true; do
    free_mib="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= required_free_mib )); then
      printf '%s gpu_ready free_mib=%s required_mib=%s\n' "$(date --iso-8601=seconds)" "$free_mib" "$required_free_mib" >> "$wait_log"
      return 0
    fi
    printf '%s waiting_for_gpu free_mib=%s required_mib=%s\n' "$(date --iso-8601=seconds)" "$free_mib" "$required_free_mib" >> "$wait_log"
    sleep "$poll_seconds"
  done
}

json_value() {
  local path="$1"
  local expr="$2"
  jq -r "$expr" "$path"
}

line_count_or_zero() {
  local path="$1"
  if [[ -f "$path" ]]; then
    wc -l < "$path"
  else
    printf '0\n'
  fi
}

write_latest_status() {
  local count="$1"
  local production_dir="$2"
  local partial_dir="$3"
  local generation_dir="$4"
  local export_dir="$5"
  env COUNT="$count" PRODUCTION_DIR="$production_dir" PARTIAL_DIR="$partial_dir" GENERATION_DIR="$generation_dir" EXPORT_DIR="$export_dir" \
    SFT_ROWS="$(line_count_or_zero "$generation_dir/sft_test_outputs.jsonl")" \
    DPO_ROWS="$(line_count_or_zero "$generation_dir/dpo_test_outputs.jsonl")" \
    PROPOSED_ROWS="$(line_count_or_zero "$generation_dir/proposed_mask_off_test_outputs.jsonl")" \
    "$python_bin" - <<'PY' > "$latest_status"
import json, os
payload = {
    "updated_at_note": "timestamp is in the log; this file records the latest materialized counts",
    "input": {"production_dir": os.environ["PRODUCTION_DIR"], "accepted_pairs_materialized": int(os.environ["COUNT"])},
    "method_completion_rows": {
        "SFT": int(os.environ["SFT_ROWS"]),
        "DPO": int(os.environ["DPO_ROWS"]),
        "Proposed": int(os.environ["PROPOSED_ROWS"]),
    },
    "paths": {
        "partial_dir": os.environ["PARTIAL_DIR"],
        "generation_dir": os.environ["GENERATION_DIR"],
        "expert_export_dir": os.environ["EXPORT_DIR"],
    },
}
print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
PY
}

refresh_progress() {
  local production_dir="$1"
  local progress_dir="$source_repo_dir/data/fullpaper_acl_pipeline/frozen_revision2_test_partial${target_max}_surface_relaxed_v1_20261008_progress"
  mkdir -p "$progress_dir"
  env PYTHONPATH="$pythonpath" "$python_bin" "$source_repo_dir/scripts/build_frozen_revision2_test_partial_export.py" \
    --production-dir "$production_dir" \
    --selection-file "$selection_file" \
    --canonical-file "$canonical_file" \
    --output-dir "$progress_dir" \
    --target-count "$target_max" >> "$pipeline_log" 2>&1 || true
  if [[ -f "$progress_dir/manifest.json" ]]; then
    json_value "$progress_dir/manifest.json" '.selected_if_frozen_now // .target_count // 0'
  elif [[ -f "$progress_dir/progress_manifest.json" ]]; then
    json_value "$progress_dir/progress_manifest.json" '.selected_if_frozen_now // .eligible_accepted_in_completed_prefix // 0'
  else
    printf '0\n'
  fi
}

ensure_partial_export() {
  local production_dir="$1"
  local count="$2"
  local partial_dir="$source_repo_dir/data/fullpaper_acl_pipeline/frozen_revision2_test_partial${count}_surface_relaxed_v1_20261008"
  if [[ ! -f "$partial_dir/manifest.json" ]]; then
    run_logged env PYTHONPATH="$pythonpath" "$python_bin" "$source_repo_dir/scripts/build_frozen_revision2_test_partial_export.py" \
      --production-dir "$production_dir" \
      --selection-file "$selection_file" \
      --canonical-file "$canonical_file" \
      --output-dir "$partial_dir" \
      --target-count "$count"
  fi
  printf '%s\n' "$partial_dir"
}

audit_sft_lengths() {
  local partial_dir="$1"
  local generation_dir="$2"
  local output="$generation_dir/sft_test_input_lengths.json"
  run_logged env PYTHONPATH="$pythonpath" "$python_bin" "$source_repo_dir/scripts/audit_frozen_test_inference_lengths.py" \
    --input "$partial_dir/sft_test.jsonl" \
    --tokenizer "$run_dir/gemma/sft/final" \
    --output "$output" \
    --stage sft \
    --frozen-limit 1024 \
    --generation-budget 1536
  json_value "$output" '.selected_complete_max_source_len'
}

audit_proposed_lengths() {
  local generation_dir="$1"
  local output="$generation_dir/proposed_mask_off_test_input_lengths.json"
  run_logged env PYTHONPATH="$pythonpath" "$python_bin" "$source_repo_dir/scripts/audit_frozen_test_inference_lengths.py" \
    --input "$generation_dir/sft_test_outputs.jsonl" \
    --tokenizer "$run_dir/gemma/sft/final" \
    --output "$output" \
    --stage proposed \
    --frozen-limit 4096 \
    --generation-budget 1536
  json_value "$output" '.selected_complete_max_source_len'
}

ensure_sft_like_outputs() {
  local method="$1"
  local adapter_dir="$2"
  local partial_dir="$3"
  local generation_dir="$4"
  local source_len="$5"
  local output="$generation_dir/${method}_test_outputs.jsonl"
  local manifest="$generation_dir/${method}_test_outputs.manifest.json"
  local missing_input="$generation_dir/${method}_missing_input.jsonl"
  local missing_status="$generation_dir/${method}_missing_status.json"
  local missing_output="$generation_dir/${method}_missing_outputs.jsonl"
  local status_name="complete_${method}_inference_for_h100_surface_relaxed_test"

  run_logged env PYTHONPATH="$pythonpath" "$python_bin" "$worktree_dir/scripts/h100_test_surface_relaxed_utils.py" prepare-missing \
    --ordered-json "$partial_dir/partial_selection.json" \
    --input-jsonl "$partial_dir/sft_test.jsonl" \
    --output-jsonl "$missing_input" \
    --status-json "$missing_status" \
    --existing-glob "$generation_root/part*/${method}_test_outputs.jsonl"
  local missing_count
  missing_count="$(json_value "$missing_status" '.missing_rows')"
  if (( missing_count > 0 )); then
    wait_for_gpu
    run_logged env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$pythonpath" "$python_bin" "$worktree_dir/scripts/build_sft_outputs_for_risk_tuning.py" \
      --base_model "$base_model" \
      --adapter_dir "$adapter_dir" \
      --input "$missing_input" \
      --output "$missing_output" \
      --max_source_len "$source_len" \
      --max_new_tokens 1536 \
      --temperature 0.0 \
      --repetition_penalty 1.15 \
      --no_repeat_ngram_size 4 \
      --sft_prompt_style sft_plain
  fi
  run_logged env PYTHONPATH="$pythonpath" "$python_bin" "$worktree_dir/scripts/h100_test_surface_relaxed_utils.py" merge-outputs \
    --ordered-json "$partial_dir/partial_selection.json" \
    --input-jsonl "$partial_dir/sft_test.jsonl" \
    --output-jsonl "$output" \
    --manifest-json "$manifest" \
    --method "$method" \
    --status "$status_name" \
    --existing-glob "$generation_root/part*/${method}_test_outputs.jsonl" \
    --new "$missing_output"
}

ensure_proposed_outputs() {
  local partial_dir="$1"
  local generation_dir="$2"
  local source_len="$3"
  local output="$generation_dir/proposed_mask_off_test_outputs.jsonl"
  local manifest="$generation_dir/proposed_mask_off_test_outputs.manifest.json"
  local missing_input="$generation_dir/proposed_missing_input.jsonl"
  local missing_status="$generation_dir/proposed_missing_status.json"
  local missing_output="$generation_dir/proposed_missing_outputs.jsonl"

  run_logged env PYTHONPATH="$pythonpath" "$python_bin" "$worktree_dir/scripts/h100_test_surface_relaxed_utils.py" prepare-missing \
    --ordered-json "$partial_dir/partial_selection.json" \
    --input-jsonl "$generation_dir/sft_test_outputs.jsonl" \
    --output-jsonl "$missing_input" \
    --status-json "$missing_status" \
    --existing-glob "$generation_root/part*/proposed_mask_off_test_outputs.jsonl"
  local missing_count
  missing_count="$(json_value "$missing_status" '.missing_rows')"
  if (( missing_count > 0 )); then
    wait_for_gpu
    run_logged env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$pythonpath" "$python_bin" "$worktree_dir/scripts/run_gemma_selective_risk_refinement.py" \
      --base_model "$base_model" \
      --sft_adapter_dir "$run_dir/gemma/sft/final" \
      --risk_adapter_dir "$run_dir/gemma/proposed_mask_off/final" \
      --router_dir "$run_dir/router/final" \
      --risk_scorer_dir "$run_dir/scorer/final" \
      --risk_contract fullpaper_v1 \
      --component_mode full \
      --input "$missing_input" \
      --output "$missing_output" \
      --reuse_sft_response \
      --sft_response_field sft_response \
      --zt_strategy no_mask \
      --T 4 \
      --timestep 3 \
      --risk_threshold "$threshold" \
      --gate_risk_threshold "$threshold" \
      --gate_strategy overall \
      --gate_focus_aspect medical_boundary \
      --gate_focus_threshold 0.15 \
      --mask_threshold "$threshold" \
      --min_risk_delta 0.01 \
      --min_focus_risk_delta 0.0 \
      --specificity_min_ratio 0.6 \
      --keyword_overlap_slack 0.15 \
      --min_word_count 20 \
      --router_max_len 512 \
      --risk_max_len 384 \
      --max_source_len "$source_len" \
      --max_new_tokens 1536 \
      --temperature 0.0 \
      --repetition_penalty 1.15 \
      --no_repeat_ngram_size 4 \
      --acceptance_policy surface_relaxed_v1
  fi
  run_logged env PYTHONPATH="$pythonpath" "$python_bin" "$worktree_dir/scripts/h100_test_surface_relaxed_utils.py" merge-outputs \
    --ordered-json "$partial_dir/partial_selection.json" \
    --input-jsonl "$generation_dir/sft_test_outputs.jsonl" \
    --output-jsonl "$output" \
    --manifest-json "$manifest" \
    --method proposed \
    --status complete_proposed_surface_relaxed_v1_inference_for_h100_test \
    --existing-glob "$generation_root/part*/proposed_mask_off_test_outputs.jsonl" \
    --new "$missing_output"
}

build_expert_export() {
  local count="$1"
  local partial_dir="$2"
  local generation_dir="$3"
  local export_dir="$expert_root/part${count}"
  run_logged env PYTHONPATH="$pythonpath" "$python_bin" "$worktree_dir/scripts/h100_test_surface_relaxed_utils.py" build-expert-export \
    --partial-dir "$partial_dir" \
    --sft "$generation_dir/sft_test_outputs.jsonl" \
    --dpo "$generation_dir/dpo_test_outputs.jsonl" \
    --proposed "$generation_dir/proposed_mask_off_test_outputs.jsonl" \
    --output-dir "$export_dir"
  printf '%s\n' "$export_dir"
}

process_count() {
  local production_dir="$1"
  local count="$2"
  local partial_dir generation_dir sft_source_len proposed_source_len export_dir
  partial_dir="$(ensure_partial_export "$production_dir" "$count")"
  generation_dir="$generation_root/part${count}"
  mkdir -p "$generation_dir"

  sft_source_len="$(audit_sft_lengths "$partial_dir" "$generation_dir")"
  ensure_sft_like_outputs "sft" "$run_dir/gemma/sft/final" "$partial_dir" "$generation_dir" "$sft_source_len"
  ensure_sft_like_outputs "dpo" "$run_dir/gemma/dpo/final" "$partial_dir" "$generation_dir" "$sft_source_len"

  proposed_source_len="$(audit_proposed_lengths "$generation_dir")"
  ensure_proposed_outputs "$partial_dir" "$generation_dir" "$proposed_source_len"

  export_dir="$(build_expert_export "$count" "$partial_dir" "$generation_dir")"
  write_latest_status "$count" "$production_dir" "$partial_dir" "$generation_dir" "$export_dir"
  log "materialized count=${count} sft=$(line_count_or_zero "$generation_dir/sft_test_outputs.jsonl") dpo=$(line_count_or_zero "$generation_dir/dpo_test_outputs.jsonl") proposed=$(line_count_or_zero "$generation_dir/proposed_mask_off_test_outputs.jsonl") export=${export_dir}"
}

log "h100 surface-relaxed TEST runner start worktree=$worktree_dir commit=$(git rev-parse HEAD) required_free_mib=$required_free_mib"

while true; do
  if ! production_dir="$(choose_production_dir)"; then
    log "waiting_for_input live_input_dir=$live_input_dir"
    sleep "$poll_seconds"
    continue
  fi
  ready_count="$(refresh_progress "$production_dir")"
  if ! [[ "$ready_count" =~ ^[0-9]+$ ]]; then
    ready_count=0
  fi
  if (( ready_count > target_max )); then
    ready_count="$target_max"
  fi
  log "input_status production_dir=$production_dir ready_accepted_pairs=$ready_count target_max=$target_max"
  if (( ready_count > 0 )); then
    process_count "$production_dir" "$ready_count"
  fi
  if (( ready_count >= target_max )); then
    log "target_complete accepted_pairs=$ready_count"
    break
  fi
  sleep "$poll_seconds"
done
