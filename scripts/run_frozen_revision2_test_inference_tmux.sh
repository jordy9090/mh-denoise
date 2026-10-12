#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/user/hsoh/mh-denoise"
production_dir="$repo_dir/data/fullpaper_acl_pipeline/frozen_revision2_test_corruption_v1_20261008"
selection_file="$repo_dir/data/fullpaper_acl_pipeline/frozen_revision2_test_selection_v1_20261008/test_generation_selection.json"
partial_dir="$repo_dir/data/fullpaper_acl_pipeline/frozen_revision2_test_partial15_v1_20261008"
run_dir="$repo_dir/outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910"
generation_dir="$run_dir/gemma/generation_1536_test_partial15_v1_20261008"
counselor_dir="$run_dir/counselor_test_partial15_v1_20261008"
base_model="/home/user/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
python_bin="/home/user/anaconda3/envs/mh-denoise/bin/python"
required_free_mib=70000
threshold="0.7215308056776166"

mkdir -p "$partial_dir" "$generation_dir" "$counselor_dir"
pipeline_log="$generation_dir/pipeline.log"
wait_log="$generation_dir/wait.log"
pause_state="$production_dir/pause_after_accepted.json"

cd "$repo_dir"

wait_for_gpu() {
  while true; do
    free_mib="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= required_free_mib )); then
      printf '%s gpu_ready free_mib=%s required_mib=%s\n' \
        "$(date --iso-8601=seconds)" "$free_mib" "$required_free_mib" >> "$wait_log"
      return
    fi
    printf '%s waiting_for_gpu free_mib=%s required_mib=%s\n' \
      "$(date --iso-8601=seconds)" "$free_mib" "$required_free_mib" >> "$wait_log"
    sleep 60
  done
}

while [[ ! -f "$pause_state" ]] || \
      [[ "$(jq -r '.status // ""' "$pause_state" 2>/dev/null || true)" != \
          "paused_after_atomic_checkpoint" ]]; do
  checkpoint="$production_dir/results_checkpoint.part0.jsonl"
  checkpoint_rows=0
  if [[ -f "$checkpoint" ]]; then
    checkpoint_rows="$(wc -l < "$checkpoint")"
  fi
  if (( checkpoint_rows >= 5 )) && [[ ! -f "$partial_dir/first5_progress.json" ]]; then
    env PYTHONPATH=scripts:. "$python_bin" \
      scripts/snapshot_frozen_revision2_test_partial_progress.py \
      --production-dir "$production_dir" \
      --selection-file "$selection_file" \
      --output "$partial_dir/first5_progress.json" \
      >> "$pipeline_log" 2>&1
  fi
  printf '%s waiting_for_15_accepted checkpoint_rows=%s\n' \
    "$(date --iso-8601=seconds)" "$checkpoint_rows" >> "$wait_log"
  sleep 15
done

env PYTHONPATH=scripts:. "$python_bin" \
  scripts/build_frozen_revision2_test_partial_export.py \
  --production-dir "$production_dir" \
  --selection-file "$selection_file" \
  --output-dir "$partial_dir" --target-count 15 >> "$pipeline_log" 2>&1

env PYTHONPATH=scripts:. "$python_bin" scripts/audit_frozen_test_inference_lengths.py \
  --input "$partial_dir/sft_test.jsonl" \
  --tokenizer "$run_dir/gemma/sft/final" \
  --output "$generation_dir/sft_test_input_lengths.json" \
  --stage sft --frozen-limit 1024 --generation-budget 1536 >> "$pipeline_log" 2>&1
sft_source_len="$(jq -r '.selected_complete_max_source_len' "$generation_dir/sft_test_input_lengths.json")"

wait_for_gpu
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:. "$python_bin" \
  scripts/build_sft_outputs_for_risk_tuning.py \
  --base_model "$base_model" --adapter_dir "$run_dir/gemma/sft/final" \
  --input "$partial_dir/sft_test.jsonl" --output "$generation_dir/sft_test_outputs.jsonl" \
  --max_source_len "$sft_source_len" --max_new_tokens 1536 \
  --temperature 0.0 --repetition_penalty 1.15 --no_repeat_ngram_size 4 \
  --sft_prompt_style sft_plain >> "$pipeline_log" 2>&1

wait_for_gpu
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:. "$python_bin" \
  scripts/build_sft_outputs_for_risk_tuning.py \
  --base_model "$base_model" --adapter_dir "$run_dir/gemma/dpo/final" \
  --input "$partial_dir/sft_test.jsonl" --output "$generation_dir/dpo_test_outputs.jsonl" \
  --max_source_len "$sft_source_len" --max_new_tokens 1536 \
  --temperature 0.0 --repetition_penalty 1.15 --no_repeat_ngram_size 4 \
  --sft_prompt_style sft_plain >> "$pipeline_log" 2>&1

env PYTHONPATH=scripts:. "$python_bin" scripts/audit_frozen_test_inference_lengths.py \
  --input "$generation_dir/sft_test_outputs.jsonl" \
  --tokenizer "$run_dir/gemma/sft/final" \
  --output "$generation_dir/proposed_mask_off_test_input_lengths.json" \
  --stage proposed --frozen-limit 4096 --generation-budget 1536 >> "$pipeline_log" 2>&1
proposed_source_len="$(jq -r '.selected_complete_max_source_len' "$generation_dir/proposed_mask_off_test_input_lengths.json")"

wait_for_gpu
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:. "$python_bin" \
  scripts/run_gemma_selective_risk_refinement.py \
  --base_model "$base_model" \
  --sft_adapter_dir "$run_dir/gemma/sft/final" \
  --risk_adapter_dir "$run_dir/gemma/proposed_mask_off/final" \
  --router_dir "$run_dir/router/final" --risk_scorer_dir "$run_dir/scorer/final" \
  --risk_contract fullpaper_v1 --component_mode full \
  --input "$generation_dir/sft_test_outputs.jsonl" \
  --output "$generation_dir/proposed_mask_off_test_outputs.jsonl" \
  --reuse_sft_response --sft_response_field sft_response --zt_strategy no_mask \
  --T 4 --timestep 3 \
  --risk_threshold "$threshold" --gate_risk_threshold "$threshold" \
  --gate_strategy overall --gate_focus_aspect medical_boundary --gate_focus_threshold 0.15 \
  --mask_threshold "$threshold" --min_risk_delta 0.01 --min_focus_risk_delta 0.0 \
  --specificity_min_ratio 0.6 --keyword_overlap_slack 0.15 --min_word_count 20 \
  --router_max_len 512 --risk_max_len 384 \
  --max_source_len "$proposed_source_len" --max_new_tokens 1536 \
  --temperature 0.0 --repetition_penalty 1.15 --no_repeat_ngram_size 4 \
  >> "$pipeline_log" 2>&1

for manifest in \
  "$generation_dir/sft_test_outputs.manifest.json" \
  "$generation_dir/dpo_test_outputs.manifest.json"; do
  [[ "$(jq -r '.output.rows' "$manifest")" == "15" ]]
  [[ "$(jq -r '.generation_completion.source_input_truncated' "$manifest")" == "0" ]]
done
[[ "$(jq -r '.output.rows' "$generation_dir/proposed_mask_off_test_outputs.manifest.json")" == "15" ]]
[[ "$(jq -r '.generation_completion.denoiser_source_input_truncated' "$generation_dir/proposed_mask_off_test_outputs.manifest.json")" == "0" ]]

env PYTHONPATH=scripts:. "$python_bin" scripts/build_frozen_revision2_counselor_export.py \
  --partial-dir "$partial_dir" \
  --sft "$generation_dir/sft_test_outputs.jsonl" \
  --dpo "$generation_dir/dpo_test_outputs.jsonl" \
  --mask-off "$generation_dir/proposed_mask_off_test_outputs.jsonl" \
  --output-dir "$counselor_dir" >> "$pipeline_log" 2>&1

printf '%s counselor_15x3_export_complete\n' "$(date --iso-8601=seconds)" >> "$pipeline_log"
