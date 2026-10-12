#!/usr/bin/env bash
set -euo pipefail

PY=/home/user/anaconda3/envs/mh-denoise/bin/python
REPO=/home/user/hsoh/mh-denoise
WORKTREE=/home/user/hsoh/mh-denoise-surface-relaxed-v1-20261008
ROOT=/mnt/ssd00/hsoh/acl_component_ablation_matrix_20261009_v1
CKPT="$REPO/outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/gemma"
BASE=/home/user/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2
ROUTER="$REPO/outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/router/final"
SCORER="$REPO/outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/scorer/final"
INPUT60="$ROOT/inputs/test60_sft_frozen.jsonl"
LOG="$ROOT/logs/gpu_inference.log"

mkdir -p "$ROOT/logs" "$ROOT/generation"
exec > >(tee -a "$LOG") 2>&1

run_selective() {
  local arm="$1"
  local expected="$2"
  local input="$3"
  local output="$4"
  local adapter="$5"
  local component="$6"
  local zt="$7"
  local threshold="$8"
  local gate="$9"
  if [[ -f "$output" ]] && [[ "$(wc -l < "$output")" -eq "$expected" ]] && [[ -f "${output%.jsonl}.manifest.json" ]]; then
    printf '{"stage":"%s","status":"reused","rows":%s}\n' "$arm" "$expected"
    return
  fi
  printf '{"stage":"%s","status":"starting","expected_rows":%s}\n' "$arm" "$expected"
  "$PY" -u "$WORKTREE/scripts/run_gemma_selective_risk_refinement.py" \
    --base_model "$BASE" \
    --sft_adapter_dir "$CKPT/sft/final" \
    --risk_adapter_dir "$adapter" \
    --router_dir "$ROUTER" \
    --risk_scorer_dir "$SCORER" \
    --risk_contract fullpaper_v1 \
    --component_mode "$component" \
    --input "$input" \
    --output "$output" \
    --reuse_sft_response \
    --sft_response_field sft_response \
    --sft_prompt_style sft_plain \
    --zt_strategy "$zt" \
    --T 4 \
    --timestep 3 \
    --risk_threshold "$threshold" \
    --gate_risk_threshold "$gate" \
    --gate_strategy overall \
    --gate_focus_aspect medical_boundary \
    --gate_focus_threshold 0.15 \
    --mask_threshold "$threshold" \
    --mask_token '<MASK>' \
    --t2_frac 0.33 \
    --t3_frac 0.66 \
    --risk_tag_format '[Risk: {dim}] {span} [/Risk]' \
    --min_risk_delta 0.01 \
    --min_focus_risk_delta 0.0 \
    --specificity_min_ratio 0.60 \
    --keyword_overlap_slack 0.15 \
    --min_word_count 20 \
    --acceptance_policy surface_relaxed_v1 \
    --router_max_len 512 \
    --risk_max_len 384 \
    --max_source_len 4096 \
    --max_new_tokens 1536 \
    --temperature 0.0 \
    --repetition_penalty 1.15 \
    --no_repeat_ngram_size 4
  if [[ "$(wc -l < "$output")" -ne "$expected" ]]; then
    printf 'Unexpected row count for %s\n' "$arm" >&2
    exit 1
  fi
  printf '{"stage":"%s","status":"complete","rows":%s}\n' "$arm" "$expected"
}

cd "$WORKTREE"
"$PY" -u "$REPO/scripts/run_test60_base_prompt_rewrite.py"

run_selective \
  always_on_missing42 42 \
  "$ROOT/inputs/test60_main_gateoff42.jsonl" \
  "$ROOT/generation/main_always_on_missing42.jsonl" \
  "$CKPT/proposed_mask_off/final" full no_mask \
  0.7215308056776166 -1.0

"$PY" -u "$REPO/scripts/run_test60_original_prompt_sft_control.py"

run_selective \
  mask_on_test60 60 "$INPUT60" \
  "$ROOT/generation/proposed_mask_on_test60.jsonl" \
  "$CKPT/proposed_mask_on/final" full staged_risk \
  0.7215308056776166 0.7215308056776166

run_selective \
  without_router_test60 60 "$INPUT60" \
  "$ROOT/generation/proposed_without_router_test60.jsonl" \
  "$CKPT/proposed_without_router/final" without_router staged_risk \
  0.948689877986908 0.948689877986908

run_selective \
  without_scorer_test60 60 "$INPUT60" \
  "$ROOT/generation/proposed_without_scorer_test60.jsonl" \
  "$CKPT/proposed_without_scorer/final" without_scorer no_mask \
  0.7560610771169199 0.7560610771169199

printf '{"status":"all_gpu_inference_complete"}\n'
