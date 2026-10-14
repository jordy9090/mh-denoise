#!/usr/bin/env bash
set -euo pipefail

# Run from the mh-denoise repository root.
# Usage: bash baselines/dpo_exp295/run_exp295.sh prepare|sweep|final|denoiser

STAGE="${1:-}"
if [[ -z "$STAGE" ]]; then
  echo "usage: $0 prepare|sweep|final|denoiser" >&2
  exit 2
fi

ROOT="baselines/dpo_exp295"
TRAIN_SPLIT="data/splits_exp295/train_mdlm.jsonl"
VALID_SPLIT="data/splits_exp295/valid_mdlm.jsonl"
TEST_SPLIT="data/splits_exp295/test.jsonl"
SFT_ADAPTER="${SFT_ADAPTER:-outputs/models/gemma4_peft_sft_plain_exp295/final}"
BASE_MODEL="${BASE_MODEL:-google/gemma-4-E4B-it}"
BETAS=(0.03 0.10 0.30 0.50)
SEEDS=(42 43 44)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

slug() {
  python - "$1" <<'PY'
import sys
x=float(sys.argv[1])
print(f"{x:.6g}".replace("-","m").replace(".","p"))
PY
}

require_file() {
  [[ -f "$1" ]] || { echo "missing file: $1" >&2; exit 1; }
}

prepare_pairs() {
  require_file "$TRAIN_SPLIT"
  require_file "$VALID_SPLIT"
  require_file "$TEST_SPLIT"
  python "$ROOT/prepare_preferences.py" \
    --train_input "$TRAIN_SPLIT" \
    --valid_input "$VALID_SPLIT" \
    --test_input "$TEST_SPLIT" \
    --output_dir data/dpo_exp295 \
    --mode both \
    --k 4 \
    --seed 42
  python -m unittest discover -s "$ROOT/tests" -p 'test_*.py' -v
}

prepare_judge_part() {
  local input="$1"
  local system="$2"
  local prefix="$3"
  local output="$4"
  python scripts/prepare_refinement_judge_input.py \
    --input "$input" \
    --output "$output" \
    --response_field response \
    --system_name "$system" \
    --id_prefix "$prefix"
}

run_beta_sweep() {
  [[ -d "$SFT_ADAPTER" ]] || { echo "missing SFT adapter: $SFT_ADAPTER" >&2; exit 1; }
  mkdir -p outputs/refinement/dpo_exp295_valid outputs/eval_inputs/dpo_exp295_beta_parts outputs/eval outputs/analysis
  rm -f outputs/eval_inputs/dpo_exp295_beta_parts/*.jsonl

  local sft_valid="outputs/refinement/dpo_exp295_valid/sft_qd_s42.jsonl"
  if [[ ! -f "$sft_valid" ]]; then
    python "$ROOT/generate.py" \
      --input "$VALID_SPLIT" \
      --output "$sft_valid" \
      --base_model "$BASE_MODEL" \
      --adapter_dir "$SFT_ADAPTER" \
      --system_name sft_qd_s42 \
      --expected_rows 174 \
      --seed 42
  fi
  prepare_judge_part "$sft_valid" sft_qd_s42 beta_sweep_sft \
    outputs/eval_inputs/dpo_exp295_beta_parts/sft_qd_s42.jsonl

  local variant config beta bslug out_dir valid_out system part
  for variant in minimal hard_k4; do
    config="$ROOT/configs/${variant}.yaml"
    for beta in "${BETAS[@]}"; do
      bslug="$(slug "$beta")"
      out_dir="outputs/models/dpo_exp295/${variant}/beta_${bslug}/seed_42"
      if [[ ! -f "$out_dir/run_manifest.json" ]]; then
        python "$ROOT/train_dpo.py" \
          --config "$config" \
          --beta "$beta" \
          --seed 42
      fi
      valid_out="outputs/refinement/dpo_exp295_valid/dpo_${variant}_b${bslug}_s42.jsonl"
      system="dpo_${variant}_b${bslug}_s42"
      if [[ ! -f "$valid_out" ]]; then
        python "$ROOT/generate.py" \
          --input "$VALID_SPLIT" \
          --output "$valid_out" \
          --base_model "$BASE_MODEL" \
          --adapter_dir "$out_dir/final" \
          --system_name "$system" \
          --expected_rows 174 \
          --seed 42
      fi
      part="outputs/eval_inputs/dpo_exp295_beta_parts/${system}.jsonl"
      prepare_judge_part "$valid_out" "$system" "beta_sweep_${variant}_${bslug}" "$part"
    done
  done

  cat outputs/eval_inputs/dpo_exp295_beta_parts/*.jsonl \
    > outputs/eval_inputs/dpo_exp295_beta_sweep.jsonl

  [[ -n "${OPENAI_API_KEY:-}" ]] || {
    echo "OPENAI_API_KEY is required for the existing refinement judge." >&2
    exit 1
  }
  python scripts/run_refinement_llm_judge.py \
    --input outputs/eval_inputs/dpo_exp295_beta_sweep.jsonl \
    --output outputs/eval/dpo_exp295_beta_sweep_judged.jsonl \
    --model gpt-4.1-mini \
    --resume \
    --sleep 0.5

  python scripts/aggregate_refinement_judge_scores.py \
    --input outputs/eval/dpo_exp295_beta_sweep_judged.jsonl \
    --output_csv outputs/analysis/dpo_exp295_beta_sweep_by_system.csv \
    --group_by system

  python "$ROOT/select_beta.py" \
    --input_csv outputs/analysis/dpo_exp295_beta_sweep_by_system.csv \
    --output_json outputs/analysis/dpo_exp295_beta_selection.json \
    --output_csv outputs/analysis/dpo_exp295_beta_selection.csv \
    --baseline_system sft_qd_s42 \
    --sweep_seed 42 \
    --expected_n 174 \
    --medical_tolerance 0.0 \
    --toxicity_tolerance 0.0 \
    --minimum_quality_delta -0.05
}

selected_beta() {
  local variant="$1"
  python - "$variant" <<'PY'
import json, sys
variant=f"dpo_{sys.argv[1]}"
p="outputs/analysis/dpo_exp295_beta_selection.json"
x=json.load(open(p, encoding="utf-8"))["selections"][variant]
if x["status"] != "selected":
    raise SystemExit(f"no feasible beta for {variant}: {x}")
print(x["beta"])
PY
}

run_final_seeds() {
  require_file outputs/analysis/dpo_exp295_beta_selection.json
  mkdir -p outputs/refinement/dpo_exp295_test
  local variant config beta bslug seed out_dir test_out system
  for variant in minimal hard_k4; do
    config="$ROOT/configs/${variant}.yaml"
    beta="$(selected_beta "$variant")"
    bslug="$(slug "$beta")"
    for seed in "${SEEDS[@]}"; do
      out_dir="outputs/models/dpo_exp295/${variant}/beta_${bslug}/seed_${seed}"
      if [[ ! -f "$out_dir/run_manifest.json" ]]; then
        python "$ROOT/train_dpo.py" \
          --config "$config" \
          --beta "$beta" \
          --seed "$seed"
      fi
      test_out="outputs/refinement/dpo_exp295_test/dpo_${variant}_selected_s${seed}.jsonl"
      system="dpo_${variant}_selected_s${seed}"
      if [[ ! -f "$test_out" ]]; then
        python "$ROOT/generate.py" \
          --input "$TEST_SPLIT" \
          --output "$test_out" \
          --base_model "$BASE_MODEL" \
          --adapter_dir "$out_dir/final" \
          --system_name "$system" \
          --expected_rows 354 \
          --seed "$seed"
      fi
    done
  done

  local sft_test="outputs/refinement/dpo_exp295_test/sft_qd_s42.jsonl"
  if [[ ! -f "$sft_test" ]]; then
    python "$ROOT/generate.py" \
      --input "$TEST_SPLIT" \
      --output "$sft_test" \
      --base_model "$BASE_MODEL" \
      --adapter_dir "$SFT_ADAPTER" \
      --system_name sft_qd_s42 \
      --expected_rows 354 \
      --seed 42
  fi
}

resolve_checkpoint() {
  local first="$1" second="$2"
  if [[ -d "$first" ]]; then echo "$first"; return; fi
  if [[ -d "$second" ]]; then echo "$second"; return; fi
  echo ""
}

run_dpo_denoiser_candidates() {
  local variant="${DPO_VARIANT:-minimal}"
  local seed="${DPO_SEED:-42}"
  local upstream="outputs/refinement/dpo_exp295_test/dpo_${variant}_selected_s${seed}.jsonl"
  require_file "$upstream"

  local router risk denoiser
  router="${ROUTER_DIR:-$(resolve_checkpoint outputs/models/router_multilabel_v1/best outputs/models/router_multilabel_v1/final)}"
  risk="${RISK_SCORER_DIR:-$(resolve_checkpoint outputs/models/span_risk_multilabel_v1/best outputs/models/span_risk_multilabel_v1/final)}"
  denoiser="${DENOISER_ADAPTER_DIR:-$(resolve_checkpoint outputs/models/gemma4_selective_sft_plain_risk_tuned_exp295_len256_lr5e6_lambda03_clean/best outputs/models/gemma4_peft_langqkvo_infermatch_exp295_main/best)}"
  [[ -n "$router" && -d "$router" ]] || { echo "set ROUTER_DIR" >&2; exit 1; }
  [[ -n "$risk" && -d "$risk" ]] || { echo "set RISK_SCORER_DIR" >&2; exit 1; }
  [[ -n "$denoiser" && -d "$denoiser" ]] || { echo "set DENOISER_ADAPTER_DIR" >&2; exit 1; }

  local bridge="outputs/refinement/dpo_exp295_test/${variant}_s${seed}_for_denoiser.jsonl"
  local candidates="outputs/refinement/dpo_exp295_test/${variant}_s${seed}_denoiser_candidates.jsonl"
  python "$ROOT/bridge_selective_denoiser.py" prepare \
    --input "$upstream" \
    --output "$bridge" \
    --upstream_response_field response \
    --upstream_system "dpo_${variant}_selected_s${seed}"

  python scripts/run_gemma_peft_real_inference.py \
    --base_model "$BASE_MODEL" \
    --adapter_dir "$denoiser" \
    --router_dir "$router" \
    --risk_scorer_dir "$risk" \
    --input "$bridge" \
    --output "$candidates" \
    --modes unsafe_t4 \
    --max_new_tokens 120 \
    --temperature 0.0 \
    --repetition_penalty 1.15 \
    --no_repeat_ngram_size 4

  cat <<EOF
Denoiser candidates created: $candidates
The committed main branch does not contain a complete selective invocation/acceptance finalizer.
Create gate decisions with the exact Proposed thresholds, then merge with:

python $ROOT/bridge_selective_denoiser.py merge \
  --upstream_input $upstream \
  --denoiser_input $candidates \
  --decisions <shared_gate_decisions.jsonl> \
  --decision_field denoiser_accepted \
  --output outputs/refinement/dpo_exp295_test/dpo_${variant}_plus_selective_denoiser_s${seed}.jsonl \
  --output_system dpo_${variant}_plus_selective_denoiser_s${seed}
EOF
}

case "$STAGE" in
  prepare) prepare_pairs ;;
  sweep) run_beta_sweep ;;
  final) run_final_seeds ;;
  denoiser) run_dpo_denoiser_candidates ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
