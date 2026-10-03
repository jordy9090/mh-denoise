#!/usr/bin/env bash
set -euo pipefail

cd ~/mh-denoise

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mh-denoise

SFT_ADAPTER=outputs/models/gemma4_peft_sft_plain_exp295/final
SFT_PROMPT_STYLE=sft_plain
VALID_OUT=outputs/refinement/sft_plain_exp295_valid_outputs_len256.jsonl

RISK_ADAPTER=outputs/models/gemma4_selective_sft_plain_risk_tuned_exp295_len256_lr5e6_lambda03_clean/best
if [ ! -d "$RISK_ADAPTER" ]; then
  RISK_ADAPTER=outputs/models/gemma4_selective_sft_plain_risk_tuned_exp295_len256_lr5e6_lambda03_clean/final
fi

RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/best
if [ ! -d "$RISK_SCORER_DIR" ]; then
  RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/final
fi

[ -f "$VALID_OUT" ] || { echo "[error] missing VALID_OUT=$VALID_OUT"; exit 1; }
[ -d "$RISK_ADAPTER" ] || { echo "[error] missing RISK_ADAPTER=$RISK_ADAPTER"; exit 1; }
[ -d outputs/models/aspect_router_exp295_multilabel/final ] || { echo "[error] missing router"; exit 1; }
[ -d "$RISK_SCORER_DIR" ] || { echo "[error] missing risk scorer"; exit 1; }

run_one () {
  local GATE="$1"
  local RISK="$2"
  local SPEC="$3"
  local TAG="gate${GATE}_risk${RISK}_spec${SPEC}"
  TAG="${TAG//./}"

  local OUT="outputs/refinement/selective_sft_plain_risk_tuned_exp295_valid_len256_${TAG}.jsonl"

  echo
  echo "============================================================"
  echo "[run] gate=$GATE risk=$RISK spec=$SPEC"
  echo "[out] $OUT"
  echo "============================================================"

  python scripts/run_gemma_selective_risk_refinement.py \
    --base_model google/gemma-4-E4B-it \
    --sft_adapter_dir "$SFT_ADAPTER" \
    --risk_adapter_dir "$RISK_ADAPTER" \
    --sft_prompt_style "$SFT_PROMPT_STYLE" \
    --router_dir outputs/models/aspect_router_exp295_multilabel/final \
    --risk_scorer_dir "$RISK_SCORER_DIR" \
    --input "$VALID_OUT" \
    --reuse_sft_response \
    --sft_response_field sft_response \
    --output "$OUT" \
    --zt_strategy staged_risk \
    --max_new_tokens 256 \
    --temperature 0.0 \
    --repetition_penalty 1.15 \
    --no_repeat_ngram_size 4 \
    --gate_risk_threshold "$GATE" \
    --risk_threshold "$RISK" \
    --specificity_min_ratio "$SPEC"
}

run_one 0.30 0.35 0.60
run_one 0.25 0.35 0.60
run_one 0.30 0.30 0.60
run_one 0.30 0.35 0.55

echo "[all done] $(date '+%F %T')"
