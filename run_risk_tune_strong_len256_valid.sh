#!/usr/bin/env bash
set -euo pipefail

cd ~/mh-denoise

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mh-denoise

SFT_ADAPTER=outputs/models/gemma4_peft_sft_plain_exp295/final
SFT_PROMPT_STYLE=sft_plain

TRAIN_OUT=outputs/refinement/sft_plain_exp295_train_outputs_len256.jsonl
VALID_OUT=outputs/refinement/sft_plain_exp295_valid_outputs_len256.jsonl

ROUTER_DIR=outputs/models/aspect_router_exp295_multilabel/final

RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/best
if [ ! -d "$RISK_SCORER_DIR" ]; then
  RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/final
fi

OUT=outputs/models/gemma4_selective_sft_plain_risk_tuned_exp295_len256_lr5e6_lambda05_os3_clean

echo "[check] SFT_ADAPTER=$SFT_ADAPTER"
echo "[check] TRAIN_OUT=$TRAIN_OUT"
echo "[check] VALID_OUT=$VALID_OUT"
echo "[check] ROUTER_DIR=$ROUTER_DIR"
echo "[check] RISK_SCORER_DIR=$RISK_SCORER_DIR"
echo "[check] OUT=$OUT"

[ -d "$SFT_ADAPTER" ] || { echo "[error] missing SFT_ADAPTER: $SFT_ADAPTER"; exit 1; }
[ -f "$TRAIN_OUT" ] || { echo "[error] missing TRAIN_OUT: $TRAIN_OUT"; exit 1; }
[ -f "$VALID_OUT" ] || { echo "[error] missing VALID_OUT: $VALID_OUT"; exit 1; }
[ -d "$ROUTER_DIR" ] || { echo "[error] missing ROUTER_DIR: $ROUTER_DIR"; exit 1; }
[ -d "$RISK_SCORER_DIR" ] || { echo "[error] missing RISK_SCORER_DIR: $RISK_SCORER_DIR"; exit 1; }

TRAIN_N=$(wc -l < "$TRAIN_OUT")
VALID_N=$(wc -l < "$VALID_OUT")

echo "[check] TRAIN rows=$TRAIN_N"
echo "[check] VALID rows=$VALID_N"

[ "$TRAIN_N" -eq 1242 ] || { echo "[error] bad train row count: $TRAIN_N"; exit 1; }
[ "$VALID_N" -eq 174 ] || { echo "[error] bad valid row count: $VALID_N"; exit 1; }

echo "[strong risk-tune] start $(date '+%F %T')"

python scripts/train_gemma_risk_tune_from_sft.py \
  --base_model google/gemma-4-E4B-it \
  --init_adapter_dir "$SFT_ADAPTER" \
  --train_file "$TRAIN_OUT" \
  --valid_file "$VALID_OUT" \
  --output_dir "$OUT" \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --zt_strategy staged_risk \
  --learning_rate 5e-6 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 1536 \
  --max_target_len 256 \
  --lambda_y 0.5 \
  --risk_oversample_threshold 0.35 \
  --risk_oversample_factor 3

echo "[strong risk-tune] done $(date '+%F %T')"

RISK_TUNED_ADAPTER="$OUT/best"
if [ ! -d "$RISK_TUNED_ADAPTER" ]; then
  RISK_TUNED_ADAPTER="$OUT/final"
fi

echo "[selective strong] adapter=$RISK_TUNED_ADAPTER"

run_selective () {
  local GATE="$1"
  local RISK="$2"
  local SPEC="$3"
  local TAG="gate${GATE}_risk${RISK}_spec${SPEC}"
  TAG="${TAG//./}"

  local SELECTIVE_OUT="outputs/refinement/selective_sft_plain_risk_tuned_exp295_valid_len256_strong_${TAG}.jsonl"

  echo
  echo "============================================================"
  echo "[selective strong] gate=$GATE risk=$RISK spec=$SPEC"
  echo "[out] $SELECTIVE_OUT"
  echo "============================================================"

  python scripts/run_gemma_selective_risk_refinement.py \
    --base_model google/gemma-4-E4B-it \
    --sft_adapter_dir "$SFT_ADAPTER" \
    --risk_adapter_dir "$RISK_TUNED_ADAPTER" \
    --sft_prompt_style "$SFT_PROMPT_STYLE" \
    --router_dir "$ROUTER_DIR" \
    --risk_scorer_dir "$RISK_SCORER_DIR" \
    --input "$VALID_OUT" \
    --reuse_sft_response \
    --sft_response_field sft_response \
    --output "$SELECTIVE_OUT" \
    --zt_strategy staged_risk \
    --max_new_tokens 256 \
    --temperature 0.0 \
    --repetition_penalty 1.15 \
    --no_repeat_ngram_size 4 \
    --gate_risk_threshold "$GATE" \
    --risk_threshold "$RISK" \
    --specificity_min_ratio "$SPEC"
}

# Best proxy setting from the weak adapter sweep.
run_selective 0.25 0.35 0.60

# Conservative reference setting, useful if gate025 becomes too aggressive.
run_selective 0.35 0.35 0.60

echo "[all done strong len256 valid] $(date '+%F %T')"
