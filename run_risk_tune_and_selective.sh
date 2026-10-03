#!/usr/bin/env bash
set -euo pipefail

cd ~/mh-denoise

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mh-denoise

SFT_ADAPTER=outputs/models/gemma4_peft_sft_plain_exp295/final
SFT_PROMPT_STYLE=sft_plain

TRAIN_OUT=outputs/refinement/sft_plain_exp295_train_outputs.jsonl
VALID_OUT=outputs/refinement/sft_plain_exp295_valid_outputs.jsonl

RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/best
if [ ! -d "$RISK_SCORER_DIR" ]; then
  RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/final
fi

OUT=outputs/models/gemma4_selective_sft_plain_risk_tuned_exp295_lr5e6_lambda03

echo "[check] SFT_ADAPTER=$SFT_ADAPTER"
echo "[check] TRAIN rows=$(wc -l < "$TRAIN_OUT")"
echo "[check] VALID rows=$(wc -l < "$VALID_OUT")"
echo "[check] RISK_SCORER_DIR=$RISK_SCORER_DIR"

echo "[risk-tune] start $(date '+%F %T')"

python scripts/train_gemma_risk_tune_from_sft.py \
  --base_model google/gemma-4-E4B-it \
  --init_adapter_dir "$SFT_ADAPTER" \
  --train_file "$TRAIN_OUT" \
  --valid_file "$VALID_OUT" \
  --output_dir "$OUT" \
  --router_dir outputs/models/aspect_router_exp295_multilabel/final \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --zt_strategy staged_risk \
  --learning_rate 5e-6 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 1536 \
  --lambda_y 0.3 \
  --risk_oversample_threshold 0.35 \
  --risk_oversample_factor 2

echo "[risk-tune] done $(date '+%F %T')"

RISK_TUNED_ADAPTER="$OUT/best"
if [ ! -d "$RISK_TUNED_ADAPTER" ]; then
  RISK_TUNED_ADAPTER="$OUT/final"
fi

echo "[selective] adapter=$RISK_TUNED_ADAPTER"
echo "[selective] start $(date '+%F %T')"

python scripts/run_gemma_selective_risk_refinement.py \
  --base_model google/gemma-4-E4B-it \
  --sft_adapter_dir "$SFT_ADAPTER" \
  --risk_adapter_dir "$RISK_TUNED_ADAPTER" \
  --sft_prompt_style "$SFT_PROMPT_STYLE" \
  --router_dir outputs/models/aspect_router_exp295_multilabel/final \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output outputs/refinement/selective_sft_plain_risk_tuned_exp295_valid_gate035.jsonl \
  --zt_strategy staged_risk \
  --max_new_tokens 160 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4 \
  --gate_risk_threshold 0.35 \
  --risk_threshold 0.35 \
  --specificity_min_ratio 0.60

echo "[all done] $(date '+%F %T')"
