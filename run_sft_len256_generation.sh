#!/usr/bin/env bash
set -euo pipefail

cd ~/mh-denoise

SFT_ADAPTER=outputs/models/gemma4_peft_sft_plain_exp295/final
SFT_PROMPT_STYLE=sft_plain

TRAIN_IN=data/splits_exp295/train_mdlm.jsonl
VALID_IN=data/splits_exp295/valid_mdlm.jsonl

TRAIN_OUT=outputs/refinement/sft_plain_exp295_train_outputs_len256.jsonl
VALID_OUT=outputs/refinement/sft_plain_exp295_valid_outputs_len256.jsonl

EXPECTED_TRAIN=$(wc -l < "$TRAIN_IN")
EXPECTED_VALID=$(wc -l < "$VALID_IN")

echo "[check] SFT_ADAPTER=$SFT_ADAPTER"
ls -lh "$SFT_ADAPTER"

echo "[check] expected train rows: $EXPECTED_TRAIN"
echo "[check] expected valid rows: $EXPECTED_VALID"

# If incomplete output exists, move it aside.
if [ -f "$TRAIN_OUT" ]; then
  TRAIN_N=$(wc -l < "$TRAIN_OUT")
  if [ "$TRAIN_N" -ne "$EXPECTED_TRAIN" ]; then
    echo "[warn] incomplete train output: $TRAIN_N/$EXPECTED_TRAIN. Moving aside."
    mv "$TRAIN_OUT" "${TRAIN_OUT}.partial.$(date +%Y%m%d_%H%M%S)"
  else
    echo "[skip] train output already complete: $TRAIN_N/$EXPECTED_TRAIN"
  fi
fi

if [ -f "$VALID_OUT" ]; then
  VALID_N=$(wc -l < "$VALID_OUT")
  if [ "$VALID_N" -ne "$EXPECTED_VALID" ]; then
    echo "[warn] incomplete valid output: $VALID_N/$EXPECTED_VALID. Moving aside."
    mv "$VALID_OUT" "${VALID_OUT}.partial.$(date +%Y%m%d_%H%M%S)"
  else
    echo "[skip] valid output already complete: $VALID_N/$EXPECTED_VALID"
  fi
fi

if [ ! -f "$TRAIN_OUT" ]; then
  echo "[train generation] start $(date '+%F %T')"
  python scripts/build_sft_outputs_for_risk_tuning.py \
    --base_model google/gemma-4-E4B-it \
    --adapter_dir "$SFT_ADAPTER" \
    --sft_prompt_style "$SFT_PROMPT_STYLE" \
    --input "$TRAIN_IN" \
    --output "$TRAIN_OUT" \
    --max_new_tokens 256 \
    --temperature 0.0 \
    --repetition_penalty 1.15 \
    --no_repeat_ngram_size 4
  echo "[train generation] done $(date '+%F %T')"
fi

if [ ! -f "$VALID_OUT" ]; then
  echo "[valid generation] start $(date '+%F %T')"
  python scripts/build_sft_outputs_for_risk_tuning.py \
    --base_model google/gemma-4-E4B-it \
    --adapter_dir "$SFT_ADAPTER" \
    --sft_prompt_style "$SFT_PROMPT_STYLE" \
    --input "$VALID_IN" \
    --output "$VALID_OUT" \
    --max_new_tokens 256 \
    --temperature 0.0 \
    --repetition_penalty 1.15 \
    --no_repeat_ngram_size 4
  echo "[valid generation] done $(date '+%F %T')"
fi

echo "[final check]"
wc -l "$TRAIN_OUT" "$VALID_OUT"

echo "[all done] $(date '+%F %T')"
