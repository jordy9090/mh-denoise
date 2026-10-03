#!/usr/bin/env bash
set -euo pipefail

cd ~/mh-denoise

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mh-denoise

SFT_ADAPTER=outputs/models/gemma4_peft_sft_plain_exp295/final
SFT_PROMPT_STYLE=sft_plain

TEST_IN=data/splits_exp295/test.jsonl
TEST_OUT=outputs/refinement/sft_plain_exp295_test_outputs_len256.jsonl

[ -f "$TEST_IN" ] || { echo "[error] missing test split: $TEST_IN"; exit 1; }
[ -d "$SFT_ADAPTER" ] || { echo "[error] missing SFT adapter: $SFT_ADAPTER"; exit 1; }

TEST_N=$(wc -l < "$TEST_IN")

echo "[check] TEST_IN=$TEST_IN"
echo "[check] TEST rows=$TEST_N"
echo "[check] SFT_ADAPTER=$SFT_ADAPTER"

if [ "$TEST_N" -ne 354 ]; then
  echo "[error] expected 354 test rows, got $TEST_N"
  exit 1
fi

echo "[test generation len256] start $(date '+%F %T')"

python scripts/build_sft_outputs_for_risk_tuning.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir "$SFT_ADAPTER" \
  --sft_prompt_style "$SFT_PROMPT_STYLE" \
  --input "$TEST_IN" \
  --output "$TEST_OUT" \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4

echo "[final check]"
wc -l "$TEST_IN" "$TEST_OUT"

echo "[all done test generation len256] $(date '+%F %T')"
