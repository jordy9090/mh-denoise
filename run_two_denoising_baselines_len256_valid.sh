#!/usr/bin/env bash
set -euo pipefail

cd ~/mh-denoise

pick_dir() {
  for d in "$@"; do
    if [ -d "$d" ]; then
      echo "$d"
      return 0
    fi
  done
  return 1
}

ROUTER_DIR=outputs/models/aspect_router_exp295_multilabel/final

RISK_SCORER_DIR=$(pick_dir \
  outputs/models/span_risk_multilabel_v1/best \
  outputs/models/span_risk_multilabel_v1/final || true)

# Risk-Aware Denoising Refiner = risk-weighted / BC denoiser
RISK_AWARE_ADAPTER=$(pick_dir \
  outputs/models/gemma4_peft_langqkvo_infermatch_exp295_v2_bc/best \
  outputs/models/gemma4_peft_langqkvo_infermatch_exp295_v2_bc/final \
  outputs/models/gemma4_peft_langqkvo_infermatch_exp295_v2_full_bc/best \
  outputs/models/gemma4_peft_langqkvo_infermatch_exp295_v2_full_bc/final \
  outputs/models/gemma4_peft_langqkvo_infermatch_exp295/best \
  outputs/models/gemma4_peft_langqkvo_infermatch_exp295/final \
  outputs/models/gemma4_peft_langqkvo_infermatch_main_a100/best \
  outputs/models/gemma4_peft_langqkvo_infermatch_main_a100/final \
  outputs/models/gemma4_peft_denoiser_exp295/best \
  outputs/models/gemma4_peft_denoiser_exp295/final || true)

# Denoising SFT w/o Risk-Weighted Loss = lambda0
NO_RISK_ADAPTER=$(pick_dir \
  outputs/models/gemma4_peft_langqkvo_infermatch_exp295_v2_bc_lambda0/best \
  outputs/models/gemma4_peft_langqkvo_infermatch_exp295_v2_bc_lambda0/final \
  outputs/models/gemma4_peft_langqkvo_infermatch_exp295_lambda0/best \
  outputs/models/gemma4_peft_langqkvo_infermatch_exp295_lambda0/final \
  outputs/models/gemma4_peft_denoiser_exp295_lambda0/best \
  outputs/models/gemma4_peft_denoiser_exp295_lambda0/final \
  outputs/models/gemma4_peft_denoiser_exp295_no_risk/best \
  outputs/models/gemma4_peft_denoiser_exp295_no_risk/final || true)

echo "ROUTER_DIR=$ROUTER_DIR"
echo "RISK_SCORER_DIR=$RISK_SCORER_DIR"
echo "RISK_AWARE_ADAPTER=$RISK_AWARE_ADAPTER"
echo "NO_RISK_ADAPTER=$NO_RISK_ADAPTER"

if [ ! -d "$ROUTER_DIR" ]; then
  echo "[error] missing router: $ROUTER_DIR"
  exit 1
fi

if [ -z "${RISK_SCORER_DIR:-}" ] || [ ! -d "$RISK_SCORER_DIR" ]; then
  echo "[error] missing risk scorer"
  find outputs/models -maxdepth 4 -type d | grep -Ei "span_risk|risk_scorer" || true
  exit 1
fi

if [ -z "${RISK_AWARE_ADAPTER:-}" ] || [ ! -d "$RISK_AWARE_ADAPTER" ]; then
  echo "[error] could not find RISK_AWARE_ADAPTER"
  find outputs/models -maxdepth 4 -name adapter_config.json \
    | sed 's#/adapter_config.json##' \
    | grep -Ei 'peft|denois|risk|lambda|infermatch|bc' || true
  exit 1
fi

if [ -z "${NO_RISK_ADAPTER:-}" ] || [ ! -d "$NO_RISK_ADAPTER" ]; then
  echo "[error] could not find NO_RISK_ADAPTER"
  find outputs/models -maxdepth 4 -name adapter_config.json \
    | sed 's#/adapter_config.json##' \
    | grep -Ei 'peft|denois|risk|lambda|infermatch|bc' || true
  exit 1
fi

echo
echo "[1/2] Risk-Aware Denoising Refiner len256 valid start $(date '+%F %T')"

python scripts/run_gemma_peft_real_inference.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir "$RISK_AWARE_ADAPTER" \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output outputs/refinement/risk_aware_denoising_refiner_exp295_valid_len256_unsafe_t3.jsonl \
  --modes unsafe_t3 \
  --zt_strategy staged_risk \
  --risk_threshold 0.35 \
  --max_source_len 896 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4

echo "[1/2] done $(date '+%F %T')"

echo
echo "[2/2] Denoising SFT w/o Risk-Weighted Loss len256 valid start $(date '+%F %T')"

python scripts/run_gemma_peft_real_inference.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir "$NO_RISK_ADAPTER" \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output outputs/refinement/denoising_sft_no_risk_weight_exp295_valid_len256_unsafe_t3.jsonl \
  --modes unsafe_t3 \
  --zt_strategy staged_risk \
  --risk_threshold 0.35 \
  --max_source_len 896 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4

echo "[2/2] done $(date '+%F %T')"

echo
echo "[final check]"
wc -l \
  outputs/refinement/risk_aware_denoising_refiner_exp295_valid_len256_unsafe_t3.jsonl \
  outputs/refinement/denoising_sft_no_risk_weight_exp295_valid_len256_unsafe_t3.jsonl

echo "[all done] $(date '+%F %T')"
