#!/usr/bin/env bash
set -euo pipefail

cd ~/mh-denoise

mkdir -p outputs/refinement outputs/eval_inputs outputs/eval outputs/analysis outputs/logs

BASE_MODEL=google/gemma-4-E4B-it
TEST_FILE=data/splits_exp295/test.jsonl
JUDGE_MODEL=gpt-4.1
[ -n "${OPENAI_API_KEY:-}" ] || { echo "[missing] OPENAI_API_KEY"; exit 1; }

SFT_ADAPTER=outputs/models/gemma4_peft_sft_plain_exp295/final

# Final selective risk-tuned adapter: len256 clean run
RISK_TUNED_ADAPTER=outputs/models/gemma4_selective_sft_plain_risk_tuned_exp295_len256_lr5e6_lambda03_clean/best
if [ ! -d "$RISK_TUNED_ADAPTER" ]; then
  RISK_TUNED_ADAPTER=outputs/models/gemma4_selective_sft_plain_risk_tuned_exp295_len256_lr5e6_lambda03_clean/final
fi

ROUTER_DIR=outputs/models/aspect_router_exp295_multilabel/final

RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/best
if [ ! -d "$RISK_SCORER_DIR" ]; then
  RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/final
fi

# Denoising SFT w/o Risk Weight = lambda0 adapter
NO_RISK_ADAPTER=outputs/models/gemma4_peft_langqkvo_infermatch_exp295_v2_bc_lambda0/best
if [ ! -d "$NO_RISK_ADAPTER" ]; then
  NO_RISK_ADAPTER=outputs/models/gemma4_peft_langqkvo_infermatch_exp295_v2_bc_lambda0/final
fi

# Risk-Aware Denoising Refiner = risk-weighted / BC adapter
RISK_AWARE_ADAPTER=outputs/models/gemma4_peft_langqkvo_infermatch_exp295_v2_bc/best
if [ ! -d "$RISK_AWARE_ADAPTER" ]; then
  RISK_AWARE_ADAPTER=outputs/models/gemma4_peft_langqkvo_infermatch_exp295_v2_bc/final
fi

echo "============================================================"
echo "[PATH CHECK]"
echo "BASE_MODEL=$BASE_MODEL"
echo "TEST_FILE=$TEST_FILE"
echo "JUDGE_MODEL=$JUDGE_MODEL"
echo "SFT_ADAPTER=$SFT_ADAPTER"
echo "RISK_TUNED_ADAPTER=$RISK_TUNED_ADAPTER"
echo "ROUTER_DIR=$ROUTER_DIR"
echo "RISK_SCORER_DIR=$RISK_SCORER_DIR"
echo "NO_RISK_ADAPTER=$NO_RISK_ADAPTER"
echo "RISK_AWARE_ADAPTER=$RISK_AWARE_ADAPTER"
echo "============================================================"

[ -f "$TEST_FILE" ] || { echo "[missing] TEST_FILE=$TEST_FILE"; exit 1; }
[ -d "$SFT_ADAPTER" ] || { echo "[missing] SFT_ADAPTER=$SFT_ADAPTER"; exit 1; }
[ -d "$RISK_TUNED_ADAPTER" ] || { echo "[missing] RISK_TUNED_ADAPTER=$RISK_TUNED_ADAPTER"; exit 1; }
[ -d "$ROUTER_DIR" ] || { echo "[missing] ROUTER_DIR=$ROUTER_DIR"; exit 1; }
[ -d "$RISK_SCORER_DIR" ] || { echo "[missing] RISK_SCORER_DIR=$RISK_SCORER_DIR"; exit 1; }
[ -d "$NO_RISK_ADAPTER" ] || { echo "[missing] NO_RISK_ADAPTER=$NO_RISK_ADAPTER"; exit 1; }
[ -d "$RISK_AWARE_ADAPTER" ] || { echo "[missing] RISK_AWARE_ADAPTER=$RISK_AWARE_ADAPTER"; exit 1; }

echo
echo "============================================================"
echo "[1/8] Build SFT test outputs len256"
echo "============================================================"

python scripts/build_sft_outputs_for_risk_tuning.py \
  --base_model "$BASE_MODEL" \
  --adapter_dir "$SFT_ADAPTER" \
  --sft_prompt_style sft_plain \
  --input "$TEST_FILE" \
  --output outputs/refinement/sft_plain_exp295_test_outputs_len256.jsonl \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4

echo
echo "============================================================"
echo "[2/8] Denoising SFT w/o Risk Weight test len256 t2"
echo "============================================================"

python scripts/run_gemma_peft_real_inference.py \
  --base_model "$BASE_MODEL" \
  --adapter_dir "$NO_RISK_ADAPTER" \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --input "$TEST_FILE" \
  --output outputs/refinement/denoising_sft_no_risk_weight_exp295_test_len256_t2.jsonl \
  --modes unsafe_t2 \
  --zt_strategy staged_risk \
  --risk_threshold 0.35 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4

echo
echo "============================================================"
echo "[3/8] Risk-Aware Denoising Refiner test len256 t2"
echo "============================================================"

python scripts/run_gemma_peft_real_inference.py \
  --base_model "$BASE_MODEL" \
  --adapter_dir "$RISK_AWARE_ADAPTER" \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --input "$TEST_FILE" \
  --output outputs/refinement/risk_aware_denoising_refiner_exp295_test_len256_t2.jsonl \
  --modes unsafe_t2 \
  --zt_strategy staged_risk \
  --risk_threshold 0.35 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4

echo
echo "============================================================"
echo "[4/8] Selective final test len256 medical-focused t2"
echo "============================================================"

python scripts/run_gemma_selective_risk_refinement.py \
  --base_model "$BASE_MODEL" \
  --sft_adapter_dir "$SFT_ADAPTER" \
  --risk_adapter_dir "$RISK_TUNED_ADAPTER" \
  --sft_prompt_style sft_plain \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --input outputs/refinement/sft_plain_exp295_test_outputs_len256.jsonl \
  --output outputs/refinement/selective_sft_plain_risk_tuned_exp295_test_len256_medical_focus_th001_t2.jsonl \
  --reuse_sft_response \
  --sft_response_field sft_response \
  --zt_strategy staged_risk \
  --timestep 2 \
  --risk_threshold 0.35 \
  --gate_strategy aspect_only \
  --gate_focus_aspect medical_advice \
  --gate_focus_threshold 0.01 \
  --min_risk_delta 0.0 \
  --min_focus_risk_delta 0.0 \
  --specificity_min_ratio 0.60 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4

echo
echo "============================================================"
echo "[5/8] Truncation fallback filtering"
echo "============================================================"

python - <<'PY'
import json, re

inp = "outputs/refinement/selective_sft_plain_risk_tuned_exp295_test_len256_medical_focus_th001_t2.jsonl"
out = "outputs/refinement/selective_sft_plain_risk_tuned_exp295_test_len256_medical_focus_th001_t2_trunc_filtered.jsonl"

end_punct = (".", "!", "?", ")", "]", "}", '"', "'")
bad_last = set("""
and or but because so if when while although though since with without for to of in on at by from into onto about
this that these those a an the your my our their his her its as than then just simply really also
be become becoming better lis unde outside crying Good good
""".split())

def last_word(text):
    toks = re.findall(r"[A-Za-z]+", str(text or "").strip())
    return toks[-1] if toks else ""

def likely_truncated(text):
    t = str(text or "").strip()
    if not t:
        return True
    if not t.endswith(end_punct):
        return True
    if last_word(t) in bad_last:
        return True
    return False

rows = [json.loads(l) for l in open(inp, encoding="utf-8") if l.strip()]
called = sum(bool(r.get("used_denoiser")) for r in rows)
accepted_raw = sum(bool(r.get("accepted_denoiser")) for r in rows)

kept = 0
fallback = 0

with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        rr = dict(r)
        trunc = bool(rr.get("accepted_denoiser")) and likely_truncated(rr.get("denoiser_response", ""))
        rr["denoiser_likely_truncated"] = trunc
        rr["accepted_denoiser_raw"] = bool(rr.get("accepted_denoiser"))
        if trunc:
            rr["final_response"] = rr.get("sft_response", "")
            rr["accepted_denoiser"] = False
            rr["truncation_fallback"] = True
            fallback += 1
        else:
            rr["truncation_fallback"] = False
            if rr.get("accepted_denoiser"):
                kept += 1
        f.write(json.dumps(rr, ensure_ascii=False) + "\n")

print("n", len(rows))
print("called", called)
print("accepted_raw", accepted_raw)
print("accepted_after_trunc_filter", kept)
print("trunc_fallback", fallback)
print("saved", out)
PY

echo
echo "============================================================"
echo "[6/8] Prepare judge inputs"
echo "============================================================"

python scripts/prepare_refinement_judge_input.py \
  --input outputs/refinement/sft_plain_exp295_test_outputs_len256.jsonl \
  --output outputs/eval_inputs/exp295_test_sft_unsafe_safe_len256_judge_input.jsonl \
  --response_field sft_response \
  --system_name sft_refiner \
  --id_prefix exp295_test \
  --include_unsafe_baseline \
  --include_safe_reference \
  --unsafe_system_name unsafe_response \
  --safe_reference_system_name gold_safe_reference

python scripts/prepare_refinement_judge_input.py \
  --input outputs/refinement/denoising_sft_no_risk_weight_exp295_test_len256_t2.jsonl \
  --output outputs/eval_inputs/exp295_test_denoising_sft_no_risk_len256_t2_judge_input.jsonl \
  --response_field peft_response \
  --system_name denoising_sft_no_risk_weight \
  --id_prefix exp295_test

python scripts/prepare_refinement_judge_input.py \
  --input outputs/refinement/risk_aware_denoising_refiner_exp295_test_len256_t2.jsonl \
  --output outputs/eval_inputs/exp295_test_risk_aware_denoising_len256_t2_judge_input.jsonl \
  --response_field peft_response \
  --system_name risk_aware_denoising_refiner \
  --id_prefix exp295_test

python scripts/prepare_refinement_judge_input.py \
  --input outputs/refinement/selective_sft_plain_risk_tuned_exp295_test_len256_medical_focus_th001_t2_trunc_filtered.jsonl \
  --output outputs/eval_inputs/exp295_test_selective_medical_focus_t2_trunc_filtered_judge_input.jsonl \
  --response_field final_response \
  --system_name selective_risk_aware_refinement \
  --id_prefix exp295_test

cat \
  outputs/eval_inputs/exp295_test_sft_unsafe_safe_len256_judge_input.jsonl \
  outputs/eval_inputs/exp295_test_denoising_sft_no_risk_len256_t2_judge_input.jsonl \
  outputs/eval_inputs/exp295_test_risk_aware_denoising_len256_t2_judge_input.jsonl \
  outputs/eval_inputs/exp295_test_selective_medical_focus_t2_trunc_filtered_judge_input.jsonl \
  > outputs/eval_inputs/exp295_test_main_table_len256_counselbench_input.jsonl

echo
echo "[judge input rows]"
wc -l outputs/eval_inputs/exp295_test_main_table_len256_counselbench_input.jsonl

echo
echo "============================================================"
echo "[7/8] CounselBench judge"
echo "============================================================"

python scripts/run_refinement_llm_judge.py \
  --input outputs/eval_inputs/exp295_test_main_table_len256_counselbench_input.jsonl \
  --output outputs/eval/exp295_test_main_table_len256_counselbench_judged.jsonl \
  --model "$JUDGE_MODEL" \
  --rubric_style counselbench \
  --resume \
  --sleep 0.5

echo
echo "============================================================"
echo "[8/8] Aggregate"
echo "============================================================"

python scripts/aggregate_refinement_judge_scores.py \
  --input outputs/eval/exp295_test_main_table_len256_counselbench_judged.jsonl \
  --output_csv outputs/analysis/exp295_test_main_table_len256_counselbench_by_system.csv \
  --group_by system

echo
echo "============================================================"
echo "[FINAL TABLE CSV]"
echo "============================================================"
cat outputs/analysis/exp295_test_main_table_len256_counselbench_by_system.csv

echo
echo "============================================================"
echo "[DONE]"
echo "============================================================"
