# Selective Risk-Aware Refinement Pipeline

This pipeline uses the stable SFT refiner as the primary response repair model and applies a weak risk-aware denoiser only when the SFT output still appears risky.

## Method

```text
question, unsafe_response
  -> SFT refiner
  -> sft_response
  -> safety/risk gate
  -> if safe: final_response = sft_response
  -> if risky:
       z_t_from_sft = C(sft_response, g_sft)
       denoiser_response = risk_refiner(question, unsafe_response, sft_response, z_t_from_sft, g_sft)
       final_response = denoiser_response only if risk decreases and specificity is preserved
```

The important design choice is that `z_t_from_sft` is constructed from the SFT response, not from the original unsafe response. The original unsafe response is retained only as context so that the risk-tuned model can recover useful details if the SFT response is too generic.

## Files

- `scripts/build_sft_outputs_for_risk_tuning.py`: runs an SFT adapter and saves `sft_response`.
- `scripts/train_gemma_risk_tune_from_sft.py`: initializes from the SFT LoRA adapter and weakly tunes on risk-corrupted SFT responses.
- `scripts/run_gemma_selective_risk_refinement.py`: runs the selective SFT-first cascade with accept/reject rules.
- `scripts/selective_risk_refinement_utils.py`: shared prompt, scoring, corruption, and selection utilities.

## 0. Train Or Reuse The SFT Refiner

If an SFT plain adapter already exists, reuse it. Otherwise train it with the existing SFT path:

```bash
python scripts/train_professor_peft_refiner.py \
  --model google/gemma-4-E4B-it \
  --train_file data/splits_exp295/train_mdlm.jsonl \
  --valid_file data/splits_exp295/valid_mdlm.jsonl \
  --output_dir outputs/models/gemma4_peft_sft_plain_exp295 \
  --prompt_style sft_plain \
  --lr 5e-5 \
  --epochs 3 \
  --batch_size 1 \
  --grad_accum 8 \
  --max_source_len 896 \
  --max_target_len 220
```

## 1. Build SFT Outputs

```bash
cd ~/mh-denoise
conda activate mh-denoise

SFT_ADAPTER=outputs/models/gemma4_peft_sft_plain_exp295/best
if [ ! -d "$SFT_ADAPTER" ]; then
  SFT_ADAPTER=outputs/models/gemma4_peft_sft_plain_exp295/final
fi

python scripts/build_sft_outputs_for_risk_tuning.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir "$SFT_ADAPTER" \
  --input data/splits_exp295/train_mdlm.jsonl \
  --output outputs/refinement/sft_refiner_train_outputs.jsonl \
  --max_new_tokens 160 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4

python scripts/build_sft_outputs_for_risk_tuning.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir "$SFT_ADAPTER" \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output outputs/refinement/sft_refiner_valid_outputs.jsonl \
  --max_new_tokens 160 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4
```

## 2. Weak Risk Tuning From SFT

```bash
RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/best
if [ ! -d "$RISK_SCORER_DIR" ]; then
  RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/final
fi

python scripts/train_gemma_risk_tune_from_sft.py \
  --base_model google/gemma-4-E4B-it \
  --init_adapter_dir "$SFT_ADAPTER" \
  --train_file outputs/refinement/sft_refiner_train_outputs.jsonl \
  --valid_file outputs/refinement/sft_refiner_valid_outputs.jsonl \
  --output_dir outputs/models/gemma4_sft_risk_tuned_exp295_lr5e6_lambda03 \
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
```

The training script scores `sft_response`, builds `z_t_from_sft`, and writes enriched copies to:

```text
outputs/models/gemma4_sft_risk_tuned_exp295_lr5e6_lambda03/risk_tune_train_enriched.jsonl
outputs/models/gemma4_sft_risk_tuned_exp295_lr5e6_lambda03/risk_tune_valid_enriched.jsonl
```

## 3. Selective Inference

```bash
RISK_TUNED_ADAPTER=outputs/models/gemma4_sft_risk_tuned_exp295_lr5e6_lambda03/best
if [ ! -d "$RISK_TUNED_ADAPTER" ]; then
  RISK_TUNED_ADAPTER=outputs/models/gemma4_sft_risk_tuned_exp295_lr5e6_lambda03/final
fi

python scripts/run_gemma_selective_risk_refinement.py \
  --base_model google/gemma-4-E4B-it \
  --sft_adapter_dir "$SFT_ADAPTER" \
  --risk_adapter_dir "$RISK_TUNED_ADAPTER" \
  --router_dir outputs/models/aspect_router_exp295_multilabel/final \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output outputs/refinement/selective_sft_risk_tuned_exp295_valid.jsonl \
  --zt_strategy staged_risk \
  --max_new_tokens 160 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4 \
  --gate_risk_threshold 0.35 \
  --risk_threshold 0.35 \
  --specificity_min_ratio 0.60
```

## Output Fields

Each output row includes:

- `sft_response`
- `z_t_from_sft`
- `denoiser_response`
- `final_response`
- `used_denoiser`
- `accepted_denoiser`
- `reject_reason`
- `sft_risk_score`
- `denoiser_risk_score`
- `final_risk_score`
- `specificity_ratio`
- `g_sft`
- `g_denoiser`

## Sanity Check

```bash
python - <<'PY'
import json, collections, statistics

p = "outputs/refinement/selective_sft_risk_tuned_exp295_valid.jsonl"
rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

print("rows", len(rows))
print("used_denoiser", collections.Counter(r.get("used_denoiser") for r in rows))
print("accepted_denoiser", collections.Counter(r.get("accepted_denoiser") for r in rows))
print("reject_reason", collections.Counter(r.get("reject_reason") for r in rows if r.get("reject_reason")))
print("avg_sft_risk", round(statistics.mean(float(r.get("sft_risk_score", 0.0)) for r in rows), 4))
print("avg_final_risk", round(statistics.mean(float(r.get("final_risk_score", 0.0)) for r in rows), 4))
ratios = [float(r.get("specificity_ratio", 1.0)) for r in rows if r.get("used_denoiser")]
print("avg_specificity_ratio_called", round(statistics.mean(ratios), 4) if ratios else 1.0)
PY
```

If the denoiser is called for nearly every example, increase `--gate_risk_threshold`. If it is almost never called, decrease it. If many denoiser outputs are rejected for `genericity_increased` or `specificity_ratio_low`, keep the SFT response and avoid increasing risk-tuning strength.

## Validation

```bash
python -m py_compile \
  scripts/selective_risk_refinement_utils.py \
  scripts/build_sft_outputs_for_risk_tuning.py \
  scripts/train_gemma_risk_tune_from_sft.py \
  scripts/run_gemma_selective_risk_refinement.py
```
