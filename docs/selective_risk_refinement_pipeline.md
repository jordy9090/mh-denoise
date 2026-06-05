# Selective Risk-Aware Refinement Pipeline

Status: current professor-proposed pipeline.

This document is the active method note for the SFT-first cascade. Older MoE, full-joint, and residual-MLP experiments are not part of this pipeline.

## Professor Proposal

```text
Stage 1: SFT Refiner
q + unsafe_response -> safe_response

Stage 2: Risk-aware Denoiser
Initialize from the SFT checkpoint.
q + unsafe_response + sft_response + g + z_t + risk_spans -> safe_response
Continue tuning with a weak learning rate.

Inference:
unsafe draft
  -> generate SFT Refiner output
  -> run safety gate
  -> if safe, use the SFT output
  -> if unsafe, call the denoiser
  -> use denoiser output only if safety improves and specificity is preserved
  -> otherwise keep the SFT output
```

The implementation follows this proposal with one important operational choice: `z_t` and `risk_spans` are constructed from `sft_response`, not from the original `unsafe_response`. The denoiser is therefore a residual repair model for risks that remain after the SFT refiner, while the original unsafe response is retained only as context.

## Active Checkpoints

Use the q+u-only SFT adapter as the first-stage anchor:

```bash
SFT_ADAPTER=outputs/models/gemma4_peft_sft_plain_exp295/final
SFT_PROMPT_STYLE=sft_plain
```

This is the intended anchor for the current pipeline. Do not use `outputs/models/professor_peft_refiner_textonly_main/final` as the main anchor; that checkpoint belongs to an older professor-style SFT refiner family with different inputs.

## Files

- `scripts/build_sft_outputs_for_risk_tuning.py`: runs the SFT adapter and saves `sft_response`.
- `scripts/train_gemma_risk_tune_from_sft.py`: initializes from the SFT LoRA adapter and weakly tunes on risk-corrupted SFT responses.
- `scripts/run_gemma_selective_risk_refinement.py`: runs the selective SFT-first cascade with accept/reject rules.
- `scripts/selective_risk_refinement_utils.py`: shared prompt, scoring, corruption, and selection utilities.

## 0. Ensure The SFT Anchor Exists

If the q+u-only SFT adapter is already available, use it directly. If it is only available on another machine, copy the model directory before continuing.

```bash
ls -lh outputs/models/gemma4_peft_sft_plain_exp295/final
```

If it is missing, reproduce it with:

```bash
OUT=outputs/models/gemma4_peft_sft_plain_exp295
LOG=outputs/logs/train_gemma4_peft_sft_plain_exp295.log
mkdir -p outputs/logs

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -u scripts/train_professor_peft_refiner_textonly.py \
  --train_file data/splits_exp295/train_mdlm.jsonl \
  --valid_file data/splits_exp295/valid_mdlm.jsonl \
  --output_dir "$OUT" \
  --model google/gemma-4-E4B-it \
  --prompt_style sft_plain \
  --epochs 3 \
  --lr 5e-5 \
  --lora_r 8 \
  --lora_alpha 16 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --grad_accum 16 \
  --target_modules q_proj,k_proj,v_proj,o_proj \
  --max_source_len 512 \
  --max_target_len 160 \
  --eval_steps 25 \
  --save_steps 100 \
  > "$LOG" 2>&1 &
```

## 1. Build SFT Outputs

```bash
cd ~/mh-denoise
conda activate mh-denoise

SFT_ADAPTER=outputs/models/gemma4_peft_sft_plain_exp295/final
SFT_PROMPT_STYLE=sft_plain

python scripts/build_sft_outputs_for_risk_tuning.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir "$SFT_ADAPTER" \
  --sft_prompt_style "$SFT_PROMPT_STYLE" \
  --input data/splits_exp295/train_mdlm.jsonl \
  --output outputs/refinement/sft_plain_exp295_train_outputs.jsonl \
  --max_new_tokens 160 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4

python scripts/build_sft_outputs_for_risk_tuning.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir "$SFT_ADAPTER" \
  --sft_prompt_style "$SFT_PROMPT_STYLE" \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output outputs/refinement/sft_plain_exp295_valid_outputs.jsonl \
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

OUT=outputs/models/gemma4_selective_sft_plain_risk_tuned_exp295_lr5e6_lambda03

python scripts/train_gemma_risk_tune_from_sft.py \
  --base_model google/gemma-4-E4B-it \
  --init_adapter_dir "$SFT_ADAPTER" \
  --train_file outputs/refinement/sft_plain_exp295_train_outputs.jsonl \
  --valid_file outputs/refinement/sft_plain_exp295_valid_outputs.jsonl \
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
```

The training script scores `sft_response`, builds `z_t_from_sft`, and writes enriched copies under the risk-tuned output directory.

## 3. Selective Inference

```bash
RISK_TUNED_ADAPTER=outputs/models/gemma4_selective_sft_plain_risk_tuned_exp295_lr5e6_lambda03/best
if [ ! -d "$RISK_TUNED_ADAPTER" ]; then
  RISK_TUNED_ADAPTER=outputs/models/gemma4_selective_sft_plain_risk_tuned_exp295_lr5e6_lambda03/final
fi

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

p = "outputs/refinement/selective_sft_plain_risk_tuned_exp295_valid_gate035.jsonl"
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
