# Exp295 B/C Pipeline

This note covers only method components B and C for the exp295/v2 denoising pipeline.
It intentionally keeps component D, aspect-specific expert LoRA MoE, out of scope.

## Method Mapping

- B, aspect router: `scripts/train_router_multilabel.py` trains `H_omega(q,u)` with `BCEWithLogitsLoss` over the six dimensions and saves sigmoid-compatible sequence-classification checkpoints.
- C, teacher-forcing aspect mixture: `scripts/prepare_overleaf_infermatch_data_v2.py` can now choose the denoiser conditioning vector `g` from either the gold vector `d` or router prediction `d_hat` during training data generation.
- E/F, source mixture and risk-aware `z_t`: already live in the v2 generator and are unchanged.
- D, expert LoRA MoE: not implemented here. The denoiser remains the existing single PEFT adapter pipeline.

Dimensions:

```text
overall_quality, empathy, specificity, medical_advice, factual_consistency, toxicity
```

## 1. Train Exp295 Multi-Label Router

```bash
mkdir -p outputs/models outputs/logs

python scripts/train_router_multilabel.py \
  --train_file data/splits_exp295/train_mdlm.jsonl \
  --valid_file data/splits_exp295/valid_mdlm.jsonl \
  --output_dir outputs/models/aspect_router_exp295_multilabel \
  --model bert-base-uncased \
  --batch_size 8 \
  --epochs 6 \
  --lr 1e-5 \
  --eval_every 25 \
  --seed 42
```

The script writes:

- `outputs/models/aspect_router_exp295_multilabel/best`
- `outputs/models/aspect_router_exp295_multilabel/final`
- `dims.json`
- `eval_metrics.json` with exact match, top1 accuracy, macro F1, and per-dimension precision/recall/F1.

Use the final router for the B/C run unless you intentionally choose `best`:

```bash
ROUTER_DIR=outputs/models/aspect_router_exp295_multilabel/final
RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/best
if [ ! -d "$RISK_SCORER_DIR" ]; then
  RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/final
fi
```

## 2. Generate V2 Train Data With Gold/Pred Mixture

Training uses the method-C mixture. With `--aspect_tf_prob 0.5`, each base example selects gold `d` about half the time and router prediction `d_hat` otherwise. All source variants from the same base example inherit the selected `g`.

```bash
mkdir -p data/overleaf_infermatch_exp295_v2_bc

python scripts/prepare_overleaf_infermatch_data_v2.py \
  --input data/splits_exp295/train_mdlm.jsonl \
  --output data/overleaf_infermatch_exp295_v2_bc/train.jsonl \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --T 4 \
  --use_gold_aspect_mixing \
  --aspect_tf_prob 0.5 \
  --seed 42
```

Expected row metadata includes:

```json
{
  "g": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
  "g_gold": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
  "g_pred": [0.12, 0.28, 0.41, 0.83, 0.35, 0.05],
  "g_source": "gold"
}
```

The manifest records `use_gold_aspect_mixing`, `aspect_tf_prob`, `g_source_distribution`, and gold-dimension warning counts.

## 3. Generate V2 Valid Data Pred-Only

Validation should match inference, where gold labels are unavailable. Do not pass `--use_gold_aspect_mixing`.

```bash
python scripts/prepare_overleaf_infermatch_data_v2.py \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output data/overleaf_infermatch_exp295_v2_bc/valid.jsonl \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --T 4 \
  --aspect_tf_prob 0.0 \
  --seed 43
```

The valid manifest should show:

```json
{
  "g_source_distribution": {
    "pred": 1218
  }
}
```

The exact count depends on the source/timestep schedule, but `g_source` should be `pred` only.

## 4. Train Existing Single-PEFT Denoiser

`scripts/train_gemma_peft_denoiser.py` already consumes `row["g"]` in the prompt and uses `target_weight_spans` for weighted CE, so no denoiser code change is needed for B/C.

```bash
OUT=outputs/models/gemma4_peft_langqkvo_infermatch_exp295_v2_bc
LOG=outputs/logs/train_gemma4_peft_langqkvo_infermatch_exp295_v2_bc.log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -u scripts/train_gemma_peft_denoiser.py \
  --train_file data/overleaf_infermatch_exp295_v2_bc/train.jsonl \
  --valid_file data/overleaf_infermatch_exp295_v2_bc/valid.jsonl \
  --output_dir "$OUT" \
  --model google/gemma-4-E4B-it \
  --epochs 3 \
  --lr 5e-5 \
  --r 8 \
  --alpha 16 \
  --batch_size 1 \
  --grad_accum 16 \
  --target_modules 'regex:.*language_model\.layers\.[0-9]+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$' \
  --max_source_len 512 \
  --max_target_len 160 \
  --eval_every 25 \
  --save_every 100 \
  > "$LOG" 2>&1 &
```

## 5. Run Existing Inference

Inference remains pred-only: the script computes `g = sigmoid(router(q,u))` and does not use gold labels.

```bash
python scripts/run_gemma_peft_real_inference.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir outputs/models/gemma4_peft_langqkvo_infermatch_exp295_v2_bc/best \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output outputs/refinement/gemma4_peft_langqkvo_infermatch_exp295_v2_bc_valid_t4.jsonl \
  --modes unsafe_t4 \
  --max_new_tokens 120 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4
```

Then run the existing judge workflow on the resulting refinement file.
