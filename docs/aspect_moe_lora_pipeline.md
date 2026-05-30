# Aspect MoE LoRA Pipeline

This is the method-D track for the exp295 denoising pipeline. It does not modify the existing B/C single-PEFT scripts.

## What Is Implemented

The custom part is limited to replacing selected Gemma self-attention projection modules with:

```text
base frozen linear
+ shared LoRA
+ sum_k tau_k(g) * aspect_expert_lora_k
```

The base model, tokenizer, quantized loading, training loop, optimizer, scheduler, router, risk scorer, and generation still use the existing HuggingFace/Transformers stack.

The wrapped modules are selected by:

```text
.*language_model\.layers\.[0-9]+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$
```

No vision/audio tower modules should match this pattern.

## Files

- `scripts/aspect_moe_lora.py`: shared `AspectMoELinear`, wrapping, gate setting, save/load helpers.
- `scripts/train_gemma_aspect_moe_denoiser.py`: training script for custom MoE LoRA adapters.
- `scripts/run_gemma_aspect_moe_real_inference.py`: inference script mirroring the existing real inference flow.

Existing B/C scripts are intentionally untouched:

- `scripts/train_gemma_peft_denoiser.py`
- `scripts/run_gemma_peft_real_inference.py`

## Smoke Train

```bash
python scripts/train_gemma_aspect_moe_denoiser.py \
  --train_file data/overleaf_infermatch_exp295_v2_bc/train.jsonl \
  --valid_file data/overleaf_infermatch_exp295_v2_bc/valid.jsonl \
  --output_dir outputs/models/gemma4_aspect_moe_smoke \
  --model google/gemma-4-E4B-it \
  --batch_size 1 \
  --grad_accum 1 \
  --epochs 1 \
  --max_train_steps 2 \
  --target_regex '.*language_model\.layers\.[0-9]+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$'
```

Expected debug prints:

- number of wrapped modules
- first 10 wrapped module names
- trainable parameter count
- source and `g_source` distribution
- whether weighted loss is active
- mean `tau(g)` for the first batch
- final adapter save path

The saved directory should contain:

- `moe_adapter.pt`
- `moe_config.json`
- tokenizer files

It should not contain a full base-model checkpoint.

## Smoke Inference

```bash
python scripts/run_gemma_aspect_moe_real_inference.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir outputs/models/gemma4_aspect_moe_smoke/final \
  --router_dir outputs/models/aspect_router_exp295_multilabel/final \
  --risk_scorer_dir outputs/models/span_risk_multilabel_v1/best \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output outputs/refinement/gemma4_aspect_moe_smoke_valid_t4.jsonl \
  --modes unsafe_t4 \
  --max_examples 3 \
  --max_new_tokens 80 \
  --temperature 0.0
```

Output rows include:

- `mode`
- `g`
- `z_t`
- `span_risks`
- `moe_response_raw`
- `moe_response`

## Full Train Template

```bash
OUT=outputs/models/gemma4_aspect_moe_exp295_v2_bc
LOG=outputs/logs/train_gemma4_aspect_moe_exp295_v2_bc.log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -u scripts/train_gemma_aspect_moe_denoiser.py \
  --train_file data/overleaf_infermatch_exp295_v2_bc/train.jsonl \
  --valid_file data/overleaf_infermatch_exp295_v2_bc/valid.jsonl \
  --output_dir "$OUT" \
  --model google/gemma-4-E4B-it \
  --batch_size 1 \
  --grad_accum 16 \
  --epochs 3 \
  --lr 5e-5 \
  --r_shared 8 \
  --r_expert 8 \
  --alpha_shared 16 \
  --alpha_expert 16 \
  --dropout 0.05 \
  --target_regex '.*language_model\.layers\.[0-9]+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$' \
  --max_source_len 512 \
  --max_target_len 160 \
  --eval_every 25 \
  --save_every 100 \
  > "$LOG" 2>&1 &
```

## Full Inference Template

```bash
python scripts/run_gemma_aspect_moe_real_inference.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir outputs/models/gemma4_aspect_moe_exp295_v2_bc/best \
  --router_dir outputs/models/aspect_router_exp295_multilabel/final \
  --risk_scorer_dir outputs/models/span_risk_multilabel_v1/best \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output outputs/refinement/gemma4_aspect_moe_exp295_v2_bc_valid_t4.jsonl \
  --modes unsafe_t4 \
  --max_new_tokens 120 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4
```

## Notes

- `g` is still serialized into the prompt for comparability with B/C.
- The same `g` is also normalized into `tau(g)` and used internally by every `AspectMoELinear`.
- During inference, `g` comes from the router prediction, matching the B/C policy.
- If local CPU testing is needed with a tiny model, pass `--no_4bit`; the exp295 Gemma run should keep the default 4-bit path.
