# Residual MLP MoE Pipeline

This is the simpler professor-requested alternative to the LoRA-expert MoE.

## Method

The adapter uses aspect-conditioned residual MLP experts:

```text
h_out = h + residual_scale * (shared_mlp(norm(h)) + sum_k tau_k(g) * expert_mlp_k(norm(h)))
```

where `tau(g)` is the normalized six-dimensional aspect vector. The base Gemma model is frozen. Only the residual MLP MoE parameters are trained.

This path is separate from the existing LoRA-MoE path:

- LoRA-MoE: `scripts/aspect_moe_lora.py`
- Residual MLP MoE: `scripts/aspect_residual_mlp_moe.py`

## Files

- `scripts/aspect_residual_mlp_moe.py`: residual MLP MoE module, layer injection, gate setting, save/load helpers.
- `scripts/train_gemma_aspect_mlp_moe_refiner.py`: training script.
- `scripts/run_gemma_aspect_mlp_moe_real_inference.py`: inference script with the same output field `moe_response`.
- `scripts/smoke_residual_mlp_moe.py`: tiny structural smoke test.

Saved checkpoints contain:

```text
mlp_moe_adapter.pt
mlp_moe_config.json
tokenizer files
```

They do not contain a full base-model checkpoint.

## Train

```bash
cd ~/mh-denoise
conda activate mh-denoise

python scripts/train_gemma_aspect_mlp_moe_refiner.py \
  --base_model google/gemma-4-E4B-it \
  --train_file data/splits_exp295/train_mdlm.jsonl \
  --valid_file data/splits_exp295/valid_mdlm.jsonl \
  --output_dir outputs/models/gemma4_aspect_mlp_moe_exp295_r64_last8 \
  --router_dir outputs/models/aspect_router_exp295_multilabel/final \
  --risk_scorer_dir outputs/models/span_risk_multilabel_v1/best \
  --num_experts 6 \
  --mlp_bottleneck_size 64 \
  --mlp_dropout 0.05 \
  --mlp_residual_scale 0.1 \
  --mlp_layers last_8 \
  --mlp_activation silu \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 1536 \
  --lambda_y 0.0
```

Useful smoke variant:

```bash
python scripts/train_gemma_aspect_mlp_moe_refiner.py \
  --base_model google/gemma-4-E4B-it \
  --train_file data/splits_exp295/train_mdlm.jsonl \
  --valid_file data/splits_exp295/valid_mdlm.jsonl \
  --output_dir outputs/models/gemma4_aspect_mlp_moe_smoke \
  --router_dir outputs/models/aspect_router_exp295_multilabel/final \
  --risk_scorer_dir outputs/models/span_risk_multilabel_v1/best \
  --mlp_bottleneck_size 64 \
  --mlp_layers last_8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --max_train_steps 2 \
  --lambda_y 0.0
```

## Inference

```bash
cd ~/mh-denoise
conda activate mh-denoise

ADAPTER_DIR=outputs/models/gemma4_aspect_mlp_moe_exp295_r64_last8/best
if [ ! -d "$ADAPTER_DIR" ]; then
  ADAPTER_DIR=outputs/models/gemma4_aspect_mlp_moe_exp295_r64_last8/final
fi

python scripts/run_gemma_aspect_mlp_moe_real_inference.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir "$ADAPTER_DIR" \
  --router_dir outputs/models/aspect_router_exp295_multilabel/final \
  --risk_scorer_dir outputs/models/span_risk_multilabel_v1/best \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output outputs/refinement/gemma4_aspect_mlp_moe_exp295_r64_last8_valid_modes.jsonl \
  --modes empty,unsafe_t2,unsafe_t3,unsafe_t4 \
  --max_new_tokens 160 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4
```

Output rows include:

- `moe_impl: residual_mlp_moe`
- `mode`
- `g`
- `z_t`
- `span_risks`
- `moe_response_raw`
- `moe_response`

## Checks

```bash
python -m py_compile \
  scripts/aspect_residual_mlp_moe.py \
  scripts/train_gemma_aspect_mlp_moe_refiner.py \
  scripts/run_gemma_aspect_mlp_moe_real_inference.py \
  scripts/smoke_residual_mlp_moe.py
```

Tiny structural smoke:

```bash
python scripts/smoke_residual_mlp_moe.py
```
