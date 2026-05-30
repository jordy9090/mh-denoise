# Denoising SFT Ablation

This ablation tests whether the risk-weighted token-level CE is responsible for gains beyond the denoising input structure itself.

Use the same denoising data, same prompt structure, same Q/K/V/O PEFT LoRA setting, and same inference mode as the risk-aware denoising refiner. The only change is:

```text
--lambda_y 0.0
```

This sets target token weights to 1.0, so the model becomes an unweighted denoising SFT baseline:

```text
question + unsafe_response + g + z_t + t -> safe_response
```

## Train Exp295 Denoising SFT Without Risk-Weighted CE

Set train/valid data to the same data used by the corresponding risk-aware denoising run. For the current exp295 B/C data:

```bash
cd ~/mh-denoise
conda activate mh-denoise

OUT=outputs/models/gemma4_peft_langqkvo_infermatch_exp295_v2_bc_lambda0
LOG=outputs/logs/train_gemma4_peft_langqkvo_infermatch_exp295_v2_bc_lambda0.log

rm -rf "$OUT"

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
  --lambda_y 0.0 \
  --eval_every 25 \
  --save_every 100 \
  > "$LOG" 2>&1 &
```

Monitor:

```bash
tail -f outputs/logs/train_gemma4_peft_langqkvo_infermatch_exp295_v2_bc_lambda0.log
```

## Test Inference

Use the same selected inference mode as the risk-aware model. If validation selected `unsafe_t2`, use:

```bash
ROUTER_DIR=outputs/models/router_multilabel_v1/best
if [ ! -d "$ROUTER_DIR" ]; then
  ROUTER_DIR=outputs/models/router_multilabel_v1/final
fi

RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/best
if [ ! -d "$RISK_SCORER_DIR" ]; then
  RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/final
fi

python scripts/run_gemma_peft_real_inference.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir outputs/models/gemma4_peft_langqkvo_infermatch_exp295_v2_bc_lambda0/best \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --input data/splits_exp295/test.jsonl \
  --output outputs/refinement/gemma4_peft_exp295_v2_bc_lambda0_test_t2.jsonl \
  --modes unsafe_t2 \
  --max_new_tokens 120 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4
```

## Judge Input

```bash
python scripts/prepare_refinement_judge_input.py \
  --input outputs/refinement/gemma4_peft_exp295_v2_bc_lambda0_test_t2.jsonl \
  --output outputs/eval_inputs/exp295_denoising_sft_lambda0_judge_input.jsonl \
  --response_field peft_response \
  --system_name denoising_sft_no_risk_weight \
  --id_prefix exp295_test \
  --mode_filter unsafe_t2
```

Use this row in an ablation table as:

```text
Denoising SFT w/o risk-weighted loss
```
