#!/usr/bin/env bash
set -euo pipefail

cd ~/mh-denoise
mkdir -p outputs/models outputs/logs

SHARED_INIT=outputs/models/gemma4_peft_langqkvo_infermatch_exp295_v2_bc_lambda0/best
if [ ! -f "$SHARED_INIT/adapter_model.safetensors" ]; then
  echo "ERROR: missing shared init adapter_model.safetensors under $SHARED_INIT"
  exit 1
fi

OUT=outputs/models/gemma4_aspect_moe_exp295_v2_bc_init_lambda0_joint_r4_lr5e6_unweighted
LOG=outputs/logs/train_gemma4_aspect_moe_exp295_v2_bc_init_lambda0_joint_r4_lr5e6_unweighted.log

rm -rf "$OUT"
mkdir -p "$OUT"

CMD=(
  python -u scripts/train_gemma_aspect_moe_denoiser.py
  --train_file data/overleaf_infermatch_exp295_v2_bc/train.jsonl
  --valid_file data/overleaf_infermatch_exp295_v2_bc/valid.jsonl
  --output_dir "$OUT"
  --model google/gemma-4-E4B-it
  --batch_size 1
  --grad_accum 16
  --epochs 1
  --lr 5e-6
  --r_shared 8
  --r_expert 4
  --alpha_shared 16
  --alpha_expert 8
  --dropout 0.1
  --init_shared_adapter_dir "$SHARED_INIT"
  --lambda_y 0.0
  --max_train_steps 160
  --eval_every 10
  --save_every 50
  --target_regex '.*language_model\.layers\.[0-9]+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$'
  --max_source_len 512
  --max_target_len 160
)

echo "SHARED_INIT=$SHARED_INIT"
echo "OUT=$OUT"
echo "LOG=$LOG"
printf 'CMD: %q ' "${CMD[@]}"
echo

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup "${CMD[@]}" > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$OUT/pid.txt"
echo "PID=$PID"
echo "LOG=$LOG"
