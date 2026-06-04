# Router-Denoiser Full Joint Training

This note documents the first full-joint training implementation aligned with `docs/professor_method_section.md`.

## What Is Joint

The joint part is the aspect router plus MoE denoising refiner:

```text
q, u -> router H_omega -> g_pred
d_gold, g_pred -> sampled aspect condition g
g -> prompt text conditioning
g -> differentiable MoE expert gate
q, u, g, z_t, t -> Gemma MoE denoiser -> y
```

The training objective follows the professor method:

```text
L = L_den + lambda_sft * L_sft + lambda_router * L_router
```

- `L_den`: trains `q,u,g,z_t,t -> safe_response`.
- `L_sft`: trains `q,u,g -> safe_response` with no corrupted draft.
- `L_router`: trains `H_omega(q,u) -> d_gold` with multi-label BCE.

The router receives denoising gradients only through the continuous MoE gate path:

```text
router logits -> sigmoid(g_pred) -> g -> tau(g) -> expert mixture -> LM logits -> CE loss
```

The serialized prompt text also contains `g` for comparability with prior runs, but prompt text construction is discrete and does not carry gradients.

In v1, router-denoiser coupling is carried by `L_den`. `L_sft` uses detached MoE gates so the same router-derived gate graph is not reused across two Gemma forwards in one step.

## What Is Not Joint Yet

The span-risk scorer is frozen in v1.

This is deliberate. The professor objective includes `L_router`, but does not define an explicit `L_risk`. Also, `z_t` construction is a discrete string corruption process, so gradients do not naturally flow through masking decisions. Training the risk scorer through token weights can create a degenerate path where the scorer reduces loss by lowering risk rather than improving corruption quality.

Use this wording in notes or paper drafts:

```text
We jointly train the aspect router and MoE denoising refiner under a unified objective. The router prediction is used as both textual conditioning and a differentiable MoE gate, while the span-risk scorer is kept frozen and used to construct online corrupted drafts.
```

Avoid this wording for v1:

```text
We train the router, risk scorer, corruption process, and denoiser fully end-to-end.
```

## Files

- `scripts/train_gemma_full_joint_denoiser.py`: new router-denoiser joint training script.
- `scripts/aspect_moe_lora.py`: adds optional differentiable gate passing via `set_moe_gates(..., detach=False)`.

Existing PEFT and MoE scripts keep their current behavior because gate detaching remains the default.

Gradient checkpointing is disabled by default in the full-joint script. The MoE gate tensor is stored as module state inside every `AspectMoELinear`; checkpoint recomputation can otherwise reread a changed or freed gate graph during backward.

## Command Template

```bash
OUT=outputs/models/gemma4_full_joint_exp295_v1
LOG=outputs/logs/train_gemma4_full_joint_exp295_v1.log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -u scripts/train_gemma_full_joint_denoiser.py \
  --train_file data/splits_exp295/train_mdlm.jsonl \
  --valid_file data/splits_exp295/valid_mdlm.jsonl \
  --output_dir "$OUT" \
  --model google/gemma-4-E4B-it \
  --router_init_dir outputs/models/aspect_router_exp295_multilabel/final \
  --risk_scorer_dir outputs/models/span_risk_multilabel_v1/best \
  --init_shared_adapter_dir outputs/models/gemma4_peft_langqkvo_infermatch_exp295_v2_bc_lambda0/best \
  --batch_size 1 \
  --grad_accum 16 \
  --epochs 1 \
  --lr_lora 5e-6 \
  --lr_expert 1e-5 \
  --lr_router 1e-5 \
  --r_shared 8 \
  --r_expert 8 \
  --alpha_shared 16 \
  --alpha_expert 16 \
  --lambda_sft 0.3 \
  --lambda_router 0.1 \
  --lambda_y 0.0 \
  --aspect_tf_prob 0.5 \
  --train_timesteps 2,3,4 \
  --valid_timesteps 2,3,4 \
  --valid_g_source pred \
  --zt_strategy threshold \
  --max_source_len 512 \
  --max_target_len 160 \
  --eval_every 25 \
  --save_every 100 \
  > "$LOG" 2>&1 &
```

## Expected Logs

Startup should report:

- base pair JSON is used, not precomputed denoising JSON
- router init path
- frozen risk scorer path
- initialized shared adapter path
- shared LoRA trainable: `True`
- expert LoRA trainable: `True`
- `MoE gates detach: False`
- `gradient checkpointing enabled: False`
- `lambda_sft`, `lambda_router`, `lambda_y`, `aspect_tf_prob`
- `zt_strategy`

Training/eval logs report:

- total loss
- `L_den`
- `L_sft`
- `L_router`
- mean predicted and used aspect vectors at eval points
- gold-vs-pred aspect usage fraction
- timestep distribution
- average masked span count

## Output Layout

Each checkpoint directory contains:

- `moe_adapter.pt`
- `moe_config.json`
- Gemma tokenizer files
- `router/`
- `joint_config.json`

For inference, use the checkpoint root as the MoE adapter directory and the nested router path:

```bash
python scripts/run_gemma_aspect_moe_real_inference.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir outputs/models/gemma4_full_joint_exp295_v1/best \
  --router_dir outputs/models/gemma4_full_joint_exp295_v1/best/router \
  --risk_scorer_dir outputs/models/span_risk_multilabel_v1/best \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output outputs/refinement/gemma4_full_joint_exp295_v1_valid_t3.jsonl \
  --modes unsafe_t3 \
  --max_new_tokens 120 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4
```

## Troubleshooting

If inference fails before generation and the next sanity script says the output JSONL does not exist, check the checkpoint paths first:

```bash
echo "$ADAPTER_DIR"
echo "$ROUTER_DIR"
echo "$RISK_SCORER_DIR"
ls -lah "$ADAPTER_DIR" "$ROUTER_DIR" "$RISK_SCORER_DIR"
```

For a full-joint checkpoint, `$ROUTER_DIR` must point to the nested `router/` directory inside the selected checkpoint, for example:

```bash
ROUTER_DIR=outputs/models/gemma4_full_joint_exp295_v1/final/router
```

The inline Python heredoc needs a space between `python` and `-`:

```bash
python - <<'PY'
print("ok")
PY
```

Common shell typos that will stop later stages are `python-`, `mkdir-p`, `--sleep0.5`, and arguments like `--adapter_dir"$ADAPTER_DIR"` without a space.
