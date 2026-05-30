# Iterative Denoising Ablation

This ablation tests the optional multi-step inference path described in the method section.

The main inference path remains a single reverse step such as `unsafe_t2`, `unsafe_t3`, or `unsafe_t4`. The iterative ablation uses the same trained PEFT denoiser but applies it across a decreasing timestep schedule:

```text
4 -> 3 -> 2
```

At each step:

1. The current draft is split into sentence spans.
2. The span-risk scorer estimates aspect-conditioned risk for the current draft.
3. Risky spans are masked according to the next timestep.
4. The denoiser generates a new response.
5. The new response becomes the draft for the next step.

The original unsafe response remains in the prompt for context, and router prediction `g` is computed once from the original question and unsafe response. We do not run a separate generation call at `t=0`; the final prediction after `t=2` is treated as the clean output.

## When To Use This

Use iterative denoising only as an ablation. It is slower than single-step inference and may over-edit or shorten responses. The main result should still be selected on validation using the single-step modes unless iterative validation clearly improves the judge scores.

## Command: Exp295/300 B/C PEFT Test Ablation

Set paths to the trained B/C PEFT adapter:

```bash
cd ~/mh-denoise
conda activate mh-denoise

ROUTER_DIR=outputs/models/router_multilabel_v1/best
if [ ! -d "$ROUTER_DIR" ]; then
  ROUTER_DIR=outputs/models/router_multilabel_v1/final
fi

RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/best
if [ ! -d "$RISK_SCORER_DIR" ]; then
  RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/final
fi

ADAPTER=outputs/models/gemma4_peft_langqkvo_infermatch_exp295_v2_bc/best
```

Run a small smoke test first:

```bash
python scripts/run_gemma_peft_iterative_inference.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir "$ADAPTER" \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output outputs/refinement/gemma4_peft_exp295_v2_bc_valid_iter_4_3_2_smoke.jsonl \
  --iter_steps 4,3,2 \
  --max_examples 5 \
  --max_new_tokens 120 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4
```

Run full validation:

```bash
python scripts/run_gemma_peft_iterative_inference.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir "$ADAPTER" \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output outputs/refinement/gemma4_peft_exp295_v2_bc_valid_iter_4_3_2.jsonl \
  --iter_steps 4,3,2 \
  --max_new_tokens 120 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4
```

If validation looks useful, run test:

```bash
python scripts/run_gemma_peft_iterative_inference.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir "$ADAPTER" \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --input data/splits_exp295/test.jsonl \
  --output outputs/refinement/gemma4_peft_exp295_v2_bc_test_iter_4_3_2.jsonl \
  --iter_steps 4,3,2 \
  --max_new_tokens 120 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4
```

Check:

```bash
wc -l \
  outputs/refinement/gemma4_peft_exp295_v2_bc_valid_iter_4_3_2.jsonl \
  outputs/refinement/gemma4_peft_exp295_v2_bc_test_iter_4_3_2.jsonl
```

Expected with the current exp295 split:

- valid: `174`
- test: `354`

Each output row contains:

- `mode`: `iter_4_3_2`
- `peft_response`: final response after the last step
- `iterative_steps`: per-step drafts, masks, raw generations, and cleaned generations
- `g`, `z_t`, `span_risks`: final-step metadata

## Judge Input

```bash
python scripts/prepare_refinement_judge_input.py \
  --input outputs/refinement/gemma4_peft_exp295_v2_bc_test_iter_4_3_2.jsonl \
  --output outputs/eval_inputs/exp295_iterative_denoising_judge_input.jsonl \
  --response_field peft_response \
  --system_name iterative_denoising_refiner \
  --id_prefix exp295_test
```

Then merge this judge input with the main table judge input only if you want iterative denoising in the ablation table, not in the primary main table.
