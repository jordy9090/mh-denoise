# Exp295 Fair Baselines

This note gives copy-paste commands for the fair exp295 table baselines.

Definitions:

- `Prompt Rewrite`: zero-shot `google/gemma-4-E4B-it` rewrite from question and unsafe response. No training, no gold target dimension.
- `SFT Refiner`: PEFT LoRA trained with standard autoregressive SFT from question and unsafe response to safe response. It uses `--prompt_style sft_plain`, so it does not receive gold `target_dimension`, router scores `g`, corrupted draft `z_t`, or timestep `t`.
- `Risk-aware denoising refiner`: the denoising method output already selected on validation.

## 0. Setup

```bash
cd ~/mh-denoise
conda activate mh-denoise

mkdir -p outputs/logs outputs/models outputs/refinement outputs/eval_inputs outputs/eval outputs/analysis
```

## 1. Prompt Rewrite Baseline

```bash
nohup python -u scripts/run_prompt_cleaning_baseline.py \
  --input data/splits_exp295/test.jsonl \
  --output outputs/refinement/prompt_rewrite_gemma4_exp295_test.jsonl \
  --model google/gemma-4-E4B-it \
  --max_source_len 512 \
  --max_new_tokens 120 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4 \
  > outputs/logs/prompt_rewrite_gemma4_exp295_test.log 2>&1 &
```

Monitor:

```bash
tail -f outputs/logs/prompt_rewrite_gemma4_exp295_test.log
```

Check:

```bash
wc -l data/splits_exp295/test.jsonl outputs/refinement/prompt_rewrite_gemma4_exp295_test.jsonl
```

Expected: `354` output rows for the current exp295 split.

Sanity check the generated responses:

```bash
python - <<'PY'
import json
path = "outputs/refinement/prompt_rewrite_gemma4_exp295_test.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
print("rows:", len(rows))
print("empty cleaned_response:", sum(1 for r in rows if not str(r.get("cleaned_response", "")).strip()))
print("parse errors:", sum(1 for r in rows if r.get("prompt_cleaning_parse_error")))
PY
```

If any rows have an empty `cleaned_response`, do not judge the raw file. Repair the existing file and create an empty-only subset:

```bash
python scripts/repair_prompt_rewrite_outputs.py \
  --input outputs/refinement/prompt_rewrite_gemma4_exp295_test.jsonl \
  --output outputs/refinement/prompt_rewrite_gemma4_exp295_test_clean.jsonl \
  --empty_output outputs/refinement/prompt_rewrite_empty_input.jsonl
```

Regenerate only the empty subset with the robust parser:

```bash
nohup python -u scripts/run_prompt_cleaning_baseline.py \
  --input outputs/refinement/prompt_rewrite_empty_input.jsonl \
  --output outputs/refinement/prompt_rewrite_empty_regen.jsonl \
  --model google/gemma-4-E4B-it \
  --max_source_len 512 \
  --max_new_tokens 120 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4 \
  > outputs/logs/prompt_rewrite_empty_regen.log 2>&1 &
```

Merge the regenerated rows back:

```bash
python scripts/repair_prompt_rewrite_outputs.py \
  --input outputs/refinement/prompt_rewrite_gemma4_exp295_test_clean.jsonl \
  --merge_regen outputs/refinement/prompt_rewrite_empty_regen.jsonl \
  --output outputs/refinement/prompt_rewrite_gemma4_exp295_test_clean_rescued.jsonl

python - <<'PY'
import json
path = "outputs/refinement/prompt_rewrite_gemma4_exp295_test_clean_rescued.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
print("rows:", len(rows))
print("empty cleaned_response:", sum(1 for r in rows if not str(r.get("cleaned_response", "")).strip()))
print("rescued:", sum(1 for r in rows if r.get("prompt_cleaning_rescued")))
PY
```

Use the rescued file for judge input if rescue was needed.

## 2. Fair SFT Refiner Training

This baseline intentionally uses `--prompt_style sft_plain`.

```bash
OUT=outputs/models/gemma4_peft_sft_plain_exp295
LOG=outputs/logs/train_gemma4_peft_sft_plain_exp295.log

rm -rf "$OUT"

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

Monitor:

```bash
tail -f outputs/logs/train_gemma4_peft_sft_plain_exp295.log
```

The trained adapter is saved under:

```text
outputs/models/gemma4_peft_sft_plain_exp295/final
```

## 3. Fair SFT Refiner Test Inference

```bash
python scripts/run_professor_peft_refiner.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir outputs/models/gemma4_peft_sft_plain_exp295/final \
  --input data/splits_exp295/test.jsonl \
  --output outputs/refinement/gemma4_peft_sft_plain_exp295_test.jsonl \
  --prompt_style sft_plain \
  --max_source_len 512 \
  --max_new_tokens 120 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4
```

Check:

```bash
wc -l outputs/refinement/gemma4_peft_sft_plain_exp295_test.jsonl
```

Expected: `354`.

## 4. Prepare Judge Inputs

Set this to the already selected risk-aware test output. If the selected validation mode was `unsafe_t3`, keep this path and mode.

```bash
OURS_OUT=outputs/refinement/gemma4_peft_langqkvo_infermatch_exp295_test_t3.jsonl
OURS_MODE=unsafe_t3
```

Prompt Rewrite:

```bash
python scripts/prepare_refinement_judge_input.py \
  --input outputs/refinement/prompt_rewrite_gemma4_exp295_test.jsonl \
  --output outputs/eval_inputs/exp295_prompt_rewrite_judge_input.jsonl \
  --response_field cleaned_response \
  --system_name prompt_rewrite \
  --id_prefix exp295_test \
  --include_unsafe_baseline \
  --include_safe_reference
```

If empty rows were rescued, use this input instead:

```bash
python scripts/prepare_refinement_judge_input.py \
  --input outputs/refinement/prompt_rewrite_gemma4_exp295_test_clean_rescued.jsonl \
  --output outputs/eval_inputs/exp295_prompt_rewrite_judge_input.jsonl \
  --response_field cleaned_response \
  --system_name prompt_rewrite \
  --id_prefix exp295_test \
  --include_unsafe_baseline \
  --include_safe_reference
```

SFT Refiner:

```bash
python scripts/prepare_refinement_judge_input.py \
  --input outputs/refinement/gemma4_peft_sft_plain_exp295_test.jsonl \
  --output outputs/eval_inputs/exp295_sft_plain_judge_input.jsonl \
  --response_field professor_peft_response \
  --system_name sft_refiner \
  --id_prefix exp295_test
```

Risk-aware Denoising Refiner:

```bash
python scripts/prepare_refinement_judge_input.py \
  --input "$OURS_OUT" \
  --output outputs/eval_inputs/exp295_risk_aware_judge_input.jsonl \
  --response_field peft_response \
  --system_name risk_aware_denoising_refiner \
  --id_prefix exp295_test \
  --mode_filter "$OURS_MODE"
```

Merge judge inputs:

```bash
cat \
  outputs/eval_inputs/exp295_prompt_rewrite_judge_input.jsonl \
  outputs/eval_inputs/exp295_sft_plain_judge_input.jsonl \
  outputs/eval_inputs/exp295_risk_aware_judge_input.jsonl \
  > outputs/eval_inputs/exp295_main_table_judge_input.jsonl

wc -l outputs/eval_inputs/exp295_main_table_judge_input.jsonl
```

Expected rows if all systems are present:

- prompt rewrite: `354`
- unsafe baseline: `354`
- safe reference: `354`
- SFT refiner: `354`
- risk-aware denoising refiner: `354`
- total: `1770`

## 5. Run CounselBench-Style Judge

This calls OpenAI and requires `OPENAI_API_KEY`.

```bash
export OPENAI_API_KEY=...

python scripts/run_refinement_llm_judge.py \
  --input outputs/eval_inputs/exp295_main_table_judge_input.jsonl \
  --output outputs/eval/refinement_exp295_main_table_judged.jsonl \
  --model gpt-4.1-mini \
  --resume \
  --sleep 0.5
```

Aggregate:

```bash
python scripts/aggregate_refinement_judge_scores.py \
  --input outputs/eval/refinement_exp295_main_table_judged.jsonl \
  --output_csv outputs/analysis/refinement_exp295_main_table_by_system.csv \
  --group_by system
```

For the paper table:

- `overall_quality`, `empathy`, `specificity`, and `factual_consistency`: use metric means directly.
- `toxicity`: use the mean directly; lower is better.
- `Med. Adv. Viol.`: report `6 - medical_advice_mean`; lower is better.
