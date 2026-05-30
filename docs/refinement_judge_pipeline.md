# Refinement LLM Judge Pipeline

These scripts evaluate refinement outputs from `scripts/run_gemma_peft_real_inference.py` without overwriting model outputs.

The judge follows the CounselBench/KDD-style toxicity convention: `toxicity=1` means not toxic at all and `toxicity=5` means highly toxic, shaming, dismissive, or harmful. Lower toxicity is better. The five other metrics are 1-5 where higher is better:

- `overall_quality`
- `empathy`
- `specificity`
- `medical_advice`
- `factual_consistency`
- `toxicity`

Set the API key in the environment. Do not hardcode it in commands or scripts.

```bash
export OPENAI_API_KEY=...
```

## Pilot Valid T4

Prepare PEFT `unsafe_t4` judge input, plus unsafe and gold safe reference systems:

```bash
python scripts/prepare_refinement_judge_input.py \
  --input outputs/refinement/gemma4_peft_langqkvo_infermatch_main_a100_valid_t4.jsonl \
  --output outputs/eval_inputs/refinement_pilot_valid_t4_judge_input.jsonl \
  --response_field peft_response \
  --system_name peft_qkvo_t4 \
  --id_prefix pilot_valid_t4 \
  --mode_filter unsafe_t4 \
  --include_unsafe_baseline \
  --include_safe_reference
```

If the input file already contains only `unsafe_t4` rows and no `mode` field, omit `--mode_filter unsafe_t4`.

Run the judge:

```bash
python scripts/run_refinement_llm_judge.py \
  --input outputs/eval_inputs/refinement_pilot_valid_t4_judge_input.jsonl \
  --output outputs/eval/refinement_pilot_valid_t4_judged.jsonl \
  --model gpt-4.1-mini \
  --resume \
  --sleep 0.5
```

Aggregate by system:

```bash
python scripts/aggregate_refinement_judge_scores.py \
  --input outputs/eval/refinement_pilot_valid_t4_judged.jsonl \
  --output_csv outputs/analysis/refinement_pilot_valid_t4_by_system.csv \
  --group_by system
```

Aggregate by system/mode/timestep:

```bash
python scripts/aggregate_refinement_judge_by_timestep.py \
  --input outputs/eval/refinement_pilot_valid_t4_judged.jsonl \
  --output_csv outputs/analysis/refinement_pilot_valid_t4_by_timestep.csv
```

## Pilot Valid Modes

Prepare a multi-mode judge input for denoising ablations:

```bash
python scripts/prepare_refinement_judge_input.py \
  --input outputs/refinement/gemma4_peft_langqkvo_infermatch_main_a100_valid_modes.jsonl \
  --output outputs/eval_inputs/refinement_pilot_valid_modes_judge_input.jsonl \
  --response_field peft_response \
  --system_name peft_qkvo \
  --id_prefix pilot_valid_modes \
  --include_unsafe_baseline \
  --include_safe_reference
```

Then run the same judge and timestep aggregate commands with the `valid_modes` paths.

## Exp295 Test T4

Prepare the current final test `unsafe_t4` judge input:

```bash
python scripts/prepare_refinement_judge_input.py \
  --input outputs/refinement/gemma4_peft_langqkvo_infermatch_exp295_test_t4.jsonl \
  --output outputs/eval_inputs/refinement_exp295_test_t4_judge_input.jsonl \
  --response_field peft_response \
  --system_name peft_qkvo_exp295_t4 \
  --id_prefix exp295_test_t4 \
  --mode_filter unsafe_t4 \
  --include_unsafe_baseline \
  --include_safe_reference
```

Run the judge:

```bash
python scripts/run_refinement_llm_judge.py \
  --input outputs/eval_inputs/refinement_exp295_test_t4_judge_input.jsonl \
  --output outputs/eval/refinement_exp295_test_t4_judged.jsonl \
  --model gpt-4.1-mini \
  --resume \
  --sleep 0.5
```

Aggregate:

```bash
python scripts/aggregate_refinement_judge_scores.py \
  --input outputs/eval/refinement_exp295_test_t4_judged.jsonl \
  --output_csv outputs/analysis/refinement_exp295_test_t4_by_system.csv \
  --group_by system

python scripts/aggregate_refinement_judge_by_timestep.py \
  --input outputs/eval/refinement_exp295_test_t4_judged.jsonl \
  --output_csv outputs/analysis/refinement_exp295_test_t4_by_timestep.csv
```

## Useful Options

Judge a small sample first:

```bash
python scripts/run_refinement_llm_judge.py \
  --input outputs/eval_inputs/refinement_pilot_valid_t4_judge_input.jsonl \
  --output outputs/eval/refinement_pilot_valid_t4_judged_sample.jsonl \
  --model gpt-4.1-mini \
  --max_examples 12
```

Use another response field:

```bash
python scripts/prepare_refinement_judge_input.py \
  --input outputs/refinement/some_method.jsonl \
  --output outputs/eval_inputs/some_method_judge_input.jsonl \
  --response_field cleaned_response \
  --system_name prompt_cleaning
```

Expected judged rows contain the original judge input fields plus:

- `judge_scores`
- `judge_rationale`
- `judge_model`
- `judge_raw`
- `judge_ok`
- `judge_error`
