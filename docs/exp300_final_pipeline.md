# Exp300 Final Denoising Pipeline

This pipeline preserves pilot files and writes the final 300-target experiment to `exp300` paths. The safe target pool is mixed:

- 99 strict safe targets from `izi-ano/CounselBench-Eval`
- 201 judged safe targets from `nbertagnolli/counsel-chat`

CounselBench-Eval has 2000 rows but only 100 unique questions, so exp300 cannot come from CounselBench-Eval alone.

Toxicity convention: CounselBench uses `toxicity_score=1` for not toxic / safe and larger values for more toxic or harmful content. The strict filter uses `toxicity_score <= 1`.

## 1. Prepare CounselBench-Eval Strict Safe Targets

```bash
python scripts/prepare_counselbench_eval_100.py \
  --n_questions 300 \
  --splits all \
  --filter_level strict \
  --shuffle \
  --seed 42 \
  --output data/raw/counselbench_eval_strict_safe_99.jsonl
```

Expected: about `99` rows.

```bash
wc -l data/raw/counselbench_eval_strict_safe_99.jsonl
```

## 2. Prepare CounselChat Candidates

```bash
python scripts/prepare_counselchat_safe_candidates.py \
  --n_candidates 800 \
  --exclude_questions_jsonl data/raw/counselbench_eval_strict_safe_99.jsonl \
  --shuffle \
  --seed 42 \
  --output data/raw/counselchat_safe_candidates.jsonl
```

```bash
wc -l data/raw/counselchat_safe_candidates.jsonl
```

## 3. Judge CounselChat Candidates

This step calls OpenAI and requires `OPENAI_API_KEY`. Do not hardcode the key.

```bash
export OPENAI_API_KEY=...

python scripts/judge_counselchat_safe_candidates.py \
  --input data/raw/counselchat_safe_candidates.jsonl \
  --output data/raw/counselchat_safe_candidates_judged.jsonl \
  --model gpt-4.1-mini \
  --resume
```

For a small paid smoke run:

```bash
python scripts/judge_counselchat_safe_candidates.py \
  --input data/raw/counselchat_safe_candidates.jsonl \
  --output data/raw/counselchat_safe_candidates_judged_sample.jsonl \
  --model gpt-4.1-mini \
  --max_examples 20 \
  --resume
```

```bash
wc -l data/raw/counselchat_safe_candidates_judged.jsonl
```

## 4. Select 201 Judged CounselChat Safe Targets

Try strict first:

```bash
python scripts/select_counselchat_safe_targets.py \
  --input data/raw/counselchat_safe_candidates_judged.jsonl \
  --output data/raw/counselchat_safe_201.jsonl \
  --n_questions 201 \
  --filter_level strict \
  --shuffle \
  --seed 42
```

If strict yields fewer than `201`, rerun with `--filter_level relaxed_1`. If still insufficient, use `relaxed_2`. The script does not silently relax.

```bash
wc -l data/raw/counselchat_safe_201.jsonl
```

## 5. Merge To 300 Safe Targets

```bash
python scripts/merge_safe_targets_exp300.py \
  --counselbench_eval data/raw/counselbench_eval_strict_safe_99.jsonl \
  --counselchat data/raw/counselchat_safe_201.jsonl \
  --output data/raw/exp300_safe_targets.jsonl \
  --expected_total 300
```

```bash
wc -l data/raw/exp300_safe_targets.jsonl

python -c "import json, collections; rows=[json.loads(l) for l in open('data/raw/exp300_safe_targets.jsonl',encoding='utf-8') if l.strip()]; print(collections.Counter(r.get('safe_target_mix_source') for r in rows))"
```

Expected:

- `counselbench_eval_strict = 99`
- `counselchat_judged = 201`
- total `300`

## 6. Generate 6-Dimension Unsafe Corruptions

```bash
python scripts/generate_unsafe_samples.py \
  --input data/raw/exp300_safe_targets.jsonl \
  --output data/synthetic_corruptions/exp300_6dim_v1.jsonl \
  --version exp300_v1 \
  --source mixed_counselbench_eval99_counselchat201
```

```bash
wc -l data/synthetic_corruptions/exp300_6dim_v1.jsonl
```

Expected: `300 x 6 = 1800`.

## 7. Split Base Pairs

```bash
python scripts/split_corruption_dataset.py \
  --input data/synthetic_corruptions/exp300_6dim_v1.jsonl \
  --out_dir data/splits_exp300 \
  --group_by_question \
  --valid_ratio 0.1 \
  --test_ratio 0.2 \
  --train_name train_mdlm.jsonl \
  --valid_name valid_mdlm.jsonl \
  --test_name test.jsonl \
  --seed 42
```

```bash
wc -l \
  data/splits_exp300/train_mdlm.jsonl \
  data/splits_exp300/valid_mdlm.jsonl \
  data/splits_exp300/test.jsonl
```

Expected: about `1260 / 180 / 360`, total `1800`.

## 8. Generate Infermatch Denoising Data

Set these to the existing trained router and span-risk scorer directories:

```bash
export ROUTER_DIR=outputs/models/aspect_router/final
export RISK_SCORER_DIR=outputs/models/span_risk_multilabel/final
```

Train:

```bash
python scripts/prepare_overleaf_infermatch_data.py \
  --input data/splits_exp300/train_mdlm.jsonl \
  --output data/overleaf_infermatch_exp300/train.jsonl \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --T 4 \
  --seed 42
```

Valid:

```bash
python scripts/prepare_overleaf_infermatch_data.py \
  --input data/splits_exp300/valid_mdlm.jsonl \
  --output data/overleaf_infermatch_exp300/valid.jsonl \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --T 4 \
  --seed 42
```

```bash
wc -l \
  data/overleaf_infermatch_exp300/train.jsonl \
  data/overleaf_infermatch_exp300/valid.jsonl

python scripts/sanity_check_denoising_jsonl.py \
  --input data/overleaf_infermatch_exp300/train.jsonl \
  --input data/overleaf_infermatch_exp300/valid.jsonl
```

Expected with 1260/180 base rows:

- train infermatch `6300`
- valid infermatch `900`
- `source`: `empty = base`, `unsafe = base x 4`
- `t`: `0 = base`, `2 = base x 2`, `3 = base`, `4 = base`

## 9. Train Q/K/V/O PEFT Denoiser

```bash
OUT=outputs/models/gemma4_peft_langqkvo_infermatch_exp300_main
LOG=outputs/logs/train_gemma4_peft_langqkvo_infermatch_exp300_main.log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -u scripts/train_gemma_peft_denoiser.py \
  --train_file data/overleaf_infermatch_exp300/train.jsonl \
  --valid_file data/overleaf_infermatch_exp300/valid.jsonl \
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

```bash
tail -f outputs/logs/train_gemma4_peft_langqkvo_infermatch_exp300_main.log
```

## 10. Validation Inference

```bash
python scripts/run_gemma_peft_real_inference.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir outputs/models/gemma4_peft_langqkvo_infermatch_exp300_main/best \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --input data/splits_exp300/valid_mdlm.jsonl \
  --output outputs/refinement/gemma4_peft_langqkvo_infermatch_exp300_valid_modes.jsonl \
  --modes empty,unsafe_t2,unsafe_t3,unsafe_t4 \
  --max_new_tokens 120 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4
```

## 11. Final Test Inference

Run this only after validation settings are fixed.

```bash
python scripts/run_gemma_peft_real_inference.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir outputs/models/gemma4_peft_langqkvo_infermatch_exp300_main/best \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --input data/splits_exp300/test.jsonl \
  --output outputs/refinement/gemma4_peft_langqkvo_infermatch_exp300_test_t4.jsonl \
  --modes unsafe_t4 \
  --max_new_tokens 120 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4
```
