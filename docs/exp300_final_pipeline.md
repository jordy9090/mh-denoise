# Exp300 Final Denoising Pipeline

This pipeline preserves the pilot files under `data/splits`, `data/overleaf_infermatch`, and existing `outputs/` paths. The expanded experiment writes to `exp300` paths only.

## 0. Safe Target Source Decision

`izi-ano/CounselBench-Eval` has 2000 rows but only 100 unique questions, so it remains a pilot/evaluation reference and is not used for exp300 safe targets.

For exp300, prepare 300 unique safe targets from `nbertagnolli/counsel-chat`, following the CounselBench preprocessing spirit: one answer per unique question, excluded sensitive topics, answer length cap, and highest-upvoted answer selection.

## 1. Prepare 300 Safe CounselChat Targets

```bash
python scripts/prepare_counselchat_safe_targets.py \
  --n_questions 300 \
  --shuffle \
  --seed 42 \
  --output data/raw/counselchat_300_safe_targets.jsonl
```

Sanity check safe target count:

```bash
wc -l data/raw/counselchat_300_safe_targets.jsonl
```

Expected target: `300`.

## 2. Generate 6-Dimension Unsafe Corruptions

This produces 300 x 6 = 1800 base unsafe-safe pairs.

```bash
python scripts/generate_unsafe_samples.py \
  --input data/raw/counselchat_300_safe_targets.jsonl \
  --output data/synthetic_corruptions/counselchat_300_6dim_v1.jsonl \
  --version exp300_v1 \
  --source counselchat_exp300
```

Sanity check synthetic corruption count:

```bash
wc -l data/synthetic_corruptions/counselchat_300_6dim_v1.jsonl
```

Expected target: `1800`.

## 3. Split Base Pairs

The split is question-grouped to reduce same-question leakage across train/valid/test. Expected counts for 300 questions and six dimensions are about train 1260, valid 180, test 360.

```bash
python scripts/split_corruption_dataset.py \
  --input data/synthetic_corruptions/counselchat_300_6dim_v1.jsonl \
  --out_dir data/splits_exp300 \
  --group_by_question \
  --valid_ratio 0.1 \
  --test_ratio 0.2 \
  --train_name train_mdlm.jsonl \
  --valid_name valid_mdlm.jsonl \
  --test_name test.jsonl \
  --seed 42
```

Sanity check split counts:

```bash
wc -l \
  data/splits_exp300/train_mdlm.jsonl \
  data/splits_exp300/valid_mdlm.jsonl \
  data/splits_exp300/test.jsonl
```

Expected target: about `1260 / 180 / 360`, total `1800`.

## 4. Generate Inference-Matched Denoising Data

Set these to the existing trained router and span-risk scorer directories before running:

```bash
export ROUTER_DIR=outputs/models/aspect_router/final
export RISK_SCORER_DIR=outputs/models/span_risk_multilabel/final
```

Train split:

```bash
python scripts/prepare_overleaf_infermatch_data.py \
  --input data/splits_exp300/train_mdlm.jsonl \
  --output data/overleaf_infermatch_exp300/train.jsonl \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --T 4 \
  --seed 42
```

Valid split:

```bash
python scripts/prepare_overleaf_infermatch_data.py \
  --input data/splits_exp300/valid_mdlm.jsonl \
  --output data/overleaf_infermatch_exp300/valid.jsonl \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --T 4 \
  --seed 42
```

Sanity check infermatch counts and source/timestep distributions:

```bash
wc -l \
  data/overleaf_infermatch_exp300/train.jsonl \
  data/overleaf_infermatch_exp300/valid.jsonl

python scripts/sanity_check_denoising_jsonl.py \
  --input data/overleaf_infermatch_exp300/train.jsonl \
  --input data/overleaf_infermatch_exp300/valid.jsonl
```

Each base row expands to five variants: one `empty` at `t=0`, two `unsafe` at `t=2`, one `unsafe` at `t=3`, and one `unsafe` at `t=4`.

Expected target:

- infermatch train = train base x 5
- infermatch valid = valid base x 5
- `source`: `empty = base`, `unsafe = base x 4`
- `t`: `0 = base`, `2 = base x 2`, `3 = base`, `4 = base`

## 5. Combined Sanity Check

```bash
python scripts/sanity_check_denoising_jsonl.py \
  data/splits_exp300/train_mdlm.jsonl \
  data/splits_exp300/valid_mdlm.jsonl \
  data/splits_exp300/test.jsonl \
  data/overleaf_infermatch_exp300/train.jsonl \
  data/overleaf_infermatch_exp300/valid.jsonl
```

## 6. Train Q/K/V/O PEFT Denoiser

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

Follow the log:

```bash
tail -f outputs/logs/train_gemma4_peft_langqkvo_infermatch_exp300_main.log
```

The main training script uses HuggingFace PEFT LoRA and risk-weighted token-level CE through `target_weight_spans`.

## 7. Validation Inference

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

## 8. Extract Validation Unsafe T4

```bash
python -c "import json; inp='outputs/refinement/gemma4_peft_langqkvo_infermatch_exp300_valid_modes.jsonl'; out='outputs/refinement/gemma4_peft_langqkvo_infermatch_exp300_valid_unsafe_t4.jsonl'; rows=[json.loads(l) for l in open(inp,encoding='utf-8') if l.strip()]; rows=[r for r in rows if r.get('mode')=='unsafe_t4']; f=open(out,'w',encoding='utf-8'); [f.write(json.dumps(r,ensure_ascii=False)+'\n') for r in rows]; f.close(); print('wrote',len(rows),out)"
```

## 9. Final Test Inference

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
