# Exp295 Final Denoising Pipeline

This file is kept at `docs/exp300_final_pipeline.md` for continuity with the earlier plan, but the actual A100 run now uses `exp295` paths.

The final safe-target pool is:

- 99 strict safe targets from `izi-ano/CounselBench-Eval`
- 196 judged CounselChat safe targets after removing 5 duplicates with CounselBench-Eval
- 295 total safe targets

The current training run uses the v1 infermatch data generator, `scripts/prepare_overleaf_infermatch_data.py`, with `source=empty` and `source=unsafe`. The v2 full-mixture generator with `bridge` and `safe` sources exists separately as `scripts/prepare_overleaf_infermatch_data_v2.py`, but it is not the data used by the current `exp295` checkpoint.

Toxicity convention: CounselBench uses `toxicity_score=1` for not toxic / safe and larger values for more toxic or harmful content. The strict filter uses `toxicity_score <= 1`.

## 0. Setup

```bash
cd ~/mh-denoise
conda activate mh-denoise-a100

mkdir -p \
  data/raw \
  data/synthetic_corruptions \
  data/splits_exp295 \
  data/overleaf_infermatch_exp295 \
  outputs/logs \
  outputs/refinement \
  outputs/models
```

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

Some rows in the generated CounselBench-Eval file can have an empty `question` field. The merge step requires `question`, so use the fixed file below for the final merge.

```bash
python - <<'PY'
import json
from pathlib import Path
from datasets import load_dataset

inp = Path("data/raw/counselbench_eval_strict_safe_99.jsonl")
out = Path("data/raw/counselbench_eval_strict_safe_99.fixed.jsonl")

ds = load_dataset("izi-ano/CounselBench-Eval")
qid_to_question = {}

def combine(title, text):
    title = title.strip() if isinstance(title, str) else ""
    text = text.strip() if isinstance(text, str) else ""
    if title and text and title != text:
        return title + "\n\n" + text
    return title or text

for split, d in ds.items():
    for ex in d:
        qid = ex.get("questionID") or ex.get("question_id")
        if not qid:
            continue
        q = combine(ex.get("questionTitle"), ex.get("questionText"))
        if q and qid not in qid_to_question:
            qid_to_question[qid] = q

n = 0
bad = []
with inp.open(encoding="utf-8") as f, out.open("w", encoding="utf-8") as w:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        q = r.get("question")
        if not isinstance(q, str) or not q.strip():
            qid = r.get("question_id") or r.get("questionID")
            q = qid_to_question.get(qid, "")
        if not q:
            bad.append((r.get("id"), r.get("question_id") or r.get("questionID")))
            continue
        r["question"] = q.strip()
        if not r.get("question_id"):
            r["question_id"] = r.get("questionID") or r.get("id")
        if not r.get("safe_response"):
            r["safe_response"] = r.get("response") or r.get("answerText") or ""
        if not r.get("source"):
            r["source"] = "CounselBench-Eval"
        if not r.get("safe_target_source"):
            r["safe_target_source"] = "izi-ano/CounselBench-Eval"
        w.write(json.dumps(r, ensure_ascii=False) + "\n")
        n += 1

print("wrote", n, "to", out)
print("bad", len(bad))
if bad:
    print("BAD:", bad[:20])
PY

wc -l data/raw/counselbench_eval_strict_safe_99.fixed.jsonl
```

Expected fixed count: `99`.

## 2. Prepare And Judge CounselChat Candidates

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

This step calls OpenAI and requires `OPENAI_API_KEY`. Do not hardcode the key.

```bash
export OPENAI_API_KEY=...

python scripts/judge_counselchat_safe_candidates.py \
  --input data/raw/counselchat_safe_candidates.jsonl \
  --output data/raw/counselchat_safe_candidates_judged.jsonl \
  --model gpt-4.1-mini \
  --resume
```

## 3. Select Judged CounselChat Safe Targets

Strict and `relaxed_1` did not provide enough final non-duplicate CounselChat examples. The current run used `relaxed_2`, while keeping toxicity, medical-advice, and factual-consistency constraints strict.

```bash
python scripts/select_counselchat_safe_targets.py \
  --input data/raw/counselchat_safe_candidates_judged.jsonl \
  --output data/raw/counselchat_safe_201.jsonl \
  --n_questions 201 \
  --filter_level relaxed_2 \
  --shuffle \
  --seed 42

wc -l data/raw/counselchat_safe_201.jsonl
```

The selected file may contain questions that duplicate CounselBench-Eval questions. The merge step removes them.

## 4. Merge To 295 Safe Targets

Use the fixed CounselBench-Eval file.

```bash
python scripts/merge_safe_targets_exp300.py \
  --counselbench_eval data/raw/counselbench_eval_strict_safe_99.fixed.jsonl \
  --counselchat data/raw/counselchat_safe_201.jsonl \
  --output data/raw/exp295_safe_targets.jsonl \
  --expected_total 295

wc -l data/raw/exp295_safe_targets.jsonl

python - <<'PY'
import json, collections
rows = [json.loads(l) for l in open("data/raw/exp295_safe_targets.jsonl", encoding="utf-8") if l.strip()]
print("n =", len(rows))
print(collections.Counter(r.get("safe_target_mix_source") for r in rows))
print("first keys:", rows[0].keys())
PY
```

Observed output:

- `counselbench_eval_strict = 99`
- `counselchat_judged = 196`
- duplicates removed: `5`
- total: `295`

## 5. Generate 6-Dimension Unsafe Corruptions

This is a long GPU generation job. The completed A100 run took about 3 hours 45 minutes.

```bash
rm -f data/synthetic_corruptions/exp295_6dim_v1.jsonl

nohup python -u scripts/generate_unsafe_samples.py \
  --input data/raw/exp295_safe_targets.jsonl \
  --output data/synthetic_corruptions/exp295_6dim_v1.jsonl \
  --version exp295_v1 \
  --source mixed_counselbench_eval99_counselchat196 \
  > outputs/logs/generate_unsafe_exp295_v1.log 2>&1 &
```

Monitor:

```bash
tail -f outputs/logs/generate_unsafe_exp295_v1.log
```

After completion:

```bash
wc -l data/synthetic_corruptions/exp295_6dim_v1.jsonl
```

Observed: `1770`, from `295 x 6`.

## 6. Split Base Pairs

```bash
rm -rf data/splits_exp295
mkdir -p data/splits_exp295

python scripts/split_corruption_dataset.py \
  --input data/synthetic_corruptions/exp295_6dim_v1.jsonl \
  --out_dir data/splits_exp295 \
  --group_by_question \
  --valid_ratio 0.1 \
  --test_ratio 0.2 \
  --train_name train_mdlm.jsonl \
  --valid_name valid_mdlm.jsonl \
  --test_name test.jsonl \
  --seed 42

wc -l \
  data/splits_exp295/train_mdlm.jsonl \
  data/splits_exp295/valid_mdlm.jsonl \
  data/splits_exp295/test.jsonl
```

Observed:

- questions: `295`
- train questions: `207`, rows: `1242`
- valid questions: `29`, rows: `174`
- test questions: `59`, rows: `354`
- total rows: `1770`

## 7. Set Router And Risk Scorer Paths

The current A100 run used the `*_v1/best` checkpoints, with fallback to `final`.

```bash
ROUTER_DIR=outputs/models/router_multilabel_v1/best
if [ ! -d "$ROUTER_DIR" ]; then
  ROUTER_DIR=outputs/models/router_multilabel_v1/final
fi

RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/best
if [ ! -d "$RISK_SCORER_DIR" ]; then
  RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/final
fi

echo "ROUTER_DIR=$ROUTER_DIR"
echo "RISK_SCORER_DIR=$RISK_SCORER_DIR"

ls -lh "$ROUTER_DIR"
ls -lh "$RISK_SCORER_DIR"
```

## 8. Generate V1 Infermatch Denoising Data

This current run uses the v1 infermatch mixture: one `empty` row and four `unsafe` rows per base pair.

```bash
rm -rf data/overleaf_infermatch_exp295
mkdir -p data/overleaf_infermatch_exp295

python scripts/prepare_overleaf_infermatch_data.py \
  --input data/splits_exp295/train_mdlm.jsonl \
  --output data/overleaf_infermatch_exp295/train.jsonl \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --T 4 \
  --seed 42

python scripts/prepare_overleaf_infermatch_data.py \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output data/overleaf_infermatch_exp295/valid.jsonl \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --T 4 \
  --seed 43
```

Sanity check:

```bash
wc -l \
  data/overleaf_infermatch_exp295/train.jsonl \
  data/overleaf_infermatch_exp295/valid.jsonl

python scripts/sanity_check_denoising_jsonl.py \
  --input data/overleaf_infermatch_exp295/train.jsonl \
  --input data/overleaf_infermatch_exp295/valid.jsonl
```

Observed:

- train: `6210` rows from `1242` base pairs
- valid: `870` rows from `174` base pairs
- train source: `empty=1242`, `unsafe=4968`
- valid source: `empty=174`, `unsafe=696`
- train timestep: `t0=1242`, `t2=2484`, `t3=1242`, `t4=1242`
- valid timestep: `t0=174`, `t2=348`, `t3=174`, `t4=174`
- empty `z_t` appears only for `source=empty`

## 9. Train Q/K/V/O PEFT Denoiser

This is the training command currently running on the A100.

```bash
OUT=outputs/models/gemma4_peft_langqkvo_infermatch_exp295_main
LOG=outputs/logs/train_gemma4_peft_langqkvo_infermatch_exp295_main.log

rm -rf "$OUT"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python -u scripts/train_gemma_peft_denoiser.py \
  --train_file data/overleaf_infermatch_exp295/train.jsonl \
  --valid_file data/overleaf_infermatch_exp295/valid.jsonl \
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

Monitor:

```bash
tail -f outputs/logs/train_gemma4_peft_langqkvo_infermatch_exp295_main.log
```

Observed at launch:

- trainable params: `4,538,368`
- all params: `7,945,639,200`
- trainable percent: `0.0571`
- first logged eval: step `25`, loss `4.6441`

## 10. Validation Inference

Run this after the training checkpoint exists.

```bash
python scripts/run_gemma_peft_real_inference.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir outputs/models/gemma4_peft_langqkvo_infermatch_exp295_main/best \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output outputs/refinement/gemma4_peft_langqkvo_infermatch_exp295_valid_modes.jsonl \
  --modes empty,unsafe_t2,unsafe_t3,unsafe_t4 \
  --max_new_tokens 120 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4
```

Optional single-mode validation output:

```bash
python scripts/run_gemma_peft_real_inference.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir outputs/models/gemma4_peft_langqkvo_infermatch_exp295_main/best \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output outputs/refinement/gemma4_peft_langqkvo_infermatch_exp295_valid_t4.jsonl \
  --modes unsafe_t4 \
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
  --adapter_dir outputs/models/gemma4_peft_langqkvo_infermatch_exp295_main/best \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --input data/splits_exp295/test.jsonl \
  --output outputs/refinement/gemma4_peft_langqkvo_infermatch_exp295_test_t4.jsonl \
  --modes unsafe_t4 \
  --max_new_tokens 120 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4
```
