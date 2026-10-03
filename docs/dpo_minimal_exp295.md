# DPO-minimal on the exp295 legacy/core dataset

## Scope and naming

This baseline uses the repository's single-turn exp295 legacy/core dataset. Its verified provenance is:

- 99 questions from `CounselBench-Eval`
- 196 questions from `nbertagnolli/counsel-chat`

Do not describe this dataset as MentalChat16K. Results on `data/splits_exp295/test.jsonl` must be labeled **exp295 legacy test**, not an independent full CounselBench evaluation.

Only DPO-minimal is in scope here. K=4 hard negatives, denoiser composition, GPU execution, beta selection, and test-set model selection are not part of this implementation.

## Fixed preference contract

The pair builder renders the existing `sft_plain` prompt using the tokenizer stored with the SFT adapter.

```text
prompt   = existing sft_plain(question, unsafe_response)
chosen   = safe_response
rejected = unsafe_response
```

`target_dimension`, `violation_vector`, and `brief_reason` are retained under `audit_metadata` but are not used to build the prompt or completions. The trainer projects every record to exactly `prompt`, `chosen`, and `rejected` before constructing the TRL dataset.

Train has 1,242 pairs and valid has 174 pairs. The 354-row test split is validated only for question leakage. The builder refuses to emit test preferences.

## Verified SFT initialization

```text
/home/user/mh-denoise/outputs/models/gemma4_peft_sft_plain_exp295/final
```

- Base model: `google/gemma-4-E4B-it`
- Adapter: text-decoder LoRA, r=8, alpha=16, dropout=0.05
- Adapter SHA-256: `f9b0c24bcd0ad05e133352e0c8fa8482b4a6b43766b84fa6c7686e2f2850bc50`

The installed and checked training stack is `torch 2.11.0`, `transformers 5.8.1`, `peft 0.19.1`, `trl 1.4.0`, `datasets 4.8.5`, `accelerate 1.13.0`, and `bitsandbytes 0.49.2`. The named frozen-reference behavior is explicitly guarded for TRL 1.4.0.

## CPU pair build and validation

Run from the repository root:

```bash
cd /home/user/mh-denoise
conda activate mh-denoise

python scripts/build_dpo_minimal_pairs.py \
  --train_file /home/user/mh-denoise/data/splits_exp295/train_mdlm.jsonl \
  --valid_file /home/user/mh-denoise/data/splits_exp295/valid_mdlm.jsonl \
  --test_file /home/user/mh-denoise/data/splits_exp295/test.jsonl \
  --provenance_file /home/user/mh-denoise/data/raw/exp295_safe_targets.jsonl \
  --tokenizer_dir /home/user/mh-denoise/outputs/models/gemma4_peft_sft_plain_exp295/final \
  --output_dir /home/user/mh-denoise/data/dpo_exp295_minimal

python scripts/train_dpo_minimal.py \
  --train_file /home/user/mh-denoise/data/dpo_exp295_minimal/train.jsonl \
  --valid_file /home/user/mh-denoise/data/dpo_exp295_minimal/valid.jsonl \
  --sft_adapter_dir /home/user/mh-denoise/outputs/models/gemma4_peft_sft_plain_exp295/final \
  --output_dir /home/user/mh-denoise/outputs/models/gemma4_dpo_minimal_exp295_validate_only \
  --validate_only
```

The pair artifacts and manifests are under ignored `data/` and are not Git changes.

The exact SFT-adapter tokenizer and its chat template give the following token-length
distribution over train and valid combined (linear percentiles):

| Segment | p50 | p90 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|
| Prompt | 317 | 409 | 475.5 | 645 | 758 |
| Chosen | 224 | 300 | 310 | 326 | 346 |
| Rejected | 72 | 89 | 94 | 104 | 114 |

At 256 completion tokens, chosen is truncated for 450/1,242 train rows and
48/174 valid rows. Both candidate budgets, 512 and 768, reduce chosen and rejected
truncation to zero. The implementation uses 512 because 768 provides no coverage
gain while raising the combined prompt-plus-completion cap from 1,280 to 1,536.
The custom collator preserves prompt and completion budgets separately and retains
EOS when truncating a completion. The machine-readable audit is saved at
`outputs/analysis/dpo_minimal_legacy_exp295_token_lengths.json`.

## GPU smoke test — do not run during CPU validation

This validates the complete train/valid contracts, then precomputes reference log-probabilities and performs one optimizer step on two train pairs. It never reads the test split.

```bash
cd /home/user/mh-denoise
conda activate mh-denoise

CUDA_VISIBLE_DEVICES=0 python scripts/train_dpo_minimal.py \
  --train_file /home/user/mh-denoise/data/dpo_exp295_minimal/train.jsonl \
  --valid_file /home/user/mh-denoise/data/dpo_exp295_minimal/valid.jsonl \
  --sft_adapter_dir /home/user/mh-denoise/outputs/models/gemma4_peft_sft_plain_exp295/final \
  --base_model google/gemma-4-E4B-it \
  --output_dir /home/user/mh-denoise/outputs/models/gemma4_dpo_minimal_exp295_smoke \
  --smoke \
  --max_steps 1 \
  --eval_steps 1 \
  --save_steps 1 \
  --max_prompt_length 768 \
  --max_completion_length 512
```

Before the optimizer step, inspect:

```text
/home/user/mh-denoise/outputs/models/gemma4_dpo_minimal_exp295_smoke/preflight_manifest.json
```

Required values:

```json
{
  "reference_snapshot": {
    "max_abs_diff": 0.0,
    "reference_trainable_tensor_count": 0,
    "unexpected_trainable_tensor_count": 0
  }
}
```

## Full DPO-minimal training — do not run during CPU validation

```bash
cd /home/user/mh-denoise
conda activate mh-denoise

CUDA_VISIBLE_DEVICES=0 python scripts/train_dpo_minimal.py \
  --train_file /home/user/mh-denoise/data/dpo_exp295_minimal/train.jsonl \
  --valid_file /home/user/mh-denoise/data/dpo_exp295_minimal/valid.jsonl \
  --sft_adapter_dir /home/user/mh-denoise/outputs/models/gemma4_peft_sft_plain_exp295/final \
  --base_model google/gemma-4-E4B-it \
  --output_dir /home/user/mh-denoise/outputs/models/gemma4_dpo_minimal_exp295_beta0p1_seed42 \
  --beta 0.1 \
  --max_steps 234 \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --warmup_ratio 0.10 \
  --max_prompt_length 768 \
  --max_completion_length 512 \
  --precompute_ref_batch_size 1 \
  --seed 42
```

The policy is loaded directly from the SFT adapter as trainable `default`. With TRL 1.4.0, `DPOTrainer` creates `ref` as an exact adapter copy. The code then checks tensor structure, zero maximum absolute difference, frozen reference tensors, and absence of trainable parameters outside `default` before calling `train()`.

## Inference after training

The existing q+d-only inference entry point can load the resulting `final` adapter. This command is for later GPU execution and labels the source split as exp295 legacy test:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/build_sft_outputs_for_risk_tuning.py \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir /home/user/mh-denoise/outputs/models/gemma4_dpo_minimal_exp295_beta0p1_seed42/final \
  --sft_prompt_style sft_plain \
  --input /home/user/mh-denoise/data/splits_exp295/test.jsonl \
  --output /home/user/mh-denoise/outputs/refinement/dpo_minimal_exp295_legacy_test_beta0p1_seed42.jsonl \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --repetition_penalty 1.15 \
  --no_repeat_ngram_size 4
```
