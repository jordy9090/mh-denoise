# Expanded Denoising Experiment Context

## Goal

We are preparing the final experiment for a CIKM short paper on risk-aware response denoising for mental-health QA.

The current pilot experiment used:

- 100 CounselBench-Eval questions
- 6 violation dimensions
- 600 base unsafe-safe pairs
- 5 inference-matched denoising variants per base pair
- 2140 train denoising examples
- 260 valid denoising examples

The pilot result showed that the best current denoiser implementation is:

- Gemma-4-E4B-it
- HuggingFace PEFT LoRA
- target modules restricted to language_model self-attention q/k/v/o
- risk-weighted token-level CE
- inference mode: unsafe_t4
- max_new_tokens around 120

Now we want to expand the base synthetic dataset from 100 questions to 300 questions.

## Existing data pipeline

Base unsafe-safe pair generation:

1. Select high-quality safe responses from CounselBench-Eval.
2. For each selected safe response, generate corrupted unsafe responses along six violation dimensions.
3. Each base row contains:
   - question
   - safe_response
   - unsafe_response
   - violation dimension / vector

Six violation dimensions:

- overall_quality
- empathy
- specificity
- medical_advice
- factual_consistency
- toxicity

Current split files:

- data/splits/train_mdlm.jsonl
- data/splits/valid_mdlm.jsonl
- data/splits/test.jsonl

Current pilot counts:

- train_mdlm: 428
- valid_mdlm: 52
- test: 120
- total: 600

Denoiser data generation:

- data/overleaf_infermatch/train.jsonl
- data/overleaf_infermatch/valid.jsonl

Each base pair is expanded into five denoising variants:

- 1 x empty-source direct refinement, t=0
- 2 x unsafe-source denoising, t=2
- 1 x unsafe-source denoising, t=3
- 1 x unsafe-source denoising, t=4

Current infermatch counts:

- train: 2140 = 428 x 5
- valid: 260 = 52 x 5

## Current model scripts

Training:

- scripts/train_gemma_peft_denoiser.py

Inference:

- scripts/run_gemma_peft_real_inference.py

Current PEFT target modules:

regex:.*language_model\.layers\.[0-9]+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$

Important implementation notes:

- Use HuggingFace PEFT LoRA.
- Do not implement custom LoRA modules.
- Do not replace PEFT with hand-written LoRA.
- Risk-weighted token-level CE is part of the method.
- Standard CE can be added later as an ablation, but the main method should keep risk-weighted CE.

## Current pilot result

The Q/K/V/O PEFT denoiser is the most stable implementation so far.

Validation max_new_tokens=120:

empty     bad 4/52, repeat 0, mask_leak 0, too_short 2
unsafe_t2 bad 8/52, repeat 0, mask_leak 0, too_short 1
unsafe_t3 bad 5/52, repeat 0, mask_leak 0, too_short 1
unsafe_t4 bad 3/52, repeat 0, mask_leak 0, too_short 1

Validation max_new_tokens=160 increased marker/leakage risk, so the current preferred decoding setting is max_new_tokens=120.

## Final expanded experiment target

Expand from:

100 questions x 6 dimensions = 600 base pairs

to:

300 questions x 6 dimensions = 1800 base pairs

Expected split:

- train: roughly 70%
- valid: roughly 10%
- test: roughly 20%

Expected infermatch expansion:

- train base pairs x 5 variants
- valid base pairs x 5 variants

## Required output naming

Use new paths so that pilot outputs are preserved.

Suggested base split paths:

- data/splits_exp300/train_mdlm.jsonl
- data/splits_exp300/valid_mdlm.jsonl
- data/splits_exp300/test.jsonl

Suggested infermatch paths:

- data/overleaf_infermatch_exp300/train.jsonl
- data/overleaf_infermatch_exp300/valid.jsonl

Suggested model path:

- outputs/models/gemma4_peft_langqkvo_infermatch_exp300_main

Suggested inference paths:

- outputs/refinement/gemma4_peft_langqkvo_infermatch_exp300_valid_modes.jsonl
- outputs/refinement/gemma4_peft_langqkvo_infermatch_exp300_test_t4.jsonl

## Evaluation

Use CounselBench-style LLM judge similar to the KDD repo:

- repo reference: jordy9090/selective-mental-health-rag
- evaluate generated mental-health answers
- metrics should match CounselBench dimensions:
  - overall_quality
  - empathy
  - specificity
  - medical_advice
  - factual_consistency
  - toxicity

Main comparison should include:

- unsafe/corrupted response
- prompt-only or base rewrite baseline
- PEFT Q/K/V/O denoiser, unsafe_t4
- gold safe response as reference / upper bound

## Constraints

- Preserve existing pilot outputs.
- Do not overwrite current data/overleaf_infermatch or outputs/models/gemma4_peft_langqkvo_infermatch_main_a100.
- Use new exp300 paths.
- Add sanity-check scripts or commands for row counts, source counts, t counts, and field names.
- The final test set should be used only after settings are fixed on validation.
