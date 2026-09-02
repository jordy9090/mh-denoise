# DPO Baselines for exp295 Single-Turn Response Refinement

This directory implements a controlled Direct Preference Optimization baseline for the existing exp295 mental-health response-refinement experiment. All files are isolated under `baselines/dpo_exp295/`; the current `scripts/`, `docs/`, `data/`, and `outputs/` trees are left unchanged.

## 1. Repository audit and experimental boundary

The committed exp295 pipeline uses:

- backbone: `google/gemma-4-E4B-it`
- grouped split: 1,242 train / 174 validation / 354 test rows
- six automatically generated flaw dimensions per question
- QLoRA with an NF4 4-bit backbone
- text-language-model attention projections as the established LoRA target family
- CounselBench-style response evaluation over overall quality, empathy, specificity, medical-boundary control, factual consistency, and toxicity

Two details require care.

1. The exp295 files documented in this repository were built from 99 strict CounselBench-Eval safe targets and 196 judged CounselChat targets, followed by six automated corruptions per question. The committed repository does not establish that these 1,770 preference rows are a direct MentalChat16K split. Paper text should describe the actual exp295 provenance unless another data artifact verifies a different source.
2. Existing SFT inference code can expose `target_dimension` to the model. The DPO condition fixed for this experiment is only `question + potentially unsafe initial draft`. The SFT comparison must therefore be regenerated with `generate.py`, which uses the same q+d prompt and decoding settings.

Generated datasets and checkpoints remain outside Git because the repository already ignores `data/` and `outputs/`.

## 2. Core experimental question

Given a mental-health question `q` and a potentially unsafe initial draft `d`, can preference optimization produce a safe final response under the same backbone, SFT initialization, LoRA capacity, split, decoding, and optimizer-step budget used by the controlled SFT baseline?

Each preference example has:

```text
x        = q + d
chosen   = safe_response
rejected = flawed/unsafe response
```

The trainable policy and reference policy both start from the same existing SFT refiner checkpoint. The reference adapter is frozen before the first DPO update.

## 3. Experimental conditions

### 3.1 SFT

Load `outputs/models/gemma4_peft_sft_plain_exp295/final` and generate directly from q+d. This is the prompt-controlled SFT baseline.

### 3.2 DPO

Load the same SFT LoRA as the trainable `default` adapter. TRL creates a frozen `ref` adapter from an exact copy of the initial SFT adapter. The implementation verifies:

- every trainable policy adapter tensor has a corresponding reference tensor;
- policy and reference adapter tensors are exactly equal before training;
- policy adapter parameters are trainable;
- reference adapter parameters are frozen.

At inference, the DPO policy directly replaces the SFT refiner output.

### 3.3 DPO + selective denoiser

Generate the DPO response first. Feed that response to the existing residual-risk router, span-risk scorer, and denoiser. Apply the same invocation rule, timestep, acceptance checks, and thresholds used for Proposed.

`bridge_selective_denoiser.py` preserves the original exp295 draft and maps the DPO output into the existing denoiser input schema. It also refuses to call an unfiltered denoiser candidate a selective result.

The current committed main branch contains candidate generation code, yet a complete invocation-and-acceptance finalizer is not present. A valid DPO+denoiser result therefore requires gate decisions produced by the exact Proposed gate implementation available on the experiment server. Reusing that policy is essential; tuning a separate gate for DPO would confound the comparison.

### 3.4 Proposed

Existing SFT-first selective denoising pipeline. Regenerate or recover its test outputs and efficiency logs under the same decoding and gate settings before building the final table.

## 4. Preference-pair variants

### Minimal DPO

One preference pair per exp295 row:

```text
safe_response > the row's own unsafe_response
```

Counts:

| Split | Source rows | Preference pairs |
|---|---:|---:|
| Train | 1,242 | 1,242 |
| Validation | 174 | 174 |
| Test audit file | 354 | 354 |

### K=4 automated hard-pair DPO

For each q+d source condition, form a four-negative pool from the six automatically generated flaw variants belonging to the same question group. The source draft is always retained. Remaining negatives are selected using:

1. closeness to the safe response;
2. flaw-dimension diversity;
3. deterministic tie-breaking by seed.

The hardness score is:

```text
0.60 * normalized sequence similarity
+ 0.30 * token Jaccard similarity
+ 0.10 * length ratio
```

Each selected negative produces a separate preference pair with the same chosen response. Counts:

| Split | Source rows | Pairs per source | Preference pairs |
|---|---:|---:|---:|
| Train | 1,242 | 4 | 4,968 |
| Validation | 174 | 4 | 696 |
| Test audit file | 354 | 4 | 1,416 |

This condition uses weak automated corruption labels and no counselor annotation. It tests whether exposure to multiple near-safe flawed alternatives changes optimization. It does not establish that all four negatives contain the intended flaw. Candidate-quality auditing remains necessary, and conclusions should be reported separately from Minimal DPO.

## 5. Controlled optimization

The configuration locks the following settings across Minimal and K=4:

| Item | Value |
|---|---|
| Backbone | `google/gemma-4-E4B-it` |
| Initialization | existing SFT refiner LoRA |
| LoRA architecture | inherited from SFT `adapter_config.json` |
| Quantization | NF4 4-bit, double quantization, BF16 compute |
| Effective batch | 16 (`batch=1`, `grad_accum=16`) |
| Optimizer updates | 234 |
| Learning rate | `5e-6` |
| Scheduler | cosine, 6% warmup |
| Weight decay | `0.01` |
| Gradient clipping | `0.3` |
| Sequence budget | 1,152 tokens |
| DPO loss | sigmoid |
| Reference log probabilities | precomputed |
| Dropout during DPO | disabled by TRL |

`234` updates correspond to three passes over the 1,242-pair Minimal condition at effective batch 16. K=4 receives the same number of optimizer updates, so extra pairs change negative diversity without increasing the update budget.

The code performs a token-length audit before loading model weights. Default behavior stops the run when any sequence exceeds the configured budget because right-side completion truncation would corrupt the DPO objective.

## 6. Beta validation and seeds

### Sweep

Run each pair variant at seed 42 with:

```text
beta in {0.03, 0.10, 0.30, 0.50}
```

The validation selector first requires:

- all 174 validation outputs are judged;
- medical-boundary score is no lower than prompt-controlled SFT;
- toxicity is no higher than prompt-controlled SFT;
- six-axis quality/safety average drops by no more than 0.05.

Among feasible betas, choose the highest six-axis quality/safety average. Ties are resolved by medical-boundary score, toxicity, then the smaller beta. When no beta satisfies the constraints, `select_beta.py` returns `no_feasible_beta`; it does not silently pick the least bad run.

### Final seeds

Retrain the selected beta using:

```text
42, 43, 44
```

Report mean and standard deviation over the three training seeds. Because the 354 rows contain six variants from 59 question groups, significance analysis should resample at the question-group level. Row-level independent tests would overstate the effective sample size.

## 7. Commands

Install a CUDA-compatible PyTorch build for the experiment host, then:

```bash
pip install -r baselines/dpo_exp295/requirements.txt
```

Prepare and validate preference files:

```bash
bash baselines/dpo_exp295/run_exp295.sh prepare
```

Run beta validation. The existing judge requires `OPENAI_API_KEY` from the environment:

```bash
export OPENAI_API_KEY=...
bash baselines/dpo_exp295/run_exp295.sh sweep
```

Run selected betas for seeds 42/43/44 and generate test responses:

```bash
bash baselines/dpo_exp295/run_exp295.sh final
```

Generate DPO denoiser candidates after setting the existing checkpoint paths when auto-detection does not find them:

```bash
export ROUTER_DIR=outputs/models/router_multilabel_v1/best
export RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/best
export DENOISER_ADAPTER_DIR=outputs/models/gemma4_selective_sft_plain_risk_tuned_exp295_len256_lr5e6_lambda03_clean/best
export DPO_VARIANT=minimal
export DPO_SEED=42
bash baselines/dpo_exp295/run_exp295.sh denoiser
```

Merge with decisions from the shared Proposed gate:

```bash
python baselines/dpo_exp295/bridge_selective_denoiser.py merge \
  --upstream_input outputs/refinement/dpo_exp295_test/dpo_minimal_selected_s42.jsonl \
  --denoiser_input outputs/refinement/dpo_exp295_test/minimal_s42_denoiser_candidates.jsonl \
  --decisions outputs/refinement/shared_gate_decisions_minimal_s42.jsonl \
  --decision_field denoiser_accepted \
  --output outputs/refinement/dpo_exp295_test/dpo_minimal_plus_selective_denoiser_s42.jsonl \
  --output_system dpo_minimal_plus_selective_denoiser_s42
```

Build a comparison catalog template:

```bash
python baselines/dpo_exp295/build_comparison_table.py \
  --write_template outputs/analysis/dpo_exp295_comparison_catalog.json
```

After filling the exact result paths and system names:

```bash
python baselines/dpo_exp295/build_comparison_table.py \
  --catalog outputs/analysis/dpo_exp295_comparison_catalog.json \
  --output_csv outputs/analysis/dpo_exp295_main_table.csv \
  --output_md outputs/analysis/dpo_exp295_main_table.md \
  --group main
```

## 8. Main comparison table

The main paper table should keep pair-construction variants outside the system-name comparison:

| System | Overall ↑ | Empathy ↑ | Specificity ↑ | Medical boundary ↑ | Factual consistency ↑ | Toxicity ↓ | Invoke % | Accept % | Latency | Peak VRAM | GPU hours |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | TBD | TBD | TBD | TBD | TBD | TBD | — | — | TBD | TBD | TBD |
| DPO | TBD | TBD | TBD | TBD | TBD | TBD | — | — | TBD | TBD | TBD |
| DPO + selective denoiser | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Proposed: SFT + selective denoiser | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Use a separate ablation table for Minimal DPO and K=4 automated hard-pair DPO. Mixing both rows into the headline table would blur the central comparison between training objectives and inference pipelines.

## 9. GPU-memory expectations

These are engineering estimates for planning, not measured results.

| Run | Expected peak GPU memory | Notes |
|---|---:|---|
| QLoRA DPO, one 4-bit backbone, copied LoRA reference, precomputed ref log-probs | roughly 24–32 GiB | `max_length=1152`, batch 1, gradient checkpointing |
| Same setup without precomputation | roughly 28–38 GiB | reference forward remains in the training loop |
| Two separately loaded policy/reference backbones | often above 45 GiB | intentionally avoided |
| Greedy 4-bit generation | roughly 10–18 GiB | depends on response length and Gemma implementation |

Every training run writes allocated and reserved CUDA peaks to `run_manifest.json`. Every generation run writes serving latency, throughput, generated tokens, and CUDA peaks to `<output>.metrics.json`. Those measurements should replace the planning ranges in the paper.

## 10. What this implementation does not claim

- No accuracy or safety improvement is asserted before the runs finish.
- Automated corruption intent is not treated as verified clinical annotation.
- K=4 is reported as an automated hard-negative experiment, with its own label-noise limitation.
- A denoiser candidate is not counted as DPO+selective-denoiser output until the shared gate explicitly invokes and accepts it.
- The existing exp295 provenance is kept distinct from raw MentalChat16K unless the underlying artifacts establish that connection.
