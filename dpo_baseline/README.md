# SFT-Initialized DPO Baselines for exp295 Mental-Health Response Refinement

## 1. Scope

This directory implements a controlled DPO baseline for the exp295 single-turn response-refinement task.

For every example,

\[
x=(q,d), \qquad y^{+}=\text{safe response}, \qquad y^{-}=\text{flawed or unsafe response},
\]

where `q` is the mental-health question and `d` is the initial draft. The model prompt contains the fixed refinement instruction, `q`, and `d`. Violation labels, counselor judgments, therapist notes, and target-dimension metadata remain outside the prompt.

The DPO objective is

\[
\mathcal{L}_{\mathrm{DPO}}
= -\log \sigma\!\left(
\beta\left[
\log\frac{\pi_{\theta}(y^{+}\mid x)}{\pi_{\mathrm{ref}}(y^{+}\mid x)}
-
\log\frac{\pi_{\theta}(y^{-}\mid x)}{\pi_{\mathrm{ref}}(y^{-}\mid x)}
\right]
\right).
\]

The trainable policy starts from the reported SFT refiner adapter. The reference is an exact frozen copy of that same adapter before the first DPO update. The code checks tensor equality and frozen status at runtime.

The data scope is fixed to exp295:

| split | questions | rows |
|---|---:|---:|
| train | 207 | 1,242 |
| validation | 29 | 174 |
| test | 59 | 354 |

Multi-turn data, MusPsy, counselor preference labels, PPO, GRPO, and reward-model training are outside this implementation.

## 2. Repository audit and control choices

The existing pipeline documents the following final denoiser controls: `google/gemma-4-E4B-it`, 4-bit NF4, bf16 compute, source/target limits of `512/160`, batch size `1`, gradient accumulation `16`, three training passes, and deterministic decoding with `max_new_tokens=120`.

The direct SFT training script currently creates a LoRA adapter with `r=16`, `alpha=32`, and `target_modules=["linear"]`. The final selective denoiser uses a separate text-attention adapter with `r=8`, `alpha=16`, and language-model `q/k/v/o` targets. DPO continuation must preserve the loaded SFT adapter architecture, because changing rank or target modules would break checkpoint initialization. The configs therefore verify the direct SFT settings. Update these expected values only after checking the exact `adapter_config.json` used for the SFT table row.

The controlled DPO settings are:

| item | setting |
|---|---|
| backbone | `google/gemma-4-E4B-it` |
| initialization | exact existing SFT adapter |
| reference | frozen clone of the initial SFT adapter |
| quantization | NF4 4-bit, double quantization, bf16 compute |
| prompt | fixed instruction + `q` + `d` |
| source / completion budget | 512 / 160 tokens |
| effective pair batch | 16 (`1 × grad_accum 16`) |
| optimizer updates | 234 |
| optimizer | AdamW, LR `1e-5`, linear schedule, warmup `0.06` |
| DPO loss | sigmoid |
| decoding | greedy, 120 new tokens, repetition penalty 1.15, no-repeat 4-gram |

`234` updates correspond to three passes over 1,242 preference pairs at an effective batch of 16. Minimal and hard-K4 experiments contain the same number of pairs, so the update count and pair exposure stay aligned. DPO still performs two completion forwards per pair and a reference precomputation pass. Training manifests record processed tokens, wall time, GPU-hours, and peak VRAM so the paper can disclose this extra cost.

## 3. Experimental arms

### 3.1 SFT

Load the existing SFT refiner checkpoint and generate one response from the shared `q+d` prompt. This is the anchor row.

### 3.2 DPO-Minimal

For each source row:

```text
prompt   = fixed instruction + q + d
chosen   = safe_response
rejected = unsafe_response
```

The DPO output is returned directly at inference. Its training lineage is `SFT checkpoint → DPO`; its serving path contains one refiner generation.

### 3.3 DPO-Hard-K4

Each source row receives an exact automated candidate budget of four rejected responses:

1. its original unsafe draft;
2. three deterministically sampled same-question corruptions from the remaining automatically generated violation variants.

The frozen SFT policy and fixed router score these four candidates. One hard rejected response is selected using

\[
h(r)=0.45\,\widetilde{\ell}_{\mathrm{SFT}}(r\mid x)
+0.35\,\mathrm{sim}(r,y^{+})
+0.20\,\mathrm{risk}(r),
\]

where `SFT likelihood` is length-normalized within the four-candidate pool, `sim` is lexical sequence similarity to the safe response, and `risk` is the maximum automated router probability. The safe response remains `chosen`. The selected flawed response becomes the single `rejected` completion. Pair counts remain 1,242/174, which isolates pair difficulty from dataset size.

The candidate-mining seed is fixed to `3407` for all DPO training seeds. Candidate files are generated once, hashed, and frozen before beta tuning.

### 3.4 DPO + selective denoiser

The DPO response becomes the residual-repair anchor. The compatibility wrapper performs:

```text
DPO response
  → fixed aspect router and span-risk scorer
  → selective invocation
  → mask high-risk spans in the DPO anchor
  → existing LoRA denoiser, conditioned on q, original d, and masked DPO anchor
  → accept/reject filter
  → final response
```

The original unsafe draft remains available to the existing denoiser as context. The corrupted draft is produced from the DPO response, which targets residual errors in the current anchor.

A denoiser candidate is accepted when all configured checks pass:

- predicted global risk does not increase;
- predicted medical-boundary risk does not increase;
- the specificity proxy stays above the validation-selected retention ratio;
- length, prompt-leak, mask-leak, and repetition checks pass.

Gate and acceptance thresholds are development settings. Select them on the 174-row validation split, freeze them, then run the 354-row test split.

### 3.5 Proposed

Use the existing SFT-first selective risk-aware denoising output unchanged. The baseline directory does not edit its training, inference, router, span scorer, acceptance code, or result files.

## 4. Beta and seed protocol

### Beta screening

Run

```text
beta ∈ {0.03, 0.10, 0.30}, seed = 42
```

for Minimal and Hard-K4 separately. Evaluate every checkpoint on validation with the same response judge and decoding settings.

The provided selector applies this rule:

1. retain candidates whose mean overall quality and specificity are within `0.05` of SFT;
2. minimize medical-advice `yes/unsure` rate;
3. minimize toxicity `≥3` rate;
4. break remaining ties by overall quality and factual consistency.

Record the selected beta for each pair regime before test inference. A shared-beta sensitivity run can be added when compute permits, using the beta selected by Minimal for both pair regimes.

### Final seeds

After beta selection, train with

```text
seeds = {42, 43, 44}
```

while keeping the pair files, hard-mining seed, beta, update budget, prompt, and decoding fixed. Report mean and standard deviation across training seeds. The aggregation script also computes 95% question-cluster bootstrap intervals and paired candidate-minus-baseline intervals. Clustering by question keeps the six corruption variants from being treated as independent questions.

The current SFT and Proposed checkpoints may represent one training seed. Such a table is preliminary. A full statistical comparison should retrain every trainable arm with the same three seeds or state the unequal seed coverage explicitly.

## 5. Files

| file | purpose |
|---|---|
| `build_preference_pairs.py` | construct Minimal or Hard-K4 TRL preference records and manifests |
| `generate_hard_candidates.py` | create exact K=4 candidate pools and select one automated hard negative |
| `train_dpo.py` | load SFT QLoRA, create a frozen SFT reference adapter, enforce budgets, and train DPO |
| `run_inference.py` | shared SFT/DPO inference for `q+d` |
| `run_selective_denoiser.py` | apply the existing residual-risk pipeline to DPO outputs |
| `run_sweep.py` | plan or execute beta and final-seed runs |
| `aggregate_comparison.py` | aggregate quality, safety, uncertainty, and efficiency results |
| `dpo_utils.py` | contracts, hashing, token budgets, prompt construction, and shared utilities |
| `tests/test_data_contract.py` | prompt leakage, pair construction, split leakage, hard selection, and paired-CI tests |

## 6. Installation

```bash
cd ~/mh-denoise
conda activate mh-denoise-a100

# Install the CUDA-matched PyTorch build first when the environment does not have one.
pip install -r dpo_baseline/requirements.txt

python -m unittest discover -s dpo_baseline/tests -v
```

TRL `1.12.0` and PEFT `0.20.0` are pinned because this implementation relies on TRL's pretrained-PEFT continuation path, which creates a `ref` adapter from the initial policy adapter. `train_dpo.py` stops when that frozen clone is absent or differs from the initial policy tensors.

## 7. Paths

Set the exact checkpoints from the completed exp295 run:

```bash
export SFT_ADAPTER=/absolute/path/to/the/reported/sft_refiner_checkpoint

export ROUTER_DIR=outputs/models/router_multilabel_v1/best
[ -d "$ROUTER_DIR" ] || export ROUTER_DIR=outputs/models/router_multilabel_v1/final

export RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/best
[ -d "$RISK_SCORER_DIR" ] || export RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/final

export DENOISER_DIR=outputs/models/gemma4_peft_langqkvo_infermatch_exp295_main/best
[ -d "$DENOISER_DIR" ] || export DENOISER_DIR=outputs/models/gemma4_peft_langqkvo_infermatch_exp295_main/final
```

Before training, inspect the SFT adapter contract:

```bash
cat "$SFT_ADAPTER/adapter_config.json"
```

The path must point to the checkpoint used for the SFT result row. A newly chosen SFT checkpoint changes the experiment.

## 8. Build Minimal pairs

```bash
mkdir -p dpo_baseline/data/exp295_minimal

python dpo_baseline/build_preference_pairs.py \
  --input data/splits_exp295/train_mdlm.jsonl \
  --output dpo_baseline/data/exp295_minimal/train.jsonl \
  --mode minimal \
  --expected_rows 1242 \
  --expected_questions 207 \
  --disjoint_with data/splits_exp295/valid_mdlm.jsonl \
  --disjoint_with data/splits_exp295/test.jsonl

python dpo_baseline/build_preference_pairs.py \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output dpo_baseline/data/exp295_minimal/valid.jsonl \
  --mode minimal \
  --expected_rows 174 \
  --expected_questions 29 \
  --disjoint_with data/splits_exp295/train_mdlm.jsonl \
  --disjoint_with data/splits_exp295/test.jsonl
```

Expected pair counts are `1,242 / 174`. Each sidecar manifest records source and output SHA-256 hashes, question counts, pair counts, and `counselor_annotation_used=false`.

## 9. Build Hard-K4 pairs

### 9.1 Mine one hard rejected response from K=4 candidates

```bash
mkdir -p dpo_baseline/data/exp295_hard_k4

python dpo_baseline/generate_hard_candidates.py \
  --input data/splits_exp295/train_mdlm.jsonl \
  --output dpo_baseline/data/exp295_hard_k4/train_candidates.jsonl \
  --base_model google/gemma-4-E4B-it \
  --sft_adapter_dir "$SFT_ADAPTER" \
  --router_dir "$ROUTER_DIR" \
  --k 4 \
  --policy_samples 0 \
  --seed 3407 \
  --expected_rows 1242 \
  --resume

python dpo_baseline/generate_hard_candidates.py \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output dpo_baseline/data/exp295_hard_k4/valid_candidates.jsonl \
  --base_model google/gemma-4-E4B-it \
  --sft_adapter_dir "$SFT_ADAPTER" \
  --router_dir "$ROUTER_DIR" \
  --k 4 \
  --policy_samples 0 \
  --seed 3407 \
  --expected_rows 174 \
  --resume
```

`policy_samples=0` keeps the primary hard-pair experiment fully tied to the existing six automated corruptions. Frozen-SFT rollout candidates are implemented as an exploratory option and should receive a separate experiment label.

### 9.2 Serialize the selected hard pairs

```bash
python dpo_baseline/build_preference_pairs.py \
  --input data/splits_exp295/train_mdlm.jsonl \
  --output dpo_baseline/data/exp295_hard_k4/train.jsonl \
  --mode hard_k4 \
  --hard_candidates dpo_baseline/data/exp295_hard_k4/train_candidates.jsonl \
  --k 4 \
  --expected_rows 1242 \
  --expected_questions 207 \
  --disjoint_with data/splits_exp295/valid_mdlm.jsonl \
  --disjoint_with data/splits_exp295/test.jsonl

python dpo_baseline/build_preference_pairs.py \
  --input data/splits_exp295/valid_mdlm.jsonl \
  --output dpo_baseline/data/exp295_hard_k4/valid.jsonl \
  --mode hard_k4 \
  --hard_candidates dpo_baseline/data/exp295_hard_k4/valid_candidates.jsonl \
  --k 4 \
  --expected_rows 174 \
  --expected_questions 29 \
  --disjoint_with data/splits_exp295/train_mdlm.jsonl \
  --disjoint_with data/splits_exp295/test.jsonl
```

Expected pair counts remain `1,242 / 174`.

## 10. Preflight

The dry run verifies hashes, row counts, question counts, split isolation, prompt contracts, and the SFT checkpoint path without loading the 4-bit model.

```bash
python dpo_baseline/train_dpo.py \
  --config dpo_baseline/configs/exp295_minimal.yaml \
  --sft_adapter_dir "$SFT_ADAPTER" \
  --dry_run

python dpo_baseline/train_dpo.py \
  --config dpo_baseline/configs/exp295_hard_k4.yaml \
  --sft_adapter_dir "$SFT_ADAPTER" \
  --dry_run
```

The full run additionally verifies the LoRA rank, alpha, target-module contract, trainable-parameter scope, exact policy/reference tensor equality, and frozen reference status.

## 11. Beta screening

Print commands first:

```bash
python dpo_baseline/run_sweep.py \
  --config dpo_baseline/configs/exp295_minimal.yaml \
  --stage beta \
  --sft_adapter_dir "$SFT_ADAPTER"

python dpo_baseline/run_sweep.py \
  --config dpo_baseline/configs/exp295_hard_k4.yaml \
  --stage beta \
  --sft_adapter_dir "$SFT_ADAPTER"
```

Execute after reviewing the manifests:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python dpo_baseline/run_sweep.py \
  --config dpo_baseline/configs/exp295_minimal.yaml \
  --stage beta \
  --sft_adapter_dir "$SFT_ADAPTER" \
  --execute

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python dpo_baseline/run_sweep.py \
  --config dpo_baseline/configs/exp295_hard_k4.yaml \
  --stage beta \
  --sft_adapter_dir "$SFT_ADAPTER" \
  --execute
```

This stage trains each beta and writes validation generations. It does not touch the test split.

Judge each `valid_inference.jsonl` with the same existing evaluation script used by the project:

```bash
export OPENAI_API_KEY=...

python scripts/run_refinement_judge_eval.py \
  --input outputs/dpo_exp295/minimal/beta/beta_0.1/seed_42/valid_inference.jsonl \
  --output outputs/dpo_exp295/minimal/beta/beta_0.1/seed_42/valid_judged.jsonl \
  --model gpt-4.1
```

Repeat for all beta values and both pair regimes. The existing judge receives method and target-dimension metadata. Keep this setup for direct comparability with prior results. A blinded sensitivity evaluation can set these two fields to a shared constant before judging.

Example beta aggregation:

```bash
python dpo_baseline/aggregate_comparison.py \
  --run SFT=outputs/refinement/sft_exp295_valid_judged.jsonl \
  --run DPO-minimal-beta0.03@42=outputs/dpo_exp295/minimal/beta/beta_0.03/seed_42/valid_judged.jsonl \
  --run DPO-minimal-beta0.1@42=outputs/dpo_exp295/minimal/beta/beta_0.1/seed_42/valid_judged.jsonl \
  --run DPO-minimal-beta0.3@42=outputs/dpo_exp295/minimal/beta/beta_0.3/seed_42/valid_judged.jsonl \
  --output_dir outputs/dpo_exp295/minimal/beta_selection \
  --select_beta \
  --baseline_method SFT \
  --candidate_prefix DPO-minimal
```

Run the same command for Hard-K4. Store the selected beta in the experiment log before launching final seeds.

## 12. Final seed runs

Example with a selected beta of `0.10`:

```bash
python dpo_baseline/run_sweep.py \
  --config dpo_baseline/configs/exp295_minimal.yaml \
  --stage seed \
  --selected_beta 0.10 \
  --seeds 42,43,44 \
  --sft_adapter_dir "$SFT_ADAPTER" \
  --execute

python dpo_baseline/run_sweep.py \
  --config dpo_baseline/configs/exp295_hard_k4.yaml \
  --stage seed \
  --selected_beta 0.10 \
  --seeds 42,43,44 \
  --sft_adapter_dir "$SFT_ADAPTER" \
  --execute
```

These commands generate validation outputs again for the fixed-beta seed runs. Confirm all manifests and validation results before opening the test split.

## 13. Test inference

Example for Minimal, beta `0.10`, seed `42`:

```bash
python dpo_baseline/run_inference.py \
  --input data/splits_exp295/test.jsonl \
  --output outputs/dpo_exp295/minimal/seed/beta_0.1/seed_42/test_inference.jsonl \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir outputs/dpo_exp295/minimal/seed/beta_0.1/seed_42/final \
  --method DPO-Minimal \
  --seed 42
```

Run the corresponding command for seeds 43/44 and Hard-K4.

Generate the SFT row through the same script and shared prompt:

```bash
python dpo_baseline/run_inference.py \
  --input data/splits_exp295/test.jsonl \
  --output outputs/dpo_exp295/sft_shared_prompt/test_inference.jsonl \
  --base_model google/gemma-4-E4B-it \
  --adapter_dir "$SFT_ADAPTER" \
  --method SFT \
  --seed 42
```

## 14. DPO + selective denoiser

Example:

```bash
python dpo_baseline/run_selective_denoiser.py \
  --input outputs/dpo_exp295/minimal/seed/beta_0.1/seed_42/test_inference.jsonl \
  --output outputs/dpo_exp295/minimal/seed/beta_0.1/seed_42/test_dpo_plus_denoiser.jsonl \
  --base_model google/gemma-4-E4B-it \
  --denoiser_adapter_dir "$DENOISER_DIR" \
  --router_dir "$ROUTER_DIR" \
  --risk_scorer_dir "$RISK_SCORER_DIR" \
  --method DPO-Minimal+Selective-Denoiser \
  --gate_global_threshold 0.50 \
  --gate_medical_threshold 0.35 \
  --risk_tolerance 0.0 \
  --medical_risk_tolerance 0.0 \
  --specificity_ratio 0.85
```

The numeric thresholds above are initial development values. Use a validation sweep for the final run, record the selected values, and reuse them unchanged on test.

## 15. Final evaluation and comparison

Run the existing response judge on every test output, including the unchanged Proposed output. Then aggregate:

```bash
python dpo_baseline/aggregate_comparison.py \
  --run SFT=outputs/dpo_exp295/sft_shared_prompt/test_judged.jsonl \
  --run DPO-Minimal@42=outputs/dpo_exp295/minimal/seed/beta_0.1/seed_42/test_judged.jsonl \
  --run DPO-Minimal+Selective-Denoiser@42=outputs/dpo_exp295/minimal/seed/beta_0.1/seed_42/test_dpo_plus_denoiser_judged.jsonl \
  --run Proposed=outputs/refinement/proposed_exp295_test_judged.jsonl \
  --train_manifest DPO-Minimal@42=outputs/dpo_exp295/minimal/seed/beta_0.1/seed_42/run_manifest.json \
  --output_dir outputs/dpo_exp295/final_comparison \
  --compare_to SFT \
  --compare_to Proposed
```

Add seed 43/44 and Hard-K4 run specifications to the same command.

The output includes:

- `comparison.md`: paper-table draft;
- `run_level_summary.json/csv`: run metrics and question-cluster confidence intervals;
- `method_seed_summary.json`: mean and standard deviation across seeds;
- `paired_differences.json`: paired question-cluster bootstrap differences against SFT and Proposed;
- training and inference efficiency fields from sidecar manifests.

## 16. Paper table design

### Main comparison

| Method | Init | Pair construction | Overall ↑ | Empathy ↑ | Specificity ↑ | Factual ↑ | Toxicity ↓ | Med. advice yes/unsure ↓ | Invoke % | Accept % | Latency ms ↓ | Train GPU h ↓ | Peak VRAM ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT | SFT | CE checkpoint | — | — | — | — | — | — | — | — | — | — | — |
| DPO | SFT | validation-selected Minimal or Hard-K4 | — | — | — | — | — | — | — | — | — | — | — |
| DPO + denoiser | SFT → DPO | same selected pair regime | — | — | — | — | — | — | — | — | — | — | — |
| Proposed | SFT + selective denoiser | existing method | — | — | — | — | — | — | — | — | — | — | — |

Report DPO values as mean ± standard deviation over seeds and provide question-cluster 95% confidence intervals in the appendix. Mark SFT/Proposed seed coverage explicitly.

### Pair-construction ablation

| Pair regime | K candidates | pairs/train | beta | Overall ↑ | Specificity ↑ | Med. yes/unsure ↓ | Preference accuracy ↑ | Reward margin ↑ | GPU h ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Minimal | 1 | 1,242 | selected on valid | — | — | — | — | — | — |
| Hard-K4 | 4 → select 1 | 1,242 | selected on valid | — | — | — | — | — | — |

TRL logs preference accuracy and reward margins. These diagnostics show whether DPO learned the supplied preference ordering. They do not establish downstream response improvement by themselves.

## 17. Expected GPU memory

These ranges are pre-run engineering estimates. CUDA version, FlashAttention availability, checkpoint target modules, response lengths, and allocator behavior can move the peak substantially. Every executable writes measured allocated and reserved VRAM.

| stage | expected peak GPU memory | planning note |
|---|---:|---|
| Minimal/Hard-K4 pair construction | CPU | no model load |
| Hard-K4 mining, 4-bit SFT + router | 10–16 GiB | reduce score batch to 1 or place router on CPU when needed |
| DPO training with `r=8` text q/k/v/o adapter | 14–22 GiB | reference log-probs precomputed; one quantized backbone in memory |
| DPO training with repository direct-SFT `r=16`, broad `linear` targets | 22–32 GiB | exact checkpoint architecture determines the true peak |
| SFT/DPO inference | 8–12 GiB | batch size 1 |
| DPO + denoiser with router and span scorer on GPU | 12–18 GiB | classifier CPU placement lowers GPU use and increases latency |

Use an A100/H100 with at least 40 GiB for the first full DPO run. After measuring `peak_memory_reserved_gib`, a smaller card can be tested with score batch `1`, classifier CPU placement, and the same scientific settings.

## 18. Interpretation boundary

This implementation supplies the baseline, controls, data audits, and result-table pipeline. It has not produced model-quality results in this branch. Reported gains require completed validation selection, sealed-test inference, judge outputs, and statistical aggregation. Preference accuracy, lower DPO loss, or a positive reward margin alone cannot support a response-quality or safety claim.
