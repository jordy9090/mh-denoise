# SFT-First Selective Risk-Aware Denoising for Mental-Health QA

This repository contains the implementation and experiment scripts for our CIKM 2026 Short Paper submission:

**SFT-First Selective Risk-Aware Denoising for Mental-Health Question Answering**

## Overview

The project studies response refinement for safety-sensitive mental-health question answering. The pipeline first applies an SFT-based refiner to improve the full response, then invokes a localized denoising stage only when residual risk remains.

The core workflow is:

1. **SFT-first refinement** of the initial response.
2. **Residual-risk detection** across safety-sensitive aspects.
3. **Aspect routing and span-level risk localization** to identify the part of the response that requires repair.
4. **LoRA-based denoising** on the localized risky span.
5. **Selective acceptance** of the candidate only when risk does not increase and response specificity is preserved.

The goal is to repair localized residual failures while preserving the empathy, specificity, and overall quality already achieved by the SFT refiner.

## Main Components

- `scripts/generate_unsafe_samples.py`: generate aspect-guided unsafe corruptions from safe targets.
- `scripts/generate_scaled_semantic_corruptions.py`: generate scaled semantic corruption variants.
- `scripts/prepare_counselbench_eval_100.py`: prepare strict safe targets from CounselBench-Eval.
- `scripts/prepare_counselchat_safe_candidates.py`: prepare candidate safe targets from CounselChat.
- `scripts/judge_counselchat_safe_candidates.py`: evaluate CounselChat candidates for use as safe targets.
- `scripts/merge_safe_targets_exp300.py`: merge the final safe-target pool.
- `scripts/build_refinement_eval_inputs.py`: construct evaluation inputs for the refinement pipeline.
- `scripts/aggregate_refinement_judge.py`: aggregate LLM-judge refinement results.
- `scripts/aggregate_refinement_judge_scores.py`: summarize evaluation scores.
- `scripts/inspect_refinement_outputs.py`: inspect final response-refinement outputs.

Detailed experiment notes are available in `docs/`.

## Data and Experimental Setup

The final documented training pipeline uses a mixed pool of safe mental-health responses drawn from CounselBench-Eval and judged CounselChat candidates. Aspect-guided corruptions are generated to create unsafe/reference pairs for training the residual-risk repair component.

The repository does not redistribute restricted data, private API credentials, large checkpoints, or full generated response files.

## Evaluation

The experiments evaluate both response quality and safety, including:

- overall response quality,
- empathy,
- specificity,
- factual consistency,
- toxicity,
- medical-boundary violations.

The evaluation pipeline compares the initial unsafe response, SFT refinement, risk-aware denoising variants, and selective risk-aware refinement.

## Repository Structure

```text
mh-denoise/
├── README.md
├── docs/
│   ├── exp295_bc_pipeline.md
│   ├── exp300_final_pipeline.md
│   ├── experiment_context_expanded_denoising.md
│   ├── professor_method_section.md
│   └── refinement_judge_pipeline.md
└── scripts/
    ├── generate_unsafe_samples.py
    ├── generate_scaled_semantic_corruptions.py
    ├── prepare_counselbench_eval_100.py
    ├── prepare_counselchat_safe_candidates.py
    ├── judge_counselchat_safe_candidates.py
    ├── merge_safe_targets_exp300.py
    ├── build_refinement_eval_inputs.py
    ├── aggregate_refinement_judge.py
    ├── aggregate_refinement_judge_scores.py
    └── inspect_refinement_outputs.py
```

## Reproducibility Notes

The experiment documentation under `docs/` records the data-preparation and training/evaluation pipeline used during development. Some scripts depend on external datasets, Hugging Face models, or OpenAI-based evaluation and therefore require the corresponding credentials and licenses.

API keys should be supplied through environment variables and must not be hard-coded in source files.

## Status

This repository accompanies a research prototype and is intended to support transparency and reproducibility of the reported experiments.