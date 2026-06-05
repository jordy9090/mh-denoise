# MH Denoise

This repository contains training, inference, and evaluation code for counseling-response refinement experiments.

## Current Method

The active pipeline is the professor-proposed SFT-first selective risk-aware refinement cascade:

```text
unsafe draft
  -> SFT refiner
  -> safety/risk gate
  -> optional risk-aware denoiser
  -> accept denoiser output only when safety improves and specificity is preserved
```

The method is documented in `docs/selective_risk_refinement_pipeline.md`.

## Active Entry Points

- `scripts/build_sft_outputs_for_risk_tuning.py`: generate first-stage SFT responses.
- `scripts/train_gemma_risk_tune_from_sft.py`: weakly tune the risk-aware denoiser from the SFT checkpoint.
- `scripts/run_gemma_selective_risk_refinement.py`: run the selective cascade at inference time.
- `scripts/selective_risk_refinement_utils.py`: shared prompt, scoring, corruption, and selection utilities.

## Supporting Components

- Aspect router: `scripts/train_aspect_router.py`
- Span-risk scorer: `scripts/train_span_risk_multilabel.py`
- SFT refiner trainer: `scripts/train_professor_peft_refiner_textonly.py`
- LLM judge utilities: `scripts/prepare_refinement_judge_input.py`, `scripts/run_refinement_llm_judge.py`, `scripts/aggregate_refinement_judge_scores.py`

Generated datasets, checkpoints, and inference outputs are expected under `data/` and `outputs/`. These large artifacts are not committed to the repository.

## Documentation

- `docs/README.md`: documentation index.
- `docs/selective_risk_refinement_pipeline.md`: current method and commands.
- `docs/exp295_fair_baselines.md`: baseline commands.
- `docs/refinement_judge_pipeline.md`: evaluation commands.

Older MoE, full-joint, residual-MLP, and denoising-ablation notes were removed from active documentation to avoid confusing them with the current method.

## Quick Validation

```bash
python -m py_compile \
  scripts/selective_risk_refinement_utils.py \
  scripts/build_sft_outputs_for_risk_tuning.py \
  scripts/train_gemma_risk_tune_from_sft.py \
  scripts/run_gemma_selective_risk_refinement.py
```
