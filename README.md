# MH Denoise

This repository contains the training and inference code for counseling-response denoising experiments. The main task is to rewrite unsafe or low-quality counseling responses into safer, more supportive responses using risk-aware corrupted drafts, aspect routing, and parameter-efficient Gemma refiners.

## Repository Layout

- `scripts/`: data preparation, router/risk-scorer training, denoiser training, inference, and evaluation utilities.
- `docs/`: reproducible experiment notes and command templates.
- `docs/professor_method_section.md`: method-section implementation notes.
- `docs/full_joint_training.md`: router-denoiser full-joint training notes.
- `docs/aspect_moe_lora_pipeline.md`: LoRA expert-MoE training and inference notes.
- `docs/residual_mlp_moe_pipeline.md`: residual MLP-MoE alternative.
- `docs/selective_risk_refinement_pipeline.md`: SFT-first selective risk-aware refinement.

Generated datasets, checkpoints, and inference outputs are expected under `data/` and `outputs/`. These large artifacts are not committed to the repository.

## Main Components

- Aspect router: `scripts/train_aspect_router.py`
- Span-risk scorer: `scripts/train_span_risk_multilabel.py`
- PEFT denoising refiner: `scripts/train_gemma_peft_denoiser.py`
- LoRA expert-MoE refiner: `scripts/train_gemma_aspect_moe_denoiser.py`
- Router-denoiser full-joint refiner: `scripts/train_gemma_full_joint_denoiser.py`
- Residual MLP-MoE refiner: `scripts/train_gemma_aspect_mlp_moe_refiner.py`
- Selective risk-aware refinement: `scripts/run_gemma_selective_risk_refinement.py`

## Quick Validation

Run a syntax check on the most recent training and inference entry points:

```bash
python -m py_compile \
  scripts/aspect_moe_lora.py \
  scripts/aspect_residual_mlp_moe.py \
  scripts/selective_risk_refinement_utils.py \
  scripts/build_sft_outputs_for_risk_tuning.py \
  scripts/train_gemma_full_joint_denoiser.py \
  scripts/train_gemma_aspect_mlp_moe_refiner.py \
  scripts/train_gemma_risk_tune_from_sft.py \
  scripts/run_gemma_aspect_moe_real_inference.py \
  scripts/run_gemma_aspect_mlp_moe_real_inference.py \
  scripts/run_gemma_selective_risk_refinement.py
```

Run the lightweight structural smoke test for the residual MLP-MoE module:

```bash
python scripts/smoke_residual_mlp_moe.py
```

## Recommended Reading Order

1. `docs/professor_method_section.md`
2. `docs/aspect_moe_lora_pipeline.md`
3. `docs/full_joint_training.md`
4. `docs/residual_mlp_moe_pipeline.md`
5. `docs/selective_risk_refinement_pipeline.md`

Each experiment document includes command templates for training, inference, expected checkpoint layout, and troubleshooting notes.
