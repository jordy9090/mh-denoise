# Legacy and Experimental Material

The following material is retained for provenance. It describes CIKM short-paper
development, exp295, pilots, and experimental alternatives rather than the final
ACL production condition. Nothing is deleted or moved by this documentation
change. Follow [ACL_REPRODUCIBILITY.md](ACL_REPRODUCIBILITY.md) for the ACL stage
mapping and check [ACL_VERIFICATION.md](ACL_VERIFICATION.md) before running it.

## Historical Documentation

| Retained document | Historical scope |
| --- | --- |
| [exp295_bc_pipeline.md](exp295_bc_pipeline.md) | exp295 Router and teacher-forced aspect-mixture development |
| [exp300_final_pipeline.md](exp300_final_pipeline.md) | historical 295-target CounselBench-Eval/CounselChat pipeline; the filename reflects an earlier plan |
| [experiment_context_expanded_denoising.md](experiment_context_expanded_denoising.md) | CIKM pilot, source mixtures, short generation budgets, and token-risk weighting |
| [professor_method_section.md](professor_method_section.md) | earlier method draft including proposed MoE and discrete denoising formulations |
| [refinement_judge_pipeline.md](refinement_judge_pipeline.md) | legacy GPT-4.1-mini six-score evaluation of PEFT denoiser outputs |

The word "final" or "current" inside a historical note refers to its own
development stage. In particular, the old safe-target pool, masked-span
reconstruction, inference time steps, risk-weighted target loss, and six-score
development composite must not be substituted for the manuscript's frozen ACL
contracts or expert-aligned pointwise evaluation.

## Experimental Code

`scripts/deprecated/` retains the misleading `full_moe` implementations already
deprecated in repository history. Names mentioning MoE are not evidence of an
implemented or production-used mixture of expert adapters.

The older `prepare_overleaf_*`, `prepare_mh_diffudetox_data.py`,
`prepare_faithful_denoise_data.py`, and `prepare_riskspan_denoise_data.py` workflows,
and the MDLM, masked-discrete, semantic-diffusion, seq2seq-span, and older
faithful/riskspan/PEFT refiner variants remain historical or experimental unless
the ACL stage mapping explicitly identifies a reused component. A filename or a
successful import does not establish that a run produced a paper result.

Failed, abandoned, or incomplete experiments are not promoted to the ACL
production protocol. Their source and notes remain available; where a failure or
completion record is unavailable, execution status is unverified rather than
inferred from the experiment name. Shared Router, Scorer, or judge utilities may
also be used by ACL runs; their actual role is determined by the frozen command,
configuration, and output manifest.

The research checkout also contains `docs/fullpaper_acl_pipeline.md`, a
calibration-era construction note with pilot blockers. It is absent from the
initial public-main snapshot and must remain labeled calibration/legacy if
imported during integration; it is not the final ACL reproduction guide.

## Separate Later Research

The later TEST100 extension, fresh base-residual training, continued-denoiser
training, and target-only improvement worktrees are separate from the
manuscript's corrected TEST60. Distinct fresh and continued adapter trainers
must retain their initialization semantics. These runs do not update the paper's
test membership, thresholds, or result tables automatically.
