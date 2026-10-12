# SFT-First Selective Refinement for Mental-Health Question Answering

Documentation for the ACL ARR 2026 manuscript and its production implementation.
The method refines a complete SFT response when residual risk warrants another
generation, then retains the candidate only when a frozen selector prefers it in
both presentation orders.

## Implementation Status

This documentation branch was started from public `main` at `756f42d`. That
snapshot contains the earlier CIKM/exp295 implementation and the manuscript
archive, but does **not** contain most ACL production entry points. The ACL code
and saved execution records were inspected in a separate research checkout.
Integration into public `main` is pending verification; this checkout must not be
treated as a complete runnable ACL release. See the dated
[verification record](docs/ACL_VERIFICATION.md) for the last checked remote
revision, missing files, and remaining checks.

## ACL Method

1. **Reference screening and corruption.** QC1 screens references from
   MentalChat16K and CounselChat. Minimal, aspect-guided corruption is followed
   by blinded QC2 assessment. Requested edits and **observed aspect labels** are
   distinct: the Router learns the degradation judged by QC2. Verified local
   evidence and observation masks provide span-risk supervision.
2. **SFT-first reconstruction.** An SFT adapter reconstructs the screened
   reference from the question and corrupted response.
3. **Router and span-risk Scorer.** The Router estimates response-level risk
   across six aspects: overall quality, empathy, specificity, medical-boundary
   control (`medical_boundary` in production supervision),
   factual consistency, and toxicity. The Scorer estimates aspect risk for local
   passages; the combined estimates support invocation and training-example
   sampling.
4. **Complete-response refinement.** A refinement adapter initialized from SFT
   reconstructs a complete response conditioned on the aspect signal. In the
   manuscript's production condition the draft field contains the full SFT
   response, masking is disabled, and target tokens have uniform loss weights.
   Local evidence supports risk estimation; generation is not a span-only edit.
5. **Selective invocation and acceptance.** Frozen validation-selected settings
   and response triggers determine whether to generate a candidate. The frozen
   `gpt-4.1-2025-04-14` selector compares SFT and the candidate in both A/B and B/A
   orders. Adoption requires the candidate to win both; a tie, disagreement, or
   invalid decision retains SFT.
6. **Automatic and expert evaluation.** Pointwise GPT-4.1 and independent
   Llama-3.1-8B-Instruct assessments measure overall quality, empathy,
   specificity, and medical-boundary advice under an explicit rubric. Identical
   question-response pairs share pointwise judgments across conditions. GPT
   selection and primary evaluation use the same model and are not independent.
   The manuscript reports counselor assessments of adopted revisions as a
   separate evaluation, not expert ratings of every test response. Completed
   expert ratings were not located in the inspected H100 artifacts.

The manuscript's frozen partitions contain **276 training, 43 validation, and
60 test questions**. A later TEST100 extension and residual-training experiments
are separate research runs, not replacements for the reported TEST60 results.
Earlier development comparisons in the manuscript also use different protocols.

## Reproducibility and Review Artifacts

- [ACL reproducibility](docs/ACL_REPRODUCIBILITY.md): production code, original
  prompt locations, model identities, settings, schemas, dependencies, execution
  order, evaluation, and evidence limits.
- [Verification record](docs/ACL_VERIFICATION.md): static checks and the status of
  reconciliation with the latest public code.
- [Anonymous artifact preparation](docs/ACL_ANONYMIZATION.md): inspected identity
  leaks, proposed source-export changes, mirror status, and history blockers.
- [Minimal Appendix addition](docs/ACL_APPENDIX_ADDITION.tex): artifact and
  reproducibility wording for author review.
- [Legacy index](docs/LEGACY.md): retained CIKM, exp295, pilot, and experimental
  material. Historical claims and commands in these notes describe those runs.

Saved local manifests support the evidence described in the reproducibility
document; they are not proof that a clean public checkout can reproduce the
entire pipeline. Training, generation, judge calls, and data access require
dependencies that are not bundled here. No new experiment was run for this
documentation update, and the frozen TEST inputs and outputs were not changed.

## Data Access and Redistribution

This change does **not** release a dataset. Users must obtain the original
MentalChat16K and CounselChat resources from their providers and satisfy their
access terms and applicable permissions. CounselBench is used as an assessment
framework in the ACL protocol; the older CounselBench-Eval target-pool workflow
belongs to the legacy experiments.

Permission to access or use original data is separate from permission to
redistribute derived question-response pairs, corruptions, evidence passages,
annotations, or generated responses. A source-code release does not establish
those redistribution rights. No derived dataset, counselor packet, checkpoint,
or raw response release is authorized or promised here. The availability and
redistribution status of each such artifact remain to be resolved separately.

Credentials belong in environment variables or local secret configuration.
Local `.env` files, datasets, checkpoints, caches, and generated experiment
outputs must stay out of the reviewer export.
