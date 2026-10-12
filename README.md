# SFT-First Selective Refinement for Mental-Health Question Answering

Documentation for the ACL ARR 2026 manuscript and its production implementation.
The method refines a complete SFT response when residual risk warrants another
generation, then retains the candidate only when a frozen selector prefers it in
both presentation orders.

## Implementation Status

The integrated code contains the ACL construction/QC pipeline, frozen-data
export, Router and Scorer training, SFT and complete-response refinement, and
corrected TEST60 GPT selection and pointwise evaluation. Their source, original
prompt definitions and recorded production settings have been checked against
the integrated implementation. See [ACL reproducibility](docs/ACL_REPRODUCIBILITY.md)
for the exact inventory and evidence limits.

This establishes source availability for those stages, not a fresh end-to-end
reproduction. Frozen counseling data, model checkpoints and execution manifests
are separate dependencies. The Llama matrix and expert packet-building sources
listed in the reproducibility record are not included in this checkout; some
manuscript evaluation claims still lack a reconciled provenance trail.

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

[ACL reproducibility](docs/ACL_REPRODUCIBILITY.md) records production code,
original prompt locations, model identities, settings, schemas, dependencies,
execution order, evaluation, and evidence limits.

Older CIKM/exp295 notes, pilots, unsuccessful or incomplete experimental
alternatives, and deprecated scripts are retained in the development repository
as **legacy**. Their claims and settings do not define the ACL production
protocol. Shared utilities are production components only where the inventory
identifies their actual use. Later TEST100 and residual-training runs also
remain separate from the manuscript's TEST60.

Saved local manifests support the evidence described in the reproducibility
document; they are not proof that a clean public checkout can reproduce the
entire pipeline. Training, generation, judge calls, and data access require
dependencies that are not bundled here. No new experiment was run for this
documentation update, and the frozen TEST inputs and outputs were not changed.

## Anonymous Review Export

Reviewer-facing documentation consists of this README and
`docs/ACL_REPRODUCIBILITY.md`. Include only the inventoried ACL production
sources and their reviewed helper dependencies in a separately checked export.
Do not mirror the development checkout wholesale.

Exclude Git metadata/history, development PRs, internal audits and verification
records, manuscript ZIP/source archives, Appendix working drafts, legacy notes,
agent/handoff records, personal workstation paths, credentials, raw or derived
data, judge queues/results, completed expert packets, checkpoints, logs and
caches. The existing history and archive remain in development and are not
review artifacts. Production scripts still contain machine-specific defaults
and sibling-worktree dependencies; an anonymous portable export and its actual
mirror access have not been verified. This update does not publish a mirror.

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
