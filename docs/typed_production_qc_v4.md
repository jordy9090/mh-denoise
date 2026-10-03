# Typed production QC v6

This contract applies only to new production runs. Existing production artifacts and checkpoints remain immutable.

## Acceptance and clean gating

- The blind paired prompt exposes only A/B. It emits the same `response_eligibility` and content-check schema for each side; code maps the hidden clean/candidate roles afterward.
- The mapped clean-side decision is stored as `clean_target_eligibility`.
- `hold` prevents stage acceptance and routes the row to `qc_holds.jsonl`; held clean targets cannot enter SFT, DPO, router, or scorer exports.
- An explicit clean-side `local_defect=1` together with clean `pass` is an unresolved structured contradiction and is routed to hold/review-required.
- Numeric clean scores and keywords in free-form reasons do not decide clean eligibility.
- Deterministic checks are limited to structural meta/identity signals. Unsupported user or family details, an invented counselling relationship or later-session promise, and editorial/meta prose require a complete question-response contextual decision; no canonical-ID or literal-phrase exception is permitted.
- A contextual decision is valid only when it is bound to the exact question and clean-response SHA-256. Missing, stale, held, or unresolved review cannot become a clean pass.

## Response-level evidence versus local scorer labels

- `local_defect` requires an exact own-response span and creates label `1`.
- `local_support` requires an exact own-response span that remains non-defective when read with the complete question and creates label `0`.
- `holistic` remains whole-response evidence. `omission` may point to an exact omitted question span or use `whole_response` with an empty span; neither creates a local label.
- Non-verbatim local evidence remains a validation failure. After the bounded retry cap, it may be quarantined as `non_verbatim_removed_no_local_label`; valid exact evidence in the same judge result remains usable.
- Every schema retry includes the concrete validation error and the previous raw output while keeping the existing attempt cap.
- Scores, score deltas, `realized_axes`, and high clean scores never create local labels by themselves.
- Every local record binds side, character offsets, source SHA-256, axis, scope, label, reason, and the scorer input contract (`complete_question_plus_exact_span`).

## Export gates

- Production files are validated in staging and the success manifest is published last. The adapter requires that manifest, the current QC contract, and matching artifact hashes.
- Scorer conflicts are keyed by the actual serialized question+span and axis, not side, offsets, or canonical ID. Opposing labels are masked to unknown and preserved in a ledger; any remaining non-null conflict hard-fails.
- Identical serialized scorer inputs with the same axis and label contribute once; the deduplication ledger retains every source record and removed contribution.
- The adapter uses a scorer tokenizer distinct from the DPO tokenizer, with the training serialization and max length. Labels whose question or evidence span is truncated are masked and ledgered.
- An all-unknown or single-class pilot can be exported for audit but is explicitly marked supervision-insufficient and not training-ready.
- Structural validity, semantic-review coverage, and explicit dataset-level training approval are separate. Merely observing both binary classes never sets `training_ready=true`.
- The main-data freeze entry point accepts only the new `training_readiness.ready=true` contract. Legacy class-count-only readiness and audit-only pilots hard-fail before any training artifacts are frozen.
- Existing legacy evidence without typed scope is recorded as excluded and is not silently upgraded.

## Small-run verification plan

1. Run a new, versioned 6–12 row pilot covering all six intended axes and both TRAIN/VALID without carry-forward from the old v3 run.
2. Enable the cached local Qwen judge, keep paid API fallback disabled, and use a new output directory and frozen run contract.
3. Verify clean holds never appear in accepted/SFT/DPO output.
4. Inspect every emitted local label for exact offset/hash and confirm the scorer serialization is question plus that span.
5. Report both label classes by TRAIN/VALID and axis, verify holistic/omission counts only in response-level evidence, and run TRAIN/VALID provenance-overlap checks. Missing supervision is reported rather than fabricated.
6. Only after this pilot passes should a bounded new production run be scheduled. This plan does not authorize model execution by itself.
