# Typed production QC v4

This contract applies only to new production runs. Existing production artifacts and checkpoints remain immutable.

## Acceptance and clean gating

- The paired judge emits an independent `clean_target_eligibility` decision.
- `hold` prevents stage acceptance and routes the row to `qc_holds.jsonl`; held clean targets cannot enter SFT, DPO, router, or scorer exports.
- Numeric clean scores and keywords in free-form reasons do not decide clean eligibility.
- Deterministic checks are limited to structural meta/identity failures. Unsupported user history is a semantic question-response comparison made through the structured QC field, not a phrase or canonical-ID exception.

## Response-level evidence versus local scorer labels

- `local_defect` requires an exact own-response span and creates label `1`.
- `local_support` requires an exact own-response span that remains non-defective when read with the complete question and creates label `0`.
- `holistic` and `omission` remain in `response_level_qc_evidence.jsonl` and never create local labels.
- Scores, score deltas, `realized_axes`, and high clean scores never create local labels by themselves.
- Every local record binds side, character offsets, source SHA-256, axis, scope, label, reason, and the scorer input contract (`complete_question_plus_exact_span`).

## Export gates

- A typed production export with accepted rows must contain at least one exact local positive and one exact local negative.
- The production training adapter reads only the explicit scope/label pair. An all-unknown or empty local scorer export hard-fails.
- Existing legacy evidence without typed scope is recorded as excluded and is not silently upgraded.

## Small-run verification plan

1. Run a new, versioned 6–12 row pilot covering all six intended axes and both TRAIN/VALID without carry-forward from the old v3 run.
2. Keep the local Qwen judge and paid API fallback disabled; use a new output directory and frozen run contract.
3. Verify clean holds never appear in accepted/SFT/DPO output.
4. Inspect every emitted local label for exact offset/hash and confirm the scorer serialization is question plus that span.
5. Require both label classes, verify holistic/omission counts only in response-level evidence, and run TRAIN/VALID provenance-overlap checks.
6. Only after this pilot passes should a bounded new production run be scheduled. This plan does not authorize model execution by itself.
