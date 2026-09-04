# Full-paper dataset pipeline: Phase 1

Phase 1 is limited to provenance inventory, integrity fingerprints, confirmed
exact/normalized contamination links, and review-only near/semantic candidates.
It does not create corruptions, train/validation/test splits, training examples,
or a final dataset-inclusion decision.

## Inputs

- CounselBench-Eval: the pinned local Arrow snapshot at revision
  `8d56a96ea1de3f3f190f77f4ca9bc3503d731af7`.
- CounselChat: `nbertagnolli/counsel-chat` revision
  `17501f72697cf8018aaf496162e9dc1408a64e67`.
- MentalChat16K: `ShenLab/MentalChat16K` revision
  `5f60cd380cfc58f0f12d44892bed41ee3670a70a`.
- Psych8k: `EmoCareAI/Psych8k` revision
  `091787feccbce3e0adfd03b1ea3063f3d938c32d`; raw access remains blocked by
  the manual Hugging Face gate.
- PsyQA: official `thu-coai/PsyQA` revision
  `e224c7e518c98a0c3df11e2fc5e6698044d8e156`; only the official 100-question
  sample is public, while full access requires the signed user agreement.

Raw source files are retained unchanged under `data/fullpaper_phase1/raw_sources`.
Generated Phase 1 manifests are written under `data/fullpaper_phase1/manifests`.

## Rebuild

Run from the repository root in an environment containing the repository's
existing `pyarrow`, `scikit-learn`, `torch`, and `transformers` dependencies:

```bash
python scripts/prepare_fullpaper_phase1.py
```

The script uses the already-local BGE model with `local_files_only=True`.  It
does not download a model.  If semantic matching needs to be deferred, run:

```bash
python scripts/prepare_fullpaper_phase1.py --skip_semantic
```

## Outputs

- `exp295_integrity_freeze.json`: branch/commit, artifact paths, configs,
  weights, results, hashes, and the exact judge prompt/rubric/settings.
- `candidate_intake_inventory.json`: source revision, license, raw schema/count,
  identifiers, grouping metadata, single-turn suitability, and blockers.
- `counselbench100_exclusion_manifest.jsonl`: one canonical record per official
  benchmark question, including confirmed direct/exact/normalized links.
- `exact_normalized_overlap_report.json`: pairwise question and response overlap.
- `manual_review_queue.jsonl`: lexical and semantic candidates only; no queue
  item is an automatic exclusion.
- `phase1_manifest.json`: hashes and row counts for the generated Phase 1 files.

The exp295 namespace remains a CIKM/legacy baseline-check lineage and is never
an input to the full-paper main dataset.
