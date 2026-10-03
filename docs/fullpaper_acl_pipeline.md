# ACL/ARR full-paper clean-QA and controlled-corruption pipeline

This pipeline is separate from the legacy exp295 lineage. It does not train an
SFT, DPO, proposed model, or refiner.

## Exact commands

Run from `/home/user/mh-denoise`.

Build and freeze the canonical clean-QA split:

```bash
python scripts/fullpaper_acl_pipeline.py build-clean \
  --seed 20260904 \
  --output-dir data/fullpaper_acl_pipeline
```

Create exactly one globally balanced corruption specification per frozen row:

```bash
python scripts/fullpaper_acl_pipeline.py assign-corruptions \
  --seed 20260904 \
  --canonical data/fullpaper_acl_pipeline/canonical_clean_qa.jsonl \
  --split-manifest data/fullpaper_acl_pipeline/split_manifest.json \
  --output-dir data/fullpaper_acl_pipeline
```

Create the isolated pilot runtime without changing the repository's base Python
environment:

```bash
python -m venv --system-site-packages .venv-fullpaper
.venv-fullpaper/bin/python -m pip install --upgrade transformers==5.16.1
```

Preview the deterministic stratified pilot selection without loading a model:

```bash
.venv-fullpaper/bin/python scripts/run_gemma_corruption_pilot.py --select-only
```

Run the 50-row Gemma generation and separate Gemma QC pass from the pinned local
snapshot only:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-fullpaper/bin/python \
  scripts/run_gemma_corruption_pilot.py \
  --pilot-size 50 \
  --batch-size 5 \
  --max-attempts 3 \
  --seed 20260904 \
  --model-snapshot /home/user/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2
```

Run the data-contract suite:

```bash
python -m pytest -q data_contract_tests
```

Preview the deterministic balanced six-axis v2 selection without loading the model:

```bash
.venv-fullpaper/bin/python scripts/run_gemma_corruption_pilot_v2.py --select-only
```

Run only the guarded 100-row sequential v2 pilot from the pinned snapshot:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-fullpaper/bin/python \
  scripts/run_gemma_corruption_pilot_v2.py \
  --pilot-size 100 \
  --batch-size 5 \
  --max-attempts 3 \
  --seed 20260906 \
  --model-snapshot /home/user/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2
```

## Design constraints

- Immediate inputs are the Phase 1 pinned MentalChat16K and CounselChat files.
- Psych8k and PsyQA exist only as explicit optional adapters; absent or gated
  optional files do not affect the default build.
- CounselChat exclusion operates on whole `questionID` groups before blank-answer
  filtering. Direct IDs and exact/normalized benchmark-question links are united.
- Duplicate components unite exact/normalized questions, exact/normalized clean
  responses, and source groups before the 80/10/10 split.
- MentalChat16K has no released transcript/session IDs. The immutable component
  row is the documented source-group fallback; duplicate linkage still operates
  globally across both immediate datasets.
- Assignment uses five atomic dimensions. CounselBench Overall is evaluation-only.
- QC is a separate model pass. The pilot records scores and realized dimensions
  without setting final pass thresholds. A later thresholds JSON can activate
  retry/pass/drop decisions without changing generation code.

The frozen canonical and legacy five-axis assignment remain unchanged. The v2
pilot is a separate six-axis experiment in which `overall_quality` is an
explicit corruption axis. Multi-axis corruptions are added one axis at a time;
each stage has an independent judge call, and all axes accumulated so far must
be evidenced before the stage is accepted. Up to three attempts are allowed.

## Current frozen run

- Canonical rows: 18,224 (`MentalChat16K`: 16,057; `CounselChat`: 2,167).
- Splits: train 14,578; valid 1,823; test 1,823.
- CounselBench exclusion: 100 directly linked question groups / 588 response
  rows, plus one additional normalized linked group (`questionID=406`) / one
  response row; 101 groups and 589 responses excluded in total.
- Axis counts: 9,112 one-axis; 6,378 two-axis; 2,734 three-axis.
- Global assignment balance: marginal max-minus-min 3; pairwise max-minus-min 3.
- Pilot: 50 retained, zero dropped; 49 passed structural generation checks on
  attempt 1 and one unchanged response was regenerated successfully on attempt 2.
- The unthresholded QC pilot found 30/50 rows missing at least one intended
  realized dimension. In particular, factual-consistency realization was 0/17.
  This is a calibration finding and blocks a full corruption run until automated
  prompt/QC calibration is completed. Experts remain reserved for held-out evaluation.

## Six-axis sequential v2 pilot

- Selection: 100 rows (`MentalChat16K`: 88; `CounselChat`: 12), with train/valid/test
  counts 80/10/10.
- Requested axis counts: 50 one-axis, 35 two-axis, 15 three-axis.
- Six-axis marginals are 27–28 each; pairwise co-occurrence max-minus-min is one.
- Result: 43 accepted and 57 automatically dropped after the per-stage retry cap.
- Full realization: 30/50 one-axis, 11/35 two-axis, and 2/15 three-axis.
- Factual consistency: 12/28 final accepted, improving on v1's 0/17.
- Overall quality: 0/28 final accepted; this is a hard production blocker.
- Additional realized-axis rate: 33/200 = 16.5% among accepted rows' unselected slots.
- No full corruption or model training was started. See
  `data/fullpaper_acl_pipeline/qc_pilot_100_v2_report.md` for calibration details.

## Paired replacement-generator diagnostic

The guarded Gemma/Qwen diagnostic is implemented in
`scripts/run_paired_generator_diagnostic.py`. It uses 24 TRAIN-only rows with
12/8/4 one-/two-/three-axis assignments and a pinned common external
`gpt-4.1-2025-04-14` judge. It has no self-QC fallback. Current pre-generation
status and the exact resume command are in
`data/fullpaper_acl_pipeline/paired_generator_diagnostic_24/pre_generation_audit.md`.

## dev120 audit repair

The original 120-input development shard is immutable. The reviewed export is
written separately under
`data/fullpaper_acl_pipeline/development_training_shard_120_corrected_v2/`.
It retains 57 of the 70 originally accepted pairs and holds 13: the ten primary
audit rows plus three separately recorded secondary integrity flags. The other
43 rejected and seven QC-conflict inputs are unchanged, so the accounting remains
120. Raw paired judge scores and evidence are copied unchanged. Masking/span
supervision is emitted only when a literal source substring and exact character
offset can be verified.

Future clean-target and candidate checks use the versioned contract in
`scripts/source_integrity_contract.py`. It narrowly covers non-response meta
commentary, false human/professional identity, question-unsupported specific
history/diagnosis/treatment history, and candidate speaker switching. Ordinary
career or relationship directiveness alone is explicitly outside the medical
boundary definition. Configuration manifests record the contract hash.

Reproduce the separate repair export without model or API calls:

```bash
.venv-fullpaper/bin/python scripts/repair_dev120_audit.py
```
