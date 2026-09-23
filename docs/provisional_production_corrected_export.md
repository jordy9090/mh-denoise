# Provisional production corrected export

`scripts/build_provisional_production_corrected_export.py` is the only export
path for the completed 652-row production run until the human review ledger is
available. It never writes into the production directory and refuses to
overwrite an existing output directory.

The older `build_fullpaper_training_data.py --source-kind production` path is
intentionally fail-closed: its historical automatic candidate/realized-axis
conversion has no reviewer evidence scope and therefore cannot produce this
production scorer supervision.

## Required input review ledger

The supplied JSONL has at most one record per accepted `canonical_id`. A
record requires `clean_disposition` (`retained`, `held`, or `unresolved`), a
non-empty `decision_source`, and non-empty `decision_evidence`. An accepted ID
absent from the ledger becomes `unresolved`; it is not retained by default.

`retained` is an explicit approval to use the existing clean response. `held`
and `unresolved` are emitted to separate ID lists and are excluded from SFT,
DPO, router, and scorer files. This is a provisional corrected export, never a
frozen dataset or a declaration that main training is ready.

## Local scorer annotations

Each optional `scorer_annotations` item must bind to an existing exact
`span_supervision` record by side, axis, start/end, text, and source SHA-256.
It also records an annotation source and evidence. Permitted scopes are:

- `local_defect`: only an explicit `label: 1` is accepted.
- `positive_support`: does not imply safety; only an explicit `label: 0` is accepted.
- `holistic` and `omission`: must use `label: null` and do not supervise a local scorer.

All absent annotations remain unknown/masked. The exporter neither transfers a
response-level paired-QC delta to a span nor invents scope for historical
evidence.

## Collision and tokenizer policy

The exporter groups every non-null label by the current scorer serialization
(`question + span`) and output axis. A 0/1 conflict is masked to unknown for
every member and written to `scorer_collision_ledger.jsonl`; any remaining
conflict hard-fails. With an explicitly supplied, already-cached tokenizer,
the same check is repeated on actual truncated token IDs at `--max-length`
(default 512), using `local_files_only=True`. Without that tokenizer the
manifest reports that actual tokenization was not run.

## Output files

The versioned output contains retained/held/unresolved IDs, copied corrected
SFT and DPO contracts by split, router files, scorer span files, the decision
ledger, collision ledger, and a manifest with input/output hashes. It also
records collision counts before filtering, removed by clean exclusion, masked,
and remaining. The source production hashes are checked again before writing
the manifest.
