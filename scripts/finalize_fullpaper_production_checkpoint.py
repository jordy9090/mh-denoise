#!/usr/bin/env python3
"""Export a preserved single-worker checkpoint after a bounded run stops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fullpaper_acl_pipeline import read_jsonl
from run_fullpaper_corruption_production import export


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-inputs", type=int, required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    checkpoint = output / "results_checkpoint.part0.jsonl"
    contract_path = output / "run_contract.json"
    status_path = output / "production_manifest.json"
    if not checkpoint.exists() or not contract_path.exists() or not status_path.exists():
        raise RuntimeError("Checkpoint, run contract, and production status are required")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") not in {"budget_exhausted", "worker_complete", "complete"}:
        raise RuntimeError(f"Run is not terminal: {status.get('status')}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    fingerprint = contract["run_fingerprint"]
    rows = list(read_jsonl(checkpoint))
    if any(row.get("run_fingerprint") != fingerprint for row in rows):
        raise RuntimeError("Checkpoint/run fingerprint mismatch")
    if len(rows) > args.expected_inputs:
        raise RuntimeError("Checkpoint exceeds the approved input count")
    materialization = {
        **status,
        "processing_terminal_status": status.get("status"),
        "approved_input_count": args.expected_inputs,
        "unprocessed_input_count": args.expected_inputs - len(rows),
        "partial_export": len(rows) != args.expected_inputs,
    }
    export(output, rows, materialization)


if __name__ == "__main__":
    main()
