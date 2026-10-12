#!/usr/bin/env python3
"""Record the immutable first-five TEST production progress and partial-15 ETA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_PRODUCTION = Path(
    "data/fullpaper_acl_pipeline/frozen_revision2_test_corruption_v1_20261008"
)
DEFAULT_SELECTION = Path(
    "data/fullpaper_acl_pipeline/frozen_revision2_test_selection_v1_20261008/"
    "test_generation_selection.json"
)
DEFAULT_OUTPUT = Path(
    "data/fullpaper_acl_pipeline/frozen_revision2_test_partial15_v1_20261008/"
    "first5_progress.json"
)
DEFAULT_RUN = Path(
    "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".tmp.", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", default=str(DEFAULT_PRODUCTION))
    parser.add_argument("--selection-file", default=str(DEFAULT_SELECTION))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--first-terminal-count", type=int, default=5)
    parser.add_argument("--target-accepted", type=int, default=15)
    parser.add_argument("--responses-per-pair", type=int, default=3)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    args = parser.parse_args()

    production = Path(args.production_dir).resolve()
    selection_path = Path(args.selection_file).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("status") != "complete_immutable_first_terminal_snapshot":
            raise RuntimeError(f"Existing progress snapshot is not complete: {output}")
        if existing.get("selection", {}).get("sha256") != sha256(selection_path):
            raise RuntimeError("Existing first-five snapshot belongs to another selection")
        print(json.dumps(existing, ensure_ascii=False, indent=2))
        return

    if args.first_terminal_count < 1 or args.target_accepted < 1:
        raise ValueError("Progress and accepted thresholds must be positive")
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_rows = selection_payload["rows"] if isinstance(selection_payload, dict) else selection_payload
    selection_ids = [str(row["canonical_id"]) for row in selection_rows]
    checkpoints = sorted(production.glob("results_checkpoint.part*.jsonl"))
    if not checkpoints:
        raise RuntimeError("No production checkpoint exists yet")
    terminal_rows = [row for path in checkpoints for row in read_jsonl(path)]
    terminal_by_id = {str(row["canonical_id"]): row for row in terminal_rows}
    if len(terminal_by_id) != len(terminal_rows):
        raise RuntimeError("Production checkpoints contain duplicate canonical IDs")
    terminal_ids_in_selection_order = [item for item in selection_ids if item in terminal_by_id]
    if terminal_ids_in_selection_order != [str(row["canonical_id"]) for row in terminal_rows]:
        raise RuntimeError("Single-worker checkpoint is not the frozen-selection prefix")
    if len(terminal_rows) < args.first_terminal_count:
        raise RuntimeError(
            f"Need {args.first_terminal_count} terminal rows, found {len(terminal_rows)}"
        )

    first_rows = terminal_rows[: args.first_terminal_count]
    accepted = sum(row.get("status") == "accepted" for row in first_rows)
    statuses = Counter(str(row.get("status") or "missing") for row in first_rows)
    row_wall_seconds = sum(float((row.get("usage") or {}).get("wall_seconds") or 0) for row in first_rows)
    generator_seconds = sum(
        float((row.get("usage") or {}).get("generator_seconds") or 0) for row in first_rows
    )
    judge_seconds = sum(
        float((row.get("usage") or {}).get("judge_seconds") or 0) for row in first_rows
    )
    mean_row_seconds = row_wall_seconds / len(first_rows)
    projected_total_terminal = (
        math.ceil(args.target_accepted * len(first_rows) / accepted) if accepted else None
    )
    projected_remaining_production_seconds = (
        max(0, projected_total_terminal - len(first_rows)) * mean_row_seconds
        if projected_total_terminal is not None else None
    )

    run_dir = Path(args.run_dir).resolve()
    historical_paths = {
        "sft": run_dir / "gemma/generation_1536/sft_valid_outputs.manifest.json",
        "dpo": run_dir / "gemma/generation_1536/dpo_valid_outputs.manifest.json",
        "mask_off": run_dir / "gemma/generation_1536/proposed_mask_off_valid_outputs.manifest.json",
    }
    historical: dict[str, Any] = {}
    scaled_inference_seconds = 0.0
    inference_estimate_available = True
    for method, path in historical_paths.items():
        if not path.exists():
            historical[method] = {"path": str(path), "available": False}
            inference_estimate_available = False
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        rows = int((manifest.get("input") or {}).get("rows") or 0)
        runtime = float(manifest.get("runtime_seconds") or 0)
        if rows <= 0 or runtime <= 0:
            inference_estimate_available = False
            historical[method] = {"path": str(path), "available": False}
            continue
        scaled = runtime * args.target_accepted / rows
        scaled_inference_seconds += scaled
        historical[method] = {
            "path": str(path),
            "sha256": sha256(path),
            "historical_rows": rows,
            "historical_runtime_seconds": runtime,
            "linear_scaled_seconds_for_15": scaled,
            "available": True,
        }

    total_eta_seconds = None
    if projected_remaining_production_seconds is not None and inference_estimate_available:
        total_eta_seconds = projected_remaining_production_seconds + scaled_inference_seconds
    captured_at = datetime.now(timezone.utc)
    run_contract = production / "run_contract.json"
    payload = {
        "status": "complete_immutable_first_terminal_snapshot",
        "captured_at_utc": captured_at.isoformat(),
        "goal": {
            "questions": args.target_accepted,
            "responses_per_question": args.responses_per_pair,
            "responses": args.target_accepted * args.responses_per_pair,
        },
        "first_terminal_rows": {
            "count": len(first_rows),
            "canonical_ids": [row["canonical_id"] for row in first_rows],
            "status_counts": dict(sorted(statuses.items())),
            "accepted": accepted,
            "acceptance_rate": accepted / len(first_rows),
            "measured_sum_row_wall_seconds": row_wall_seconds,
            "measured_mean_row_wall_seconds": mean_row_seconds,
            "measured_sum_generator_seconds": generator_seconds,
            "measured_sum_judge_seconds": judge_seconds,
            "note": (
                "Row wall_seconds is the saved per-row model/QC elapsed time; model-load and GPU-wait "
                "time is not stored per row and is not presented as measured processing time."
            ),
        },
        "eta": {
            "method": (
                "first-five observed acceptance rate and saved per-row wall time, plus a linear "
                "15/43 scaling of the three frozen VALID inference runtimes"
            ),
            "projected_total_terminal_rows_for_15_accepted": projected_total_terminal,
            "projected_remaining_production_seconds_after_first_five": projected_remaining_production_seconds,
            "historical_inference_seconds_scaled_for_45_responses": (
                scaled_inference_seconds if inference_estimate_available else None
            ),
            "estimated_seconds_until_45_responses_ready": total_eta_seconds,
            "estimated_ready_at_utc": (
                (captured_at + timedelta(seconds=total_eta_seconds)).isoformat()
                if total_eta_seconds is not None else None
            ),
            "estimate_not_guarantee": True,
        },
        "historical_inference_manifests": historical,
        "selection": {
            "path": str(selection_path),
            "sha256": sha256(selection_path),
            "rows": len(selection_rows),
        },
        "production": {
            "directory": str(production),
            "run_contract": (
                {"path": str(run_contract), "sha256": sha256(run_contract)}
                if run_contract.exists() else None
            ),
            "checkpoint_files": [
                {"path": str(path), "sha256_at_snapshot": sha256(path)} for path in checkpoints
            ],
            "terminal_rows_available_at_snapshot": len(terminal_rows),
        },
    }
    atomic_write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
