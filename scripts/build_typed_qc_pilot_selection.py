#!/usr/bin/env python3
"""Build a fixed-seed, held-clean-excluded TRAIN/VALID typed-QC pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from corruption_contract_v2 import AXES
from fullpaper_acl_pipeline import read_jsonl, sha256_file, write_json


VERSION = "typed-qc-pilot-selection-v2-20260926"


def stable_key(seed: int, canonical_id: str) -> str:
    return hashlib.sha256(f"{seed}:{canonical_id}".encode("utf-8")).hexdigest()


def read_id_file(path: Path) -> set[str]:
    if path.suffix == ".jsonl":
        return {row["canonical_id"] for row in read_jsonl(path)}
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(f"ID JSON must be a list or an object with rows: {path}")
        return {row["canonical_id"] for row in rows}
    return {
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-selection", required=True)
    parser.add_argument("--canonical-file", required=True)
    parser.add_argument("--clean-qc-file", required=True)
    parser.add_argument("--known-holds-file", action="append", default=[])
    parser.add_argument("--exclude-ids-file", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--valid-count", type=int, default=4)
    parser.add_argument(
        "--required-source",
        action="append",
        default=[],
        help="Require at least one selected row from this source when an eligible row exists.",
    )
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite pilot selection directory: {output}")
    source_path = Path(args.source_selection).resolve()
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_rows = source_payload.get("rows", source_payload)
    if not isinstance(source_rows, list):
        raise ValueError("Source selection must be a list or an object with rows")
    canonical_path = Path(args.canonical_file).resolve()
    qc_path = Path(args.clean_qc_file).resolve()
    canonical = {row["canonical_id"]: row for row in read_jsonl(canonical_path)}
    qc = {row["canonical_id"]: row for row in read_jsonl(qc_path)}
    hold_paths = [Path(path).resolve() for path in args.known_holds_file]
    exclude_paths = [Path(path).resolve() for path in args.exclude_ids_file]
    held_ids = set().union(*(read_id_file(path) for path in hold_paths)) if hold_paths else set()
    excluded_ids = set().union(*(read_id_file(path) for path in exclude_paths)) if exclude_paths else set()

    eligible: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    for spec in source_rows:
        canonical_id = spec.get("canonical_id")
        row, clean_qc = canonical.get(canonical_id), qc.get(canonical_id)
        if row is None or clean_qc is None:
            rejection_counts["missing_canonical_or_clean_qc"] += 1
            continue
        if canonical_id in held_ids:
            rejection_counts["known_hold"] += 1
            continue
        if canonical_id in excluded_ids:
            rejection_counts["excluded_prior_review_batch"] += 1
            continue
        if (
            not clean_qc.get("qc_ok")
            or not clean_qc.get("eligible")
            or clean_qc.get("baseline_degraded_axes") != []
        ):
            rejection_counts["clean_qc_not_pass"] += 1
            continue
        if row.get("split") not in {"train", "valid"}:
            rejection_counts["non_train_valid"] += 1
            continue
        eligible.append({
            **spec,
            "source": row["source"],
            "source_component": row["source_component"],
        })

    eligible.sort(key=lambda row: stable_key(args.seed, row["canonical_id"]))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_questions: set[str] = set()

    def add(row: dict[str, Any]) -> bool:
        canonical_id = row["canonical_id"]
        question_key = canonical[canonical_id]["question_normalized_sha256"]
        if canonical_id in selected_ids or question_key in selected_questions:
            return False
        selected.append(row)
        selected_ids.add(canonical_id)
        selected_questions.add(question_key)
        return True

    # Cover every intended axis from TRAIN first, then fill fixed split quotas.
    covered: set[str] = set()
    for axis in AXES:
        candidates = [
            row for row in eligible
            if row.get("split") == "train" and axis in row.get("intended_axes", [])
            and row["canonical_id"] not in selected_ids
        ]
        if not candidates or not add(candidates[0]):
            raise RuntimeError(f"Cannot cover pilot axis from eligible TRAIN pool: {axis}")
        covered.update(candidates[0]["intended_axes"])
    for source in args.required_source:
        if any(row.get("source") == source for row in selected):
            continue
        candidate = next((
            row for row in eligible
            if row.get("source") == source
            and sum(item["split"] == row["split"] for item in selected)
            < (args.train_count if row["split"] == "train" else args.valid_count)
            and add(row)
        ), None)
        if candidate is None:
            raise RuntimeError(f"Cannot cover required source in pilot: {source}")
    for split, target in (("train", args.train_count), ("valid", args.valid_count)):
        while sum(row["split"] == split for row in selected) < target:
            candidate = next((row for row in eligible if row["split"] == split and add(row)), None)
            if candidate is None:
                raise RuntimeError(f"Insufficient unique eligible rows for {split} quota")
    if len(selected) != args.train_count + args.valid_count:
        raise AssertionError("Pilot selection count mismatch")
    covered = {axis for row in selected for axis in row.get("intended_axes", [])}
    if covered != set(AXES):
        raise AssertionError(f"Pilot does not cover all axes: {covered}")
    selected.sort(key=lambda row: (0 if row["split"] == "train" else 1, stable_key(args.seed, row["canonical_id"])))

    output.mkdir(parents=True)
    selection_path = output / "selection.json"
    write_json(selection_path, {"rows": selected})
    manifest = {
        "version": VERSION,
        "status": "frozen",
        "seed": args.seed,
        "purpose": "bounded typed-QC execution validation; not a prevalence sample",
        "counts": dict(Counter(row["split"] for row in selected)),
        "axis_counts": dict(Counter(axis for row in selected for axis in row["intended_axes"])),
        "source_counts": dict(Counter(row.get("source") for row in selected)),
        "required_sources": list(args.required_source),
        "canonical_ids": [row["canonical_id"] for row in selected],
        "known_hold_ids_loaded": len(held_ids),
        "prior_review_ids_excluded": len(excluded_ids),
        "pool_rejection_counts": dict(rejection_counts),
        "inputs": {
            "source_selection": {"path": str(source_path), "sha256": sha256_file(source_path)},
            "canonical": {"path": str(canonical_path), "sha256": sha256_file(canonical_path)},
            "clean_qc": {"path": str(qc_path), "sha256": sha256_file(qc_path)},
            "known_holds": [{"path": str(path), "sha256": sha256_file(path)} for path in hold_paths],
            "excluded_prior_review_batches": [
                {"path": str(path), "sha256": sha256_file(path)} for path in exclude_paths
            ],
        },
        "selection_sha256": sha256_file(selection_path),
        "test_split_used": False,
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
