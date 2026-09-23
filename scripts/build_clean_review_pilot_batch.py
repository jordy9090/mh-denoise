#!/usr/bin/env python3
"""Select a deterministic 50-ID Codex clean/candidate review work batch."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SEED = 20260923
VERSION = "codex-clean-review-pilot-batch-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_key(canonical_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{canonical_id}".encode()).hexdigest()


def qc_warning(row: dict[str, Any]) -> bool:
    scores = row["final_grade"]["clean_scores"]
    return bool(
        scores["factual_consistency"]["score"] <= 2
        or scores["overall_quality"]["score"] <= 2
        or scores["empathy"]["score"] == 1
        or scores["toxicity_or_harm"]["score"] >= 3
        or scores["specificity"]["score"] == 1
        or scores["medical_boundary"]["violation"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", required=True)
    parser.add_argument("--reviewed-clean-cases", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    production, output = Path(args.production_dir).resolve(), Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    accepted = read_jsonl(production / "accepted.jsonl")
    by_id = {row["canonical_id"]: row for row in accepted}
    reviewed = read_jsonl(Path(args.reviewed_clean_cases).resolve())
    required = sorted(
        row["canonical_id"] for row in reviewed
        if row["recommendation"] == "no_obvious_integrity_defect_in_this_read"
    )
    if len(required) != 16 or not set(required) <= set(by_id):
        raise RuntimeError("Expected the 16 limited-integrity-review IDs in production accepted")
    prior_holds = {row["canonical_id"] for row in reviewed if row["recommendation"] == "hold_clean_integrity"}
    pool = [row for row in accepted if row["canonical_id"] not in set(required) | prior_holds]
    strata: dict[tuple[str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        strata[(row["source"], row["split"], qc_warning(row))].append(row)
    for rows in strata.values():
        rows.sort(key=lambda row: stable_key(row["canonical_id"]))

    # One from every non-empty source/split/warning stratum, then fill by the
    # globally stable key.  This is a workflow-validation batch, not a
    # prevalence sample and no sampling weights are implied.
    sampled: list[dict[str, Any]] = []
    for key in sorted(strata, key=str):
        if strata[key]:
            sampled.append(strata[key].pop(0))
    remaining = sorted((row for rows in strata.values() for row in rows), key=lambda row: stable_key(row["canonical_id"]))
    sampled.extend(remaining[: 34 - len(sampled)])
    if len(sampled) != 34 or len({row["canonical_id"] for row in sampled}) != 34:
        raise RuntimeError("Failed to select 34 unique unresolved rows")

    ordered_ids = required + [row["canonical_id"] for row in sampled]
    sft_by_id = {}
    for split in ("train", "valid"):
        for row in read_jsonl(production / f"{split}_sft.jsonl"):
            sft_by_id[row["metadata"]["canonical_id"]] = row
    work = []
    for index, canonical_id in enumerate(ordered_ids):
        row, sft = by_id[canonical_id], sft_by_id[canonical_id]
        work.append({
            "batch_index": index,
            "canonical_id": canonical_id,
            "selection_role": "required_limited_integrity_recheck" if canonical_id in required else "seeded_unresolved_sample",
            "selection_seed": SEED,
            "selection_stratum": {"source": row["source"], "split": row["split"], "qc_warning": qc_warning(row)},
            "source": row["source"], "split": row["split"], "question": row["question"],
            "clean_response": row["clean_response"], "candidate_response": row["corrupted_response"],
            "intended_axes": row["intended_axes"], "realized_axes": row["realized_axes"],
            "paired_qc": row["final_grade"], "span_supervision": sft["metadata"].get("span_supervision", []),
        })
    write_jsonl(output / "review_batch.jsonl", work)
    manifest = {
        "version": VERSION, "status": "selected_not_reviewed", "seed": SEED,
        "purpose": "workflow and labeling procedure validation; not a prevalence sample",
        "selection": "all 16 limited-integrity reads plus one per non-empty source/split/QC-warning stratum, then SHA256(seed:canonical_id) order to 34",
        "counts": {"required": 16, "seeded_unresolved": 34, "total": 50},
        "strata": {"|".join(map(str, key)): count for key, count in Counter((row["source"], row["split"], qc_warning(row)) for row in sampled).items()},
        "canonical_ids": ordered_ids,
    }
    (output / "selection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
