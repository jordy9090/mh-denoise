#!/usr/bin/env python3
"""Freeze candidate-only regeneration inputs from a completed reuse export."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from fullpaper_acl_pipeline import read_jsonl, sha256_file, write_json


VERSION = "production-candidate-regeneration-selection-v1-20261001"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", required=True)
    parser.add_argument("--reuse-export-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    production = Path(args.production_dir).resolve()
    reuse = Path(args.reuse_export_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / "selection.json"
    manifest_path = output / "selection_manifest.json"
    if selection_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite frozen regeneration selection")

    accepted_rows = list(read_jsonl(production / "accepted.jsonl"))
    accepted = {row["canonical_id"]: row for row in accepted_rows}
    decisions = list(read_jsonl(reuse / "regenerate_candidate.jsonl"))
    if len(decisions) != len({row["canonical_id"] for row in decisions}):
        raise RuntimeError("Regeneration decisions contain duplicate IDs")
    rows: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for decision in decisions:
        canonical_id = decision["canonical_id"]
        if decision.get("clean_assessment") != "suitable":
            raise RuntimeError(f"Candidate regeneration includes non-suitable clean: {canonical_id}")
        source = accepted[canonical_id]
        question_group = source["question_normalized_sha256"]
        if question_group in seen_questions:
            raise RuntimeError(f"Duplicate normalized question in regeneration: {question_group}")
        seen_questions.add(question_group)
        rows.append({
            "canonical_id": canonical_id,
            "split": source["split"],
            "intended_axes": source["intended_axes"],
            "generation_seeds": source.get("generation_seeds", []),
            "generation_objective": "response_pair",
            "immutable_text_sha256": {
                "question": hashlib.sha256(source["question"].encode()).hexdigest(),
                "clean_response": hashlib.sha256(source["clean_response"].encode()).hexdigest(),
                "failed_candidate": hashlib.sha256(source["corrupted_response"].encode()).hexdigest(),
            },
            "regeneration_reason": decision.get("pair_assessment"),
            "decision_source": decision.get("decision_source"),
        })
    payload = {"version": VERSION, "rows": rows}
    write_json(selection_path, payload)
    manifest = {
        "version": VERSION,
        "rows": len(rows),
        "by_split": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "valid")
        },
        "production_accepted_sha256": sha256_file(production / "accepted.jsonl"),
        "reuse_decisions_sha256": sha256_file(reuse / "regenerate_candidate.jsonl"),
        "selection_sha256": sha256_file(selection_path),
        "policy": {
            "candidate_only_regeneration": True,
            "question_clean_split_and_intended_axes_preserved": True,
            "failed_candidate_not_reused_as_training_pair": True,
            "no_replacement_sampling": True,
        },
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
