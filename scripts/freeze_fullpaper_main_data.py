#!/usr/bin/env python3
"""Merge, convert, validate, and hash the completed bounded production run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "fullpaper-main-data-freeze-v2-20260926"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, env={**os.environ, "PYTHONPATH": "scripts"})


def require_training_ready(manifest: dict[str, Any]) -> None:
    """Refuse legacy/class-count-only or semantically unapproved exports."""
    readiness = manifest.get("training_readiness")
    if not isinstance(readiness, dict) or readiness.get("ready") is not True:
        blockers = readiness.get("blocking_reasons", []) if isinstance(readiness, dict) else []
        raise RuntimeError(
            "Training export is audit-only or lacks explicit semantic training approval; "
            f"blocking_reasons={blockers!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", required=True)
    parser.add_argument("--corrected-dir", required=True)
    parser.add_argument("--merged-raw-dir", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--canonical-file", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--expected-inputs", type=int, default=1200)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    production = Path(args.production_dir).resolve()
    corrected = Path(args.corrected_dir).resolve()
    merged = Path(args.merged_raw_dir).resolve()
    training = Path(args.training_dir).resolve()
    status = json.loads((production / "production_manifest.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete" or status.get("processed_inputs") != args.expected_inputs:
        raise RuntimeError("Production is not a complete exact-size bounded run")

    if not merged.exists():
        run([
            sys.executable, "scripts/merge_fullpaper_training_sources.py",
            "--production-dir", str(production),
            "--corrected-dir", str(corrected),
            "--output-dir", str(merged),
        ], repo)
    if not training.exists():
        run([
            sys.executable, "scripts/build_fullpaper_training_data.py",
            "--source-kind", "production",
            "--source-dir", str(merged),
            "--output-dir", str(training),
            "--canonical-file", str(Path(args.canonical_file).resolve()),
            "--tokenizer-dir", str(Path(args.tokenizer_dir).resolve()),
            "--development-eval-size", "0",
        ], repo)

    required = [
        "sft_train.jsonl", "sft_valid.jsonl", "dpo_train.jsonl", "dpo_valid.jsonl",
        "router_train.jsonl", "router_valid.jsonl",
        "scorer_train_verified_spans.jsonl", "scorer_valid_verified_spans.jsonl",
        "manifest.json",
    ]
    missing = [name for name in required if not (training / name).exists()]
    if missing:
        raise RuntimeError(f"Missing converted training artifacts: {missing}")
    training_manifest = json.loads((training / "manifest.json").read_text(encoding="utf-8"))
    require_training_ready(training_manifest)
    train = read_jsonl(training / "sft_train.jsonl")
    valid = read_jsonl(training / "sft_valid.jsonl")
    train_questions = {row["question_normalized_sha256"] for row in train}
    valid_questions = {row["question_normalized_sha256"] for row in valid}
    train_clusters = {row["duplicate_cluster_id"] for row in train}
    valid_clusters = {row["duplicate_cluster_id"] for row in valid}
    train_groups = {row["source_group_id"] for row in train}
    valid_groups = {row["source_group_id"] for row in valid}
    if train_questions & valid_questions or train_clusters & valid_clusters or train_groups & valid_groups:
        raise RuntimeError("Frozen TRAIN/VALID leakage check failed")
    if len(train_questions) != len(train) or len(valid_questions) != len(valid):
        raise RuntimeError("More than one final response remains for a normalized question")

    cases = []
    used_axes: set[str] = set()
    for row in train:
        candidate_spans = [
            span for span in row.get("verified_spans", [])
            if span.get("side") == "candidate"
            and span.get("axis") in row.get("realized_axes", [])
        ]
        if not candidate_spans:
            continue
        axes = set(row.get("intended_axes", []))
        if cases and axes <= used_axes:
            continue
        cases.append({
            "status": "author_review_candidate_not_expert_annotated",
            "canonical_id": row["canonical_id"],
            "source": row["source"],
            "question": row["question"],
            "clean_reference": row["safe_response"],
            "corrupted_response": row["unsafe_response"],
            "intended_axes": row["intended_axes"],
            "realized_axes": row["realized_axes"],
            "source_verified_candidate_evidence": candidate_spans,
        })
        used_axes.update(axes)
        if len(cases) == 2:
            break
    case_path = training / "case_study_author_review_candidates.jsonl"
    with case_path.open("w", encoding="utf-8") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    files = {name: {"sha256": sha256(training / name), "rows": len(read_jsonl(training / name))}
             for name in required if name.endswith(".jsonl")}
    files["manifest.json"] = {"sha256": sha256(training / "manifest.json")}
    files[case_path.name] = {"sha256": sha256(case_path), "rows": len(cases)}
    manifest = {
        "version": VERSION,
        "status": "frozen_complete",
        "production_manifest_sha256": sha256(production / "production_manifest.json"),
        "production_run_fingerprint": status["run_fingerprint"],
        "production_processed_inputs": status["processed_inputs"],
        "production_terminal_counts": {
            key: status.get(key, 0) for key in ("accepted", "rejected", "qc_conflicts", "qc_holds")
        },
        "merged_manifest_sha256": sha256(merged / "manifest.json"),
        "counts": {
            "train_pairs": len(train), "valid_pairs": len(valid),
            "train_by_source": dict(Counter(row["source"] for row in train)),
            "valid_by_source": dict(Counter(row["source"] for row in valid)),
        },
        "checks": {
            "one_response_per_normalized_question": True,
            "train_valid_question_overlap": 0,
            "train_valid_duplicate_cluster_overlap": 0,
            "train_valid_source_group_overlap": 0,
            "source_span_offsets_transferred_to_future_sft_outputs": False,
        },
        "files": files,
    }
    (training / "frozen_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
