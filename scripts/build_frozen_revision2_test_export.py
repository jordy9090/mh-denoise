#!/usr/bin/env python3
"""Freeze accepted canonical TEST corruption pairs for evaluation only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fullpaper_acl_pipeline import read_jsonl, sha256_file, write_json, write_jsonl


VERSION = "frozen-revision2-test-pair-export-v1-20261008"
DEFAULT_PRODUCTION = Path(
    "data/fullpaper_acl_pipeline/frozen_revision2_test_corruption_v1_20261008"
)
DEFAULT_SELECTION = Path(
    "data/fullpaper_acl_pipeline/frozen_revision2_test_selection_v1_20261008"
)
DEFAULT_FROZEN_TRAIN_VALID = Path(
    "data/fullpaper_acl_pipeline/production_accepted652_reuse_export_reviewed_v2_20261004"
)
DEFAULT_OUTPUT = Path(
    "data/fullpaper_acl_pipeline/frozen_revision2_test_pairs_v1_20261008"
)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unique_values(rows: list[dict[str, Any]], field: str) -> set[str]:
    return {str(row[field]) for row in rows if row.get(field) is not None}


def failure_reason(row: dict[str, Any]) -> str:
    return str(
        row.get("failure_reason")
        or row.get("conflict_reason")
        or row.get("hold_reason")
        or row.get("status")
        or "unknown"
    )


def pair_row(
    row: dict[str, Any],
    spans: list[dict[str, Any]],
    production_dir: Path,
    run_fingerprint: str,
) -> dict[str, Any]:
    question = str(row["question"])
    unsafe = str(row["corrupted_response"])
    safe = str(row["clean_response"])
    metadata = {
        "canonical_id": row["canonical_id"],
        "split": "test",
        "source": row.get("source"),
        "source_component": row.get("source_component"),
        "question_normalized_sha256": row["question_normalized_sha256"],
        "duplicate_cluster_id": row["duplicate_cluster_id"],
        "source_group_id": row["source_group_id"],
        "intended_axes": row.get("intended_axes", []),
        "realized_axes": row.get("realized_axes", []),
        "unintended_axes": row.get("unintended_axes", []),
        "axis_count": row.get("axis_count"),
        "generation_objective": row.get("generation_objective"),
        "production_run_fingerprint": run_fingerprint,
        "production_source_path": str((production_dir / "accepted.jsonl").resolve()),
        "question_sha256": text_sha256(question),
        "unsafe_response_sha256": text_sha256(unsafe),
        "safe_response_sha256": text_sha256(safe),
        "generator": {
            "repo": row.get("generator_repo"),
            "revision": row.get("generator_revision"),
            "prompt_version": row.get("generator_prompt_version"),
        },
        "qc": {
            "model": row.get("judge_model"),
            "repo": row.get("judge_repo"),
            "revision": row.get("judge_revision"),
            "prompt_version": row.get("judge_prompt_version"),
        },
    }
    return {
        "canonical_id": row["canonical_id"],
        "id": row["canonical_id"],
        "split": "test",
        "source": row.get("source"),
        "source_component": row.get("source_component"),
        "question_normalized_sha256": row["question_normalized_sha256"],
        "duplicate_cluster_id": row["duplicate_cluster_id"],
        "source_group_id": row["source_group_id"],
        "question": question,
        "unsafe_response": unsafe,
        "safe_response": safe,
        "intended_axes": row.get("intended_axes", []),
        "realized_axes": row.get("realized_axes", []),
        "unintended_axes": row.get("unintended_axes", []),
        "verified_spans": spans,
        "generation_provenance": {
            "generation_seeds": row.get("generation_seeds", []),
            "stage_history": row.get("stage_history", []),
            "usage": row.get("usage", {}),
        },
        "qc_provenance": {
            "automatic_clean_preflight": row.get("automatic_clean_preflight"),
            "final_grade": row.get("final_grade"),
            "infrastructure_failures": row.get("infrastructure_failures", []),
        },
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", default=str(DEFAULT_PRODUCTION))
    parser.add_argument("--selection-dir", default=str(DEFAULT_SELECTION))
    parser.add_argument("--frozen-train-valid-dir", default=str(DEFAULT_FROZEN_TRAIN_VALID))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    production = Path(args.production_dir).resolve()
    selection = Path(args.selection_dir).resolve()
    frozen = Path(args.frozen_train_valid_dir).resolve()
    output = Path(args.output_dir).resolve()

    production_manifest_path = production / "production_manifest.json"
    production_manifest = json.loads(production_manifest_path.read_text(encoding="utf-8"))
    if production_manifest.get("status") != "complete":
        raise RuntimeError("TEST corruption production is not complete")
    if production_manifest.get("allowed_splits") != ["test"]:
        raise RuntimeError("Production manifest is not a TEST-only run")

    selection_payload = json.loads(
        (selection / "test_generation_selection.json").read_text(encoding="utf-8")
    )
    selected_ids = [str(row["canonical_id"]) for row in selection_payload["rows"]]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("Selection contains duplicate canonical IDs")

    artifact_names = ("accepted.jsonl", "rejected.jsonl", "qc_conflicts.jsonl", "qc_holds.jsonl")
    artifacts = {name: list(read_jsonl(production / name)) for name in artifact_names}
    terminal_rows = [row for name in artifact_names for row in artifacts[name]]
    terminal_by_id = {str(row["canonical_id"]): row for row in terminal_rows}
    if len(terminal_by_id) != len(terminal_rows):
        raise RuntimeError("Terminal production artifacts contain duplicate canonical IDs")
    if set(terminal_by_id) != set(selected_ids):
        raise RuntimeError("Terminal production partition does not match frozen TEST selection")
    if any(row.get("split") != "test" for row in terminal_rows):
        raise RuntimeError("A terminal production row lost canonical split=test")

    accepted = artifacts["accepted.jsonl"]
    spans_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for span in read_jsonl(production / "source_verified_spans.jsonl"):
        spans_by_id[str(span["canonical_id"])].append(span)
    pairs = [
        pair_row(
            row,
            spans_by_id.get(str(row["canonical_id"]), []),
            production,
            str(production_manifest["run_fingerprint"]),
        )
        for row in accepted
    ]
    if len({row["canonical_id"] for row in pairs}) != len(pairs):
        raise RuntimeError("Accepted TEST pairs contain duplicate canonical IDs")
    if len({row["question_normalized_sha256"] for row in pairs}) != len(pairs):
        raise RuntimeError("Accepted TEST pairs contain duplicate normalized questions")

    frozen_rows = [
        row
        for name in ("sft_train.jsonl", "sft_valid.jsonl")
        for row in read_jsonl(frozen / name)
    ]
    overlap = {
        field: len(unique_values(pairs, field) & unique_values(frozen_rows, field))
        for field in (
            "canonical_id",
            "question_normalized_sha256",
            "duplicate_cluster_id",
            "source_group_id",
        )
    }
    if any(overlap.values()):
        raise RuntimeError(f"Accepted TEST export overlaps frozen TRAIN/VALID: {overlap}")

    excluded = [
        {
            "canonical_id": row["canonical_id"],
            "split": row["split"],
            "source": row.get("source"),
            "source_component": row.get("source_component"),
            "question": row.get("question"),
            "safe_response": row.get("clean_response"),
            "status": row.get("status"),
            "failure_reason": failure_reason(row),
            "intended_axes": row.get("intended_axes", []),
            "automatic_clean_preflight": row.get("automatic_clean_preflight"),
            "infrastructure_failures": row.get("infrastructure_failures", []),
            "production_source_path": str(
                (
                    production
                    / next(
                        name for name in artifact_names
                        if any(item["canonical_id"] == row["canonical_id"] for item in artifacts[name])
                    )
                ).resolve()
            ),
        }
        for row in terminal_rows
        if row.get("status") != "accepted"
    ]

    output.mkdir(parents=True, exist_ok=True)
    sft_path = output / "sft_test.jsonl"
    dpo_path = output / "dpo_test.jsonl"
    pair_path = output / "test_pairs.jsonl"
    excluded_path = output / "excluded_test_candidates.jsonl"
    write_jsonl(sft_path, pairs)
    write_jsonl(pair_path, pairs)
    write_jsonl(
        dpo_path,
        [
            {
                "input": {"question": row["question"], "corrupted_response": row["unsafe_response"]},
                "chosen": row["safe_response"],
                "rejected": row["unsafe_response"],
                "metadata": row["metadata"],
            }
            for row in pairs
        ],
    )
    write_jsonl(excluded_path, excluded)

    failure_counts = Counter(row["failure_reason"] for row in excluded)
    checks = {
        "terminal_partition_matches_selection": len(terminal_rows) == len(selected_ids),
        "accepted_plus_excluded_matches_selection": len(pairs) + len(excluded) == len(selected_ids),
        "all_split_test": all(row["split"] == "test" for row in pairs + excluded),
        "unique_pair_ids": len({row["canonical_id"] for row in pairs}) == len(pairs),
        "unique_pair_questions": len({row["question_normalized_sha256"] for row in pairs}) == len(pairs),
        "pair_text_nonempty": all(
            row["question"].strip() and row["unsafe_response"].strip() and row["safe_response"].strip()
            for row in pairs
        ),
        "safe_unsafe_differ": all(row["safe_response"] != row["unsafe_response"] for row in pairs),
        "zero_frozen_train_valid_overlap": not any(overlap.values()),
        "production_structural_validation_passed": bool(
            production_manifest.get("structural_validation", {}).get("passed")
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)

    manifest = {
        "version": VERSION,
        "status": "complete_evaluation_only_test_export",
        "purpose": "frozen_revision2_held_out_test_evaluation_not_training",
        "counts": {
            "selected_candidates": len(selected_ids),
            "accepted_pairs": len(pairs),
            "excluded_candidates": len(excluded),
            "failure_reasons": dict(sorted(failure_counts.items())),
            "by_source": dict(sorted(Counter(row["source"] for row in pairs).items())),
        },
        "checks": checks,
        "frozen_train_valid_overlap": overlap,
        "inputs": {
            "production_manifest": {
                "path": str(production_manifest_path),
                "sha256": sha256_file(production_manifest_path),
                "run_fingerprint": production_manifest["run_fingerprint"],
            },
            "selection": {
                "path": str((selection / "test_generation_selection.json").resolve()),
                "sha256": sha256_file(selection / "test_generation_selection.json"),
            },
            "pilot_exclusions": {
                "path": str((selection / "excluded_development_pilot_test_ids.txt").resolve()),
                "sha256": sha256_file(selection / "excluded_development_pilot_test_ids.txt"),
            },
        },
        "outputs": {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in (pair_path, sft_path, dpo_path, excluded_path)
        },
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
