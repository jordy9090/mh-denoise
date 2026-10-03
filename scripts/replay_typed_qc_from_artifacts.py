#!/usr/bin/env python3
"""Replay typed QC from immutable generator/judge artifacts without model calls.

The script verifies the source run, re-parses every recorded paired-judge raw
output with the current contract, applies hash-bound complete-context clean
reviews, and writes a new versioned production derivative.  It never invokes a
model, API, generator, or trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fullpaper_acl_pipeline import read_jsonl
from local_qwen_production_qc_v4 import (
    JUDGE_REPO,
    JUDGE_REVISION,
    VERSION,
    content_disposition,
    validate_and_map,
)
from run_fullpaper_corruption_production import (
    LOCAL_JUDGE_VERSION,
    candidate_a,
    export,
    parse_json_object,
    selection_hash,
    sha256_file,
    write_json,
    write_jsonl,
)
from source_integrity_contract import validate_contextual_clean_review


def load_unique_jsonl(path: Path, key: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = list(read_jsonl(path))
    indexed = {str(row.get(key) or ""): row for row in rows}
    if "" in indexed or len(indexed) != len(rows):
        raise RuntimeError(f"{path} needs unique non-empty {key}")
    return rows, indexed


def complete_grade(
    grade: dict[str, Any], *, raw: str, usage: dict[str, Any], a: str, b: str,
    intended_axes: list[str], source_call: str,
) -> dict[str, Any]:
    grade.update({
        "raw_output": raw,
        "raw_output_source_call": source_call,
        "local_judge_usage": usage,
        "response_a_sha256": hashlib.sha256(a.encode()).hexdigest(),
        "response_b_sha256": hashlib.sha256(b.encode()).hexdigest(),
        "response_a_chars": len(a),
        "response_b_chars": len(b),
    })
    grade["unintended_axes"] = [
        axis for axis in grade["realized_axes"] if axis not in intended_axes
    ]
    return grade


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--selection-file", required=True)
    parser.add_argument("--clean-review-ledger", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source = Path(args.source_run).resolve()
    selection_path = Path(args.selection_file).resolve()
    review_path = Path(args.clean_review_ledger).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite replay output: {output}")

    source_manifest_path = source / "production_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "complete":
        raise RuntimeError("Source production run is not complete")
    artifact_mismatches = []
    for name, expected in source_manifest.get("artifact_sha256", {}).items():
        path = source / name
        if not path.is_file() or sha256_file(path) != expected:
            artifact_mismatches.append(name)
    if artifact_mismatches:
        raise RuntimeError(f"Source production artifact mismatch: {artifact_mismatches}")

    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_rows = selection_payload["rows"] if isinstance(selection_payload, dict) else selection_payload
    frozen_rows = [
        {
            "canonical_id": row["canonical_id"],
            "split": row["split"],
            "question_normalized_sha256": row["question_normalized_sha256"],
            "duplicate_cluster_id": row["duplicate_cluster_id"],
            "source_group_id": row["source_group_id"],
            "intended_axes": row["intended_axes"],
            "generation_seeds": row["generation_seeds"],
        }
        for row in selection_rows
    ]
    selection_file_sha = sha256_file(selection_path)
    selection_execution_sha = selection_hash(frozen_rows)
    if selection_file_sha != source_manifest.get("selection_file_sha256"):
        raise RuntimeError("Selection file bytes differ from source run")
    if selection_execution_sha != source_manifest.get("selection_sha256"):
        raise RuntimeError("Expanded selection execution contract differs from source run")

    checkpoint_rows: list[dict[str, Any]] = []
    for path in sorted(source.glob("results_checkpoint.part*.jsonl")):
        checkpoint_rows.extend(read_jsonl(path))
    checkpoint = {row["canonical_id"]: row for row in checkpoint_rows}
    selected_ids = [row["canonical_id"] for row in selection_rows]
    if len(checkpoint) != len(checkpoint_rows) or set(checkpoint) != set(selected_ids):
        raise RuntimeError("Checkpoint IDs do not exactly match frozen selection")

    review_rows, reviews = load_unique_jsonl(review_path, "canonical_id")
    if set(reviews) != set(selected_ids):
        raise RuntimeError("Replay requires one clean-context review per selected ID")
    validated_reviews = {}
    for canonical_id in selected_ids:
        row = checkpoint[canonical_id]
        validated_reviews[canonical_id] = validate_contextual_clean_review(
            row["question"], row["clean_response"], reviews[canonical_id]
        )

    calls = []
    for path in sorted((source / "local_calls").glob("call_*.json")):
        call = json.loads(path.read_text(encoding="utf-8"))
        call["_source_file"] = path.name
        calls.append(call)
    generator_calls: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    judge_calls: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        context = call.get("context", {})
        canonical_id = context.get("canonical_id")
        stage = context.get("stage_index")
        semantic_attempt = context.get("semantic_attempt")
        if canonical_id not in checkpoint or not isinstance(stage, int) or not isinstance(semantic_attempt, int):
            continue
        key = (canonical_id, stage, semantic_attempt)
        if call.get("call_kind") == "generator":
            generator_calls[key].append(call)
        elif call.get("call_kind") == "judge" and context.get("call_purpose") == "paired_stage_qc":
            judge_calls[key].append(call)

    chosen_grades: dict[tuple[str, int, int], dict[str, Any]] = {}
    revalidation_rows: list[dict[str, Any]] = []
    for key, grouped_calls in sorted(judge_calls.items()):
        canonical_id, stage, semantic_attempt = key
        row = checkpoint[canonical_id]
        usable_generations = [
            call for call in generator_calls.get(key, [])
            if call.get("status") == "returned"
            and not call.get("usage", {}).get("truncation_reason")
        ]
        if not usable_generations:
            continue
        candidate = usable_generations[-1]["raw_output"]
        candidate_label = "A" if candidate_a(canonical_id, stage, semantic_attempt) else "B"
        clean_label = "B" if candidate_label == "A" else "A"
        a = candidate if candidate_label == "A" else row["clean_response"]
        b = row["clean_response"] if candidate_label == "A" else candidate
        strict_choice = None
        quarantined_choices = []
        for call in grouped_calls:
            record = {
                "canonical_id": canonical_id,
                "stage_index": stage,
                "semantic_attempt": semantic_attempt,
                "infrastructure_attempt": call.get("context", {}).get("infrastructure_attempt"),
                "source_call": call["_source_file"],
                "raw_output_sha256": call.get("raw_output_sha256"),
                "source_prompt_sha256": call.get("prompt_sha256"),
                "strict_status": None,
                "strict_error": None,
                "quarantine_status": None,
                "invalid_evidence": [],
            }
            try:
                payload = parse_json_object(call.get("raw_output", ""))
                grade, invalid = validate_and_map(
                    payload, a, b, clean_label, candidate_label, question=row["question"]
                )
                grade = complete_grade(
                    grade, raw=call["raw_output"], usage=call.get("usage", {}),
                    a=a, b=b, intended_axes=row["intended_axes"],
                    source_call=call["_source_file"],
                )
                record["strict_status"] = "valid"
                strict_choice = strict_choice or grade
            except Exception as exc:
                record["strict_status"] = "invalid"
                record["strict_error"] = f"{type(exc).__name__}: {exc}"
                try:
                    payload = parse_json_object(call.get("raw_output", ""))
                    grade, invalid = validate_and_map(
                        payload, a, b, clean_label, candidate_label,
                        question=row["question"], quarantine_invalid_evidence=True,
                    )
                    grade = complete_grade(
                        grade, raw=call["raw_output"], usage=call.get("usage", {}),
                        a=a, b=b, intended_axes=row["intended_axes"],
                        source_call=call["_source_file"],
                    )
                    grade["strict_validation_attempts_exhausted"] = True
                    grade["quarantined_invalid_evidence_count"] = len(invalid)
                    record["quarantine_status"] = "valid_with_non_verbatim_removed"
                    record["invalid_evidence"] = invalid
                    quarantined_choices.append((len(invalid), call["_source_file"], grade))
                except Exception as fallback_exc:
                    record["quarantine_status"] = "invalid"
                    record["quarantine_error"] = f"{type(fallback_exc).__name__}: {fallback_exc}"
            revalidation_rows.append(record)
        if strict_choice is not None:
            chosen_grades[key] = strict_choice
        elif quarantined_choices:
            chosen_grades[key] = min(quarantined_choices, key=lambda item: (item[0], item[1]))[2]

    replayed: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for spec in selection_rows:
        canonical_id = spec["canonical_id"]
        old = checkpoint[canonical_id]
        review = validated_reviews[canonical_id]
        result = {
            key: value for key, value in old.items()
            if key not in {
                "status", "failure_reason", "corrupted_response", "realized_axes",
                "unintended_axes", "final_grade", "stage_history", "infrastructure_failures",
            }
        }
        result.update({
            "judge_model": f"{JUDGE_REPO}@{JUDGE_REVISION}",
            "judge_repo": JUDGE_REPO,
            "judge_revision": JUDGE_REVISION,
            "judge_prompt_version": VERSION,
            "replay_source_status": old["status"],
            "contextual_clean_review": review,
            "stage_history": [],
            "infrastructure_failures": [],
        })
        if review["disposition"] != "pass":
            result.update({
                "status": "qc_hold",
                "failure_reason": "contextual_clean_review_required",
            })
        else:
            current = old["clean_response"]
            final_grade = None
            terminal: tuple[str, str] | None = None
            for stage, _axis in enumerate(spec["intended_axes"], 1):
                stage_ok = False
                recorded_candidate = False
                last_reason = "no_recorded_candidate"
                for semantic_attempt in (1, 2):
                    key = (canonical_id, stage, semantic_attempt)
                    usable_generations = [
                        call for call in generator_calls.get(key, [])
                        if call.get("status") == "returned"
                        and not call.get("usage", {}).get("truncation_reason")
                    ]
                    if not usable_generations:
                        continue
                    recorded_candidate = True
                    candidate = usable_generations[-1]["raw_output"]
                    grade = chosen_grades.get(key)
                    if grade is None:
                        terminal = ("technical_failure", "replay_judge_schema_unresolved")
                        break
                    conflict = []
                    if grade["clean_scores"]["specificity"]["score"] == 1:
                        conflict.append("clean_specificity_score_1")
                    if grade["clean_scores"]["medical_boundary"]["violation"]:
                        conflict.append("clean_medical_boundary_violation")
                    if grade["text_reason_contradiction"]["detected"]:
                        conflict.append("judge_text_reason_contradiction")
                    if conflict:
                        terminal = ("qc_conflict", ";".join(conflict))
                        break
                    disposition, reason = content_disposition(
                        old["question"], candidate, grade, clean=old["clean_response"],
                        contextual_clean_review=review,
                    )
                    missing = [
                        intended for intended in spec["intended_axes"][:stage]
                        if intended not in grade["realized_axes"]
                    ]
                    if disposition == "pass" and missing:
                        disposition = "reject"
                        reason = "missing_intended_axes:" + ",".join(missing)
                    result["stage_history"].append({
                        "stage_index": stage,
                        "semantic_attempt": semantic_attempt,
                        "candidate_response": candidate,
                        "grade": grade,
                        "accepted": disposition == "pass",
                        "reason": reason,
                        "replayed_from_existing_artifacts": True,
                    })
                    if disposition == "hold":
                        terminal = ("qc_hold", reason or "clean_review_required")
                        break
                    if disposition == "pass":
                        current = candidate
                        final_grade = grade
                        stage_ok = True
                        break
                    last_reason = reason or "recorded_candidate_rejected"
                if terminal is not None:
                    break
                if not stage_ok:
                    terminal = (
                        ("rejected", last_reason)
                        if recorded_candidate else
                        ("technical_failure", "replay_missing_recorded_generation")
                    )
                    break
            if terminal is None and final_grade is not None:
                result.update({
                    "status": "accepted",
                    "corrupted_response": current,
                    "realized_axes": final_grade["realized_axes"],
                    "unintended_axes": final_grade["unintended_axes"],
                    "final_grade": final_grade,
                })
            else:
                status, reason = terminal or ("technical_failure", "replay_incomplete")
                result.update({"status": status, "failure_reason": reason})
        replayed.append(result)
        transitions.append({
            "canonical_id": canonical_id,
            "split": spec["split"],
            "before_status": old["status"],
            "after_status": result["status"],
            "after_reason": result.get("failure_reason"),
            "clean_review_disposition": review["disposition"],
        })

    output.mkdir(parents=True)
    write_jsonl(output / "results_replayed.jsonl", replayed)
    write_jsonl(output / "judge_revalidation.jsonl", revalidation_rows)
    write_jsonl(output / "status_transitions.jsonl", transitions)
    write_jsonl(output / "clean_context_review.jsonl", review_rows)

    source_manifest_sha = sha256_file(source_manifest_path)
    manifest = {
        "version": LOCAL_JUDGE_VERSION,
        "replay_contract_version": "typed-qc-artifact-replay-v1-20260926",
        "replay_source": {
            "path": str(source),
            "production_manifest_sha256": source_manifest_sha,
            "production_version": source_manifest.get("version"),
            "artifacts_verified": len(source_manifest.get("artifact_sha256", {})),
            "artifact_mismatches": artifact_mismatches,
        },
        "selection_file": str(selection_path),
        "selection_file_sha256": selection_file_sha,
        "selection_execution_contract_sha256": selection_execution_sha,
        "selection_sha256": selection_execution_sha,
        "selection_hash_semantics": {
            "selection_file_sha256": "exact selection.json file bytes",
            "selection_execution_contract_sha256": (
                "compact canonical JSON of the expanded fields used by execution"
            ),
        },
        "processed_inputs": len(replayed),
        "allowed_splits": ["train", "valid"],
        "generator": source_manifest.get("generator"),
        "judge": {
            **source_manifest.get("judge", {}),
            "qc_version": VERSION,
            "model_calls_during_replay": 0,
            "raw_outputs_reused": len(revalidation_rows),
        },
        "semantic_review": {
            "ledger_path": str(review_path),
            "ledger_sha256": sha256_file(review_path),
            "coverage_complete": True,
            "pass": sum(row["disposition"] == "pass" for row in review_rows),
            "hold": sum(row["disposition"] == "hold" for row in review_rows),
            "unresolved": sum(row["disposition"] == "unresolved" for row in review_rows),
            "approved_for_training": False,
            "reviewer": "Codex (not human or clinical expert)",
        },
        "budget": {"paid_api_calls": 0, "paid_api_disabled": True},
        "model_or_api_calls": 0,
        "replay_artifact_sha256": {
            name: sha256_file(output / name)
            for name in (
                "results_replayed.jsonl", "judge_revalidation.jsonl",
                "status_transitions.jsonl", "clean_context_review.jsonl",
            )
        },
        "before_status_counts": dict(Counter(row["status"] for row in checkpoint_rows)),
        "after_status_counts": dict(Counter(row["status"] for row in replayed)),
        "judge_revalidation_counts": dict(Counter(
            (
                "strict_valid" if row["strict_status"] == "valid" else
                "quarantine_valid" if row["quarantine_status"] == "valid_with_non_verbatim_removed" else
                "invalid"
            )
            for row in revalidation_rows
        )),
    }
    export(output, replayed, manifest)
    # export publishes the production manifest last.  Do not mutate it after
    # publication; every replay-only artifact hash is already embedded above.
    print(json.dumps({
        "output": str(output),
        "before": manifest["before_status_counts"],
        "after": manifest["after_status_counts"],
        "selection_file_sha256": selection_file_sha,
        "selection_execution_contract_sha256": selection_execution_sha,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
