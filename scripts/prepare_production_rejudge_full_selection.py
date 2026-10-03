#!/usr/bin/env python3
"""Freeze the full production-reuse rejudge selection without model calls.

The immutable production accepted rows are the text authority.  Complete v5
judgments are reused only after question/clean/candidate hashes and the saved
A/B response hashes match.  Technical failures are deliberately not seeded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from fullpaper_acl_pipeline import read_jsonl, sha256_file, write_json
from rejudge_production_reuse_subset import adjudicate_grade


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRODUCTION = ROOT / "data/fullpaper_acl_pipeline/production_run_train1000_valid200_local_qwen35_27b_v3r1_20260910"
DEFAULT_REJUDGE = ROOT / "data/fullpaper_acl_pipeline/production_accepted652_reuse_rejudge_v5_20260927"
DEFAULT_AUDIT_EXPORT = ROOT / "data/fullpaper_acl_pipeline/scorer_supervision_audit_20260929/current_policy_production_reuse_export"
VERSION = "production-rejudge-full-selection-v1-20261001"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", default=str(DEFAULT_PRODUCTION))
    parser.add_argument("--previous-rejudge-dir", default=str(DEFAULT_REJUDGE))
    parser.add_argument("--audit-export-dir", default=str(DEFAULT_AUDIT_EXPORT))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    production = Path(args.production_dir).resolve()
    previous = Path(args.previous_rejudge_dir).resolve()
    audit = Path(args.audit_export_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_outputs = (
        "explicit_ids.txt", "seed_rejudged_checkpoint.jsonl",
        "selection_preparation_manifest.json", "clean_gate_conflict_trace.json",
    )
    existing = [name for name in expected_outputs if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite prepared artifacts: {existing}")

    accepted = list(read_jsonl(production / "accepted.jsonl"))
    accepted_by_id = {row["canonical_id"]: row for row in accepted}
    if len(accepted) != 652 or len(accepted_by_id) != 652:
        raise RuntimeError("Expected 652 unique immutable production rows")

    previous_selection_payload = json.loads((previous / "selection.json").read_text(encoding="utf-8"))
    previous_selection = {row["canonical_id"]: row for row in previous_selection_payload["rows"]}
    previous_checkpoint = list(read_jsonl(previous / "rejudged_checkpoint.jsonl"))
    if len(previous_checkpoint) != len({row["canonical_id"] for row in previous_checkpoint}):
        raise RuntimeError("Previous rejudge checkpoint contains duplicate IDs")
    audit_rows = list(read_jsonl(audit / "adjudication_ledger.jsonl"))
    legacy_ids = {
        row["canonical_id"] for row in audit_rows
        if row["decision_source"] == "legacy_accepted_pair_plus_no_clean_warning"
    }
    if len(legacy_ids) != 488:
        raise RuntimeError(f"Expected 488 unreviewed legacy rows, found {len(legacy_ids)}")

    reused, retry_ids = [], set()
    validation_rows = []
    for old in previous_checkpoint:
        canonical_id = old["canonical_id"]
        source = accepted_by_id[canonical_id]
        frozen = previous_selection.get(canonical_id)
        if frozen is None:
            raise RuntimeError(f"Previous checkpoint ID absent from frozen selection: {canonical_id}")
        actual = {
            "question_sha256": digest(source["question"]),
            "clean_response_sha256": digest(source["clean_response"]),
            "corrupted_response_sha256": digest(source["corrupted_response"]),
        }
        if any(frozen[key] != value for key, value in actual.items()):
            raise RuntimeError(f"Immutable text hash mismatch: {canonical_id}")
        if old.get("status") != "complete":
            retry_ids.add(canonical_id)
            validation_rows.append({"canonical_id": canonical_id, **actual, "reuse": False,
                                    "reason": old.get("failure_reason", "non_complete")})
            continue
        grade = old["grade"]
        saved_ab = {grade["response_a_sha256"], grade["response_b_sha256"]}
        expected_ab = {actual["clean_response_sha256"], actual["corrupted_response_sha256"]}
        if saved_ab != expected_ab:
            raise RuntimeError(f"Saved A/B response hash mismatch: {canonical_id}")
        revised = adjudicate_grade(source, grade)
        carried = {
            **old,
            "decision_history": [{
                "policy": "production-reuse-subset-rejudge-v2-20260927",
                "clean_status": old.get("clean_status"),
                "pair_status": old.get("pair_status"),
                "pair_reason": old.get("pair_reason"),
            }],
            **revised,
            "decision_policy_version": "production-reuse-subset-rejudge-v3-20261001",
            "reused_without_model_call": True,
            "validated_text_sha256": actual,
        }
        reused.append(carried)
        validation_rows.append({"canonical_id": canonical_id, **actual, "reuse": True,
                                "reason": "same_id_and_question_clean_candidate_hashes"})

    if len(reused) != 78 or len(retry_ids) != 5:
        raise RuntimeError(f"Unexpected prior terminal split: reused={len(reused)} retry={len(retry_ids)}")
    selected_ids = legacy_ids | set(previous_selection)
    if len(selected_ids) != 571:
        raise RuntimeError(f"Expected 571 selected IDs, found {len(selected_ids)}")
    pending_call_ids = selected_ids - {row["canonical_id"] for row in reused}
    if pending_call_ids != legacy_ids | retry_ids:
        raise RuntimeError("Pending calls differ from 488 legacy plus 5 technical failures")

    ordered_ids = [row["canonical_id"] for row in accepted if row["canonical_id"] in selected_ids]
    (output / "explicit_ids.txt").write_text("\n".join(ordered_ids) + "\n", encoding="utf-8")
    write_jsonl(output / "seed_rejudged_checkpoint.jsonl", reused)
    write_jsonl(output / "reused_hash_validation.jsonl", validation_rows)

    conflict = None
    for carried in reused:
        history = carried["decision_history"][0]
        gate = carried["grade"].get("clean_target_eligibility", {})
        if (
            history["clean_status"] == "original_answer_issue"
            and carried["clean_status"] == "unresolved"
            and gate.get("disposition") == "pass"
        ):
            source = accepted_by_id[carried["canonical_id"]]
            conflict = {
                "canonical_id": carried["canonical_id"],
                "question": source["question"],
                "clean_response": source["clean_response"],
                "candidate_response": source["corrupted_response"],
                "intended_axes": source["intended_axes"],
                "legacy_realized_axes": source["realized_axes"],
                "clean_target_eligibility": gate,
                "clean_scores": carried["grade"].get("clean_scores", {}),
                "candidate_scores": carried["grade"].get("candidate_scores", {}),
                "clean_local_defects": [item for item in carried["grade"].get("local_supervision", [])
                                        if item.get("side") == "clean" and item.get("label") == 1],
                "previous_decision": history,
                "revised_decision": {
                    "clean_status": carried["clean_status"],
                    "pair_status": carried["pair_status"],
                    "pair_reason": carried["pair_reason"],
                },
                "learning_rows_under_revised_decision": {
                    "sft": 0, "dpo": 0, "router": 0, "scorer": 0,
                    "reason": "held as unresolved pending resolution of the clean gate/local-label contradiction",
                },
            }
            break
    if conflict is None:
        raise RuntimeError("No observed pass-plus-clean-local-defect conflict found")
    write_json(output / "clean_gate_conflict_trace.json", conflict)

    inputs = {
        "accepted.jsonl": production / "accepted.jsonl",
        "previous_selection.json": previous / "selection.json",
        "previous_checkpoint.jsonl": previous / "rejudged_checkpoint.jsonl",
        "audit_adjudication_ledger.jsonl": audit / "adjudication_ledger.jsonl",
    }
    manifest = {
        "version": VERSION,
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in inputs.items()},
        "counts": {
            "production_accepted": len(accepted),
            "frozen_selection": len(selected_ids),
            "reused_complete_same_hash": len(reused),
            "pending_model_calls": len(pending_call_ids),
            "pending_no_current_qc": len(legacy_ids),
            "pending_prior_technical_failure": len(retry_ids),
            "not_selected_existing_review_or_hold": len(accepted) - len(selected_ids),
            "reused_decisions_after_policy": dict(Counter(row["clean_status"] for row in reused)),
        },
        "selection_sha256": sha256_file(output / "explicit_ids.txt"),
        "seed_checkpoint_sha256": sha256_file(output / "seed_rejudged_checkpoint.jsonl"),
        "pending_ids_sha256": hashlib.sha256("\n".join(sorted(pending_call_ids)).encode()).hexdigest(),
        "policy": {
            "complete_same_hash_rows_reused_without_model_call": True,
            "technical_failures_retried": True,
            "question_and_both_answers_unchanged": True,
            "paid_api_enabled": False,
            "generation_enabled": False,
        },
    }
    write_json(output / "selection_preparation_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
