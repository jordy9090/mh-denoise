#!/usr/bin/env python3
"""Separate completed response-pair suitability from local scorer positives."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from corruption_contract_v2 import AXES
from fullpaper_risk_contract import scorer_input_text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".tmp.", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def input_sha(question: str, span: str) -> str:
    value = scorer_input_text(question, span)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True,
        help="NAME:RESULTS_JSONL:CLEAN_REVIEW_JSONL; may be repeated",
    )
    parser.add_argument("--model-input-review", action="append", default=[])
    parser.add_argument("--physical-label-review", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_reviews: dict[str, dict[str, Any]] = {}
    for value in args.model_input_review:
        for row in read_jsonl(Path(value).resolve()):
            key = str(row["model_input_sha256"])
            if key in model_reviews:
                raise RuntimeError(f"Duplicate model-input review: {key}")
            model_reviews[key] = row
    physical_reviews: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
    for value in args.physical_label_review:
        for row in read_jsonl(Path(value).resolve()):
            key = (
                str(row["canonical_id"]), str(row["span_side"]),
                int(row["span_start"]), int(row["span_end"]), str(row["axis"]),
            )
            if key in physical_reviews:
                raise RuntimeError(f"Duplicate physical scorer review: {key}")
            physical_reviews[key] = row

    cases: list[dict[str, Any]] = []
    target_events: list[dict[str, Any]] = []
    completed_pairs: list[dict[str, Any]] = []
    input_files = []
    for spec in args.run:
        name, results_value, clean_value = spec.split(":", 2)
        results_path, clean_path = Path(results_value).resolve(), Path(clean_value).resolve()
        clean_reviews = {
            row["canonical_id"]: row for row in read_jsonl(clean_path)
        }
        rows = read_jsonl(results_path)
        input_files.append({
            "name": name, "results": str(results_path), "clean_review": str(clean_path),
            "rows": len(rows),
        })
        for row in rows:
            canonical_id = row["canonical_id"]
            clean_disposition = clean_reviews[canonical_id]["disposition"]
            cases.append({
                "run": name, "canonical_id": canonical_id, "split": row["split"],
                "status": row["status"], "clean_disposition": clean_disposition,
            })
            if clean_disposition == "pass":
                for event in row.get("stage_history", []):
                    stage = int(event.get("stage_index", 1))
                    target_axis = event.get("target_axis") or row["intended_axes"][stage - 1]
                    grade = event.get("grade")
                    score = (grade or {}).get("candidate_scores", {}).get(target_axis, {})
                    target_events.append({
                        "run": name, "canonical_id": canonical_id,
                        "target_axis": target_axis,
                        "semantic_attempt": event.get("semantic_attempt"),
                        "accepted_stage": bool(event.get("accepted")),
                        "reason": event.get("reason"),
                        "judge_grade_present": grade is not None,
                        "response_axis_realized": bool(
                            grade and target_axis in grade.get("realized_axes", [])
                        ),
                        "judge_candidate_scope": score.get("evidence_scope"),
                        "judge_reported_local_positive": bool(grade and any(
                            item.get("side") == "candidate"
                            and item.get("axis") == target_axis
                            and item.get("label") == 1
                            for item in grade.get("local_supervision", [])
                        )),
                    })
            if row["status"] != "accepted" or clean_disposition != "pass":
                continue
            grade = row["final_grade"]
            axis_records = []
            for axis in grade.get("realized_axes", []):
                annotations = [
                    item for item in grade.get("local_supervision", [])
                    if item.get("side") == "candidate"
                    and item.get("axis") == axis
                    and item.get("label") == 1
                ]
                classification = None
                evidence = []
                for annotation in annotations:
                    physical_key = (
                        canonical_id, "candidate", int(annotation["start"]),
                        int(annotation["end"]), axis,
                    )
                    physical = physical_reviews.get(physical_key)
                    model = model_reviews.get(input_sha(row["question"], annotation["text"]))
                    if physical is not None:
                        decision = physical["disposition"]
                        reason = physical["reason"]
                        reviewed_label = annotation["label"] if decision == "retain" else None
                    elif model is not None:
                        reviewed_label = model.get("labels", {}).get(axis)
                        reason = model.get("decisions", {}).get(axis, {}).get("reason")
                        decision = "retain" if reviewed_label == 1 else "mask_unknown"
                    else:
                        reviewed_label, decision = None, "unreviewed"
                        reason = "No question-plus-span semantic review was found."
                    evidence.append({
                        "span": annotation["text"], "judge_scope": annotation["scope"],
                        "review_decision": decision, "reviewed_label": reviewed_label,
                        "review_reason": reason,
                    })
                    if reviewed_label == 1:
                        classification = "local_positive"
                if classification is None:
                    candidate_scope = grade["candidate_scores"][axis].get("evidence_scope")
                    if candidate_scope in {"holistic", "omission"} or any(
                        item["review_decision"] == "mask_unknown" for item in evidence
                    ):
                        classification = "response_level_only"
                    else:
                        classification = "insufficient_local_evidence"
                axis_records.append({
                    "axis": axis, "classification": classification,
                    "judge_candidate_scope": grade["candidate_scores"][axis].get("evidence_scope"),
                    "evidence": evidence,
                })
            completed_pairs.append({
                "run": name, "canonical_id": canonical_id, "split": row["split"],
                "realized_axes": grade.get("realized_axes", []),
                "axis_records": axis_records,
            })

    axis_summary: dict[str, dict[str, int]] = {}
    for axis in AXES:
        records = [
            item for pair in completed_pairs for item in pair["axis_records"]
            if item["axis"] == axis
        ]
        counts = Counter(item["classification"] for item in records)
        axis_summary[axis] = {
            "response_pair_axes": len(records),
            "local_positive": counts["local_positive"],
            "response_level_only": counts["response_level_only"],
            "insufficient_local_evidence": counts["insufficient_local_evidence"],
        }
    target_summary = {}
    for axis in AXES:
        events = [item for item in target_events if item["target_axis"] == axis]
        target_summary[axis] = {
            "attempts": len(events),
            "judge_graded": sum(item["judge_grade_present"] for item in events),
            "response_axis_realized": sum(item["response_axis_realized"] for item in events),
            "stage_accepted": sum(item["accepted_stage"] for item in events),
            "judge_reported_local_positive": sum(
                item["judge_reported_local_positive"] for item in events
            ),
            "failure_reasons": dict(Counter(
                str(item["reason"]) for item in events if not item["accepted_stage"]
            )),
        }

    payload = {
        "version": "typed-qc-pair-local-evidence-audit-v1-20260927",
        "status": "complete",
        "policy": {
            "pair_suitability": "accepted terminal pair with an independent complete-context clean pass",
            "local_positive": "candidate local_defect exact span retained by question-plus-span semantic review",
            "response_level_only": "holistic/omission evidence or a purported local label masked because its reason needs surrounding response context",
            "insufficient_local_evidence": "no retained local positive and no supported holistic/omission classification",
            "pair_not_discarded_without_local_positive": True,
        },
        "inputs": input_files,
        "case_counts": {
            "all": len(cases),
            "clean_pass": sum(item["clean_disposition"] == "pass" for item in cases),
            "clean_hold": sum(item["clean_disposition"] == "hold" for item in cases),
            "completed_usable_pairs": len(completed_pairs),
            "terminal_status": dict(Counter(item["status"] for item in cases)),
        },
        "completed_pair_axis_summary": axis_summary,
        "completed_pairs": completed_pairs,
        "clean_pass_target_attempt_summary": target_summary,
        "clean_pass_target_events": target_events,
    }
    atomic_json(Path(args.output).resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
