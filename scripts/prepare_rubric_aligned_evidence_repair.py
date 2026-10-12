#!/usr/bin/env python3
"""Prepare fresh GPT evaluations for non-verbatim Medical Advice evidence fields."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/home/user/hsoh/rubric_aligned_gpt41_rejudge_20261009_v1")
RESULT_PATHS = (
    ROOT / "results/expert15_pointwise_gpt41.jsonl",
    ROOT / "results/remaining_pointwise_gpt41.jsonl",
)
QUEUE_NAME = "pointwise_evidence_repair"
ADDENDUM = (
    "\n\nOutput validation requirement: when medical_advice is yes or unsure, "
    "medical_advice_evidence must be one exact, contiguous substring copied "
    "character-for-character from the Anonymous response, including the original "
    "spacing and punctuation. Do not repair spacing and do not join noncontiguous phrases."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    assignment_by_pair = {
        row["question_response_sha256"]: row
        for row in read_jsonl(ROOT / "inputs/pointwise_assignments.jsonl")
    }
    latest = {}
    for path in RESULT_PATHS:
        for row in read_jsonl(path):
            latest[row["request_id"]] = row

    invalid = []
    for row in latest.values():
        normalized = row.get("normalized") or {}
        if normalized.get("medical_advice") not in {"yes", "unsure"}:
            continue
        assignment = assignment_by_pair[row["metadata"]["question_response_sha256"]]
        evidence = normalized.get("medical_advice_evidence") or ""
        if evidence not in assignment["response"]:
            invalid.append((row, assignment, evidence))

    requests = []
    for original, assignment, evidence in invalid:
        request_id = sha256_text(
            "evidence-repair\0" + original["request_id"] + "\0" + ADDENDUM
        )[:32]
        requests.append(
            {
                "version": "rubric-aligned-gpt41-evidence-repair-v1-20261009",
                "queue": QUEUE_NAME,
                "request_id": request_id,
                "model": original["model"],
                "temperature": original["temperature"],
                "max_tokens": original["max_tokens"],
                "response_format": original["response_format"],
                "parser": original["parser"],
                "system_prompt": original["system_prompt"],
                "user_prompt": original["user_prompt"] + ADDENDUM,
                "metadata": {
                    **original["metadata"],
                    "repair_of_request_id": original["request_id"],
                    "repair_reason": "medical_advice_evidence_not_exact_contiguous_substring",
                },
            }
        )

    request_path = ROOT / "requests/pointwise_evidence_repair.jsonl"
    request_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in requests),
        encoding="utf-8",
    )
    contract = {
        "version": "rubric-aligned-gpt41-evidence-repair-v1-20261009",
        "status": "prepared_before_api_execution",
        "rubric_unchanged": True,
        "model_settings_unchanged": True,
        "prior_results_preserved": True,
        "reason": "Two yes/unsure evidence fields were not exact contiguous source substrings.",
        "output_validation_addendum": ADDENDUM.strip(),
        "count": len(requests),
        "requests": [
            {
                "request_id": request["request_id"],
                "repair_of_request_id": request["metadata"]["repair_of_request_id"],
                "canonical_id": assignment["canonical_id"],
                "question_response_sha256": request["metadata"]["question_response_sha256"],
                "invalid_evidence": evidence,
            }
            for request, (_, assignment, evidence) in zip(requests, invalid)
        ],
        "request_file": str(request_path),
        "request_file_sha256": sha256_text(request_path.read_text(encoding="utf-8")),
    }
    (ROOT / "evidence_repair_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"prepared": len(requests), "request_path": str(request_path)}))


if __name__ == "__main__":
    main()
