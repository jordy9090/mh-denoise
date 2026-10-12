#!/usr/bin/env python3
"""Prepare pointwise GPT-4.1 requests for TEST candidates absent from the final-response cache."""

from __future__ import annotations

import json
from pathlib import Path

from prepare_gpt41_posthoc_test import (
    LOCAL_EVAL,
    OUTPUT_DIR,
    build_request,
    encode_jsonl,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_new_or_verify,
)


def main() -> None:
    contract_path = LOCAL_EVAL / "evaluation_contract.json"
    rubric = json.loads(contract_path.read_text(encoding="utf-8"))["judge"]
    cached_representatives = read_jsonl(LOCAL_EVAL / "judge_representatives.jsonl")
    cached_pairs = {
        row["question_response_sha256"] for row in cached_representatives
    }
    candidates = read_jsonl(OUTPUT_DIR / "test_called_candidates.jsonl")
    requests = []
    for row in candidates:
        pair_hash = sha256_text(row["question"] + "\0" + row["candidate_response"])
        if pair_hash in cached_pairs:
            continue
        user_prompt = rubric["user_prompt_template"].format(
            question=row["question"], response=row["candidate_response"]
        )
        metadata = {
            "canonical_id": row["canonical_id"],
            "case_id": row["case_id"],
            "question_sha256": sha256_text(row["question"]),
            "response_sha256": sha256_text(row["candidate_response"]),
            "question_response_sha256": pair_hash,
            "response_role": "stored_called_denoiser_candidate",
            "stored_current_accepted": row["stored_current_accepted"],
        }
        requests.append(
            build_request(
                "test_candidate_pointwise",
                "pointwise_four_axis",
                rubric["system_prompt"],
                user_prompt,
                metadata,
                384,
            )
        )
    if len(requests) != 4:
        raise RuntimeError(f"Expected four uncached rejected TEST candidates, got {len(requests)}")
    path = OUTPUT_DIR / "requests/test_candidate_pointwise.jsonl"
    write_new_or_verify(path, encode_jsonl(requests))
    extension = {
        "status": "prepared",
        "queue": "test_candidate_pointwise",
        "requests": len(requests),
        "request_path": str(path),
        "request_sha256": sha256_file(path),
        "rubric_contract": str(contract_path),
        "rubric_contract_sha256": sha256_file(contract_path),
        "dedup_rule": "exclude exact (question, response) pairs already in the frozen TEST47 pointwise cache",
    }
    write_new_or_verify(
        OUTPUT_DIR / "test_candidate_pointwise_contract.json",
        (json.dumps(extension, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    print(json.dumps(extension, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
