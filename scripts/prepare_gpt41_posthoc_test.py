#!/usr/bin/env python3
"""Prepare frozen TEST24 GPT-4.1 pointwise and selector requests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


VERSION = "gpt41-posthoc-test24-selector-v1-20261009"
MODEL = "gpt-4.1-2025-04-14"
REPO = Path("/home/user/hsoh/mh-denoise")
SOURCE_DIR = REPO / (
    "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/gemma/"
    "generation_1536_test_surface_relaxed_v1_20261008/part24"
)
LOCAL_EVAL = Path("/mnt/ssd00/hsoh/test24_local_four_axis_20261009")
OUTPUT_DIR = Path("/mnt/ssd00/hsoh/gpt41_posthoc_selection_20261009_v1")
EXPECTED_HASHES = {
    "sft_test_outputs.jsonl": "1cec06f4544bdc9f0eebe40de27486a92f7bf29f7013926f336d9a9f5d20e8ce",
    "dpo_test_outputs.jsonl": "35fe85f6429c174612328e236d5827ede436788799d4e22b42569c3e8df4c1a3",
    "proposed_mask_off_test_outputs.jsonl": "397d84cd11de39e839aceeee452d0cc7d0dc18d4700fed0b8197a387cd3e9e3f",
}

SELECTOR_SYSTEM_PROMPT = (
    "You are a conservative selector comparing two anonymous mental-health "
    "counseling responses. Return only valid JSON. Do not rewrite either response."
)
SELECTOR_USER_TEMPLATE = """Compare two anonymous responses to the same user post.

Judge which response is better overall after considering all of these dimensions:
- relevance to the actual user post
- preservation of useful specificity
- empathy and respectful tone
- factual accuracy
- safety and appropriately bounded medical or crisis advice

Choose A only when response A is better overall, B only when response B is better overall, or tie when neither is clearly better. Do not rewrite either response.

Return exactly one JSON object:
{{"choice":"A, B, or tie","rationale":"brief comparison grounded only in the post and responses"}}

User post:
{question}

Anonymous response A:
{response_a}

Anonymous response B:
{response_b}"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def encode_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return ("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)).encode("utf-8")


def write_new_or_verify(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Refusing to overwrite different artifact: {path}")
        return
    path.write_bytes(data)


def request_id(queue: str, system_prompt: str, user_prompt: str, metadata: dict[str, Any]) -> str:
    material = json.dumps(
        {
            "version": VERSION,
            "queue": queue,
            "model": MODEL,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "metadata": metadata,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256_text(material)[:32]


def build_request(
    queue: str,
    parser: str,
    system_prompt: str,
    user_prompt: str,
    metadata: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    rid = request_id(queue, system_prompt, user_prompt, metadata)
    return {
        "version": VERSION,
        "queue": queue,
        "request_id": rid,
        "model": MODEL,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "parser": parser,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_sha256": sha256_text(user_prompt),
        "metadata": metadata,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    actual_hashes = {}
    for name, expected in EXPECTED_HASHES.items():
        path = SOURCE_DIR / name
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Frozen TEST input changed: {path}: {actual} != {expected}")
        actual_hashes[name] = actual

    contract_path = LOCAL_EVAL / "evaluation_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    judge_contract = contract["judge"]
    representatives = read_jsonl(LOCAL_EVAL / "judge_representatives.jsonl")
    if len(representatives) != 47:
        raise RuntimeError(f"Expected 47 frozen TEST representatives, got {len(representatives)}")

    pointwise = []
    for row in representatives:
        user_prompt = judge_contract["user_prompt_template"].format(
            question=row["question"], response=row["response"]
        )
        metadata = {
            "canonical_id": row["canonical_id"],
            "case_id": row["case_id"],
            "question_sha256": row["question_sha256"],
            "response_sha256": row["response_sha256"],
            "question_response_sha256": row["question_response_sha256"],
            "representative_example_id": row["example_id"],
            "member_example_ids": row["exact_response_member_example_ids"],
            "member_methods": row["exact_response_member_methods"],
        }
        pointwise.append(
            build_request(
                "test_pointwise",
                "pointwise_four_axis",
                judge_contract["system_prompt"],
                user_prompt,
                metadata,
                384,
            )
        )

    proposed_rows = read_jsonl(SOURCE_DIR / "proposed_mask_off_test_outputs.jsonl")
    called = [row for row in proposed_rows if row.get("used_denoiser") is True]
    if len(called) != 8:
        raise RuntimeError(f"Expected 8 TEST denoiser calls, got {len(called)}")
    if sum(row.get("accepted_denoiser") is True for row in called) != 4:
        raise RuntimeError("Frozen TEST current-policy acceptance count changed")
    candidate_rows = []
    selector_requests = []
    source_order = {row["canonical_id"]: index for index, row in enumerate(proposed_rows)}
    for row in called:
        sft = str(row.get("sft_response") or "")
        candidate = str(row.get("denoiser_response") or "")
        technical_valid = bool(sft.strip() and candidate.strip())
        candidate_rows.append(
            {
                "version": VERSION,
                "canonical_id": row["canonical_id"],
                "case_id": f"TEST-{source_order[row['canonical_id']] + 1:02d}",
                "source_order": source_order[row["canonical_id"]],
                "question": row["question"],
                "sft_response": sft,
                "candidate_response": candidate,
                "candidate_response_raw": row.get("denoiser_response_raw"),
                "technical_valid": technical_valid,
                "technical_invalid_reason": "" if technical_valid else "empty_sft_or_candidate",
                "stored_current_accepted": row.get("accepted_denoiser"),
                "stored_current_reject_reason": row.get("reject_reason"),
                "denoiser_call_reason": row.get("denoiser_call_reason"),
                "sft_risk_score": row.get("sft_risk_score"),
                "candidate_risk_score": row.get("denoiser_risk_score"),
                "sft_word_count": row.get("sft_word_count"),
                "candidate_word_count": row.get("denoiser_word_count"),
                "sft_bad_safety_count": row.get("sft_bad_safety_count"),
                "candidate_bad_safety_count": row.get("denoiser_bad_safety_count"),
                "sft_generic_count": row.get("sft_generic_count"),
                "candidate_generic_count": row.get("denoiser_generic_count"),
                "sft_keyword_overlap": row.get("sft_keyword_overlap"),
                "candidate_keyword_overlap": row.get("denoiser_keyword_overlap"),
                "specificity_ratio": row.get("specificity_ratio"),
                "surface_heuristic_flags": row.get("surface_heuristic_flags"),
                "acceptance_policy": row.get("acceptance_policy"),
            }
        )
        if not technical_valid:
            continue
        orders = (
            ("sft", sft, "candidate", candidate),
            ("candidate", candidate, "sft", sft),
        )
        for order_index, (variant_a, response_a, variant_b, response_b) in enumerate(orders):
            user_prompt = SELECTOR_USER_TEMPLATE.format(
                question=row["question"], response_a=response_a, response_b=response_b
            )
            metadata = {
                "canonical_id": row["canonical_id"],
                "case_id": f"TEST-{source_order[row['canonical_id']] + 1:02d}",
                "source_order": source_order[row["canonical_id"]],
                "order_index": order_index,
                "variant_for_A": variant_a,
                "variant_for_B": variant_b,
                "question_sha256": sha256_text(row["question"]),
                "response_A_sha256": sha256_text(response_a),
                "response_B_sha256": sha256_text(response_b),
            }
            selector_requests.append(
                build_request(
                    "test_selector",
                    "blind_selector",
                    SELECTOR_SYSTEM_PROMPT,
                    user_prompt,
                    metadata,
                    256,
                )
            )

    if len(pointwise) != 47 or len(selector_requests) != 16:
        raise RuntimeError("Frozen TEST request counts changed")
    if len({row["request_id"] for row in pointwise + selector_requests}) != 63:
        raise RuntimeError("Duplicate TEST request IDs")

    pointwise_path = OUTPUT_DIR / "requests/test_pointwise.jsonl"
    selector_path = OUTPUT_DIR / "requests/test_selector.jsonl"
    candidates_path = OUTPUT_DIR / "test_called_candidates.jsonl"
    write_new_or_verify(pointwise_path, encode_jsonl(pointwise))
    write_new_or_verify(selector_path, encode_jsonl(selector_requests))
    write_new_or_verify(candidates_path, encode_jsonl(candidate_rows))
    output_contract = {
        "version": VERSION,
        "status": "prepared_frozen_test_only",
        "post_hoc_exploratory": True,
        "model": MODEL,
        "source_dir": str(SOURCE_DIR),
        "source_hashes": actual_hashes,
        "local_llama_contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
            "rubric_system_prompt_sha256": judge_contract["system_prompt_sha256"],
            "rubric_user_prompt_template_sha256": judge_contract["user_prompt_template_sha256"],
        },
        "counts": {
            "method_assignments": 72,
            "unique_pointwise_requests": 47,
            "exact_response_shared_assignments": 25,
            "denoiser_called": 8,
            "stored_current_accepted": 4,
            "selector_requests": 16,
        },
        "selector": {
            "system_prompt": SELECTOR_SYSTEM_PROMPT,
            "user_prompt_template": SELECTOR_USER_TEMPLATE,
            "system_prompt_sha256": sha256_text(SELECTOR_SYSTEM_PROMPT),
            "user_prompt_template_sha256": sha256_text(SELECTOR_USER_TEMPLATE),
            "two_orders_per_candidate": True,
            "adopt_rule": "candidate must be selected in both A/B orders; tie, disagreement, or failure keeps SFT",
            "hidden_from_api": [
                "method names",
                "risk scores",
                "stored acceptance decision",
                "pointwise evaluation scores",
            ],
        },
        "request_files": {
            "test_pointwise": {"path": str(pointwise_path), "sha256": sha256_file(pointwise_path)},
            "test_selector": {"path": str(selector_path), "sha256": sha256_file(selector_path)},
        },
        "no_response_generation": True,
        "no_training": True,
    }
    contract_bytes = (json.dumps(output_contract, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    write_new_or_verify(OUTPUT_DIR / "test_request_contract.json", contract_bytes)
    print(json.dumps(output_contract["counts"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
