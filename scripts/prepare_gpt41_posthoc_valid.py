#!/usr/bin/env python3
"""Prepare frozen VALID43/VALID9 GPT-4.1 and local comparison requests."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from prepare_gpt41_posthoc_test import (
    MODEL,
    OUTPUT_DIR,
    SELECTOR_SYSTEM_PROMPT,
    SELECTOR_USER_TEMPLATE,
    encode_jsonl,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_new_or_verify,
)


VERSION = "gpt41-posthoc-valid43-valid9-complete-v1-20261009"
REPO = Path("/home/user/hsoh/mh-denoise")
VALID_ROOT = REPO / "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/gemma/generation_1536"
FEEDBACK_ROOT = Path(
    "/home/user/hsoh/mh-denoise-feedback-v1-valid-20261008/"
    "outputs/valid_feedback_v1_20261008"
)
BC_ROOT = Path("/mnt/ssd00/hsoh/feedback_v1_bc_20261009")
POINTWISE_CONTRACT = Path("/mnt/ssd00/hsoh/test24_local_four_axis_20261009/evaluation_contract.json")


def pair_hash(question: str, response: str) -> str:
    return sha256_text(question + "\0" + response)


def build_request(
    queue: str,
    parser: str,
    system_prompt: str,
    user_prompt: str,
    metadata: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
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
    return {
        "version": VERSION,
        "queue": queue,
        "request_id": sha256_text(material)[:32],
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


def standard_candidate(row: dict[str, Any], candidate_type: str) -> dict[str, Any]:
    return {
        "version": VERSION,
        "canonical_id": row["canonical_id"],
        "source_order": row["source_order"],
        "candidate_type": candidate_type,
        "source_variant": row["variant"],
        "question": row["question"],
        "sft_response": row["sft_response"],
        "candidate_response": row["candidate_text"],
        "candidate_raw_decoded_output": row.get("candidate_raw_decoded_output"),
        "technical_valid": bool(
            str(row.get("sft_response") or "").strip()
            and str(row.get("candidate_text") or "").strip()
        ),
        "current_rules_accepted": row.get("accepted_candidate"),
        "current_rules_reject_reason": row.get("reject_reason"),
        "current_rules_final_response": row.get("policy_final_response"),
        "gate_called": row.get("gate_replay_called"),
        "gate_reason": row.get("gate_replay_reason"),
        "sft_metrics": row.get("sft_metrics"),
        "candidate_metrics": row.get("candidate_metrics"),
        "surface_heuristic_flags": row.get("surface_heuristic_flags"),
        "draft_source": row.get("draft_source"),
        "generation_adapter_path": row.get("generation_adapter_path"),
    }


def main() -> None:
    requests_dir = OUTPUT_DIR / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)
    rubric = json.loads(POINTWISE_CONTRACT.read_text(encoding="utf-8"))["judge"]
    feedback_contract = json.loads((FEEDBACK_ROOT / "experiment_contract.json").read_text(encoding="utf-8"))
    pair_system = feedback_contract["judge"]["system_prompt"]
    pair_template = feedback_contract["judge"]["user_prompt_template"]

    final_specs = {
        "SFT": ("sft_valid_outputs.jsonl", "sft_response"),
        "DPO": ("dpo_valid_outputs.jsonl", "sft_response"),
        "Mask_ON_current": ("proposed_mask_on_valid_outputs.jsonl", "final_response"),
        "Mask_OFF_current": ("proposed_mask_off_valid_outputs.jsonl", "final_response"),
        "without_router_current": ("proposed_without_router_valid_outputs.jsonl", "final_response"),
        "without_scorer_current": ("proposed_without_scorer_valid_outputs.jsonl", "final_response"),
    }
    final_rows = {}
    input_hashes = {}
    ordered_ids = None
    assignments = []
    for method, (filename, response_field) in final_specs.items():
        path = VALID_ROOT / filename
        rows = read_jsonl(path)
        by_id = {row["canonical_id"]: row for row in rows}
        if len(rows) != 43 or len(by_id) != 43:
            raise RuntimeError(f"Expected 43 unique rows in {path}")
        ids = [row["canonical_id"] for row in rows]
        if ordered_ids is None:
            ordered_ids = ids
        elif set(ids) != set(ordered_ids):
            raise RuntimeError("VALID43 method membership mismatch")
        final_rows[method] = by_id
        input_hashes[filename] = sha256_file(path)
        for source_order, canonical_id in enumerate(ordered_ids):
            row = by_id[canonical_id]
            response = str(row.get(response_field) or "")
            if not response.strip():
                raise RuntimeError(f"Empty final response: {method} {canonical_id}")
            assignments.append(
                {
                    "assignment_id": f"valid43::{canonical_id}::{method}",
                    "scope": "VALID43_final",
                    "canonical_id": canonical_id,
                    "source_order": source_order,
                    "condition": method,
                    "question": row["question"],
                    "response": response,
                    "question_response_sha256": pair_hash(row["question"], response),
                    "source_path": str(path),
                    "source_response_field": response_field,
                }
            )

    base_candidate_rows = read_jsonl(FEEDBACK_ROOT / "candidate_results.jsonl")
    c_candidate_rows = read_jsonl(BC_ROOT / "c_candidate_results.jsonl")
    type_mapping = {
        "existing_denoiser": "original_denoiser",
        "feedback_A_existing_candidate": "feedback_A",
        "feedback_B_sft": "feedback_B",
        "feedback_C_sft_adapter": "feedback_C",
    }
    standardized = [standard_candidate(row, type_mapping[row["variant"]]) for row in base_candidate_rows]
    standardized += [standard_candidate(row, type_mapping[row["variant"]]) for row in c_candidate_rows]
    if len(standardized) != 36:
        raise RuntimeError(f"Expected 36 VALID9 candidate rows, got {len(standardized)}")
    valid9_ids = [row["canonical_id"] for row in sorted(standardized, key=lambda item: item["source_order"]) if row["candidate_type"] == "original_denoiser"]
    if len(valid9_ids) != 9 or len(set(valid9_ids)) != 9:
        raise RuntimeError("VALID9 membership changed")
    candidate_by = {(row["canonical_id"], row["candidate_type"]): row for row in standardized}
    if len(candidate_by) != 36:
        raise RuntimeError("Duplicate VALID9 candidates")
    for row in standardized:
        assignments.append(
            {
                "assignment_id": f"valid9_candidate::{row['canonical_id']}::{row['candidate_type']}",
                "scope": "VALID9_candidate_raw",
                "canonical_id": row["canonical_id"],
                "source_order": row["source_order"],
                "condition": row["candidate_type"],
                "question": row["question"],
                "response": row["candidate_response"],
                "question_response_sha256": pair_hash(row["question"], row["candidate_response"]),
                "source_path": (
                    str(BC_ROOT / "c_candidate_results.jsonl")
                    if row["candidate_type"] == "feedback_C"
                    else str(FEEDBACK_ROOT / "candidate_results.jsonl")
                ),
                "source_response_field": "candidate_text",
            }
        )
    if len(assignments) != 294:
        raise RuntimeError(f"Expected 294 VALID assignments, got {len(assignments)}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignments:
        grouped[row["question_response_sha256"]].append(row)
    representatives = []
    pointwise_requests = []
    seen = set()
    for row in assignments:
        key = row["question_response_sha256"]
        if key in seen:
            continue
        seen.add(key)
        members = grouped[key]
        representative = {
            **row,
            "member_assignment_ids": [item["assignment_id"] for item in members],
            "member_conditions": [item["condition"] for item in members],
            "member_scopes": [item["scope"] for item in members],
        }
        representatives.append(representative)
        user_prompt = rubric["user_prompt_template"].format(
            question=row["question"], response=row["response"]
        )
        metadata = {
            "canonical_id": row["canonical_id"],
            "source_order": row["source_order"],
            "question_response_sha256": key,
            "question_sha256": sha256_text(row["question"]),
            "response_sha256": sha256_text(row["response"]),
            "representative_assignment_id": row["assignment_id"],
            "member_assignment_ids": representative["member_assignment_ids"],
            "member_conditions": representative["member_conditions"],
            "member_scopes": representative["member_scopes"],
        }
        pointwise_requests.append(
            build_request(
                "valid_pointwise",
                "pointwise_four_axis",
                rubric["system_prompt"],
                user_prompt,
                metadata,
                384,
            )
        )
    if len(representatives) != 126:
        raise RuntimeError(f"Expected 126 VALID unique question-response pairs, got {len(representatives)}")

    original_pair_requests = read_jsonl(FEEDBACK_ROOT / "blind_judge_requests.jsonl")
    bc_pair_requests = read_jsonl(BC_ROOT / "bc_blind_judge_requests.jsonl")
    if len(original_pair_requests) != 54 or len(bc_pair_requests) != 9:
        raise RuntimeError("Existing VALID pairwise request counts changed")
    pairwise_replay = []
    for row in original_pair_requests + bc_pair_requests:
        metadata = {
            "source_comparison_id": row["comparison_id"],
            "canonical_id": row["canonical_id"],
            "source_order": row["source_order"],
            "pair": row["pair"],
            "variant_for_A": row["variant_for_A"],
            "variant_for_B": row["variant_for_B"],
            "response_A_sha256": row["response_A_sha256"],
            "response_B_sha256": row["response_B_sha256"],
            "source_llama_request_file": str(
                FEEDBACK_ROOT / "blind_judge_requests.jsonl"
                if row in original_pair_requests
                else BC_ROOT / "bc_blind_judge_requests.jsonl"
            ),
        }
        pairwise_replay.append(
            build_request(
                "valid_pairwise_replay63",
                "pairwise_four_axis",
                row["system_prompt"],
                row["user_prompt"],
                metadata,
                768,
            )
        )

    missing_pairs = (
        ("feedback_C", "sft"),
        ("feedback_C", "original_denoiser"),
        ("feedback_C", "feedback_A"),
    )
    new_pairwise = []
    new_pairwise_local = []
    for pair_index, (left, right) in enumerate(missing_pairs):
        for source_order, canonical_id in enumerate(valid9_ids):
            sft_row = final_rows["SFT"][canonical_id]
            response_map = {
                "sft": sft_row["sft_response"],
                "original_denoiser": candidate_by[(canonical_id, "original_denoiser")]["candidate_response"],
                "feedback_A": candidate_by[(canonical_id, "feedback_A")]["candidate_response"],
                "feedback_C": candidate_by[(canonical_id, "feedback_C")]["candidate_response"],
            }
            if (source_order + pair_index) % 2 == 0:
                variant_a, variant_b = left, right
            else:
                variant_a, variant_b = right, left
            response_a, response_b = response_map[variant_a], response_map[variant_b]
            user_prompt = pair_template.format(
                question=sft_row["question"],
                response_a=response_a,
                response_b=response_b,
            )
            metadata = {
                "source_comparison_id": sha256_text(
                    f"{VERSION}|{canonical_id}|{left}|{right}"
                )[:24],
                "canonical_id": canonical_id,
                "source_order": source_order,
                "pair": [left, right],
                "variant_for_A": variant_a,
                "variant_for_B": variant_b,
                "response_A_sha256": sha256_text(response_a),
                "response_B_sha256": sha256_text(response_b),
                "position_scheme": "(source_order + pair_index) parity",
            }
            request = build_request(
                "valid_pairwise_new27",
                "pairwise_four_axis",
                pair_system,
                user_prompt,
                metadata,
                768,
            )
            new_pairwise.append(request)
            new_pairwise_local.append({**request, "queue": "valid_pairwise_new27_local"})
    if len(new_pairwise) != 27:
        raise RuntimeError("Missing-pair request count changed")

    selector_requests = []
    for source_order, canonical_id in enumerate(valid9_ids):
        sft_row = final_rows["SFT"][canonical_id]
        sft = sft_row["sft_response"]
        for candidate_type in ("original_denoiser", "feedback_A", "feedback_B", "feedback_C"):
            candidate = candidate_by[(canonical_id, candidate_type)]
            if not candidate["technical_valid"]:
                continue
            for order_index, (variant_a, response_a, variant_b, response_b) in enumerate(
                (("sft", sft, "candidate", candidate["candidate_response"]), ("candidate", candidate["candidate_response"], "sft", sft))
            ):
                user_prompt = SELECTOR_USER_TEMPLATE.format(
                    question=sft_row["question"], response_a=response_a, response_b=response_b
                )
                metadata = {
                    "canonical_id": canonical_id,
                    "source_order": source_order,
                    "candidate_type": candidate_type,
                    "order_index": order_index,
                    "variant_for_A": variant_a,
                    "variant_for_B": variant_b,
                    "question_sha256": sha256_text(sft_row["question"]),
                    "response_A_sha256": sha256_text(response_a),
                    "response_B_sha256": sha256_text(response_b),
                }
                selector_requests.append(
                    build_request(
                        "valid_selector",
                        "blind_selector",
                        SELECTOR_SYSTEM_PROMPT,
                        user_prompt,
                        metadata,
                        256,
                    )
                )
    if len(selector_requests) != 72:
        raise RuntimeError(f"Expected 72 VALID selector requests, got {len(selector_requests)}")

    artifacts = {
        "valid_pointwise_assignments.jsonl": assignments,
        "valid_pointwise_representatives.jsonl": representatives,
        "valid9_candidates_standardized.jsonl": standardized,
        "requests/valid_pointwise.jsonl": pointwise_requests,
        "requests/valid_pairwise_replay63.jsonl": pairwise_replay,
        "requests/valid_pairwise_new27.jsonl": new_pairwise,
        "requests/valid_pairwise_new27_local.jsonl": new_pairwise_local,
        "requests/valid_selector.jsonl": selector_requests,
    }
    for name, rows in artifacts.items():
        write_new_or_verify(OUTPUT_DIR / name, encode_jsonl(rows))

    contract = {
        "version": VERSION,
        "status": "prepared",
        "post_hoc_exploratory_test_comparison": True,
        "input_hashes": input_hashes,
        "feedback_inputs": {
            str(FEEDBACK_ROOT / "candidate_results.jsonl"): sha256_file(FEEDBACK_ROOT / "candidate_results.jsonl"),
            str(BC_ROOT / "c_candidate_results.jsonl"): sha256_file(BC_ROOT / "c_candidate_results.jsonl"),
            str(FEEDBACK_ROOT / "blind_judge_requests.jsonl"): sha256_file(FEEDBACK_ROOT / "blind_judge_requests.jsonl"),
            str(BC_ROOT / "bc_blind_judge_requests.jsonl"): sha256_file(BC_ROOT / "bc_blind_judge_requests.jsonl"),
        },
        "counts": {
            "valid43_ids": 43,
            "valid43_final_method_assignments": 258,
            "valid9_ids": 9,
            "valid9_raw_candidate_assignments": 36,
            "pointwise_assignments": 294,
            "pointwise_unique_question_response": 126,
            "existing_pairwise_llama_preserved": 63,
            "gpt_pairwise_replay": 63,
            "new_pairwise_both_judges": 27,
            "complete_pairwise_coverage": 90,
            "selector_requests": 72,
        },
        "models": {
            "gpt": MODEL,
            "llama": "/home/user/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        },
        "pointwise_rubric_contract": str(POINTWISE_CONTRACT),
        "pointwise_rubric_contract_sha256": sha256_file(POINTWISE_CONTRACT),
        "pairwise_prompt_contract": str(FEEDBACK_ROOT / "experiment_contract.json"),
        "pairwise_prompt_contract_sha256": sha256_file(FEEDBACK_ROOT / "experiment_contract.json"),
        "selector_prompt_hashes": {
            "system": sha256_text(SELECTOR_SYSTEM_PROMPT),
            "user_template": sha256_text(SELECTOR_USER_TEMPLATE),
        },
        "request_files": {
            name: {"sha256": sha256_file(OUTPUT_DIR / name), "rows": len(rows)}
            for name, rows in artifacts.items()
            if name.startswith("requests/")
        },
        "no_feedback_regeneration": True,
        "no_candidate_regeneration": True,
        "no_training": True,
    }
    write_new_or_verify(
        OUTPUT_DIR / "valid_request_contract.json",
        (json.dumps(contract, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    print(json.dumps(contract["counts"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
