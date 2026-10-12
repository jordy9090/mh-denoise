#!/usr/bin/env python3
"""Prepare, run-cache, and aggregate the frozen selector for TEST60 ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "acl-component-ablation-matrix-v1-20261009"
ROOT = Path("/mnt/ssd00/hsoh/acl_component_ablation_matrix_20261009_v1")
CORRECTED = Path("/home/user/hsoh/test60_rubric_aligned_final_selection_20261009_v2_test40_fix")
FROZEN = Path("/home/user/hsoh/rubric_aligned_gpt41_rejudge_20261009_v1")
MODEL = "gpt-4.1-2025-04-14"
QUEUE = "acl_component_selector"
FINAL_SOURCE = CORRECTED / "test60_final_outputs.jsonl"
MAIN_SOURCE = ROOT / "inputs/test60_main_proposed_frozen.jsonl"
ALWAYS_NEW = ROOT / "generation/main_always_on_missing42.jsonl"
CONTROL = ROOT / "generation/original_prompt_sft_control_test18.jsonl"
COMPONENT_FILES = {
    "MaskON": ROOT / "generation/proposed_mask_on_test60.jsonl",
    "WithoutRouter": ROOT / "generation/proposed_without_router_test60.jsonl",
    "WithoutScorer": ROOT / "generation/proposed_without_scorer_test60.jsonl",
}
CACHE_ROOTS = (
    FROZEN,
    Path("/home/user/hsoh/test60_rubric_aligned_final_selection_20261009_v1"),
    CORRECTED,
    Path("/mnt/ssd00/hsoh/original_prompt_sft_control_20261009_v1"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_key(row: dict[str, Any]) -> str:
    material = {
        "model": row.get("model"),
        "temperature": row.get("temperature"),
        "max_tokens": row.get("max_tokens"),
        "response_format": row.get("response_format"),
        "parser": row.get("parser"),
        "system_prompt": row.get("system_prompt"),
        "user_prompt": row.get("user_prompt"),
    }
    return sha256_text(json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def response_hash(question: str, response: str) -> str:
    return sha256_text(json.dumps([question, response], ensure_ascii=False, separators=(",", ":")))


def technical_valid(response: str) -> bool:
    text = str(response or "").strip()
    return bool(text) and text not in {"**Final", "Final", "**Final Response:**"}


def frozen_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final = sorted(read_jsonl(FINAL_SOURCE), key=lambda row: int(row["source_order"]))
    main = sorted(read_jsonl(MAIN_SOURCE), key=lambda row: int(row["source_order"]))
    if len(final) != 60 or len(main) != 60:
        raise RuntimeError("Corrected final and main generation must each contain 60 rows")
    for expected, (final_row, main_row) in enumerate(zip(final, main)):
        if int(final_row["source_order"]) != expected or int(main_row["source_order"]) != expected:
            raise RuntimeError("TEST60 order is not 0..59")
        if final_row["canonical_id"] != main_row["canonical_id"]:
            raise RuntimeError("Corrected final and main generation ID order differs")
        if final_row["question"] != main_row["question"] or final_row["sft_response"] != main_row["sft_response"]:
            raise RuntimeError(f"Frozen input mismatch for {final_row['canonical_id']}")
    return final, main


def standard_candidate(
    arm: str,
    row: dict[str, Any],
    response: str,
    raw: str,
    generation: dict[str, Any] | None,
    risk: float | None,
    focus_risk: float | None,
    call_reason: str,
    source: str,
) -> dict[str, Any]:
    generation_valid = True
    if generation is not None:
        generation_valid = (
            not bool(generation.get("length_limit_reached"))
            and not bool(generation.get("source_input_truncated"))
            and generation.get("finish_reason") != "length"
        )
    return {
        "version": VERSION,
        "split": "TEST",
        "arm": arm,
        "source_order": int(row["source_order"]),
        "case_id": row["case_id"],
        "canonical_id": row["canonical_id"],
        "question": row["question"],
        "sft_response": row["sft_response"],
        "candidate_response": response,
        "candidate_response_raw": raw,
        "candidate_generation": generation,
        "sft_risk_score": row.get("sft_risk_score"),
        "candidate_risk_score": risk,
        "candidate_focus_risk_score": focus_risk,
        "call_reason": call_reason,
        "technical_valid": technical_valid(response) and generation_valid,
        "source": source,
        "question_sha256": sha256_text(row["question"]),
        "sft_response_sha256": sha256_text(row["sft_response"]),
        "candidate_response_sha256": sha256_text(response),
        "sft_question_response_sha256": response_hash(row["question"], row["sft_response"]),
        "candidate_question_response_sha256": response_hash(row["question"], response),
    }


def load_candidates() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    final, main = frozen_rows()
    final_by_id = {row["canonical_id"]: row for row in final}
    main_by_id = {row["canonical_id"]: row for row in main}
    original = []
    for row in main:
        if not row.get("used_denoiser"):
            continue
        original.append(
            standard_candidate(
                "OriginalDenoiser",
                row,
                str(row.get("denoiser_response") or ""),
                str(row.get("denoiser_response_raw") or ""),
                row.get("denoiser_generation"),
                row.get("denoiser_risk_score"),
                row.get("denoiser_focus_risk_score"),
                str(row.get("denoiser_call_reason") or ""),
                str(MAIN_SOURCE),
            )
        )
    if len(original) != 18:
        raise RuntimeError(f"Expected 18 original candidates, got {len(original)}")

    always_new_rows = sorted(read_jsonl(ALWAYS_NEW), key=lambda row: int(row["source_order"]))
    if len(always_new_rows) != 42 or any(not row.get("used_denoiser") for row in always_new_rows):
        raise RuntimeError("Always-on missing output must contain 42 called rows")
    always_by_id = {row["canonical_id"]: row for row in always_new_rows}
    for row in main:
        if row.get("used_denoiser"):
            always_by_id[row["canonical_id"]] = row
    if len(always_by_id) != 60:
        raise RuntimeError("Merged always-on candidate set is not complete")
    always = []
    for row in main:
        candidate = always_by_id[row["canonical_id"]]
        if candidate["question"] != row["question"] or candidate["sft_response"] != row["sft_response"]:
            raise RuntimeError(f"Always-on input mismatch for {row['canonical_id']}")
        always.append(
            standard_candidate(
                "AlwaysOnDenoiser",
                row,
                str(candidate.get("denoiser_response") or ""),
                str(candidate.get("denoiser_response_raw") or ""),
                candidate.get("denoiser_generation"),
                candidate.get("denoiser_risk_score"),
                candidate.get("denoiser_focus_risk_score"),
                str(candidate.get("denoiser_call_reason") or "always_on"),
                str(MAIN_SOURCE if row.get("used_denoiser") else ALWAYS_NEW),
            )
        )

    controls = []
    for row in sorted(read_jsonl(CONTROL), key=lambda item: int(item["source_order"])):
        source = main_by_id[row["canonical_id"]]
        if row["question"] != source["question"] or row["sft_response"] != source["sft_response"]:
            raise RuntimeError(f"Control input mismatch for {row['canonical_id']}")
        controls.append(
            standard_candidate(
                "OriginalPromptSFTControl",
                source,
                str(row.get("control_response") or ""),
                str(row.get("control_raw_decoded_output") or ""),
                row.get("control_generation"),
                (row.get("control_metrics") or {}).get("risk_score"),
                (row.get("control_metrics") or {}).get("focus_risk_score"),
                str(source.get("denoiser_call_reason") or ""),
                str(CONTROL),
            )
        )
    if len(controls) != 18:
        raise RuntimeError(f"Expected 18 SFT-adapter controls, got {len(controls)}")

    component_sets: dict[str, list[dict[str, Any]]] = {}
    for arm, path in COMPONENT_FILES.items():
        rows = sorted(read_jsonl(path), key=lambda row: int(row["source_order"]))
        if len(rows) != 60:
            raise RuntimeError(f"{arm} output has {len(rows)} rows")
        candidates = []
        for row in rows:
            final_row = final_by_id[row["canonical_id"]]
            if row["question"] != final_row["question"] or row["sft_response"] != final_row["sft_response"]:
                raise RuntimeError(f"{arm} input mismatch for {row['canonical_id']}")
            if not row.get("used_denoiser"):
                continue
            row_with_case = {**row, "case_id": final_row["case_id"]}
            candidates.append(
                standard_candidate(
                    arm,
                    row_with_case,
                    str(row.get("denoiser_response") or ""),
                    str(row.get("denoiser_response_raw") or ""),
                    row.get("denoiser_generation"),
                    row.get("denoiser_risk_score"),
                    row.get("denoiser_focus_risk_score"),
                    str(row.get("denoiser_call_reason") or ""),
                    str(path),
                )
            )
        component_sets[arm] = candidates
    selector_sets = {
        "AlwaysOnDenoiser": always,
        "OriginalPromptSFTControl": controls,
        **component_sets,
    }
    all_candidates = original + always + controls
    for rows in component_sets.values():
        all_candidates.extend(rows)
    write_jsonl(ROOT / "inputs/standardized_candidates.jsonl", all_candidates)
    write_json(
        ROOT / "inputs/candidate_coverage.json",
        {
            "OriginalDenoiser": len(original),
            "AlwaysOnDenoiser": len(always),
            "OriginalPromptSFTControl": len(controls),
            **{arm: len(rows) for arm, rows in component_sets.items()},
        },
    )
    return original, selector_sets


def cache_results() -> tuple[dict[str, tuple[Path, dict[str, Any]]], list[dict[str, Any]]]:
    by_key: dict[str, tuple[Path, dict[str, Any]]] = {}
    conflicts = []
    seen_paths: set[Path] = set()
    for root in CACHE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            in_results_dir = any(part.lower() == "results" for part in path.parts)
            if path in seen_paths or ("result" not in path.name.lower() and not in_results_dir):
                continue
            seen_paths.add(path)
            try:
                rows = read_jsonl(path)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            for row in rows:
                if row.get("parser") != "blind_selector" or row.get("ok") is not True:
                    continue
                if row.get("model") != MODEL or row.get("returned_model") != MODEL:
                    continue
                if row.get("temperature") != 0 or row.get("max_tokens") != 256:
                    continue
                if row.get("response_format") != {"type": "json_object"}:
                    continue
                choice = str((row.get("normalized") or {}).get("choice") or "")
                if choice not in {"A", "B", "tie"}:
                    continue
                key = request_key(row)
                prior = by_key.get(key)
                if prior is not None and (prior[1].get("normalized") or {}).get("choice") != choice:
                    conflicts.append(
                        {
                            "request_key": key,
                            "first_path": str(prior[0]),
                            "second_path": str(path),
                            "first_choice": (prior[1].get("normalized") or {}).get("choice"),
                            "second_choice": choice,
                        }
                    )
                    continue
                by_key[key] = (path, row)
    return by_key, conflicts


def make_request(
    candidate: dict[str, Any],
    order_index: int,
    system_prompt: str,
    user_template: str,
) -> dict[str, Any]:
    if order_index == 0:
        response_a, response_b = candidate["sft_response"], candidate["candidate_response"]
        variant_a, variant_b = "sft", "candidate"
    else:
        response_a, response_b = candidate["candidate_response"], candidate["sft_response"]
        variant_a, variant_b = "candidate", "sft"
    user_prompt = user_template.format(
        question=candidate["question"], response_a=response_a, response_b=response_b
    )
    metadata = {
        "split": "TEST",
        "arm": candidate["arm"],
        "canonical_id": candidate["canonical_id"],
        "case_id": candidate["case_id"],
        "source_order": candidate["source_order"],
        "order_index": order_index,
        "variant_for_A": variant_a,
        "variant_for_B": variant_b,
        "question_sha256": candidate["question_sha256"],
        "response_A_sha256": sha256_text(response_a),
        "response_B_sha256": sha256_text(response_b),
    }
    request_id = sha256_text(
        json.dumps(
            {"version": VERSION, "queue": QUEUE, "metadata": metadata, "user_prompt": user_prompt},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )[:32]
    return {
        "version": VERSION,
        "queue": QUEUE,
        "request_id": request_id,
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
        "parser": "blind_selector",
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_sha256": sha256_text(user_prompt),
        "metadata": metadata,
    }


def prepare() -> None:
    original, selector_sets = load_candidates()
    contract = read_json(FROZEN / "run_contract.json")
    settings = contract["model_settings"]
    prompts = contract["prompts"]
    if settings["model"] != MODEL or settings["temperature"] != 0 or settings["selector_max_tokens"] != 256:
        raise RuntimeError("Frozen selector settings changed")
    system_prompt = prompts["selector_system"]
    user_template = prompts["selector_user_template"]
    if sha256_text(system_prompt) != prompts["selector_system_sha256"]:
        raise RuntimeError("Frozen selector system prompt hash mismatch")
    if sha256_text(user_template) != prompts["selector_user_template_sha256"]:
        raise RuntimeError("Frozen selector user prompt hash mismatch")

    expected = []
    technical_fallbacks = []
    for arm, candidates in selector_sets.items():
        for candidate in candidates:
            if not candidate["technical_valid"]:
                technical_fallbacks.append(
                    {
                        "arm": arm,
                        "canonical_id": candidate["canonical_id"],
                        "case_id": candidate["case_id"],
                        "reason": "candidate_technical_invalid",
                    }
                )
                continue
            if candidate["candidate_response"] == candidate["sft_response"]:
                technical_fallbacks.append(
                    {
                        "arm": arm,
                        "canonical_id": candidate["canonical_id"],
                        "case_id": candidate["case_id"],
                        "reason": "candidate_identical_to_sft",
                    }
                )
                continue
            expected.extend(
                make_request(candidate, order, system_prompt, user_template) for order in (0, 1)
            )
    if len({row["request_id"] for row in expected}) != len(expected):
        raise RuntimeError("Duplicate selector request IDs")

    cache, conflicts = cache_results()
    reused = []
    new = []
    for request in expected:
        key = request_key(request)
        cached = cache.get(key)
        if cached is None:
            new.append(request)
            continue
        source_path, result = cached
        reused.append(
            {
                "target_request_id": request["request_id"],
                "target_request_key": key,
                "target_metadata": request["metadata"],
                "source_result_path": str(source_path),
                "source_request_id": result.get("request_id"),
                "source_request_sha256": result.get("request_sha256"),
                "normalized": result["normalized"],
                "raw_json_text": result.get("raw_json_text"),
                "returned_model": result.get("returned_model"),
                "system_fingerprint": result.get("system_fingerprint"),
                "completed_at_utc": result.get("completed_at_utc"),
            }
        )
    write_jsonl(ROOT / "requests/expected_selector_requests.jsonl", expected)
    write_jsonl(ROOT / "requests/new_selector_requests.jsonl", new)
    write_jsonl(ROOT / "results/reused_selector_cache.jsonl", reused)
    write_jsonl(ROOT / "results/selector_cache_conflicts.jsonl", conflicts)
    write_jsonl(ROOT / "results/selector_technical_fallbacks.jsonl", technical_fallbacks)
    selector_contract = {
        "version": VERSION,
        "status": "prepared_before_api_execution",
        "created_at_utc": utc_now(),
        "frozen_contract": {
            "path": str((FROZEN / "run_contract.json").resolve()),
            "sha256": sha256_file(FROZEN / "run_contract.json"),
        },
        "settings": {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
            "parser": "blind_selector",
        },
        "prompts": {
            "system_sha256": sha256_text(system_prompt),
            "user_template_sha256": sha256_text(user_template),
        },
        "rule": {
            "orders": 2,
            "adopt": "candidate selected in both reversed orders",
            "fallback": "tie, disagreement, final API/parse failure, or technical-invalid candidate keeps SFT",
            "risk_or_length_constraint": False,
        },
        "stored_candidate_policy_replays": {
            "legacy_strict": "historical risk, 20-word, bad-safety, specificity-ratio, genericity, and keyword-overlap conditions",
            "current_rules": "stored surface_relaxed_v1 final output",
            "no_min_length": "surface_relaxed_v1 acceptance with only the final 20-word candidate condition removed",
            "risk_only": "finite sft_risk - candidate_risk >= 0.01",
            "accept_generated": "all technically valid candidates on the frozen called set; represented by GatedNoSelector",
            "gpt_selector": "candidate selected in both reversed orders under the frozen expert-aligned prompt",
        },
        "candidate_counts": {arm: len(rows) for arm, rows in selector_sets.items()},
        "original_candidate_count_not_reselected": len(original),
        "requests": {
            "expected": len(expected),
            "reused_exact": len(reused),
            "new": len(new),
            "selector_skipped_cases": len(technical_fallbacks),
            "cache_conflicts_excluded": len(conflicts),
        },
    }
    write_json(ROOT / "selector_contract.json", selector_contract)
    print(json.dumps(selector_contract["requests"], sort_keys=True))


def result_map() -> dict[str, dict[str, Any]]:
    mapped = {}
    for row in read_jsonl(ROOT / "results/reused_selector_cache.jsonl"):
        mapped[row["target_request_id"]] = {
            "ok": True,
            "normalized": row["normalized"],
            "provenance": "exact_cache_reuse",
            "source_result_path": row["source_result_path"],
        }
    for path in sorted((ROOT / "results").glob("new_selector*_results.jsonl")):
        for row in read_jsonl(path):
            mapped[row["request_id"]] = {**row, "provenance": "new_api"}
    return mapped


def build_final_methods(
    decisions: list[dict[str, Any]],
    original: list[dict[str, Any]],
    selector_sets: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final, main = frozen_rows()
    main_by_id = {row["canonical_id"]: row for row in main}
    decisions_by_key = {(row["arm"], row["canonical_id"]): row for row in decisions}
    candidates_by_arm = {
        arm: {row["canonical_id"]: row for row in rows} for arm, rows in selector_sets.items()
    }
    original_by_id = {row["canonical_id"]: row for row in original}
    base_by_id = {
        row["canonical_id"]: row
        for row in read_jsonl(ROOT / "generation/base_prompt_rewrite_test60.jsonl")
    }
    roles = {
        "BasePromptRewrite": "main_baseline",
        "SFT": "main_baseline",
        "DPO": "main_baseline",
        "Proposed": "main_method",
        "GatedNoSelector": "ablation",
        "AlwaysOnNoSelector": "ablation",
        "AlwaysOnGPTSelector": "ablation",
        "OriginalPromptSFTControlGPTSelector": "ablation",
        "MaskONGPTSelector": "ablation",
        "WithoutRouterGPTSelector": "ablation",
        "WithoutScorerGPTSelector": "ablation",
        "LegacyStrict": "policy_sensitivity",
        "CurrentRules": "policy_sensitivity",
        "NoMinLength": "policy_sensitivity",
        "RiskOnly": "policy_sensitivity",
        "UnsafeResponse": "diagnostic_lower_bound",
        "ReferenceResponse": "diagnostic_oracle",
    }
    outputs = []
    candidate_quality = []

    def selected(arm: str, row: dict[str, Any]) -> tuple[str, str, bool, str]:
        candidate = candidates_by_arm[arm].get(row["canonical_id"])
        if candidate is None:
            return row["sft_response"], "SFT_gate_off", False, "gate_not_called"
        decision = decisions_by_key.get((arm, row["canonical_id"]))
        if decision is None or not decision["accepted_candidate"]:
            reason = decision["selection_reason"] if decision else "technical_invalid"
            return row["sft_response"], "SFT_fallback", False, reason
        return candidate["candidate_response"], "candidate", True, "selector_consensus"

    for row in final:
        row_id = row["canonical_id"]
        main_row = main_by_id[row_id]
        original_candidate = original_by_id.get(row_id)
        always_candidate = candidates_by_arm["AlwaysOnDenoiser"][row_id]
        current_rules = str(main_row["final_response"])
        risk_only_accept = False
        no_min_length_accept = False
        legacy_strict_accept = False
        if original_candidate is not None and original_candidate["technical_valid"]:
            before = original_candidate.get("sft_risk_score")
            after = original_candidate.get("candidate_risk_score")
            risk_only_accept = (
                isinstance(before, (int, float))
                and isinstance(after, (int, float))
                and math.isfinite(float(before))
                and math.isfinite(float(after))
                and float(before) - float(after) >= 0.01
            )
            source_metrics = main_by_id[row_id]
            finite_risk_improvement = risk_only_accept
            bad_safety_ok = int(source_metrics.get("denoiser_bad_safety_count") or 0) <= int(
                source_metrics.get("sft_bad_safety_count") or 0
            )
            min_overlap = max(0.0, float(source_metrics.get("sft_keyword_overlap") or 0.0) - 0.15)
            overlap_ok = float(source_metrics.get("denoiser_keyword_overlap") or 0.0) >= min_overlap
            no_min_length_accept = finite_risk_improvement and bad_safety_ok and overlap_ok
            legacy_strict_accept = (
                no_min_length_accept
                and int(source_metrics.get("denoiser_word_count") or 0) >= 20
                and float(source_metrics.get("specificity_ratio") or 0.0) >= 0.60
                and int(source_metrics.get("denoiser_generic_count") or 0)
                <= int(source_metrics.get("sft_generic_count") or 0)
            )
        risk_only_response = (
            original_candidate["candidate_response"] if risk_only_accept else row["sft_response"]
        )
        selector_values = {
            "AlwaysOnGPTSelector": selected("AlwaysOnDenoiser", row),
            "OriginalPromptSFTControlGPTSelector": selected("OriginalPromptSFTControl", row),
            "MaskONGPTSelector": selected("MaskON", row),
            "WithoutRouterGPTSelector": selected("WithoutRouter", row),
            "WithoutScorerGPTSelector": selected("WithoutScorer", row),
        }
        methods: dict[str, tuple[str, str, bool, str]] = {
            "BasePromptRewrite": (
                base_by_id[row_id]["base_prompt_response"],
                "base_model_same_sft_prompt",
                True,
                "no_selector",
            ),
            "SFT": (row["sft_response"], "SFT", False, "baseline"),
            "DPO": (row["dpo_response"], "DPO", True, "baseline"),
            "Proposed": (
                row["final_response"],
                row["final_response_source"],
                bool(row["gpt_selector_accepted_candidate"]),
                row["gpt_selector_selection_reason"],
            ),
            "GatedNoSelector": (
                original_candidate["candidate_response"] if original_candidate else row["sft_response"],
                "candidate" if original_candidate else "SFT_gate_off",
                original_candidate is not None,
                "accept_generated" if original_candidate else "gate_not_called",
            ),
            "AlwaysOnNoSelector": (
                always_candidate["candidate_response"] if always_candidate["technical_valid"] else row["sft_response"],
                "candidate" if always_candidate["technical_valid"] else "SFT_technical_fallback",
                bool(always_candidate["technical_valid"]),
                "accept_generated" if always_candidate["technical_valid"] else "candidate_technical_invalid",
            ),
            **selector_values,
            "LegacyStrict": (
                original_candidate["candidate_response"] if legacy_strict_accept else row["sft_response"],
                "candidate" if legacy_strict_accept else "SFT_fallback",
                legacy_strict_accept,
                "legacy_strict_accepted" if legacy_strict_accept else "legacy_strict_rejected_or_gate_off",
            ),
            "CurrentRules": (
                current_rules,
                "candidate" if current_rules != row["sft_response"] else "SFT_fallback",
                current_rules != row["sft_response"],
                str(main_row.get("reject_reason") or "current_rules_accepted"),
            ),
            "NoMinLength": (
                original_candidate["candidate_response"] if no_min_length_accept else row["sft_response"],
                "candidate" if no_min_length_accept else "SFT_fallback",
                no_min_length_accept,
                "surface_relaxed_without_min_length_accepted"
                if no_min_length_accept
                else "surface_relaxed_without_min_length_rejected_or_gate_off",
            ),
            "RiskOnly": (
                risk_only_response,
                "candidate" if risk_only_accept else "SFT_fallback",
                risk_only_accept,
                "risk_delta_at_least_0.01" if risk_only_accept else "risk_delta_below_0.01_or_gate_off",
            ),
            "UnsafeResponse": (str(main_row.get("unsafe_response") or ""), "unsafe_source", True, "diagnostic"),
            "ReferenceResponse": (str(main_row.get("safe_response") or ""), "stored_reference", True, "diagnostic"),
        }
        if set(methods) != set(roles):
            raise RuntimeError("Final method set differs from frozen roles")
        for method, (response, source, accepted, reason) in methods.items():
            outputs.append(
                {
                    "version": VERSION,
                    "split": "TEST",
                    "scope": "TEST60_final_policy_output",
                    "source_order": row["source_order"],
                    "case_id": row["case_id"],
                    "canonical_id": row_id,
                    "method": method,
                    "role": roles[method],
                    "question": row["question"],
                    "response": response,
                    "final_source": source,
                    "candidate_accepted": accepted,
                    "selection_reason": reason,
                    "text_differs_from_sft": response != row["sft_response"],
                    "question_sha256": sha256_text(row["question"]),
                    "response_sha256": sha256_text(response),
                    "question_response_sha256": response_hash(row["question"], response),
                }
            )

    candidate_method_names = {
        "OriginalDenoiser": "OriginalDenoiserCandidate",
        "AlwaysOnDenoiser": "AlwaysOnDenoiserCandidate",
        "OriginalPromptSFTControl": "OriginalPromptSFTControlCandidate",
        "MaskON": "MaskONCandidate",
        "WithoutRouter": "WithoutRouterCandidate",
        "WithoutScorer": "WithoutScorerCandidate",
    }
    all_sets = {"OriginalDenoiser": original, **selector_sets}
    for arm, rows in all_sets.items():
        for candidate in rows:
            candidate_quality.append(
                {
                    "version": VERSION,
                    "split": "TEST",
                    "scope": "called_candidate_quality",
                    "source_order": candidate["source_order"],
                    "case_id": candidate["case_id"],
                    "canonical_id": candidate["canonical_id"],
                    "method": candidate_method_names[arm],
                    "arm": arm,
                    "question": candidate["question"],
                    "response": candidate["candidate_response"],
                    "technical_valid": candidate["technical_valid"],
                    "sft_risk_score": candidate["sft_risk_score"],
                    "candidate_risk_score": candidate["candidate_risk_score"],
                    "question_sha256": candidate["question_sha256"],
                    "response_sha256": candidate["candidate_response_sha256"],
                    "question_response_sha256": candidate["candidate_question_response_sha256"],
                }
            )
    return outputs, candidate_quality


def aggregate() -> None:
    original, selector_sets = load_candidates()
    expected = read_jsonl(ROOT / "requests/expected_selector_requests.jsonl")
    expected_by_id = {row["request_id"]: row for row in expected}
    results = result_map()
    decisions = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for request in expected:
        grouped[(request["metadata"]["arm"], request["metadata"]["canonical_id"])].append(request)
    technical = {
        (row["arm"], row["canonical_id"]): row
        for row in read_jsonl(ROOT / "results/selector_technical_fallbacks.jsonl")
    }
    all_case_keys = []
    for arm, candidates in selector_sets.items():
        all_case_keys.extend((arm, row["canonical_id"], row) for row in candidates)
    for arm, row_id, candidate in all_case_keys:
        if (arm, row_id) in technical:
            skip_reason = technical[(arm, row_id)]["reason"]
            decisions.append(
                {
                    "arm": arm,
                    "source_order": candidate["source_order"],
                    "case_id": candidate["case_id"],
                    "canonical_id": row_id,
                    "accepted_candidate": False,
                    "selection_reason": skip_reason,
                    "orders": [],
                }
            )
            continue
        requests = sorted(grouped[(arm, row_id)], key=lambda row: int(row["metadata"]["order_index"]))
        if len(requests) != 2:
            raise RuntimeError(f"Expected two selector requests for {arm}/{row_id}")
        orders = []
        candidate_selected = []
        failures = []
        choices = []
        for request in requests:
            result = results.get(request["request_id"])
            if result is None or result.get("ok") is not True:
                failures.append("missing_or_failed_result")
                orders.append({"request_id": request["request_id"], "ok": False})
                continue
            choice = str((result.get("normalized") or {}).get("choice") or "")
            if choice not in {"A", "B", "tie"}:
                failures.append("invalid_choice")
                orders.append({"request_id": request["request_id"], "ok": False})
                continue
            order_index = int(request["metadata"]["order_index"])
            selected = choice == ("B" if order_index == 0 else "A")
            choices.append(choice)
            candidate_selected.append(selected)
            orders.append(
                {
                    "request_id": request["request_id"],
                    "order_index": order_index,
                    "choice": choice,
                    "candidate_selected": selected,
                    "provenance": result.get("provenance"),
                    "rationale": (result.get("normalized") or {}).get("rationale"),
                }
            )
        accepted = len(candidate_selected) == 2 and all(candidate_selected) and not failures
        if failures:
            reason = "technical_failure"
        elif "tie" in choices:
            reason = "tie"
        elif candidate_selected[0] != candidate_selected[1]:
            reason = "order_disagreement"
        elif accepted:
            reason = "candidate_selected_both_orders"
        else:
            reason = "sft_selected_both_orders"
        decisions.append(
            {
                "arm": arm,
                "source_order": candidate["source_order"],
                "case_id": candidate["case_id"],
                "canonical_id": row_id,
                "accepted_candidate": accepted,
                "selection_reason": reason,
                "orders": orders,
            }
        )
    decisions.sort(key=lambda row: (row["arm"], int(row["source_order"])))
    write_jsonl(ROOT / "results/selector_decisions.jsonl", decisions)
    summaries = []
    for arm in selector_sets:
        rows = [row for row in decisions if row["arm"] == arm]
        summaries.append(
            {
                "arm": arm,
                "called_candidates": len(rows),
                "accepted": sum(row["accepted_candidate"] for row in rows),
                "reasons": dict(Counter(row["selection_reason"] for row in rows)),
            }
        )
    write_jsonl(ROOT / "results/selector_summary.jsonl", summaries)
    outputs, candidates = build_final_methods(decisions, original, selector_sets)
    write_jsonl(ROOT / "results/final_method_outputs.jsonl", outputs)
    write_jsonl(ROOT / "results/candidate_quality_outputs.jsonl", candidates)
    status = {
        "status": "complete" if all(row["selection_reason"] != "technical_failure" for row in decisions) else "complete_with_failures",
        "completed_at_utc": utc_now(),
        "selector_cases": len(decisions),
        "selector_requests": len(expected),
        "new_api_results_seen": sum(
            result.get("provenance") == "new_api" for result in results.values()
        ),
        "exact_cache_results_seen": sum(
            result.get("provenance") == "exact_cache_reuse" for result in results.values()
        ),
        "final_method_assignments": len(outputs),
        "final_methods": len({row["method"] for row in outputs}),
        "candidate_quality_assignments": len(candidates),
        "selector_summary": summaries,
    }
    write_json(ROOT / "results/selector_completion.json", status)
    print(json.dumps(status, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "aggregate"])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        aggregate()


if __name__ == "__main__":
    main()
