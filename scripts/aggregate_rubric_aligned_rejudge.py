#!/usr/bin/env python3
"""Aggregate aligned-rubric GPT/Llama scores, selector replay, policy quality, and risk."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VERSION = "rubric-aligned-gpt41-rejudge-complete-v1-20261009"
ROOT = Path("/home/user/hsoh/rubric_aligned_gpt41_rejudge_20261009_v1")
REPO = Path("/home/user/hsoh/mh-denoise")
PRIOR = Path("/mnt/ssd00/hsoh/gpt41_posthoc_selection_20261009_v1")
CONTROL = Path("/mnt/ssd00/hsoh/original_prompt_sft_control_20261009_v1")
CHECKPOINT_ROOT = REPO / "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/gemma"
TEST_ROOT = CHECKPOINT_ROOT / "generation_1536_test_surface_relaxed_v1_20261008/part24"
VALID_ROOT = CHECKPOINT_ROOT / "generation_1536"
QUALITY_AXES = ("overall_quality", "empathy", "specificity")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def pair_hash(question: str, response: str) -> str:
    return sha256_text(question + "\0" + response)


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def latest_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    latest = {}
    for path in paths:
        for row in read_jsonl(path):
            latest[row["request_id"]] = row
    return latest


def quality_mean(scores: dict[str, Any]) -> float:
    return statistics.mean(float(scores[axis]) for axis in QUALITY_AXES)


def score_maps() -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    response_by_pair = {
        row["question_response_sha256"]: row["response"]
        for row in read_jsonl(ROOT / "inputs/pointwise_assignments.jsonl")
    }
    gpt_paths = [
        ROOT / "results/expert15_pointwise_gpt41.jsonl",
        ROOT / "results/remaining_pointwise_gpt41.jsonl",
        ROOT / "results/pointwise_evidence_repair_gpt41.jsonl",
    ]
    gpt_latest = latest_rows(gpt_paths)
    gpt = {}
    for row in gpt_latest.values():
        if row.get("ok") is not True:
            continue
        gpt[row["metadata"]["question_response_sha256"]] = row["normalized"]
    if len(gpt) != 193:
        raise RuntimeError(f"GPT pointwise coverage incomplete: {len(gpt)}/193")
    nonverbatim_evidence = [
        response_pair_hash
        for response_pair_hash, result in gpt.items()
        if result["medical_advice"] in {"yes", "unsure"}
        and result["medical_advice_evidence"] not in response_by_pair[response_pair_hash]
    ]
    if nonverbatim_evidence:
        raise RuntimeError(f"Non-verbatim Medical Advice evidence remains: {nonverbatim_evidence}")

    llama_paths = [
        ROOT / "results/expert15_pointwise_llama.jsonl",
        ROOT / "results/remaining_pointwise_llama.jsonl",
    ]
    llama_latest = latest_rows(llama_paths)
    llama = {}
    llama_evidence_validation_failures = []
    for row in llama_latest.values():
        if row.get("judge_ok") is not True:
            continue
        response_pair_hash = row["metadata"]["question_response_sha256"]
        result = row["judge_normalized"]
        if (
            result["medical_advice"] in {"yes", "unsure"}
            and result["medical_advice_evidence"] not in response_by_pair[response_pair_hash]
        ):
            llama_evidence_validation_failures.append(row["request_id"])
            continue
        llama[response_pair_hash] = result
    llama_manifest_path = ROOT / "results/llama_rubric_aligned_manifest.json"
    llama_status = (
        json.loads(llama_manifest_path.read_text(encoding="utf-8"))
        if llama_manifest_path.exists()
        else {"status": "running_or_unavailable"}
    )
    maps = {"GPT-4.1": gpt}
    if llama_latest:
        maps["Llama"] = llama
    return maps, {
        "gpt_unique_scored": len(gpt),
        "llama_unique_results": len(llama_latest),
        "llama_unique_scored": len(llama),
        "llama_parse_failures": len(llama_latest) - len(llama),
        "llama_runner_parse_failures": sum(row.get("judge_ok") is not True for row in llama_latest.values()),
        "llama_evidence_validation_failures": len(llama_evidence_validation_failures),
        "llama_evidence_validation_failure_request_ids": llama_evidence_validation_failures,
        "llama_manifest": llama_status,
    }


def selector_decisions() -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    paths = [
        ROOT / "results/expert15_original_selector_gpt41.jsonl",
        ROOT / "results/remaining_selector_gpt41.jsonl",
    ]
    latest = latest_rows(paths)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in latest.values():
        metadata = row["metadata"]
        grouped[(metadata["split"], metadata["canonical_id"], metadata["candidate_type"])].append(row)
    candidates = read_jsonl(ROOT / "inputs/selector_candidates.jsonl")
    decisions = {}
    failures = []
    for candidate in candidates:
        key = (candidate["split"], candidate["canonical_id"], candidate["candidate_type"])
        rows = grouped.get(key, [])
        if not candidate["technical_valid"]:
            decisions[key] = {
                "accepted": False, "reason": "technical_invalid", "mapped_choices": [],
                "request_ids": [], "rationales": [],
            }
            continue
        if len(rows) != 2 or any(row.get("ok") is not True for row in rows):
            decisions[key] = {
                "accepted": False, "reason": "selector_failure", "mapped_choices": [],
                "request_ids": [row["request_id"] for row in rows], "rationales": [],
            }
            failures.append({"split": key[0], "canonical_id": key[1], "candidate_type": key[2], "reason": "missing_or_failed_order"})
            continue
        ordered = sorted(rows, key=lambda row: row["metadata"]["order_index"])
        mapped = []
        for row in ordered:
            choice = row["normalized"]["choice"]
            mapped.append("tie" if choice == "tie" else row["metadata"][f"variant_for_{choice}"])
        accepted = mapped == ["candidate", "candidate"]
        decisions[key] = {
            "accepted": accepted,
            "reason": "candidate_selected_both_orders" if accepted else (
                "selector_tie" if "tie" in mapped else "selector_kept_sft_or_order_disagreement"
            ),
            "mapped_choices": mapped,
            "request_ids": [row["request_id"] for row in ordered],
            "rationales": [row["normalized"]["rationale"] for row in ordered],
        }
    if len(decisions) != 61:
        raise RuntimeError(f"Selector decision coverage incomplete: {len(decisions)}/61")
    return decisions, failures


def old_selector_acceptance() -> dict[tuple[str, str, str], bool]:
    old = {}
    for row in read_jsonl(PRIOR / "test_policy_replay.jsonl"):
        if row["policy"] == "gpt_selector" and row["gate_called"]:
            old[("TEST", row["canonical_id"], "original_denoiser")] = bool(row["accepted"])
    for row in read_jsonl(PRIOR / "valid9_policy_replay.jsonl"):
        if row["policy"] == "gpt_selector":
            old[("VALID", row["canonical_id"], row["candidate_type"])] = bool(row["accepted"])
    for row in read_jsonl(CONTROL / "results/policy_replay.jsonl"):
        if row["policy"] == "gpt_selector" and row["arm"] == "original_prompt_sft_control":
            old[(row["split"], row["canonical_id"], row["arm"])] = bool(row["accepted"])
    if len(old) != 61:
        raise RuntimeError(f"Old selector coverage incomplete: {len(old)}/61")
    return old


def summarize_response_map(
    response_map: dict[str, tuple[str, str]], score_map: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    scored = []
    failures = []
    for canonical_id, (question, response) in response_map.items():
        result = score_map.get(pair_hash(question, response))
        if result is None:
            failures.append(canonical_id)
            continue
        scored.append((canonical_id, result))
    medical = Counter(result["medical_advice"] for _, result in scored)
    denominator = medical["yes"] + medical["no"]
    return {
        "N_expected": len(response_map),
        "N_scored": len(scored),
        "parse_or_missing_failures": len(failures),
        "failure_ids": failures,
        "means": {
            axis: mean([float(result["scores"][axis]) for _, result in scored])
            for axis in QUALITY_AXES
        },
        "quality_3axis_mean": mean([quality_mean(result["scores"]) for _, result in scored]),
        "medical_advice_counts": {label: medical[label] for label in ("yes", "no", "unsure")},
        "medical_advice_yes_rate": round(medical["yes"] / denominator, 6) if denominator else None,
        "medical_advice_yes_rate_denominator": denominator,
    }


def compare_with_sft(
    sft_map: dict[str, tuple[str, str]],
    method_map: dict[str, tuple[str, str]],
    score_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter()
    details = []
    for canonical_id in sft_map:
        if canonical_id not in method_map:
            continue
        sft = score_map.get(pair_hash(*sft_map[canonical_id]))
        method = score_map.get(pair_hash(*method_map[canonical_id]))
        if sft is None or method is None:
            counts["parse_or_missing_failure"] += 1
            continue
        delta = quality_mean(method["scores"]) - quality_mean(sft["scores"])
        outcome = "win" if delta > 0 else ("loss" if delta < 0 else "tie")
        counts[outcome] += 1
        details.append({"canonical_id": canonical_id, "quality_3axis_delta": round(delta, 4), "outcome": outcome})
    return {
        "N": len(details),
        "win_tie_loss": {key: counts[key] for key in ("win", "tie", "loss")},
        "parse_or_missing_failure": counts["parse_or_missing_failure"],
        "mean_quality_3axis_delta": mean([row["quality_3axis_delta"] for row in details]),
        "details": details,
    }


def risk_summary(
    ordered_ids: list[str],
    sft_risks: dict[str, Any],
    final_risks: dict[str, Any] | None,
    sft_map: dict[str, tuple[str, str]],
    final_map: dict[str, tuple[str, str]],
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    if final_risks is None:
        return {
            "status": "unavailable", "reason": unavailable_reason,
            "fixed_denominator_N": len(ordered_ids), "complete_N": 0,
            "missing_risk_ids": ordered_ids,
        }
    initial_values, final_values, reductions, missing = [], [], [], []
    for canonical_id in ordered_ids:
        initial = sft_risks.get(canonical_id)
        final = final_risks.get(canonical_id)
        if not finite(initial) or not finite(final):
            missing.append(canonical_id)
            continue
        initial_float = float(initial)
        final_float = initial_float if final_map[canonical_id][1] == sft_map[canonical_id][1] else float(final)
        initial_values.append(initial_float)
        final_values.append(final_float)
        reductions.append(initial_float - final_float)
    complete = len(reductions)
    return {
        "status": "complete_fixed_denominator" if complete == len(ordered_ids) else "partial_missing_absolute_risk",
        "fixed_denominator_N": len(ordered_ids),
        "complete_N": complete,
        "missing_risk_ids": missing,
        "mean_sft_risk": mean(initial_values),
        "mean_final_risk": mean(final_values),
        "mean_risk_reduction_fixed_denominator": mean(reductions) if complete == len(ordered_ids) else None,
        "unchanged_response_reduction_forced_zero": sum(
            final_map[canonical_id][1] == sft_map[canonical_id][1] for canonical_id in ordered_ids
        ),
    }


def main() -> None:
    contract = json.loads((ROOT / "run_contract.json").read_text(encoding="utf-8"))
    score_map_by_judge, judge_status = score_maps()
    new_decisions, selector_failures = selector_decisions()
    old_acceptance = old_selector_acceptance()
    candidates = read_jsonl(ROOT / "inputs/selector_candidates.jsonl")
    candidate_by = {
        (row["split"], row["canonical_id"], row["candidate_type"]): row for row in candidates
    }
    selector_rows = []
    for key in sorted(new_decisions):
        candidate = candidate_by[key]
        decision = new_decisions[key]
        old = old_acceptance[key]
        selector_rows.append(
            {
                "split": key[0], "canonical_id": key[1], "candidate_type": key[2],
                "source_order": candidate["source_order"], "old_five_item_accepted": old,
                "new_four_item_accepted": decision["accepted"],
                "selection_changed": old != decision["accepted"],
                "new_reason": decision["reason"], "mapped_choices": decision["mapped_choices"],
                "request_ids": decision["request_ids"], "rationales": decision["rationales"],
                "question": candidate["question"], "sft_response": candidate["sft_response"],
                "candidate_response": candidate["candidate_response"],
                "sft_risk_score": candidate.get("sft_risk_score"),
                "candidate_risk_score": candidate.get("candidate_risk_score"),
            }
        )

    test_sft_rows = read_jsonl(TEST_ROOT / "sft_test_outputs.jsonl")
    test_dpo_rows = {row["canonical_id"]: row for row in read_jsonl(TEST_ROOT / "dpo_test_outputs.jsonl")}
    test_proposed_rows = read_jsonl(TEST_ROOT / "proposed_mask_off_test_outputs.jsonl")
    test_proposed_by = {row["canonical_id"]: row for row in test_proposed_rows}
    test_ids = [row["canonical_id"] for row in test_sft_rows]
    test_sft = {row["canonical_id"]: (row["question"], row["sft_response"]) for row in test_sft_rows}
    test_dpo = {
        canonical_id: (test_dpo_rows[canonical_id]["question"], test_dpo_rows[canonical_id]["sft_response"])
        for canonical_id in test_ids
    }
    test_sft_risk = {row["canonical_id"]: row.get("sft_risk_score") for row in test_proposed_rows}
    test_replay_rows = read_jsonl(PRIOR / "test_policy_replay.jsonl")
    test_replay = {(row["policy"], row["canonical_id"]): row for row in test_replay_rows}

    assignments = read_jsonl(ROOT / "inputs/pointwise_assignments.jsonl")
    valid_final_rows = [row for row in assignments if row["scope"] == "VALID43_final"]
    valid_conditions = defaultdict(dict)
    for row in valid_final_rows:
        valid_conditions[row["condition"]][row["canonical_id"]] = (row["question"], row["response"])
    valid_sft = valid_conditions["SFT"]
    valid_ids = list(valid_sft)
    if len(test_ids) != 24 or len(valid_ids) != 43:
        raise RuntimeError("Fixed split denominator changed")
    valid_mask_off_rows = read_jsonl(VALID_ROOT / "proposed_mask_off_valid_outputs.jsonl")
    valid_sft_risk = {row["canonical_id"]: row.get("sft_risk_score") for row in valid_mask_off_rows}

    method_maps: dict[tuple[str, str], dict[str, tuple[str, str]]] = {}
    method_acceptance: dict[tuple[str, str], dict[str, bool] | None] = {}
    final_risk_maps: dict[tuple[str, str], dict[str, Any] | None] = {}
    risk_unavailable: dict[tuple[str, str], str] = {}

    def register(
        scope: str,
        method: str,
        responses: dict[str, tuple[str, str]],
        acceptance: dict[str, bool] | None = None,
        risks: dict[str, Any] | None = None,
        unavailable: str | None = None,
    ) -> None:
        key = (scope, method)
        if key in method_maps:
            raise RuntimeError(f"Duplicate method map: {key}")
        method_maps[key] = responses
        method_acceptance[key] = acceptance
        final_risk_maps[key] = risks
        if unavailable:
            risk_unavailable[key] = unavailable

    register("TEST24_final", "SFT", test_sft, {key: False for key in test_ids}, test_sft_risk)
    register("TEST24_final", "DPO", test_dpo, None, None, "no stored scorer risk for DPO responses")
    for policy, label in (
        ("current_rules", "current_rules"),
        ("risk_only", "risk_only"),
        ("gpt_selector", "existing_5item_selector"),
        ("no_min_length", "no_min_length"),
        ("legacy_strict", "legacy_strict"),
        ("accept_generated", "accept_generated"),
    ):
        responses, accepted, risks = {}, {}, {}
        for canonical_id in test_ids:
            row = test_replay[(policy, canonical_id)]
            responses[canonical_id] = (test_sft[canonical_id][0], row["final_response"])
            accepted[canonical_id] = bool(row["accepted"])
            risks[canonical_id] = row["candidate_risk_score"] if row["accepted"] else row["sft_risk_score"]
        register("TEST24_final", label, responses, accepted, risks)

    def new_selector_full(
        split: str,
        ordered_ids: list[str],
        sft_map: dict[str, tuple[str, str]],
        candidate_type: str,
    ) -> tuple[dict[str, tuple[str, str]], dict[str, bool], dict[str, Any]]:
        responses = dict(sft_map)
        accepted = {canonical_id: False for canonical_id in ordered_ids}
        if split == "TEST":
            sft_risks = test_sft_risk
        else:
            sft_risks = valid_sft_risk
        risks = dict(sft_risks)
        for key, candidate in candidate_by.items():
            if key[0] != split or key[2] != candidate_type:
                continue
            decision = new_decisions[key]
            if decision["accepted"]:
                responses[key[1]] = (candidate["question"], candidate["candidate_response"])
                accepted[key[1]] = True
                risks[key[1]] = candidate.get("candidate_risk_score")
        return responses, accepted, risks

    responses, accepted, risks = new_selector_full("TEST", test_ids, test_sft, "original_denoiser")
    register("TEST24_final", "new_4item_selector", responses, accepted, risks)

    control_replay_rows = read_jsonl(CONTROL / "results/policy_replay.jsonl")
    control_replay = {
        (row["split"], row["arm"], row["policy"], row["canonical_id"]): row
        for row in control_replay_rows
    }
    for policy, label in (
        ("current_rules", "control__current_rules"),
        ("no_min_length", "control__no_min_length"),
        ("risk_only", "control__risk_only"),
        ("gpt_selector", "control__existing_5item_selector"),
    ):
        responses, accepted, risks = dict(test_sft), {key: False for key in test_ids}, dict(test_sft_risk)
        for key, candidate in candidate_by.items():
            if key[0] != "TEST" or key[2] != "original_prompt_sft_control":
                continue
            row = control_replay[("TEST", key[2], policy, key[1])]
            responses[key[1]] = (candidate["question"], row["final_response"])
            accepted[key[1]] = bool(row["accepted"])
            risks[key[1]] = candidate["candidate_risk_score"] if row["accepted"] else candidate["sft_risk_score"]
        register("TEST24_final", label, responses, accepted, risks)
    responses, accepted, risks = new_selector_full("TEST", test_ids, test_sft, "original_prompt_sft_control")
    register("TEST24_final", "control__new_4item_selector", responses, accepted, risks)

    test15_ids = contract["test15"]["ordered_ids"]
    expert_mapping = read_jsonl(ROOT / "researcher_model_mapping_TEST15x3.jsonl")
    expert_text = {
        (row["canonical_id"], row["method"]): next(
            response["response_text"]
            for case in read_jsonl(ROOT / "expert_packet_TEST15x3/expert_blinded_cases_TEST15x3.jsonl")
            if case["case_id"] == row["case_id"]
            for response in case["responses"]
            if response["response_id"] == row["response_id"]
        )
        for row in expert_mapping
    }
    for method in ("SFT", "DPO", "Proposed"):
        register(
            "TEST15_expert_packet", method,
            {
                canonical_id: (test_sft[canonical_id][0], expert_text[(canonical_id, method)])
                for canonical_id in test15_ids
            },
        )

    valid_source_risk_files = {
        "Mask_ON_current": VALID_ROOT / "proposed_mask_on_valid_outputs.jsonl",
        "Mask_OFF_current": VALID_ROOT / "proposed_mask_off_valid_outputs.jsonl",
        "without_router_current": VALID_ROOT / "proposed_without_router_valid_outputs.jsonl",
        "without_scorer_current": VALID_ROOT / "proposed_without_scorer_valid_outputs.jsonl",
    }
    register("VALID43_final", "SFT", valid_sft, {key: False for key in valid_ids}, valid_sft_risk)
    register("VALID43_final", "DPO", valid_conditions["DPO"], None, None, "no stored scorer risk for DPO responses")
    for condition in ("Mask_ON_current", "Mask_OFF_current", "without_router_current", "without_scorer_current"):
        rows = read_jsonl(valid_source_risk_files[condition])
        risks = {row["canonical_id"]: row.get("final_risk_score") for row in rows}
        register("VALID43_final", condition, valid_conditions[condition], None, risks)

    valid_replay_rows = read_jsonl(PRIOR / "valid9_policy_replay.jsonl")
    valid_replay = {
        (row["candidate_type"], row["policy"], row["canonical_id"]): row
        for row in valid_replay_rows
    }
    valid_candidate_types = ("original_denoiser", "feedback_A", "feedback_B", "feedback_C")
    for candidate_type in valid_candidate_types:
        for policy, label_suffix in (
            ("current_rules", "current_rules"),
            ("no_min_length", "no_min_length"),
            ("risk_only", "risk_only"),
            ("gpt_selector", "existing_5item_selector"),
            ("legacy_strict", "legacy_strict"),
            ("accept_generated", "accept_generated"),
        ):
            responses, accepted, risks = dict(valid_sft), {key: False for key in valid_ids}, dict(valid_sft_risk)
            for key, candidate in candidate_by.items():
                if key[0] != "VALID" or key[2] != candidate_type:
                    continue
                row = valid_replay[(candidate_type, policy, key[1])]
                responses[key[1]] = (candidate["question"], row["final_response"])
                accepted[key[1]] = bool(row["accepted"])
                risks[key[1]] = candidate["candidate_risk_score"] if row["accepted"] else candidate["sft_risk_score"]
            register("VALID43_final", f"{candidate_type}__{label_suffix}", responses, accepted, risks)
        responses, accepted, risks = new_selector_full("VALID", valid_ids, valid_sft, candidate_type)
        register("VALID43_final", f"{candidate_type}__new_4item_selector", responses, accepted, risks)

    for policy, label_suffix in (
        ("current_rules", "current_rules"),
        ("no_min_length", "no_min_length"),
        ("risk_only", "risk_only"),
        ("gpt_selector", "existing_5item_selector"),
    ):
        responses, accepted, risks = dict(valid_sft), {key: False for key in valid_ids}, dict(valid_sft_risk)
        for key, candidate in candidate_by.items():
            if key[0] != "VALID" or key[2] != "original_prompt_sft_control":
                continue
            row = control_replay[("VALID", key[2], policy, key[1])]
            responses[key[1]] = (candidate["question"], row["final_response"])
            accepted[key[1]] = bool(row["accepted"])
            risks[key[1]] = candidate["candidate_risk_score"] if row["accepted"] else candidate["sft_risk_score"]
        register("VALID43_final", f"original_prompt_sft_control__{label_suffix}", responses, accepted, risks)
    responses, accepted, risks = new_selector_full("VALID", valid_ids, valid_sft, "original_prompt_sft_control")
    register("VALID43_final", "original_prompt_sft_control__new_4item_selector", responses, accepted, risks)

    valid_called_ids = sorted(
        {row["canonical_id"] for row in candidates if row["split"] == "VALID"},
        key=lambda canonical_id: candidate_by[("VALID", canonical_id, "original_denoiser")]["source_order"],
    )
    test_called_ids = sorted(
        {row["canonical_id"] for row in candidates if row["split"] == "TEST"},
        key=lambda canonical_id: candidate_by[("TEST", canonical_id, "original_denoiser")]["source_order"],
    )
    register("VALID9_candidate_raw", "SFT", {key: valid_sft[key] for key in valid_called_ids})
    for candidate_type in (*valid_candidate_types, "original_prompt_sft_control"):
        register(
            "VALID9_candidate_raw", candidate_type,
            {
                key: (
                    candidate_by[("VALID", key, candidate_type)]["question"],
                    candidate_by[("VALID", key, candidate_type)]["candidate_response"],
                )
                for key in valid_called_ids
            },
        )
    register("TEST8_candidate_raw", "SFT", {key: test_sft[key] for key in test_called_ids})
    register("TEST8_candidate_raw", "DPO", {key: test_dpo[key] for key in test_called_ids})
    for candidate_type in ("original_denoiser", "original_prompt_sft_control"):
        register(
            "TEST8_candidate_raw", candidate_type,
            {
                key: (
                    candidate_by[("TEST", key, candidate_type)]["question"],
                    candidate_by[("TEST", key, candidate_type)]["candidate_response"],
                )
                for key in test_called_ids
            },
        )

    quality_rows, wtl_rows, per_item_rows = [], [], []
    for judge, score_map in score_map_by_judge.items():
        for (scope, method), response_map in sorted(method_maps.items()):
            summary = summarize_response_map(response_map, score_map)
            acceptance = method_acceptance[(scope, method)]
            quality_rows.append(
                {
                    "judge": judge, "scope": scope, "method": method,
                    "accepted": sum(acceptance.values()) if acceptance is not None else None,
                    **summary,
                }
            )
            sft_key = (scope, "SFT")
            if method != "SFT" and sft_key in method_maps:
                comparison = compare_with_sft(method_maps[sft_key], response_map, score_map)
                details = comparison.pop("details")
                wtl_rows.append({"judge": judge, "scope": scope, "method": method, **comparison})
                for detail in details:
                    question, response = response_map[detail["canonical_id"]]
                    scored = score_map[pair_hash(question, response)]
                    per_item_rows.append(
                        {
                            "judge": judge, "scope": scope, "method": method,
                            "canonical_id": detail["canonical_id"], **detail,
                            "scores": scored["scores"], "medical_advice": scored["medical_advice"],
                            "response_sha256": sha256_text(response),
                        }
                    )

    risk_rows = []
    for (scope, method), response_map in sorted(method_maps.items()):
        if scope not in {"TEST24_final", "VALID43_final"}:
            continue
        ordered_ids = test_ids if scope == "TEST24_final" else valid_ids
        sft_map = test_sft if scope == "TEST24_final" else valid_sft
        sft_risks = test_sft_risk if scope == "TEST24_final" else valid_sft_risk
        risk_rows.append(
            {
                "scope": scope, "method": method,
                **risk_summary(
                    ordered_ids, sft_risks, final_risk_maps[(scope, method)], sft_map, response_map,
                    risk_unavailable.get((scope, method)),
                ),
            }
        )

    selection_summary = []
    for split in ("TEST", "VALID"):
        candidate_types = sorted({key[2] for key in new_decisions if key[0] == split})
        for candidate_type in candidate_types:
            subset = [row for row in selector_rows if row["split"] == split and row["candidate_type"] == candidate_type]
            selection_summary.append(
                {
                    "split": split, "candidate_type": candidate_type, "called": len(subset),
                    "old_five_item_accepted": sum(row["old_five_item_accepted"] for row in subset),
                    "new_four_item_accepted": sum(row["new_four_item_accepted"] for row in subset),
                    "selection_changed": sum(row["selection_changed"] for row in subset),
                    "changed_ids": [row["canonical_id"] for row in subset if row["selection_changed"]],
                    "new_reason_counts": dict(Counter(row["new_reason"] for row in subset)),
                }
            )

    assignment_score_rows = []
    for judge, score_map in score_map_by_judge.items():
        for row in assignments:
            score = score_map.get(row["question_response_sha256"])
            assignment_score_rows.append(
                {
                    **row, "judge": judge, "judge_ok": score is not None,
                    "scores": score["scores"] if score else None,
                    "quality_3axis_mean": round(quality_mean(score["scores"]), 4) if score else None,
                    "medical_advice": score["medical_advice"] if score else None,
                    "rationale": score["rationale"] if score else None,
                    "medical_advice_evidence": score["medical_advice_evidence"] if score else None,
                    "medical_advice_rationale": score["medical_advice_rationale"] if score else None,
                }
            )

    assignments_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignments:
        assignments_by_pair[row["question_response_sha256"]].append(row)
    judge_comparison_rows = []
    gpt_score_map = score_map_by_judge["GPT-4.1"]
    llama_score_map = score_map_by_judge.get("Llama", {})
    for response_pair_hash, members in sorted(assignments_by_pair.items()):
        gpt_score = gpt_score_map.get(response_pair_hash)
        llama_score = llama_score_map.get(response_pair_hash)
        both_available = gpt_score is not None and llama_score is not None
        axis_deltas = (
            {
                axis: int(gpt_score["scores"][axis]) - int(llama_score["scores"][axis])
                for axis in QUALITY_AXES
            }
            if both_available else None
        )
        judge_comparison_rows.append(
            {
                "question_response_sha256": response_pair_hash,
                "canonical_id": members[0]["canonical_id"],
                "member_assignment_ids": [
                    f"{row['scope']}::{row['canonical_id']}::{row['condition']}" for row in members
                ],
                "member_scopes": sorted({row["scope"] for row in members}),
                "member_conditions": sorted({row["condition"] for row in members}),
                "gpt_available": gpt_score is not None,
                "llama_available": llama_score is not None,
                "both_available": both_available,
                "gpt_scores": gpt_score["scores"] if gpt_score else None,
                "llama_scores": llama_score["scores"] if llama_score else None,
                "gpt_minus_llama": axis_deltas,
                "gpt_minus_llama_quality_3axis_mean": (
                    round(quality_mean(gpt_score["scores"]) - quality_mean(llama_score["scores"]), 4)
                    if both_available else None
                ),
                "gpt_medical_advice": gpt_score["medical_advice"] if gpt_score else None,
                "llama_medical_advice": llama_score["medical_advice"] if llama_score else None,
                "medical_advice_agrees": (
                    gpt_score["medical_advice"] == llama_score["medical_advice"]
                    if both_available else None
                ),
            }
        )
    paired_judge_rows = [row for row in judge_comparison_rows if row["both_available"]]
    judge_comparison_summary = {
        "unique_pairs_expected": len(assignments_by_pair),
        "paired_available": len(paired_judge_rows),
        "llama_unavailable_or_parse_failure": len(assignments_by_pair) - len(paired_judge_rows),
        "mean_gpt_minus_llama": {
            axis: mean([float(row["gpt_minus_llama"][axis]) for row in paired_judge_rows])
            for axis in QUALITY_AXES
        },
        "mean_gpt_minus_llama_quality_3axis_mean": mean(
            [float(row["gpt_minus_llama_quality_3axis_mean"]) for row in paired_judge_rows]
        ),
        "medical_advice_agreement": sum(row["medical_advice_agrees"] is True for row in paired_judge_rows),
        "medical_advice_disagreement": sum(row["medical_advice_agrees"] is False for row in paired_judge_rows),
    }

    gpt_paths = list((ROOT / "results").glob("*gpt41.jsonl"))
    gpt_api_rows = [row for path in gpt_paths for row in read_jsonl(path)]
    gpt_api_latest = {row["request_id"]: row for row in gpt_api_rows}.values()
    gpt_ok = [row for row in gpt_api_latest if row.get("ok") is True]
    api_usage = {
        "calls": len(gpt_api_latest),
        "ok": len(gpt_ok),
        "failed": len(gpt_api_latest) - len(gpt_ok),
        "input_tokens": sum(int((row.get("usage") or {}).get("prompt_tokens") or 0) for row in gpt_ok),
        "output_tokens": sum(int((row.get("usage") or {}).get("completion_tokens") or 0) for row in gpt_ok),
        "estimated_cost_usd": round(sum(float(row.get("estimated_cost_usd") or 0) for row in gpt_ok), 6),
        "models": sorted({row.get("returned_model") for row in gpt_ok}),
        "attempt_counts": dict(Counter(str(row.get("attempt")) for row in gpt_api_latest)),
        "retry_events": sum(len(row.get("retry_events") or []) for row in gpt_api_latest),
    }

    llama_latest_rows = latest_rows(
        [
            ROOT / "results/expert15_pointwise_llama.jsonl",
            ROOT / "results/remaining_pointwise_llama.jsonl",
        ]
    )
    llama_evidence_failure_ids = set(judge_status["llama_evidence_validation_failure_request_ids"])
    llama_failure_rows = [
        {
            "request_id": row["request_id"],
            "failure_type": (
                "nonverbatim_medical_advice_evidence"
                if row["request_id"] in llama_evidence_failure_ids
                else "runner_parse_or_schema_failure"
            ),
            "judge_error": row.get("judge_error"),
            "metadata": row.get("metadata"),
            "judge_raw": row.get("judge_raw"),
            "judge_payload": row.get("judge_payload"),
        }
        for row in llama_latest_rows.values()
        if row.get("judge_ok") is not True or row["request_id"] in llama_evidence_failure_ids
    ]

    results_dir = ROOT / "aggregated"
    artifacts = {
        "pointwise_assignment_scores.jsonl": assignment_score_rows,
        "selector_new4_decisions_and_old5_comparison.jsonl": selector_rows,
        "selector_summary.jsonl": selection_summary,
        "method_quality_summary.jsonl": quality_rows,
        "method_sft_wtl.jsonl": wtl_rows,
        "method_per_item_scores.jsonl": per_item_rows,
        "method_risk_summary.jsonl": risk_rows,
        "judge_direct_comparison.jsonl": judge_comparison_rows,
        "llama_failures.jsonl": llama_failure_rows,
        "selector_failures.jsonl": selector_failures,
    }
    for name, rows in artifacts.items():
        write_jsonl(results_dir / name, rows)
    write_json(results_dir / "api_usage.json", api_usage)
    write_json(results_dir / "judge_status.json", judge_status)
    write_json(results_dir / "judge_direct_comparison_summary.json", judge_comparison_summary)
    known_source_issues = [
        {
            "scope": "TEST", "case": "TEST-01 DPO", "status": "recovered_export_used",
            "issue": "The obsolete seven-character export artifact was excluded; the latest stored sft_response is used.",
        },
        {
            "scope": "TEST", "case": "TEST-15", "canonical_id": "qa_627a5555b89cc1151b371b2f",
            "status": "preserved_generation_failure",
            "issue": "Stored SFT and DPO have collapsed word spacing. Stored current_rules also falls back to the affected SFT; the new four-item selector independently accepts the already-stored intact original-denoiser candidate for Proposed.",
        },
        {
            "scope": "TEST", "case": "TEST-24", "canonical_id": "qa_03e93d92b6e415e7344811d7",
            "status": "preserved_factual_error",
            "issue": "The original-denoiser and matching control candidate say U.S. crisis number 989 instead of 988; current_rules, no_min_length, risk_only, and accept_generated retain it. Both GPT selector variants keep SFT.",
        },
        {
            "scope": "VALID", "case": "qa_24f1b77a40c9b22203b8f337 feedback_B",
            "status": "preserved_known_failure", "issue": "Known specificity-loss acceptance is retained.",
        },
        {
            "scope": "VALID", "case": "qa_b243b4c887feb8e3594dd74c feedback_B",
            "status": "preserved_known_failure", "issue": "Known SFT fallback is retained.",
        },
        {
            "scope": "VALID", "case": "qa_24f1b77a40c9b22203b8f337 control",
            "status": "preserved_known_failure", "issue": "Collapsed spacing/malformed markup is accepted by current_rules.",
        },
        {
            "scope": "TEST", "case": "TEST-10 control", "status": "preserved_known_failure",
            "issue": "The control substitutes incorrect/malformed crisis contacts for the treatment-discontinuation question and is accepted by current_rules.",
        },
        {
            "scope": "TEST", "case": "TEST-22 control", "status": "preserved_known_failure",
            "issue": "The control omits the high-place/trance safety context and is accepted by current_rules.",
        },
    ]
    write_json(ROOT / "KNOWN_SOURCE_ISSUES.json", known_source_issues)

    def find_quality(judge: str, scope: str, method: str) -> dict[str, Any]:
        matches = [row for row in quality_rows if row["judge"] == judge and row["scope"] == scope and row["method"] == method]
        if len(matches) != 1:
            raise RuntimeError(f"Quality lookup failed: {judge}/{scope}/{method}")
        return matches[0]

    def find_risk(scope: str, method: str) -> dict[str, Any]:
        matches = [row for row in risk_rows if row["scope"] == scope and row["method"] == method]
        if len(matches) != 1:
            raise RuntimeError(f"Risk lookup failed: {scope}/{method}")
        return matches[0]

    def find_wtl(judge: str, scope: str, method: str) -> dict[str, Any]:
        matches = [
            row for row in wtl_rows
            if row["judge"] == judge and row["scope"] == scope and row["method"] == method
        ]
        if len(matches) != 1:
            raise RuntimeError(f"W/T/L lookup failed: {judge}/{scope}/{method}")
        return matches[0]

    def format_score(value: Any) -> str:
        return "unavailable" if value is None else f"{float(value):.4f}"

    headline_methods = {
        "TEST24_final": ["SFT", "DPO", "current_rules", "risk_only", "existing_5item_selector", "new_4item_selector"],
        "VALID43_final": [
            "SFT", "DPO", "Mask_OFF_current", "original_denoiser__current_rules",
            "original_denoiser__risk_only", "original_denoiser__existing_5item_selector",
            "original_denoiser__new_4item_selector",
        ],
        "TEST15_expert_packet": ["SFT", "DPO", "Proposed"],
    }
    summary = {
        "version": VERSION,
        "status": "complete" if judge_status["llama_manifest"].get("status") == "complete" else "gpt_complete_llama_pending_or_partial",
        "rubric": contract["rubric"],
        "gpt_api_usage": api_usage,
        "evidence_repair": {
            "count": 2,
            "status": "complete",
            "rubric_unchanged": True,
            "model_settings_unchanged": True,
            "contract": str(ROOT / "evidence_repair_contract.json"),
        },
        "judge_status": judge_status,
        "selector_failures": len(selector_failures),
        "selector_summary": selection_summary,
        "comparison_coverage": {
            "pointwise_method_assignments": len(assignments),
            "unique_question_response_pairs": 193,
            "gpt_pointwise_scored": judge_status["gpt_unique_scored"],
            "selector_candidates": len(candidates),
            "selector_ordered_comparisons": len(candidates) * 2,
            "selector_failures": len(selector_failures),
            "llama_pointwise_attempted": judge_status["llama_unique_results"],
            "llama_pointwise_scored": judge_status["llama_unique_scored"],
            "llama_parse_or_schema_failures": judge_status["llama_parse_failures"],
            "expert_packet_questions": 15,
            "expert_packet_responses": 45,
        },
        "judge_direct_comparison": judge_comparison_summary,
        "known_source_issues": known_source_issues,
        "headline_quality": {
            judge: {
                scope: {method: find_quality(judge, scope, method) for method in methods}
                for scope, methods in headline_methods.items()
            }
            for judge in score_map_by_judge
        },
        "headline_risk": {
            scope: {method: find_risk(scope, method) for method in methods if (scope, method) in method_maps}
            for scope, methods in headline_methods.items() if scope in {"TEST24_final", "VALID43_final"}
        },
        "headline_sft_wtl": {
            judge: {
                scope: {
                    method: find_wtl(judge, scope, method)
                    for method in methods if method != "SFT"
                }
                for scope, methods in headline_methods.items()
            }
            for judge in score_map_by_judge
        },
        "expert_packet": json.loads((ROOT / "expert_packet_manifest.json").read_text(encoding="utf-8")),
        "metric_separation": {
            "quality_metric": "quality_3axis_mean only",
            "medical_advice": "categorical yes/no/unsure; never converted to numeric or added to quality",
            "legacy_four_axis_mean_compared_as_same_metric": False,
        },
    }
    write_json(ROOT / "RESULTS_SUMMARY.json", summary)

    lines = [
        "# Expert-rubric-aligned GPT-4.1 rejudge",
        "",
        "## Status",
        "",
        f"GPT pointwise: 193/193 unique question-response pairs. GPT selector: 122/122 ordered comparisons. Selector failures: {len(selector_failures)}.",
        f"Llama status: {judge_status['llama_manifest'].get('status')}; parsed {judge_status['llama_unique_scored']}/{judge_status['llama_unique_results']} completed rows.",
        "",
        "Medical Advice is categorical (`yes`/`no`/`unsure`) and is not part of `quality_3axis_mean`. Legacy four-axis means are not compared as the same metric.",
        "",
        "## Comparison coverage",
        "",
        "| Component | Complete | Expected | Notes |",
        "|---|---:|---:|---|",
        f"| Pointwise method assignments | {len(assignments)} | {len(assignments)} | 193 deduplicated question-response pairs |",
        f"| GPT-4.1 pointwise pairs | {judge_status['gpt_unique_scored']} | 193 | final rubric-aligned score map |",
        f"| GPT-4.1 selector ordered comparisons | {len(candidates) * 2 - len(selector_failures)} | {len(candidates) * 2} | 61 candidates in both A/B orders |",
        "| Counselor packet | 45 | 45 | fixed TEST15, three anonymous responses each |",
        f"| Llama pointwise attempts | {judge_status['llama_unique_results']} | 193 | one local model instance |",
        f"| Llama parsed and schema-valid pairs | {judge_status['llama_unique_scored']} | 193 | {judge_status['llama_parse_failures']} failures excluded, not treated as ties |",
        "",
        "## GPT-4.1 headline quality",
        "",
        "| Scope | Method | N | Overall | Empathy | Specificity | quality_3axis_mean | Medical yes/no/unsure | yes rate (denominator) |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for scope, methods in headline_methods.items():
        for method in methods:
            row = find_quality("GPT-4.1", scope, method)
            counts = row["medical_advice_counts"]
            means = row["means"]
            rate = row["medical_advice_yes_rate"]
            rate_text = "unavailable" if rate is None else f"{rate:.4f} ({row['medical_advice_yes_rate_denominator']})"
            lines.append(
                f"| {scope} | {method} | {row['N_scored']} | {means['overall_quality']:.4f} | "
                f"{means['empathy']:.4f} | {means['specificity']:.4f} | {row['quality_3axis_mean']:.4f} | "
                f"{counts['yes']}/{counts['no']}/{counts['unsure']} | {rate_text} |"
            )
    lines += [
        "",
        "## GPT-4.1 SFT-relative quality W/T/L",
        "",
        "W/T/L is based only on `quality_3axis_mean`; Medical Advice remains categorical and separate.",
        "",
        "| Scope | Method | Compared N | Win | Tie | Loss | Mean delta | Parse/missing |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scope, methods in headline_methods.items():
        for method in methods:
            if method == "SFT":
                continue
            row = find_wtl("GPT-4.1", scope, method)
            counts = row["win_tie_loss"]
            lines.append(
                f"| {scope} | {method} | {row['N']} | {counts['win']} | {counts['tie']} | "
                f"{counts['loss']} | {format_score(row['mean_quality_3axis_delta'])} | "
                f"{row['parse_or_missing_failure']} |"
            )

    if "Llama" in score_map_by_judge:
        lines += [
            "",
            "## Llama auxiliary quality",
            "",
            "| Scope | Method | Scored / expected | Overall | Empathy | Specificity | quality_3axis_mean | Medical yes/no/unsure | Failures |",
            "|---|---|---:|---:|---:|---:|---:|---|---:|",
        ]
        for scope, methods in headline_methods.items():
            for method in methods:
                row = find_quality("Llama", scope, method)
                counts = row["medical_advice_counts"]
                means = row["means"]
                lines.append(
                    f"| {scope} | {method} | {row['N_scored']} / {row['N_expected']} | "
                    f"{format_score(means['overall_quality'])} | {format_score(means['empathy'])} | "
                    f"{format_score(means['specificity'])} | {format_score(row['quality_3axis_mean'])} | "
                    f"{counts['yes']}/{counts['no']}/{counts['unsure']} | {row['parse_or_missing_failures']} |"
                )
        lines += [
            "",
            "## Llama SFT-relative quality W/T/L",
            "",
            "| Scope | Method | Compared N | Win | Tie | Loss | Mean delta | Parse/missing |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for scope, methods in headline_methods.items():
            for method in methods:
                if method == "SFT":
                    continue
                row = find_wtl("Llama", scope, method)
                counts = row["win_tie_loss"]
                lines.append(
                    f"| {scope} | {method} | {row['N']} | {counts['win']} | {counts['tie']} | "
                    f"{counts['loss']} | {format_score(row['mean_quality_3axis_delta'])} | "
                    f"{row['parse_or_missing_failure']} |"
                )
        lines += [
            "",
            "## Judge alignment",
            "",
            f"Paired parsed responses: {judge_comparison_summary['paired_available']}/{judge_comparison_summary['unique_pairs_expected']}; "
            f"Llama parse/missing failures: {judge_comparison_summary['llama_unavailable_or_parse_failure']}.",
            "",
            "| Comparison | Overall | Empathy | Specificity | quality_3axis_mean |",
            "|---|---:|---:|---:|---:|",
            f"| Mean GPT-4.1 minus Llama | {format_score(judge_comparison_summary['mean_gpt_minus_llama']['overall_quality'])} | "
            f"{format_score(judge_comparison_summary['mean_gpt_minus_llama']['empathy'])} | "
            f"{format_score(judge_comparison_summary['mean_gpt_minus_llama']['specificity'])} | "
            f"{format_score(judge_comparison_summary['mean_gpt_minus_llama_quality_3axis_mean'])} |",
            "",
            f"Medical Advice label agreement/disagreement: {judge_comparison_summary['medical_advice_agreement']}/"
            f"{judge_comparison_summary['medical_advice_disagreement']} among paired parsed responses.",
        ]
    lines += [
        "",
        "## Selector changes",
        "",
        "| Split | Candidate arm | Called | Old 5-item accepted | New 4-item accepted | Changed IDs |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in selection_summary:
        lines.append(
            f"| {row['split']} | {row['candidate_type']} | {row['called']} | "
            f"{row['old_five_item_accepted']} | {row['new_four_item_accepted']} | {row['selection_changed']} |"
        )
    lines += [
        "",
        "## Fixed-denominator risk",
        "",
        "| Scope | Method | Complete / fixed N | Mean SFT risk | Mean final risk | Mean reduction |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scope in ("TEST24_final", "VALID43_final"):
        for method in headline_methods[scope]:
            row = find_risk(scope, method)
            if row["status"] == "unavailable":
                lines.append(f"| {scope} | {method} | 0 / {row['fixed_denominator_N']} | unavailable | unavailable | unavailable |")
            else:
                reduction = row["mean_risk_reduction_fixed_denominator"]
                lines.append(
                    f"| {scope} | {method} | {row['complete_N']} / {row['fixed_denominator_N']} | "
                    f"{row['mean_sft_risk']:.6f} | {row['mean_final_risk']:.6f} | "
                    f"{reduction:.6f} |" if reduction is not None else
                    f"| {scope} | {method} | {row['complete_N']} / {row['fixed_denominator_N']} | partial | partial | unavailable |"
                )
    changed = [row for row in selector_rows if row["selection_changed"]]
    lines += [
        "",
        "## Changed selector cases",
        "",
    ]
    for row in changed:
        lines.append(
            f"- {row['split']} `{row['canonical_id']}` {row['candidate_type']}: old={row['old_five_item_accepted']}, new={row['new_four_item_accepted']} ({row['new_reason']})."
        )
    lines += [
        "",
        "## Execution",
        "",
        f"GPT calls: {api_usage['calls']} total, {api_usage['ok']} successful, {api_usage['failed']} failed; "
        f"input {api_usage['input_tokens']} tokens, output {api_usage['output_tokens']} tokens, estimated cost ${api_usage['estimated_cost_usd']:.6f}.",
        "",
        "Two pointwise responses were freshly re-evaluated under the unchanged rubric/model settings because the initial Medical Advice evidence was not an exact contiguous source substring. Both initial raw results and the separate repair contract are preserved; the final score map uses one complete repaired result per affected question-response pair.",
        "",
        "## Preserved known source issues",
        "",
        "- TEST-01 DPO uses the recovered latest `sft_response`; the prior seven-character export artifact is not used.",
        "- TEST-15 (`qa_627a5555b89cc1151b371b2f`) retains collapsed word spacing in stored SFT and DPO. The new selector independently selected the already-stored intact original-denoiser candidate for Proposed; no text was edited or regenerated.",
        "- TEST-24 (`qa_03e93d92b6e415e7344811d7`) retains the stored original-denoiser factual error `989` in policies that accept that candidate. The new four-item selector keeps SFT for this case.",
        "- Additional previously documented feedback/control failures are preserved in `KNOWN_SOURCE_ISSUES.json`; they were not corrected or regenerated.",
        "",
        "The TEST analysis is post-hoc exploratory. Stored responses, candidate gates, risks, and old policy outputs were not regenerated or edited.",
    ]
    report_path = ROOT / "RUBRIC_ALIGNED_REJUDGE_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if summary["status"] == "complete":
        code_dir = ROOT / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        for script_name in (
            "prepare_rubric_aligned_gpt41_rejudge.py",
            "build_rubric_aligned_expert_packet.py",
            "run_gpt41_checkpoint_queue.py",
            "run_llama_rubric_aligned_rejudge.py",
            "aggregate_rubric_aligned_rejudge.py",
            "prepare_rubric_aligned_evidence_repair.py",
        ):
            shutil.copy2(REPO / "scripts" / script_name, code_dir / script_name)

        final_manifest_path = ROOT / "FINAL_MANIFEST.json"
        research_zip_path = ROOT / "rubric_aligned_gpt41_rejudge_research_bundle.zip"
        delivery_path = ROOT / "DELIVERY.json"
        excluded_names = {final_manifest_path.name, research_zip_path.name, delivery_path.name}
        bundle_files = sorted(
            path for path in ROOT.rglob("*")
            if path.is_file()
            and path.name not in excluded_names
            and path.suffix not in {".lock", ".zip"}
        )
        final_manifest = {
            "version": VERSION,
            "status": "complete",
            "file_count_excluding_manifest": len(bundle_files),
            "files": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in bundle_files
            ],
            "counselor_zip": {
                "path": str(ROOT / "counselor_expert_packet_TEST15x3_rubric_aligned.zip"),
                "sha256": sha256_file(ROOT / "counselor_expert_packet_TEST15x3_rubric_aligned.zip"),
            },
        }
        write_json(final_manifest_path, final_manifest)
        with zipfile.ZipFile(research_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in [*bundle_files, final_manifest_path]:
                archive.write(path, path.relative_to(ROOT))
        write_json(
            delivery_path,
            {
                "status": "complete",
                "counselor_zip": {
                    "path": str(ROOT / "counselor_expert_packet_TEST15x3_rubric_aligned.zip"),
                    "sha256": sha256_file(ROOT / "counselor_expert_packet_TEST15x3_rubric_aligned.zip"),
                },
                "research_zip": {
                    "path": str(research_zip_path),
                    "sha256": sha256_file(research_zip_path),
                },
                "report": str(report_path),
            },
        )

    print(
        json.dumps(
            {
                "status": summary["status"], "judges": list(score_map_by_judge),
                "gpt_calls": api_usage["calls"], "selector_changes": len(changed),
                "judge_pairs": judge_comparison_summary["paired_available"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
