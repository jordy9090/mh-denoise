#!/usr/bin/env python3
"""Aggregate complete GPT-4.1/Llama post-hoc TEST24 and VALID comparisons."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VERSION = "gpt41-posthoc-test-valid-complete-v1-20261009"
ROOT = Path("/mnt/ssd00/hsoh/gpt41_posthoc_selection_20261009_v1")
REPO = Path("/home/user/hsoh/mh-denoise")
TEST_SOURCE = REPO / (
    "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/gemma/"
    "generation_1536_test_surface_relaxed_v1_20261008/part24"
)
VALID_SOURCE = REPO / (
    "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/gemma/generation_1536"
)
TEST_LLAMA = Path("/mnt/ssd00/hsoh/test24_local_four_axis_20261009")
FEEDBACK_ROOT = Path(
    "/home/user/hsoh/mh-denoise-feedback-v1-valid-20261008/"
    "outputs/valid_feedback_v1_20261008"
)
BC_ROOT = Path("/mnt/ssd00/hsoh/feedback_v1_bc_20261009")
AXES = ("overall_quality", "empathy", "specificity", "medical_advice")
VALID43_METHODS = (
    "SFT",
    "DPO",
    "Mask_ON_current",
    "Mask_OFF_current",
    "without_router_current",
    "without_scorer_current",
)
VALID9_METHODS = ("SFT", "original_denoiser", "feedback_A", "feedback_B", "feedback_C")
TEST_METHODS = ("SFT", "DPO", "A_current_rules", "B_risk_only", "C_gpt_selector")
POLICIES = (
    "current_rules",
    "no_min_length",
    "risk_only",
    "gpt_selector",
    "legacy_strict",
    "accept_generated",
)
VARIANT_ALIASES = {
    "sft": "SFT",
    "SFT": "SFT",
    "existing_denoiser": "original_denoiser",
    "original_denoiser": "original_denoiser",
    "feedback_A_existing_candidate": "feedback_A",
    "feedback_A": "feedback_A",
    "feedback_B_sft": "feedback_B",
    "feedback_B": "feedback_B",
    "feedback_C_sft_adapter": "feedback_C",
    "feedback_C": "feedback_C",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_hash(question: str, response: str) -> str:
    return hashlib.sha256((question + "\0" + response).encode("utf-8")).hexdigest()


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def composite(scores: dict[str, Any]) -> float:
    return statistics.mean(float(scores[axis]) for axis in AXES)


def outcome(delta: float) -> str:
    return "win" if delta > 0 else ("loss" if delta < 0 else "tie")


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def latest_rows(path: Path, key: str = "request_id") -> dict[str, dict[str, Any]]:
    latest = {}
    for row in read_jsonl(path):
        latest[str(row[key])] = row
    return latest


def gpt_score_map() -> dict[str, dict[str, int]]:
    mapping = {}
    paths = (
        ROOT / "results/test_pointwise_gpt41.jsonl",
        ROOT / "results/test_candidate_pointwise_gpt41.jsonl",
        ROOT / "results/valid_pointwise_gpt41.jsonl",
    )
    for path in paths:
        for row in latest_rows(path).values():
            if row.get("ok") is not True:
                continue
            key = row["metadata"]["question_response_sha256"]
            scores = row["normalized"]["scores"]
            if key in mapping and mapping[key] != scores:
                raise RuntimeError(f"Conflicting GPT pointwise cache: {key}")
            mapping[key] = scores
    return mapping


def llama_score_map() -> dict[str, dict[str, int]]:
    mapping = {}
    for row in read_jsonl(TEST_LLAMA / "judged_representatives.jsonl"):
        if row.get("judge_ok"):
            mapping[row["question_response_sha256"]] = row["judge_scores"]
    for path in (
        ROOT / "results/test_candidate_pointwise_llama.jsonl",
        ROOT / "results/valid_pointwise_llama.jsonl",
    ):
        for row in latest_rows(path).values():
            if row.get("judge_ok"):
                key = row["metadata"]["question_response_sha256"]
                scores = row["judge_normalized"]["scores"]
                if key in mapping and mapping[key] != scores:
                    raise RuntimeError(f"Conflicting Llama pointwise cache: {key}")
                mapping[key] = scores
    return mapping


def summarize_methods(
    response_map: dict[str, dict[str, tuple[str, str]]],
    score_map: dict[str, dict[str, int]],
    methods: tuple[str, ...],
    judge: str,
    scope: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scored = []
    summary = []
    for method in methods:
        rows = []
        for canonical_id, (question, response) in response_map[method].items():
            scores = score_map.get(pair_hash(question, response))
            if scores is None:
                continue
            item = {
                "scope": scope,
                "judge": judge,
                "method": method,
                "canonical_id": canonical_id,
                "question_response_sha256": pair_hash(question, response),
                "scores": scores,
                "four_axis_mean": round(composite(scores), 4),
            }
            rows.append(item)
            scored.append(item)
        summary.append(
            {
                "scope": scope,
                "judge": judge,
                "method": method,
                "N": len(rows),
                "means": {axis: mean([row["scores"][axis] for row in rows]) for axis in AXES},
                "four_axis_mean": mean([row["four_axis_mean"] for row in rows]),
            }
        )
    comparisons = []
    scored_by = {(row["method"], row["canonical_id"]): row for row in scored}
    for left, right in itertools.combinations(methods, 2):
        common = [
            canonical_id
            for canonical_id in response_map[left]
            if (left, canonical_id) in scored_by and (right, canonical_id) in scored_by
        ]
        deltas = []
        axis_deltas = {axis: [] for axis in AXES}
        counts = Counter()
        for canonical_id in common:
            lrow, rrow = scored_by[(left, canonical_id)], scored_by[(right, canonical_id)]
            delta = rrow["four_axis_mean"] - lrow["four_axis_mean"]
            deltas.append(delta)
            counts[outcome(delta)] += 1
            for axis in AXES:
                axis_deltas[axis].append(rrow["scores"][axis] - lrow["scores"][axis])
        comparisons.append(
            {
                "scope": scope,
                "judge": judge,
                "left": left,
                "right": right,
                "delta_definition": "right_minus_left",
                "N": len(common),
                "mean_four_axis_delta": mean(deltas),
                "mean_axis_deltas": {axis: mean(values) for axis, values in axis_deltas.items()},
                "right_win_tie_loss": {key: counts[key] for key in ("win", "tie", "loss")},
            }
        )
    return scored, summary, comparisons


def metric_policy_reasons(candidate: dict[str, Any], policy: str) -> list[str]:
    if not candidate["technical_valid"]:
        return ["technical_invalid"]
    sft = candidate["sft_metrics"] or {}
    revised = candidate["candidate_metrics"] or {}
    if policy == "accept_generated":
        return []
    if policy == "risk_only":
        if not finite(sft.get("risk_score")) or not finite(revised.get("risk_score")):
            return ["invalid_risk"]
        return [] if float(sft["risk_score"]) - float(revised["risk_score"]) >= 0.01 else ["risk_not_improved"]
    if policy not in {"current_rules", "no_min_length", "legacy_strict"}:
        raise ValueError(policy)
    reasons = []
    if policy != "no_min_length" and int(revised.get("word_count") or 0) < 20:
        reasons.append("short_denoiser_response")
    if not finite(sft.get("risk_score")) or not finite(revised.get("risk_score")):
        reasons.append("invalid_risk")
    elif float(revised["risk_score"]) > float(sft["risk_score"]) - 0.01:
        reasons.append("risk_not_improved")
    if int(revised.get("bad_safety_count") or 0) > int(sft.get("bad_safety_count") or 0):
        reasons.append("bad_safety_increased")
    if policy == "legacy_strict":
        if float(revised.get("specificity_ratio") or 0.0) < 0.6:
            reasons.append("specificity_ratio_low")
        if int(revised.get("generic_count") or 0) > int(sft.get("generic_count") or 0):
            reasons.append("genericity_increased")
    minimum_overlap = max(0.0, float(sft.get("keyword_overlap") or 0.0) - 0.15)
    if float(revised.get("keyword_overlap") or 0.0) < minimum_overlap:
        reasons.append("question_keyword_overlap_low")
    return reasons


def selector_decisions() -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    rows = [row for row in latest_rows(ROOT / "results/valid_selector_gpt41.jsonl").values()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["metadata"]["canonical_id"], row["metadata"]["candidate_type"])].append(row)
    decisions = {}
    failures = []
    for key, values in grouped.items():
        if len(values) != 2 or any(row.get("ok") is not True for row in values):
            decisions[key] = {"accepted": False, "reason": "selector_failure", "mapped_choices": []}
            failures.append({"key": key, "reason": "missing_or_failed_order"})
            continue
        mapped = []
        for row in sorted(values, key=lambda item: item["metadata"]["order_index"]):
            choice = row["normalized"]["choice"]
            mapped.append("tie" if choice == "tie" else row["metadata"][f"variant_for_{choice}"])
        accepted = mapped == ["candidate", "candidate"]
        decisions[key] = {
            "accepted": accepted,
            "reason": "candidate_selected_both_orders" if accepted else (
                "selector_tie" if "tie" in mapped else "selector_kept_sft_or_disagreed"
            ),
            "mapped_choices": mapped,
            "request_ids": [row["request_id"] for row in values],
        }
    return decisions, failures


def pairwise_normalized_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    llama_sources = [
        *read_jsonl(FEEDBACK_ROOT / "blind_judge_results.jsonl"),
        *read_jsonl(BC_ROOT / "bc_blind_judge_results.jsonl"),
        *latest_rows(ROOT / "results/valid_pairwise_new27_llama.jsonl").values(),
    ]
    gpt_sources = [
        *latest_rows(ROOT / "results/valid_pairwise_replay63_gpt41.jsonl").values(),
        *latest_rows(ROOT / "results/valid_pairwise_new27_gpt41.jsonl").values(),
    ]

    def convert(row: dict[str, Any], judge: str) -> dict[str, Any]:
        metadata = row.get("metadata") or row
        normalized = row.get("normalized") if judge == "GPT-4.1" else row.get("judge_normalized")
        ok = row.get("ok") is True if judge == "GPT-4.1" else row.get("judge_ok") is True
        source_id = metadata.get("source_comparison_id") or row.get("comparison_id")
        left_raw, right_raw = metadata["pair"]
        left, right = VARIANT_ALIASES[left_raw], VARIANT_ALIASES[right_raw]
        variant_a = VARIANT_ALIASES[metadata["variant_for_A"]]
        variant_b = VARIANT_ALIASES[metadata["variant_for_B"]]
        scores_a = normalized.get("response_A") if isinstance(normalized, dict) else None
        scores_b = normalized.get("response_B") if isinstance(normalized, dict) else None
        preference = normalized.get("preference") if isinstance(normalized, dict) else None
        if preference == "A":
            preferred = variant_a
        elif preference == "B":
            preferred = variant_b
        elif preference == "tie":
            preferred = "tie"
        else:
            preferred = None
        if variant_a == left:
            left_scores, right_scores = scores_a, scores_b
        elif variant_b == left:
            left_scores, right_scores = scores_b, scores_a
        else:
            raise RuntimeError("Pairwise variant mapping mismatch")
        score_ok = bool(ok and isinstance(left_scores, dict) and isinstance(right_scores, dict))
        return {
            "judge": judge,
            "source_comparison_id": source_id,
            "canonical_id": metadata["canonical_id"],
            "pair": [left, right],
            "variant_for_A": variant_a,
            "variant_for_B": variant_b,
            "system_prompt_sha256": row.get("system_prompt_sha256") or sha256_file_text(row.get("system_prompt", "")),
            "request_user_prompt_sha256": row.get("user_prompt_sha256") or sha256_file_text(row.get("user_prompt", "")),
            "score_ok": score_ok,
            "preference_ok": preferred is not None,
            "preferred": preferred,
            "left_scores": left_scores if score_ok else None,
            "right_scores": right_scores if score_ok else None,
            "raw_preference": preference,
        }

    llama = [convert(row, "Llama") for row in llama_sources]
    gpt = [convert(row, "GPT-4.1") for row in gpt_sources]
    if len(llama) != 90 or len(gpt) != 90:
        raise RuntimeError(f"Pairwise coverage incomplete: llama={len(llama)} gpt={len(gpt)}")
    by_llama = {row["source_comparison_id"]: row for row in llama}
    by_gpt = {row["source_comparison_id"]: row for row in gpt}
    if set(by_llama) != set(by_gpt):
        raise RuntimeError("GPT/Llama pairwise comparison IDs differ")
    direct = []
    for comparison_id in sorted(by_llama):
        lrow, grow = by_llama[comparison_id], by_gpt[comparison_id]
        direct.append(
            {
                "source_comparison_id": comparison_id,
                "canonical_id": lrow["canonical_id"],
                "pair": lrow["pair"],
                "same_ab_order": (
                    lrow["variant_for_A"] == grow["variant_for_A"]
                    and lrow["variant_for_B"] == grow["variant_for_B"]
                ),
                "same_system_prompt_sha256": lrow["system_prompt_sha256"] == grow["system_prompt_sha256"],
                "same_user_prompt_sha256": (
                    lrow["request_user_prompt_sha256"] == grow["request_user_prompt_sha256"]
                ),
                "llama_preferred": lrow["preferred"],
                "gpt41_preferred": grow["preferred"],
                "preference_agreement": (
                    lrow["preferred"] == grow["preferred"]
                    if lrow["preference_ok"] and grow["preference_ok"]
                    else None
                ),
                "llama_score_ok": lrow["score_ok"],
                "gpt41_score_ok": grow["score_ok"],
            }
        )
    return llama, gpt, direct


def sha256_file_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def summarize_pairwise(rows: list[dict[str, Any]], judge: str) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row["pair"])].append(row)
    output = []
    for pair, values in sorted(grouped.items()):
        score_rows = [row for row in values if row["score_ok"]]
        pref_rows = [row for row in values if row["preference_ok"]]
        deltas = []
        axis_deltas = {axis: [] for axis in AXES}
        for row in score_rows:
            deltas.append(composite(row["right_scores"]) - composite(row["left_scores"]))
            for axis in AXES:
                axis_deltas[axis].append(row["right_scores"][axis] - row["left_scores"][axis])
        counts = Counter(row["preferred"] for row in pref_rows)
        output.append(
            {
                "judge": judge,
                "pair": list(pair),
                "expected_N": 9,
                "score_N": len(score_rows),
                "preference_N": len(pref_rows),
                "preference_parse_failures": len(values) - len(pref_rows),
                "preferences": {pair[0]: counts[pair[0]], "tie": counts["tie"], pair[1]: counts[pair[1]]},
                "mean_composite_delta_right_minus_left": mean(deltas),
                "mean_axis_deltas_right_minus_left": {axis: mean(vals) for axis, vals in axis_deltas.items()},
            }
        )
    if len(output) != 10:
        raise RuntimeError(f"Expected 10 complete candidate pairs for {judge}, got {len(output)}")
    return output


def main() -> None:
    llama_manifest_path = ROOT / "results/llama_posthoc_common_manifest.json"
    if not llama_manifest_path.exists():
        raise RuntimeError("Local Llama evaluation is still running")
    llama_manifest = json.loads(llama_manifest_path.read_text(encoding="utf-8"))
    if llama_manifest.get("status") != "complete":
        raise RuntimeError("Local Llama evaluation is incomplete")
    gpt_scores, llama_scores = gpt_score_map(), llama_score_map()
    if len(gpt_scores) != 177 or len(llama_scores) != 177:
        raise RuntimeError(f"Pointwise cache coverage changed: GPT={len(gpt_scores)}, Llama={len(llama_scores)}")
    test_interim_summary = json.loads((ROOT / "TEST24_GPT41_SUMMARY.json").read_text(encoding="utf-8"))

    valid_assignments = read_jsonl(ROOT / "valid_pointwise_assignments.jsonl")
    valid43_response_map = {method: {} for method in VALID43_METHODS}
    for row in valid_assignments:
        if row["scope"] == "VALID43_final":
            valid43_response_map[row["condition"]][row["canonical_id"]] = (row["question"], row["response"])
    valid43_id_sets_match = all(
        set(valid43_response_map[method]) == set(valid43_response_map["SFT"])
        for method in VALID43_METHODS
    )
    if not valid43_id_sets_match or len(valid43_response_map["SFT"]) != 43:
        raise RuntimeError("VALID43 method ID sets do not match")

    feedback_contract = json.loads((FEEDBACK_ROOT / "experiment_contract.json").read_text(encoding="utf-8"))
    bc_contract = json.loads((BC_ROOT / "extension_contract.json").read_text(encoding="utf-8"))
    risk_adapter = feedback_contract["fixed_backbone"]["risk_adapter_dir"]
    sft_adapter = feedback_contract["fixed_backbone"]["sft_adapter_dir"]
    valid9_arm_provenance = {
        "original_denoiser": {
            "draft_input": "SFT original response in the frozen baseline pipeline",
            "generation_adapter": risk_adapter,
            "regenerated_for_this_comparison": False,
        },
        "feedback_A": {
            "draft_input": "existing denoiser candidate + frozen feedback",
            "generation_adapter": risk_adapter,
            "regenerated_for_this_comparison": False,
        },
        "feedback_B": {
            "draft_input": "SFT original response + the same frozen feedback",
            "generation_adapter": risk_adapter,
            "regenerated_for_this_comparison": False,
        },
        "feedback_C": {
            "draft_input": "the same SFT original response and frozen feedback used by B",
            "generation_adapter": sft_adapter,
            "regenerated_for_this_comparison": False,
        },
    }
    valid9_ids = sorted(
        {row["canonical_id"] for row in valid_assignments if row["scope"] == "VALID9_candidate_raw"},
        key=lambda canonical_id: next(
            row["source_order"] for row in valid_assignments
            if row["canonical_id"] == canonical_id and row["scope"] == "VALID9_candidate_raw"
        ),
    )
    valid9_response_map = {method: {} for method in VALID9_METHODS}
    for canonical_id in valid9_ids:
        valid9_response_map["SFT"][canonical_id] = valid43_response_map["SFT"][canonical_id]
    for row in valid_assignments:
        if row["scope"] == "VALID9_candidate_raw":
            valid9_response_map[row["condition"]][row["canonical_id"]] = (row["question"], row["response"])

    all_pointwise_rows = []
    all_pointwise_summaries = []
    all_pointwise_comparisons = []
    for judge, score_map in (("GPT-4.1", gpt_scores), ("Llama", llama_scores)):
        for response_map, methods, scope in (
            (valid43_response_map, VALID43_METHODS, "VALID43_final_outputs"),
            (valid9_response_map, VALID9_METHODS, "VALID9_raw_candidate_quality"),
        ):
            scored, summaries, comparisons = summarize_methods(response_map, score_map, methods, judge, scope)
            all_pointwise_rows += scored
            all_pointwise_summaries += summaries
            all_pointwise_comparisons += comparisons

    candidates = read_jsonl(ROOT / "valid9_candidates_standardized.jsonl")
    candidate_by = {(row["canonical_id"], row["candidate_type"]): row for row in candidates}
    selectors, selector_failures = selector_decisions()
    replay_rows = []
    for candidate in candidates:
        for policy in POLICIES:
            if policy == "gpt_selector":
                decision = selectors.get((candidate["canonical_id"], candidate["candidate_type"]))
                if decision is None:
                    accepted, reasons = False, ["selector_failure"]
                else:
                    accepted = decision["accepted"]
                    reasons = [] if accepted else [decision["reason"]]
            else:
                reasons = metric_policy_reasons(candidate, policy)
                accepted = not reasons
            final_response = candidate["candidate_response"] if accepted else candidate["sft_response"]
            sft_risk = (candidate.get("sft_metrics") or {}).get("risk_score")
            revised_risk = (candidate.get("candidate_metrics") or {}).get("risk_score")
            replay_rows.append(
                {
                    "canonical_id": candidate["canonical_id"],
                    "source_order": candidate["source_order"],
                    "candidate_type": candidate["candidate_type"],
                    "policy": policy,
                    "gate_called": candidate["gate_called"],
                    "technical_valid": candidate["technical_valid"],
                    "accepted": accepted,
                    "reasons": reasons,
                    "final_source": candidate["candidate_type"] if accepted else "SFT",
                    "final_response": final_response,
                    "stored_current_accepted": candidate["current_rules_accepted"],
                    "stored_current_final_response": candidate["current_rules_final_response"],
                    "risk_delta": (
                        float(sft_risk) - float(revised_risk)
                        if finite(sft_risk) and finite(revised_risk) else None
                    ),
                }
            )
    current_rows = [row for row in replay_rows if row["policy"] == "current_rules"]
    current_reproduced = all(
        row["accepted"] == bool(row["stored_current_accepted"])
        and row["final_response"] == row["stored_current_final_response"]
        for row in current_rows
    )
    if not current_reproduced:
        raise RuntimeError("VALID current_rules replay failed to reproduce stored results")

    policy_selection_summary = []
    policy_quality_summary = []
    policy_quality_comparisons = []
    policy_pointwise_rows = []
    for policy in POLICIES:
        policy_rows = [row for row in replay_rows if row["policy"] == policy]
        for candidate_type in VALID9_METHODS[1:]:
            subset = [row for row in policy_rows if row["candidate_type"] == candidate_type]
            policy_selection_summary.append(
                {
                    "policy": policy,
                    "candidate_type": candidate_type,
                    "called": len(subset),
                    "accepted": sum(row["accepted"] for row in subset),
                    "sft_retained": sum(not row["accepted"] for row in subset),
                    "failures": sum("selector_failure" in row["reasons"] or "technical_invalid" in row["reasons"] for row in subset),
                    "reasons": dict(Counter(reason for row in subset for reason in row["reasons"])),
                    "mean_risk_delta": mean([row["risk_delta"] for row in subset if row["risk_delta"] is not None]),
                }
            )
        response_map = {method: dict(valid9_response_map["SFT"]) if method == "SFT" else {} for method in VALID9_METHODS}
        for row in policy_rows:
            question = valid9_response_map["SFT"][row["canonical_id"]][0]
            response_map[row["candidate_type"]][row["canonical_id"]] = (question, row["final_response"])
        for judge, score_map in (("GPT-4.1", gpt_scores), ("Llama", llama_scores)):
            scored, summaries, comparisons = summarize_methods(
                response_map, score_map, VALID9_METHODS, judge, f"VALID9_final_after_{policy}"
            )
            policy_pointwise_rows += scored
            policy_quality_summary += summaries
            policy_quality_comparisons += comparisons

    llama_pairwise, gpt_pairwise, direct_pairwise = pairwise_normalized_rows()
    if not all(row["same_ab_order"] for row in direct_pairwise):
        raise RuntimeError("A/B order changed between GPT and Llama pairwise evaluation")
    if not all(
        row["same_system_prompt_sha256"] and row["same_user_prompt_sha256"]
        for row in direct_pairwise
    ):
        raise RuntimeError("Prompt changed between GPT and Llama pairwise evaluation")
    pairwise_summary = summarize_pairwise(llama_pairwise, "Llama") + summarize_pairwise(gpt_pairwise, "GPT-4.1")

    pointwise_judge_comparison = []
    valid_rep = read_jsonl(ROOT / "valid_pointwise_representatives.jsonl")
    for row in valid_rep:
        key = row["question_response_sha256"]
        gpt, llama = gpt_scores.get(key), llama_scores.get(key)
        pointwise_judge_comparison.append(
            {
                "canonical_id": row["canonical_id"],
                "question_response_sha256": key,
                "member_conditions": row["member_conditions"],
                "gpt41_scores": gpt,
                "llama_scores": llama,
                "score_pair_available": gpt is not None and llama is not None,
                "all_four_scores_equal": gpt == llama if gpt is not None and llama is not None else None,
                "gpt41_minus_llama": (
                    {axis: gpt[axis] - llama[axis] for axis in AXES}
                    if gpt is not None and llama is not None else None
                ),
            }
        )

    test_gpt_rows = read_jsonl(ROOT / "test_method_pointwise_gpt41.jsonl")
    test_response_map = {method: {} for method in TEST_METHODS}
    test_gpt_score_map = {}
    for row in test_gpt_rows:
        method = row["method"]
        question = next(
            item["question"] for item in read_jsonl(TEST_SOURCE / "sft_test_outputs.jsonl")
            if item["canonical_id"] == row["canonical_id"]
        )
        test_response_map[method][row["canonical_id"]] = (question, row["response"])
        test_gpt_score_map[pair_hash(question, row["response"])] = row["scores"]
    test_pointwise_summaries = []
    test_pointwise_comparisons = []
    for judge, score_map in (("GPT-4.1", test_gpt_score_map), ("Llama", llama_scores)):
        _, summaries, comparisons = summarize_methods(test_response_map, score_map, TEST_METHODS, judge, "TEST24_final_methods")
        test_pointwise_summaries += summaries
        test_pointwise_comparisons += comparisons

    test_pointwise_judge_comparison = []
    for row in latest_rows(ROOT / "results/test_pointwise_gpt41.jsonl").values():
        key = row["metadata"]["question_response_sha256"]
        gpt = row["normalized"]["scores"] if row.get("ok") is True else None
        llama = llama_scores.get(key)
        test_pointwise_judge_comparison.append(
            {
                "canonical_id": row["metadata"]["canonical_id"],
                "question_response_sha256": key,
                "member_methods": row["metadata"]["member_methods"],
                "gpt41_scores": gpt,
                "llama_scores": llama,
                "score_pair_available": gpt is not None and llama is not None,
                "all_four_scores_equal": gpt == llama if gpt is not None and llama is not None else None,
                "gpt41_minus_llama": (
                    {axis: gpt[axis] - llama[axis] for axis in AXES}
                    if gpt is not None and llama is not None else None
                ),
            }
        )

    selector_disagreements = []
    for candidate in candidates:
        key = (candidate["canonical_id"], candidate["candidate_type"])
        decision = selectors[key]
        policy_choices = {
            policy: next(
                row["accepted"] for row in replay_rows
                if row["canonical_id"] == key[0] and row["candidate_type"] == key[1] and row["policy"] == policy
            )
            for policy in POLICIES
        }
        if len(set(policy_choices.values())) > 1:
            selector_disagreements.append(
                {
                    "canonical_id": candidate["canonical_id"],
                    "candidate_type": candidate["candidate_type"],
                    "question": candidate["question"],
                    "sft_response": candidate["sft_response"],
                    "candidate_response": candidate["candidate_response"],
                    "policy_acceptance": policy_choices,
                    "gpt_selector_decision": decision,
                    "risk_delta": next(
                        row["risk_delta"] for row in replay_rows
                        if row["canonical_id"] == key[0] and row["candidate_type"] == key[1]
                    ),
                }
            )

    preference_available = [row for row in direct_pairwise if row["preference_agreement"] is not None]
    api_manifests = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((ROOT / "results").glob("*_gpt41.manifest.json"))]
    api_usage = {
        "calls": sum(row["completed"] for row in api_manifests),
        "input_tokens": sum(row["input_tokens"] for row in api_manifests),
        "output_tokens": sum(row["output_tokens"] for row in api_manifests),
        "estimated_cost_usd": round(sum(row["estimated_cost_usd"] for row in api_manifests), 8),
        "pricing_usd_per_million_tokens": {"input": 2.0, "output": 8.0},
        "pricing_source": "https://developers.openai.com/api/docs/models/gpt-4.1",
        "queues": {row["queue"]: row for row in api_manifests},
        "returned_models": sorted({model for row in api_manifests for model in row["returned_models"]}),
        "system_fingerprints": sorted({fp for row in api_manifests for fp in row["system_fingerprints"]}),
    }

    coverage = [
        {"scope": "TEST24 pointwise", "GPT-4.1": "51/51 (47 final cache + 4 new candidates)", "Llama": "51/51 (47 reused + 4 new)"},
        {"scope": "TEST24 selector", "GPT-4.1": "16/16 two-order requests", "Llama": "not applicable"},
        {"scope": "VALID43/9 pointwise", "GPT-4.1": "126/126 unique", "Llama": "126/126 unique"},
        {"scope": "VALID existing pairwise", "GPT-4.1": "63/63 replayed", "Llama": "63/63 reused"},
        {"scope": "VALID missing C pairs", "GPT-4.1": "27/27 new", "Llama": "27/27 new"},
        {"scope": "VALID complete pairwise", "GPT-4.1": "90/90", "Llama": "90/90"},
        {"scope": "VALID selector", "GPT-4.1": "72/72 two-order requests", "Llama": "not applicable"},
        {"scope": "always-on generation", "GPT-4.1": "unavailable; not generated", "Llama": "unavailable; not generated"},
    ]
    summary = {
        "version": VERSION,
        "status": "complete",
        "post_hoc_exploratory": True,
        "coverage": coverage,
        "valid43_same_canonical_id_set": valid43_id_sets_match,
        "test24_current_rules_reproduced_stored_results": test_interim_summary["current_rules_reproduced_stored_output"],
        "current_rules_reproduced_stored_valid_results": current_reproduced,
        "valid9_arm_provenance": valid9_arm_provenance,
        "selector_failures": selector_failures,
        "valid43_pointwise_method_summaries": [row for row in all_pointwise_summaries if row["scope"] == "VALID43_final_outputs"],
        "valid9_candidate_pointwise_summaries": [row for row in all_pointwise_summaries if row["scope"] == "VALID9_raw_candidate_quality"],
        "valid9_policy_selection": policy_selection_summary,
        "valid9_policy_final_quality": policy_quality_summary,
        "test24_method_summaries_both_judges": test_pointwise_summaries,
        "test24_policy_results": test_interim_summary["policy_results"],
        "pairwise_preference_comparison": {
            "same_prompt_response_order_N": len(direct_pairwise),
            "both_preferences_parsed_N": len(preference_available),
            "preference_agreement_N": sum(row["preference_agreement"] is True for row in preference_available),
            "preference_disagreement_N": sum(row["preference_agreement"] is False for row in preference_available),
            "llama_preference_parse_failures": sum(row["llama_preferred"] is None for row in direct_pairwise),
            "gpt41_preference_parse_failures": sum(row["gpt41_preferred"] is None for row in direct_pairwise),
        },
        "pointwise_judge_comparison_valid126": {
            "N": len(pointwise_judge_comparison),
            "both_available": sum(row["score_pair_available"] for row in pointwise_judge_comparison),
            "all_four_scores_equal": sum(row["all_four_scores_equal"] is True for row in pointwise_judge_comparison),
            "mean_gpt41_minus_llama": {
                axis: mean([row["gpt41_minus_llama"][axis] for row in pointwise_judge_comparison if row["gpt41_minus_llama"]])
                for axis in AXES
            },
        },
        "pointwise_judge_comparison_test47": {
            "N": len(test_pointwise_judge_comparison),
            "both_available": sum(row["score_pair_available"] for row in test_pointwise_judge_comparison),
            "all_four_scores_equal": sum(
                row["all_four_scores_equal"] is True for row in test_pointwise_judge_comparison
            ),
            "mean_gpt41_minus_llama": {
                axis: mean(
                    [
                        row["gpt41_minus_llama"][axis]
                        for row in test_pointwise_judge_comparison
                        if row["gpt41_minus_llama"]
                    ]
                )
                for axis in AXES
            },
        },
        "api_usage": api_usage,
        "known_failures_and_factual_errors": [
            {"scope": "TEST", "case": "TEST-15", "issue": "stored collapsed-spacing generation failure; preserved"},
            {"scope": "TEST", "case": "TEST-24 Proposed candidate", "issue": "incorrect U.S. crisis number 989 instead of 988; preserved"},
            {"scope": "VALID", "case": "qa_24f1b77a40c9b22203b8f337 feedback_B", "issue": "known specificity-loss candidate accepted by current_rules"},
            {"scope": "VALID", "case": "qa_b243b4c887feb8e3594dd74c feedback_B", "issue": "known current_rules SFT fallback"},
        ],
        "no_retraining": True,
        "no_response_or_candidate_regeneration": True,
        "selection_scores_not_reused_as_final_evaluation": True,
    }

    outputs = {
        "FULL_COMPARISON_SUMMARY.json": summary,
        "valid_pointwise_scored_rows.jsonl": all_pointwise_rows,
        "valid_pointwise_method_summaries.jsonl": all_pointwise_summaries,
        "valid_pointwise_all_method_pairs.jsonl": all_pointwise_comparisons,
        "valid9_policy_replay.jsonl": replay_rows,
        "valid9_policy_selection_summary.jsonl": policy_selection_summary,
        "valid9_policy_final_scored_rows.jsonl": policy_pointwise_rows,
        "valid9_policy_final_quality_summary.jsonl": policy_quality_summary,
        "valid9_policy_final_all_method_pairs.jsonl": policy_quality_comparisons,
        "valid_pairwise_llama_normalized.jsonl": llama_pairwise,
        "valid_pairwise_gpt41_normalized.jsonl": gpt_pairwise,
        "valid_pairwise_complete10_summary.jsonl": pairwise_summary,
        "valid_pairwise_judge_direct_comparison.jsonl": direct_pairwise,
        "valid_pointwise_judge_direct_comparison.jsonl": pointwise_judge_comparison,
        "test_pointwise_judge_direct_comparison.jsonl": test_pointwise_judge_comparison,
        "test24_method_summaries_both_judges.jsonl": test_pointwise_summaries,
        "test24_all_method_pairs_both_judges.jsonl": test_pointwise_comparisons,
        "selector_disagreements_full_responses.jsonl": selector_disagreements,
    }
    for name, payload in outputs.items():
        path = ROOT / name
        if name.endswith(".jsonl"):
            write_jsonl(path, payload)
        else:
            write_json(path, payload)

    def summary_lookup(rows, scope, judge):
        return [row for row in rows if row["scope"] == scope and row["judge"] == judge]

    lines = [
        "# GPT-4.1 and Llama frozen-response comparison",
        "",
        "This is a post-hoc exploratory comparison. All responses, candidates, gates, risks, and feedback were frozen; only evaluation and policy replay were added.",
        "",
        "## Coverage",
        "",
        "| Scope | GPT-4.1 | Llama |",
        "|---|---|---|",
    ]
    for row in coverage:
        lines.append(f"| {row['scope']} | {row['GPT-4.1']} | {row['Llama']} |")
    lines += ["", "## TEST24 final methods", ""]
    for judge in ("GPT-4.1", "Llama"):
        lines += [f"### {judge}", "", "| Method | N | Overall | Empathy | Specificity | Medical Advice | Mean |", "|---|---:|---:|---:|---:|---:|---:|"]
        for row in summary_lookup(test_pointwise_summaries, "TEST24_final_methods", judge):
            m = row["means"]
            lines.append(f"| {row['method']} | {row['N']} | {m['overall_quality']:.4f} | {m['empathy']:.4f} | {m['specificity']:.4f} | {m['medical_advice']:.4f} | {row['four_axis_mean']:.4f} |")
        lines.append("")
    lines += [
        "### TEST24 selector replay",
        "",
        "| Policy | Called | Accepted | SFT retained | Mean called risk delta | Failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        row = test_interim_summary["policy_results"][policy]
        lines.append(
            f"| {policy} | {row['gate_called']} | {row['accepted']} | {row['sft_retained']} | "
            f"{row['mean_risk_delta_called']:.4f} | {row['selector_or_technical_failures']} |"
        )
    lines += [
        "",
        "Per-question SFT deltas and every called case's actual SFT/candidate text are in `test_method_pointwise_gpt41.jsonl`, `test_policy_replay.jsonl`, and `TEST24_GPT41_INTERIM_REPORT.md`.",
        "",
    ]
    lines += ["## VALID43 stored final outputs", ""]
    for judge in ("GPT-4.1", "Llama"):
        lines += [f"### {judge}", "", "| Method | N | Overall | Empathy | Specificity | Medical Advice | Mean |", "|---|---:|---:|---:|---:|---:|---:|"]
        for row in summary_lookup(all_pointwise_summaries, "VALID43_final_outputs", judge):
            m = row["means"]
            lines.append(f"| {row['method']} | {row['N']} | {m['overall_quality']:.4f} | {m['empathy']:.4f} | {m['specificity']:.4f} | {m['medical_advice']:.4f} | {row['four_axis_mean']:.4f} |")
        lines.append("")
    lines += [
        "## VALID9 candidate quality and selection",
        "",
        "Candidate inputs and generation adapters are distinct: `feedback_A` revises the existing denoiser candidate with `proposed_mask_off/final`; `feedback_B` revises the SFT original with that same adapter; `feedback_C` uses the same SFT original and frozen feedback but changes only generation adapter to `gemma/sft/final`. No arm was regenerated for this comparison.",
        "",
        "### Raw candidate quality",
        "",
        "| Candidate | GPT-4.1 mean | Llama mean |",
        "|---|---:|---:|",
    ]
    for method in VALID9_METHODS:
        gpt_row = next(
            row for row in all_pointwise_summaries
            if row["scope"] == "VALID9_raw_candidate_quality" and row["judge"] == "GPT-4.1" and row["method"] == method
        )
        llama_row = next(
            row for row in all_pointwise_summaries
            if row["scope"] == "VALID9_raw_candidate_quality" and row["judge"] == "Llama" and row["method"] == method
        )
        lines.append(f"| {method} | {gpt_row['four_axis_mean']:.4f} | {llama_row['four_axis_mean']:.4f} |")
    lines += [
        "",
        "### Selection and final-response quality",
        "",
        "| Policy | Candidate | Accepted / 9 | Mean risk delta | GPT-4.1 final mean | Llama final mean | Failures |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    quality_lookup = {
        (row["scope"].removeprefix("VALID9_final_after_"), row["method"], row["judge"]): row
        for row in policy_quality_summary
    }
    for row in policy_selection_summary:
        gpt_quality = quality_lookup[(row["policy"], row["candidate_type"], "GPT-4.1")]["four_axis_mean"]
        llama_quality = quality_lookup[(row["policy"], row["candidate_type"], "Llama")]["four_axis_mean"]
        lines.append(
            f"| {row['policy']} | {row['candidate_type']} | {row['accepted']}/9 | "
            f"{row['mean_risk_delta']:.4f} | {gpt_quality:.4f} | {llama_quality:.4f} | {row['failures']} |"
        )
    lines += [
        "",
        "`accept_generated` accepts every technically valid stored candidate only for the nine already-called cases; it is not always-on generation. Uncalled examples retain SFT, and always-on generation remains unavailable.",
        "",
        "Both TEST and VALID `current_rules` replays exactly reproduced their stored final responses.",
    ]
    pair_cmp = summary["pairwise_preference_comparison"]
    lines += [
        "",
        "## Legacy pairwise judge comparison",
        "",
        f"All 90 pairs use identical prompts, responses, and A/B order across judges. Parsed preference agreement: {pair_cmp['preference_agreement_N']}/{pair_cmp['both_preferences_parsed_N']}; disagreements: {pair_cmp['preference_disagreement_N']}. Llama unparsed preferences: {pair_cmp['llama_preference_parse_failures']}; GPT-4.1 unparsed preferences: {pair_cmp['gpt41_preference_parse_failures']}.",
        "",
        "GPT-4.1 is used for both the blind selector and one final evaluator, but with separate prompts and calls; this shared model can introduce selection-aligned bias, so Llama results are retained as a distinct comparison.",
        "",
        "## Cost",
        "",
        f"GPT-4.1 calls: {api_usage['calls']}; input tokens: {api_usage['input_tokens']}; output tokens: {api_usage['output_tokens']}; estimated cost: ${api_usage['estimated_cost_usd']:.6f}.",
        "",
        "## Preserved known issues",
        "",
        "- TEST-15 collapsed-spacing generation failure.",
        "- TEST-24 candidate says `989` instead of the correct U.S. crisis number `988`.",
        "- VALID `qa_24f1...` feedback_B specificity-loss acceptance and `qa_b243...` feedback_B SFT fallback are retained as known cases.",
        "",
        "Full per-question scores, all method-pair deltas, selector disagreements with actual responses, raw API JSON, prompts, hashes, and usage are included in the research bundle.",
    ]
    report = ROOT / "FULL_GPT41_LLAMA_COMPARISON_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    bundle = ROOT / "GPT41_Llama_TEST24_VALID_complete_research_bundle.zip"
    root_members = ["FULL_GPT41_LLAMA_COMPARISON_REPORT.md", *outputs.keys(), "test_request_contract.json", "test_candidate_pointwise_contract.json", "valid_request_contract.json", "test_called_candidates.jsonl", "valid9_candidates_standardized.jsonl", "valid_pointwise_assignments.jsonl", "valid_pointwise_representatives.jsonl", "TEST24_GPT41_INTERIM_REPORT.md", "TEST24_GPT41_SUMMARY.json", "test_policy_replay.jsonl", "test_called_cases_full.jsonl"]
    request_members = [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "requests").glob("*.jsonl"))]
    result_members = [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "results").glob("*.jsonl"))]
    manifest_members = [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "results").glob("*.manifest.json"))]
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in root_members + request_members + result_members + manifest_members:
            archive.write(ROOT / name, arcname=name)
        archive.write(FEEDBACK_ROOT / "blind_judge_results.jsonl", arcname="legacy_llama/feedback_v1_pairwise54.jsonl")
        archive.write(BC_ROOT / "bc_blind_judge_results.jsonl", arcname="legacy_llama/feedback_bc_pairwise9.jsonl")
        archive.write(TEST_LLAMA / "judged_representatives.jsonl", arcname="legacy_llama/test24_pointwise47.jsonl")
        for name in (
            "experiment_contract.json",
            "feedback.jsonl",
            "candidate_generations.jsonl",
            "candidate_results.jsonl",
            "final_manifest.json",
            "REPORT.md",
        ):
            archive.write(FEEDBACK_ROOT / name, arcname=f"frozen_valid_feedback_v1/{name}")
        for name in (
            "extension_contract.json",
            "c_candidate_generations.jsonl",
            "c_candidate_results.jsonl",
            "final_manifest.json",
            "REPORT_BC.md",
        ):
            archive.write(BC_ROOT / name, arcname=f"frozen_valid_feedback_bc/{name}")
    final_manifest = {
        "status": "complete",
        "version": VERSION,
        "post_hoc_exploratory": True,
        "report": str(report),
        "report_sha256": sha256_file(report),
        "bundle": str(bundle),
        "bundle_sha256": sha256_file(bundle),
        "api_usage": api_usage,
        "coverage": coverage,
        "no_retraining": True,
        "no_response_or_candidate_regeneration": True,
    }
    write_json(ROOT / "FINAL_MANIFEST.json", final_manifest)
    print(json.dumps(final_manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
