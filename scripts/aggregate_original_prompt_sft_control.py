#!/usr/bin/env python3
"""Aggregate the original-prompt SFT-adapter control and create the research bundle."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VERSION = "original-prompt-sft-control-complete-v1-20261009"
ROOT = Path("/mnt/ssd00/hsoh/original_prompt_sft_control_20261009_v1")
PRIOR = Path("/mnt/ssd00/hsoh/gpt41_posthoc_selection_20261009_v1")
REPO = Path("/home/user/hsoh/mh-denoise")
WORKTREE = Path("/home/user/hsoh/mh-denoise-original-prompt-sft-control-20261009")
CHECKPOINT_ROOT = REPO / "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/gemma"
VALID_ROOT = CHECKPOINT_ROOT / "generation_1536"
TEST_ROOT = CHECKPOINT_ROOT / "generation_1536_test_surface_relaxed_v1_20261008/part24"
LOCAL_TEST = Path("/mnt/ssd00/hsoh/test24_local_four_axis_20261009")
FEEDBACK_ROOT = Path(
    "/home/user/hsoh/mh-denoise-feedback-v1-valid-20261008/"
    "outputs/valid_feedback_v1_20261008"
)
BC_ROOT = Path("/mnt/ssd00/hsoh/feedback_v1_bc_20261009")
AXES = ("overall_quality", "empathy", "specificity", "medical_advice")
POLICIES = ("current_rules", "no_min_length", "risk_only", "gpt_selector")
ARMS = ("original_denoiser", "original_prompt_sft_control")


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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_hash(question: str, response: str) -> str:
    return sha256_text(question + "\0" + response)


def latest(path: Path) -> list[dict[str, Any]]:
    output = {}
    for index, row in enumerate(read_jsonl(path)):
        output[row.get("request_id") or row.get("comparison_id") or index] = row
    return list(output.values())


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def composite(scores: dict[str, Any]) -> float:
    return statistics.mean(float(scores[axis]) for axis in AXES)


def outcome(delta: float) -> str:
    return "improved" if delta > 0 else ("worsened" if delta < 0 else "tie")


def gpt_score_map() -> dict[str, dict[str, int]]:
    output = {}
    paths = (
        PRIOR / "results/test_pointwise_gpt41.jsonl",
        PRIOR / "results/test_candidate_pointwise_gpt41.jsonl",
        PRIOR / "results/valid_pointwise_gpt41.jsonl",
        ROOT / "results/control_pointwise_gpt41.jsonl",
    )
    for path in paths:
        for row in latest(path):
            if row.get("ok") is not True or row.get("parser") != "pointwise_four_axis":
                continue
            key = row["metadata"]["question_response_sha256"]
            scores = row["normalized"]["scores"]
            if key in output and output[key] != scores:
                raise RuntimeError(f"Conflicting GPT pointwise score for {key}")
            output[key] = scores
    for row in read_jsonl(ROOT / "results/evaluation_cache_reuse.jsonl"):
        if row["evaluation"] != "pointwise" or row["judge"] != "GPT-4.1":
            continue
        scores = row["result"]["normalized"]["scores"]
        key = row["question_response_sha256"]
        if key in output and output[key] != scores:
            raise RuntimeError(f"Conflicting reused GPT pointwise score for {key}")
        output[key] = scores
    return output


def llama_score_map() -> dict[str, dict[str, int]]:
    output = {}
    for row in read_jsonl(LOCAL_TEST / "judged_representatives.jsonl"):
        if row.get("judge_ok") is True:
            output[row["question_response_sha256"]] = row["judge_scores"]
    for path in (
        PRIOR / "results/test_candidate_pointwise_llama.jsonl",
        PRIOR / "results/valid_pointwise_llama.jsonl",
        ROOT / "results/control_pointwise_llama.jsonl",
    ):
        for row in latest(path):
            if row.get("judge_ok") is not True or row.get("parser") != "pointwise_four_axis":
                continue
            key = row["metadata"]["question_response_sha256"]
            scores = row["judge_normalized"]["scores"]
            if key in output and output[key] != scores:
                raise RuntimeError(f"Conflicting Llama pointwise score for {key}")
            output[key] = scores
    for row in read_jsonl(ROOT / "results/evaluation_cache_reuse.jsonl"):
        if row["evaluation"] != "pointwise" or row["judge"] != "Llama":
            continue
        result = row["result"]
        scores = result.get("judge_scores") or result["judge_normalized"]["scores"]
        key = row["question_response_sha256"]
        if key in output and output[key] != scores:
            raise RuntimeError(f"Conflicting reused Llama pointwise score for {key}")
        output[key] = scores
    return output


def selector_result_rows() -> list[dict[str, Any]]:
    output = []
    for path in (
        PRIOR / "results/test_selector_gpt41.jsonl",
        PRIOR / "results/valid_selector_gpt41.jsonl",
    ):
        for row in latest(path):
            metadata = row["metadata"]
            if path.name.startswith("valid") and metadata.get("candidate_type") != "original_denoiser":
                continue
            split = "VALID" if path.name.startswith("valid") else "TEST"
            output.append({"split": split, "arm": "original_denoiser", "metadata": metadata, "result": row})
    for row in latest(ROOT / "results/control_selector_gpt41.jsonl"):
        output.append({"split": row["metadata"]["split"], "arm": "original_prompt_sft_control", "metadata": row["metadata"], "result": row})
    for row in read_jsonl(ROOT / "results/evaluation_cache_reuse.jsonl"):
        if row["evaluation"] == "selector":
            metadata = {key: row[key] for key in (
                "split", "canonical_id", "case_id", "source_order", "candidate_type",
                "order_index", "variant_for_A", "variant_for_B", "question_sha256",
                "response_A_sha256", "response_B_sha256",
            )}
            output.append({"split": row["split"], "arm": "original_prompt_sft_control", "metadata": metadata, "result": row["result"]})
    return output


def selector_decisions() -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for item in selector_result_rows():
        metadata = item["metadata"]
        grouped[(item["split"], metadata["canonical_id"], item["arm"])].append(item)
    decisions = {}
    failures = []
    for key, values in grouped.items():
        values = sorted(values, key=lambda item: item["metadata"]["order_index"])
        if len(values) != 2 or any(item["result"].get("ok") is not True for item in values):
            decisions[key] = {"accepted": False, "reason": "selector_failure", "mapped_choices": []}
            failures.append({"split": key[0], "canonical_id": key[1], "arm": key[2], "reason": "missing_or_failed_order"})
            continue
        mapped = []
        for item in values:
            choice = item["result"]["normalized"]["choice"]
            mapped.append("tie" if choice == "tie" else item["metadata"][f"variant_for_{choice}"])
        accepted = mapped == ["candidate", "candidate"]
        reason = "candidate_selected_both_orders" if accepted else (
            "selector_tie" if "tie" in mapped else "selector_kept_sft_or_order_disagreement"
        )
        decisions[key] = {
            "accepted": accepted,
            "reason": reason,
            "mapped_choices": mapped,
            "request_ids": [item["result"].get("request_id") for item in values],
        }
    return decisions, failures


def candidate_rows() -> list[dict[str, Any]]:
    frozen = {row["canonical_id"]: row for row in read_jsonl(ROOT / "frozen_called_cases.jsonl")}
    generated = {row["canonical_id"]: row for row in read_jsonl(ROOT / "results/control_generations.jsonl")}
    if set(frozen) != set(generated) or len(frozen) != 17:
        raise RuntimeError("Frozen/control candidate membership mismatch")
    output = []
    for canonical_id, source in frozen.items():
        control = generated[canonical_id]
        original_generation = source["original_generation"]
        original_technical = bool(source["original_candidate"].strip()) and bool(
            original_generation.get("eos_reached")
        ) and not bool(original_generation.get("source_input_truncated"))
        base = {
            "split": source["split"],
            "canonical_id": canonical_id,
            "case_id": source.get("case_id"),
            "source_order": source["source_order"],
            "called_order": source["called_order"],
            "question": source["question"],
            "sft_response": source["sft_response"],
            "sft_metrics": source["sft_metrics_stored"],
            "call_reason": source["call_reason"],
        }
        output.append(
            {
                **base,
                "arm": "original_denoiser",
                "candidate_response": source["original_candidate"],
                "candidate_raw": source["original_candidate_raw"],
                "candidate_metrics": source["original_metrics_stored"],
                "technical_valid": original_technical,
                "generation": original_generation,
                "generation_adapter": json.loads((ROOT / "experiment_contract.json").read_text())["adapter_only_change"]["original_denoiser"],
                "stored_accepted": source["original_stored_accepted"],
                "stored_reject_reason": source["original_stored_reject_reason"],
                "stored_acceptance_policy": source["original_acceptance_policy"],
            }
        )
        output.append(
            {
                **base,
                "arm": "original_prompt_sft_control",
                "candidate_response": control["control_response"],
                "candidate_raw": control["control_raw_decoded_output"],
                "candidate_metrics": control["control_metrics"],
                "technical_valid": control["technical_valid"],
                "generation": control["control_generation"],
                "generation_adapter": control["control_adapter"],
                "stored_accepted": None,
                "stored_reject_reason": None,
                "stored_acceptance_policy": None,
            }
        )
    return sorted(output, key=lambda row: (row["split"], row["source_order"], row["arm"]))


def policy_reasons(candidate: dict[str, Any], policy: str) -> list[str]:
    if not candidate["technical_valid"]:
        return ["technical_invalid"]
    sft = candidate["sft_metrics"]
    revised = candidate["candidate_metrics"]
    if policy == "risk_only":
        if not finite(sft.get("risk_score")) or not finite(revised.get("risk_score")):
            return ["invalid_risk"]
        return [] if float(sft["risk_score"]) - float(revised["risk_score"]) >= 0.01 else ["risk_not_improved"]
    if policy not in {"current_rules", "no_min_length"}:
        raise ValueError(policy)
    reasons = []
    if policy == "current_rules" and int(revised.get("word_count") or 0) < 20:
        reasons.append("short_denoiser_response")
    if not finite(sft.get("risk_score")) or not finite(revised.get("risk_score")):
        reasons.append("invalid_risk")
    elif float(revised["risk_score"]) > float(sft["risk_score"]) - 0.01:
        reasons.append("risk_not_improved")
    if int(revised.get("bad_safety_count") or 0) > int(sft.get("bad_safety_count") or 0):
        reasons.append("bad_safety_increased")
    minimum_overlap = max(0.0, float(sft.get("keyword_overlap") or 0.0) - 0.15)
    if float(revised.get("keyword_overlap") or 0.0) < minimum_overlap:
        reasons.append("question_keyword_overlap_low")
    return reasons


def summarize_responses(
    response_map: dict[str, tuple[str, str]], score_map: dict[str, dict[str, int]]
) -> dict[str, Any]:
    rows = []
    for canonical_id, (question, response) in response_map.items():
        scores = score_map.get(pair_hash(question, response))
        if scores is None:
            raise RuntimeError(f"Missing pointwise score for {canonical_id}: {pair_hash(question, response)}")
        rows.append({"canonical_id": canonical_id, "scores": scores, "composite": composite(scores)})
    return {
        "N": len(rows),
        "means": {axis: mean([row["scores"][axis] for row in rows]) for axis in AXES},
        "four_axis_mean": mean([row["composite"] for row in rows]),
    }


def paired_summary(
    left_map: dict[str, tuple[str, str]],
    right_map: dict[str, tuple[str, str]],
    score_map: dict[str, dict[str, int]],
) -> dict[str, Any]:
    common = [canonical_id for canonical_id in left_map if canonical_id in right_map]
    deltas = []
    axis_deltas = {axis: [] for axis in AXES}
    counts = Counter()
    details = []
    for canonical_id in common:
        lq, lr = left_map[canonical_id]
        rq, rr = right_map[canonical_id]
        if lq != rq:
            raise RuntimeError(f"Question mismatch for {canonical_id}")
        ls = score_map[pair_hash(lq, lr)]
        rs = score_map[pair_hash(rq, rr)]
        delta = composite(rs) - composite(ls)
        counts[outcome(delta)] += 1
        deltas.append(delta)
        for axis in AXES:
            axis_deltas[axis].append(rs[axis] - ls[axis])
        details.append({"canonical_id": canonical_id, "delta": round(delta, 4), "outcome": outcome(delta), "left_scores": ls, "right_scores": rs})
    return {
        "N": len(common),
        "mean_four_axis_delta_right_minus_left": mean(deltas),
        "mean_axis_deltas_right_minus_left": {axis: mean(values) for axis, values in axis_deltas.items()},
        "right_improved_tie_worsened": {key: counts[key] for key in ("improved", "tie", "worsened")},
        "details": details,
    }


def normalized_pairwise_rows() -> list[dict[str, Any]]:
    raw = []
    for judge, path in (
        ("GPT-4.1", ROOT / "results/control_pairwise_gpt41.jsonl"),
        ("Llama", ROOT / "results/control_pairwise_llama.jsonl"),
    ):
        for row in latest(path):
            raw.append({"judge": judge, "metadata": row["metadata"], "result": row})
    for row in read_jsonl(ROOT / "results/evaluation_cache_reuse.jsonl"):
        if row["evaluation"] != "pairwise":
            continue
        metadata = {key: row[key] for key in (
            "split", "canonical_id", "case_id", "source_order", "comparison_id", "pair",
            "order_index", "variant_for_A", "variant_for_B", "response_A_sha256", "response_B_sha256",
        )}
        raw.append({"judge": row["judge"], "metadata": metadata, "result": row["result"]})
    output = []
    for item in raw:
        result, metadata, judge = item["result"], item["metadata"], item["judge"]
        ok = result.get("ok") is True if judge == "GPT-4.1" else result.get("judge_ok") is True
        normalized = result.get("normalized") if judge == "GPT-4.1" else result.get("judge_normalized")
        preference = normalized.get("preference") if ok and isinstance(normalized, dict) else None
        preferred = None
        if preference in {"A", "B"}:
            preferred = metadata[f"variant_for_{preference}"]
        elif preference == "tie":
            preferred = "tie"
        scores_a = normalized.get("response_A") if ok and isinstance(normalized, dict) else None
        scores_b = normalized.get("response_B") if ok and isinstance(normalized, dict) else None
        by_variant = {
            metadata["variant_for_A"]: scores_a,
            metadata["variant_for_B"]: scores_b,
        }
        output.append(
            {
                **metadata,
                "judge": judge,
                "parse_ok": bool(ok and preferred is not None and scores_a and scores_b),
                "preferred": preferred,
                "original_scores": by_variant.get("original_denoiser"),
                "control_scores": by_variant.get("original_prompt_sft_control"),
                "raw_preference": preference,
                "request_id": result.get("request_id"),
            }
        )
    return sorted(output, key=lambda row: (row["judge"], row["split"], row["source_order"], row["order_index"]))


def pairwise_summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["judge"], row["split"], row["canonical_id"])].append(row)
    cases = []
    for (judge, split, canonical_id), values in grouped.items():
        values = sorted(values, key=lambda row: row["order_index"])
        parsed = [row for row in values if row["parse_ok"]]
        if len(values) != 2 or len(parsed) != 2:
            consensus = "parse_failure"
        else:
            preferences = [row["preferred"] for row in parsed]
            if preferences == ["original_denoiser", "original_denoiser"]:
                consensus = "original_denoiser"
            elif preferences == ["original_prompt_sft_control", "original_prompt_sft_control"]:
                consensus = "original_prompt_sft_control"
            elif preferences == ["tie", "tie"]:
                consensus = "tie"
            else:
                consensus = "order_disagreement"
        cases.append(
            {
                "judge": judge,
                "split": split,
                "canonical_id": canonical_id,
                "orders": len(values),
                "parsed_orders": len(parsed),
                "mapped_preferences": [row["preferred"] for row in values],
                "consensus": consensus,
            }
        )
    summaries = []
    for judge in ("GPT-4.1", "Llama"):
        for split in ("VALID", "TEST", "ALL"):
            subset = [row for row in cases if row["judge"] == judge and (split == "ALL" or row["split"] == split)]
            counts = Counter(row["consensus"] for row in subset)
            summaries.append({"judge": judge, "split": split, "N": len(subset), "consensus_counts": dict(counts)})
    return cases, summaries


def api_usage() -> dict[str, Any]:
    rows = []
    for path in (
        ROOT / "results/control_pointwise_gpt41.jsonl",
        ROOT / "results/control_pairwise_gpt41.jsonl",
        ROOT / "results/control_selector_gpt41.jsonl",
    ):
        rows.extend(row for row in latest(path) if row.get("ok") is True)
    input_tokens = sum(int((row.get("usage") or {}).get("prompt_tokens") or 0) for row in rows)
    output_tokens = sum(int((row.get("usage") or {}).get("completion_tokens") or 0) for row in rows)
    return {
        "new_paid_calls": len(rows),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(sum(float(row.get("estimated_cost_usd") or 0) for row in rows), 8),
        "pricing_recorded_by_runner_usd_per_million": {"input": 2.0, "output": 8.0},
    }


def main() -> None:
    generation_manifest = json.loads((ROOT / "results/control_generation_manifest.json").read_text(encoding="utf-8"))
    evaluation_contract = json.loads((ROOT / "evaluation_contract.json").read_text(encoding="utf-8"))
    queue_marker = json.loads((ROOT / "results/evaluation_queues_complete.json").read_text(encoding="utf-8"))
    if generation_manifest.get("status") != "complete" or queue_marker.get("status") != "complete":
        raise RuntimeError("Generation or evaluation queues are incomplete")
    gpt_scores, llama_scores = gpt_score_map(), llama_score_map()
    score_maps = {"GPT-4.1": gpt_scores, "Llama": llama_scores}
    candidates = candidate_rows()
    if len(candidates) != 34 or any(not row["technical_valid"] for row in candidates):
        raise RuntimeError("Expected 34 technically valid arm assignments")
    selectors, selector_failures = selector_decisions()
    expected_selector_keys = {(row["split"], row["canonical_id"], row["arm"]) for row in candidates}
    if set(selectors) != expected_selector_keys:
        raise RuntimeError("Selector coverage mismatch")

    # Raw candidate score table and direct arm/SFT comparisons.
    raw_scored = []
    raw_summaries = []
    raw_comparisons = []
    for judge, score_map in score_maps.items():
        for split in ("VALID", "TEST", "ALL"):
            subset = [row for row in candidates if split == "ALL" or row["split"] == split]
            maps = {"SFT": {}, **{arm: {} for arm in ARMS}}
            for row in subset:
                maps["SFT"][row["canonical_id"]] = (row["question"], row["sft_response"])
                maps[row["arm"]][row["canonical_id"]] = (row["question"], row["candidate_response"])
                scores = score_map[pair_hash(row["question"], row["candidate_response"])]
                raw_scored.append({"judge": judge, "split": row["split"], "canonical_id": row["canonical_id"], "arm": row["arm"], "scores": scores, "four_axis_mean": round(composite(scores), 4)})
            for method, response_map in maps.items():
                raw_summaries.append({"judge": judge, "split": split, "method": method, **summarize_responses(response_map, score_map)})
            for left, right in (("SFT", ARMS[0]), ("SFT", ARMS[1]), (ARMS[0], ARMS[1])):
                result = paired_summary(maps[left], maps[right], score_map)
                details = result.pop("details")
                raw_comparisons.append({"judge": judge, "split": split, "left": left, "right": right, **result})
                for detail in details:
                    detail.update({"judge": judge, "split": split, "left": left, "right": right})

    # Pairwise two-order results.
    pair_rows = normalized_pairwise_rows()
    if len(pair_rows) != 68:
        raise RuntimeError(f"Expected 68 pairwise order results, got {len(pair_rows)}")
    pair_cases, pair_summaries = pairwise_summary(pair_rows)
    direct_judge_pairwise = []
    by_case = {(row["judge"], row["split"], row["canonical_id"]): row for row in pair_cases}
    for split, canonical_id in sorted({(row["split"], row["canonical_id"]) for row in pair_cases}):
        grow = by_case[("GPT-4.1", split, canonical_id)]
        lrow = by_case[("Llama", split, canonical_id)]
        direct_judge_pairwise.append({"split": split, "canonical_id": canonical_id, "gpt41_consensus": grow["consensus"], "llama_consensus": lrow["consensus"], "same_consensus": grow["consensus"] == lrow["consensus"]})

    # Replay the four requested policies for both arms.
    replay = []
    for row in candidates:
        sft_risk = row["sft_metrics"].get("risk_score")
        candidate_risk = row["candidate_metrics"].get("risk_score")
        risk_delta = float(sft_risk) - float(candidate_risk) if finite(sft_risk) and finite(candidate_risk) else None
        for policy in POLICIES:
            if policy == "gpt_selector":
                decision = selectors[(row["split"], row["canonical_id"], row["arm"])]
                accepted = row["technical_valid"] and decision["accepted"]
                reasons = [] if accepted else (["technical_invalid"] if not row["technical_valid"] else [decision["reason"]])
                selector_detail = decision
            else:
                reasons = policy_reasons(row, policy)
                accepted = not reasons
                selector_detail = None
            replay.append(
                {
                    "split": row["split"],
                    "canonical_id": row["canonical_id"],
                    "case_id": row.get("case_id"),
                    "source_order": row["source_order"],
                    "arm": row["arm"],
                    "policy": policy,
                    "gate_called": True,
                    "call_reason": row["call_reason"],
                    "technical_valid": row["technical_valid"],
                    "accepted": accepted,
                    "reasons": reasons,
                    "final_source": row["arm"] if accepted else "SFT",
                    "final_response": row["candidate_response"] if accepted else row["sft_response"],
                    "risk_delta": risk_delta,
                    "selector_detail": selector_detail,
                }
            )
    test_original_current = [row for row in replay if row["split"] == "TEST" and row["arm"] == "original_denoiser" and row["policy"] == "current_rules"]
    source_by = {(row["split"], row["canonical_id"], row["arm"]): row for row in candidates}
    test_current_reproduced = all(
        row["accepted"] == bool(source_by[("TEST", row["canonical_id"], "original_denoiser")]["stored_accepted"])
        for row in test_original_current
    )
    if not test_current_reproduced:
        raise RuntimeError("TEST original current_rules replay does not reproduce stored surface_relaxed_v1")
    prior_valid_policy = {
        (row["canonical_id"], row["policy"]): row
        for row in read_jsonl(PRIOR / "valid9_policy_replay.jsonl")
        if row.get("candidate_type") == "original_denoiser" and row.get("policy") in POLICIES
    }
    prior_test_policy = {
        (row["canonical_id"], row["policy"]): row
        for row in read_jsonl(PRIOR / "test_policy_replay.jsonl")
        if row.get("policy") in POLICIES
    }
    original_replay_matches_prior = True
    for row in (item for item in replay if item["arm"] == "original_denoiser"):
        prior = (prior_valid_policy if row["split"] == "VALID" else prior_test_policy).get(
            (row["canonical_id"], row["policy"])
        )
        if prior is None or prior["accepted"] != row["accepted"] or prior["final_response"] != row["final_response"]:
            original_replay_matches_prior = False
            break
    if not original_replay_matches_prior:
        raise RuntimeError("Original-arm policy replay differs from preserved prior comparison")

    # Full split response maps retain SFT on uncalled examples.
    split_sources = {
        "VALID": read_jsonl(VALID_ROOT / "sft_valid_outputs.jsonl"),
        "TEST": read_jsonl(TEST_ROOT / "sft_test_outputs.jsonl"),
    }
    full_sft = {}
    for split, rows in split_sources.items():
        full_sft[split] = {row["canonical_id"]: (row["question"], row["sft_response"]) for row in rows}
    if len(full_sft["VALID"]) != 43 or len(full_sft["TEST"]) != 24:
        raise RuntimeError("Full split baseline membership changed")

    policy_selection_summary = []
    policy_quality_summary = []
    policy_sft_comparisons = []
    policy_arm_comparisons = []
    policy_score_rows = []
    for split in ("VALID", "TEST"):
        called_ids = {row["canonical_id"] for row in candidates if row["split"] == split}
        called_sft = {canonical_id: value for canonical_id, value in full_sft[split].items() if canonical_id in called_ids}
        for judge, score_map in score_maps.items():
            policy_quality_summary.append({"split": split, "scope": "called", "judge": judge, "arm": "SFT", "policy": "baseline", **summarize_responses(called_sft, score_map)})
            policy_quality_summary.append({"split": split, "scope": "full", "judge": judge, "arm": "SFT", "policy": "baseline", **summarize_responses(full_sft[split], score_map)})
    for split in ("VALID", "TEST"):
        split_ids = set(full_sft[split])
        for policy in POLICIES:
            response_maps = {arm: dict(full_sft[split]) for arm in ARMS}
            for arm in ARMS:
                subset = [row for row in replay if row["split"] == split and row["policy"] == policy and row["arm"] == arm]
                if len(subset) != (9 if split == "VALID" else 8):
                    raise RuntimeError(f"Called policy coverage mismatch: {split} {arm} {policy}")
                for row in subset:
                    question = full_sft[split][row["canonical_id"]][0]
                    response_maps[arm][row["canonical_id"]] = (question, row["final_response"])
                policy_selection_summary.append(
                    {
                        "split": split,
                        "arm": arm,
                        "policy": policy,
                        "full_N": len(split_ids),
                        "called": len(subset),
                        "accepted": sum(row["accepted"] for row in subset),
                        "sft_retained_called": sum(not row["accepted"] for row in subset),
                        "uncalled_sft_retained": len(split_ids) - len(subset),
                        "reason_counts": dict(Counter(reason for row in subset for reason in row["reasons"])),
                        "mean_risk_delta_all_called": mean([row["risk_delta"] for row in subset if row["risk_delta"] is not None]),
                        "mean_risk_delta_accepted": mean([row["risk_delta"] for row in subset if row["accepted"] and row["risk_delta"] is not None]),
                    }
                )
            for judge, score_map in score_maps.items():
                for arm in ARMS:
                    called_map = {canonical_id: response_maps[arm][canonical_id] for canonical_id in response_maps[arm] if canonical_id in {row["canonical_id"] for row in replay if row["split"] == split}}
                    policy_quality_summary.append({"split": split, "scope": "called", "judge": judge, "arm": arm, "policy": policy, **summarize_responses(called_map, score_map)})
                    policy_quality_summary.append({"split": split, "scope": "full", "judge": judge, "arm": arm, "policy": policy, **summarize_responses(response_maps[arm], score_map)})
                    for scope, base_map, final_map in (
                        ("called", {key: full_sft[split][key] for key in called_map}, called_map),
                        ("full", full_sft[split], response_maps[arm]),
                    ):
                        result = paired_summary(base_map, final_map, score_map)
                        details = result.pop("details")
                        policy_sft_comparisons.append({"split": split, "scope": scope, "judge": judge, "arm": arm, "policy": policy, "left": "SFT", "right": arm, **result})
                        for detail in details:
                            policy_score_rows.append({"split": split, "scope": scope, "judge": judge, "arm": arm, "policy": policy, **detail})
                for scope, left_map, right_map in (
                    ("called", {key: response_maps[ARMS[0]][key] for key in response_maps[ARMS[0]] if key in {row["canonical_id"] for row in replay if row["split"] == split}}, {key: response_maps[ARMS[1]][key] for key in response_maps[ARMS[1]] if key in {row["canonical_id"] for row in replay if row["split"] == split}}),
                    ("full", response_maps[ARMS[0]], response_maps[ARMS[1]]),
                ):
                    result = paired_summary(left_map, right_map, score_map)
                    result.pop("details")
                    policy_arm_comparisons.append({"split": split, "scope": scope, "judge": judge, "policy": policy, "left": ARMS[0], "right": ARMS[1], **result})

    # Same-ID comparison tables requested for VALID and TEST.
    valid_existing = read_jsonl(PRIOR / "valid9_candidates_standardized.jsonl")
    valid_existing_by = {(row["canonical_id"], row["candidate_type"]): row for row in valid_existing}
    candidate_by = {(row["split"], row["canonical_id"], row["arm"]): row for row in candidates}
    valid_table = []
    valid_ids = sorted({row["canonical_id"] for row in candidates if row["split"] == "VALID"}, key=lambda cid: candidate_by[("VALID", cid, ARMS[0])]["source_order"])
    for canonical_id in valid_ids:
        source = candidate_by[("VALID", canonical_id, ARMS[0])]
        responses = {"SFT": source["sft_response"], "original_denoiser": source["candidate_response"]}
        for method in ("feedback_A", "feedback_B", "feedback_C"):
            responses[method] = valid_existing_by[(canonical_id, method)]["candidate_response"]
        responses["original_prompt_sft_control"] = candidate_by[("VALID", canonical_id, ARMS[1])]["candidate_response"]
        valid_table.append({"canonical_id": canonical_id, "source_order": source["source_order"], "question": source["question"], "responses": responses, "scores": {judge: {method: score_map[pair_hash(source["question"], response)] for method, response in responses.items()} for judge, score_map in score_maps.items()}})

    dpo_test = {row["canonical_id"]: row for row in read_jsonl(TEST_ROOT / "dpo_test_outputs.jsonl")}
    test_table = []
    test_ids = sorted({row["canonical_id"] for row in candidates if row["split"] == "TEST"}, key=lambda cid: candidate_by[("TEST", cid, ARMS[0])]["source_order"])
    for canonical_id in test_ids:
        source = candidate_by[("TEST", canonical_id, ARMS[0])]
        responses = {
            "SFT": source["sft_response"],
            "DPO": dpo_test[canonical_id]["sft_response"],
            "original_denoiser": source["candidate_response"],
            "original_prompt_sft_control": candidate_by[("TEST", canonical_id, ARMS[1])]["candidate_response"],
        }
        test_table.append({"canonical_id": canonical_id, "case_id": source["case_id"], "source_order": source["source_order"], "question": source["question"], "responses": responses, "scores": {judge: {method: score_map[pair_hash(source["question"], response)] for method, response in responses.items()} for judge, score_map in score_maps.items()}})

    same_id_method_summaries = []
    for scope, table in (("VALID9", valid_table), ("TEST8", test_table)):
        methods = list(table[0]["responses"])
        for judge in score_maps:
            for method in methods:
                scores = [row["scores"][judge][method] for row in table]
                same_id_method_summaries.append(
                    {
                        "scope": scope,
                        "judge": judge,
                        "method": method,
                        "N": len(scores),
                        "means": {axis: mean([item[axis] for item in scores]) for axis in AXES},
                        "four_axis_mean": mean([composite(item) for item in scores]),
                    }
                )

    generation_input_comparison = {
        "verified_from_code": str(WORKTREE / "scripts/selective_risk_refinement_utils.py"),
        "initial_sft_generation": {
            "renderer": "build_sft_prompt/build_sft_user_content",
            "inputs": ["question", "unsafe_response"],
            "adapter": str(CHECKPOINT_ROOT / "sft/final"),
            "max_source_len": 1024,
            "max_new_tokens": 1536,
            "decoding": {"temperature": 0, "repetition_penalty": 1.15, "no_repeat_ngram_size": 4},
        },
        "denoiser_revision_generation_both_arms": {
            "renderer": "build_risk_tune_prompt/build_risk_tune_user_content",
            "inputs": ["question", "unsafe_response", "sft_response", "z_t_from_sft", "g_sft", "risk_component_mode", "aspect_conditioning_status"],
            "zt_strategy": "no_mask",
            "timestep": 3,
            "T": 4,
            "max_source_len": 4096,
            "max_new_tokens": 1536,
            "decoding": {"temperature": 0, "repetition_penalty": 1.15, "no_repeat_ngram_size": 4},
            "original_adapter": str(CHECKPOINT_ROOT / "proposed_mask_off/final"),
            "control_adapter": str(CHECKPOINT_ROOT / "sft/final"),
            "additional_natural_language_feedback": False,
        },
        "confirmed_equal_between_revision_arms": ["base-model checkpoint", "tokenizer", "chat template", "special tokens", "rendered prompt", "input token IDs", "truncation", "decoding", "EOS contract", "software versions", "GPU model"],
        "only_intended_difference": "active generation adapter",
        "residual_run_level_difference": "The original candidates are frozen outputs from the historical run, while controls were generated in this run. Representative replays show that 4-bit greedy output is not bit-reproducible, so numerical run variation remains a documented confound.",
    }

    usage = api_usage()
    generation_issues = []
    for row in candidates:
        if not row["technical_valid"]:
            generation_issues.append({"split": row["split"], "canonical_id": row["canonical_id"], "arm": row["arm"], "issue": "technical_invalid"})
    known_issues = [
        {"scope": "TEST", "case": "TEST-15", "issue": "preserved collapsed-spacing SFT/DPO/Proposed generation failure"},
        {"scope": "TEST", "case": "TEST-24", "issue": "preserved original denoiser candidate and exact-matching control say U.S. crisis number 989 instead of 988; current_rules accepts it"},
        {"scope": "VALID", "case": "qa_24f1b77a40c9b22203b8f337 feedback_B", "issue": "preserved known specificity-loss acceptance"},
        {"scope": "VALID", "case": "qa_b243b4c887feb8e3594dd74c feedback_B", "issue": "preserved known SFT fallback"},
        {"scope": "VALID", "case": "qa_24f1b77a40c9b22203b8f337 control", "issue": "new control has collapsed spacing/malformed markup and is accepted by current_rules"},
        {"scope": "TEST", "case": "TEST-10 control", "issue": "new control substitutes incorrect/malformed crisis contacts for the treatment-discontinuation question and is accepted by current_rules"},
        {"scope": "TEST", "case": "TEST-22 control", "issue": "new control omits the high-place/trance safety context and is accepted by current_rules"},
    ]
    manual_issue_specs = {
        "qa_75e82899b8bd0d0c0137d720": [
            {"type": "unsupported_external_resource", "evidence": "please see our [Moving Guide](https://www.caregivingguide.com/moving-guide)", "note": "Introduces an unverified external guide and implied affiliation not grounded in the prompt."},
        ],
        "qa_9f10354e318c925cbee2f65c": [
            {"type": "relevance_loss", "evidence": "Instead of viewing these relationships solely as transactional", "note": "Reframes the user's constructive outreach plan as transactional/pushy without support from the question."},
        ],
        "qa_b243b4c887feb8e3594dd74c": [
            {"type": "generic_medical_redirection", "evidence": "ask your doctor about potential medications or other types of treatments", "note": "Moves quickly to tests/medication discussion while adding little question-specific support."},
        ],
        "qa_24f1b77a40c9b22203b8f337": [
            {"type": "collapsed_spacing_and_malformed_markup", "evidence": "mother'mathrm{Quality of Life}through improved advocacy", "note": "Large portions lose word boundaries and expose malformed markup."},
        ],
        "qa_7e6fd89045be08081e775de7": [
            {"type": "specificity_loss", "evidence": "struggle with chronic pain or illness", "note": "The concrete constipation concern is replaced by generic chronic pain/illness language."},
        ],
        "qa_27a352405e61695e38cbf95f": [
            {"type": "unsupported_prescriptive_timing", "evidence": "try texting tomorrow afternoon or evening", "note": "Gives a specific timing recommendation without evidence from the post."},
        ],
        "qa_c7c476cb621af0d845bbee09": [
            {"type": "malformed_text", "evidence": "I can'ten give you personal advice", "note": "Contains multiple visible language corruptions, including body'clock and it''s."},
        ],
        "qa_4e761249efc266a1874bfb11": [
            {"type": "question_mismatch", "evidence": "people available to listen and assist you", "note": "Substitutes crisis resources for the question about discontinuation of psychiatric treatment."},
            {"type": "incorrect_or_malformed_crisis_contact", "evidence": "1-800-273-TALK (1-866-886-4827)", "note": "The parenthetical number does not match the named legacy TALK number or current 988 service.", "verification_source": "https://988lifeline.org/faq/about-us/faq-what-is-988-and-is-it-active/"},
            {"type": "malformed_url", "evidence": "https://wwwofeline.org", "note": "Displayed link target is malformed."},
        ],
        "qa_9f1146f51f7e73e58bb51962": [
            {"type": "question_mismatch", "evidence": "the strongest partnerships are built on two whole and healthy individuals", "note": "Ignores the antibiotics/heart-condition decision and gives generic relationship advice."},
        ],
        "qa_4eccc9d0fe512fafe4304501": [
            {"type": "specificity_and_safety_omission", "evidence": "There are many resources available", "note": "Does not address the reported trance-like state in a high place or assess immediate danger."},
        ],
        "qa_7e61d2d90766eff47e00d31f": [
            {"type": "inappropriate_refusal", "evidence": "I cannot fulfill this request.", "note": "Refuses a benign request for Alzheimer's information and caregiving guidance."},
        ],
        "qa_03e93d92b6e415e7344811d7": [
            {"type": "factual_error", "evidence": "calling or texting 989", "note": "Uses 989 instead of the U.S. 988 Suicide & Crisis Lifeline number.", "verification_source": "https://988lifeline.org/contact-us/"},
        ],
    }
    manual_response_review = []
    for row in sorted((item for item in candidates if item["arm"] == "original_prompt_sft_control"), key=lambda item: (item["split"], item["source_order"])):
        issues = manual_issue_specs.get(row["canonical_id"], [])
        manual_response_review.append(
            {
                "split": row["split"],
                "canonical_id": row["canonical_id"],
                "case_id": row.get("case_id"),
                "reviewed_full_response": True,
                "status": "issue_found" if issues else "no_clear_defect_in_manual_read",
                "issues": issues,
                "response": row["candidate_response"],
            }
        )

    artifacts = {
        "results/control_candidate_pointwise_scores.jsonl": raw_scored,
        "results/control_candidate_quality_summary.jsonl": raw_summaries,
        "results/control_candidate_comparisons.jsonl": raw_comparisons,
        "results/control_pairwise_normalized.jsonl": pair_rows,
        "results/control_pairwise_case_consensus.jsonl": pair_cases,
        "results/control_pairwise_summary.jsonl": pair_summaries,
        "results/control_pairwise_judge_agreement.jsonl": direct_judge_pairwise,
        "results/control_selector_decisions.jsonl": [{"split": key[0], "canonical_id": key[1], "arm": key[2], **value} for key, value in sorted(selectors.items())],
        "results/control_selector_failures.jsonl": selector_failures,
        "results/policy_replay.jsonl": replay,
        "results/policy_selection_summary.jsonl": policy_selection_summary,
        "results/policy_quality_summary.jsonl": policy_quality_summary,
        "results/policy_sft_comparisons.jsonl": policy_sft_comparisons,
        "results/policy_arm_comparisons.jsonl": policy_arm_comparisons,
        "results/policy_per_item_scores.jsonl": policy_score_rows,
        "results/valid9_six_arm_comparison.jsonl": valid_table,
        "results/test8_four_arm_comparison.jsonl": test_table,
        "results/valid9_test8_same_id_method_summaries.jsonl": same_id_method_summaries,
    }
    for relative, rows in artifacts.items():
        write_jsonl(ROOT / relative, rows)
    write_json(ROOT / "results/generation_input_contract_comparison.json", generation_input_comparison)
    write_json(ROOT / "results/api_usage_and_cost.json", usage)
    write_json(ROOT / "results/known_issues.json", {"new_generation_issues": generation_issues, "preserved_known_issues": known_issues})
    write_jsonl(ROOT / "results/manual_control_response_review.jsonl", manual_response_review)

    def lookup(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
        matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one row for {criteria}, got {len(matches)}")
        return matches[0]

    headline = {
        "raw_original_vs_control": {
            judge: {
                split: lookup(raw_comparisons, judge=judge, split=split, left=ARMS[0], right=ARMS[1])
                for split in ("VALID", "TEST", "ALL")
            }
            for judge in score_maps
        },
        "raw_vs_sft": {
            judge: {
                split: {
                    arm: lookup(raw_comparisons, judge=judge, split=split, left="SFT", right=arm)
                    for arm in ARMS
                }
                for split in ("VALID", "TEST", "ALL")
            }
            for judge in score_maps
        },
        "current_rules_final_original_vs_control": {
            judge: {
                split: {
                    scope: lookup(policy_arm_comparisons, judge=judge, split=split, scope=scope, policy="current_rules")
                    for scope in ("called", "full")
                }
                for split in ("VALID", "TEST")
            }
            for judge in score_maps
        },
        "pairwise_two_order": pair_summaries,
        "pairwise_judge_consensus_agreement": {
            "N": len(direct_judge_pairwise),
            "same": sum(row["same_consensus"] for row in direct_judge_pairwise),
            "different": sum(not row["same_consensus"] for row in direct_judge_pairwise),
        },
    }

    # Select two transparent examples by largest absolute GPT pointwise arm delta.
    example_deltas = []
    for split in ("VALID", "TEST"):
        comparison = lookup(raw_comparisons, judge="GPT-4.1", split=split, left=ARMS[0], right=ARMS[1])
        maps = {
            arm: {row["canonical_id"]: row for row in candidates if row["split"] == split and row["arm"] == arm}
            for arm in ARMS
        }
        for canonical_id in maps[ARMS[0]]:
            left = maps[ARMS[0]][canonical_id]
            right = maps[ARMS[1]][canonical_id]
            left_scores = gpt_scores[pair_hash(left["question"], left["candidate_response"])]
            right_scores = gpt_scores[pair_hash(right["question"], right["candidate_response"])]
            delta = composite(right_scores) - composite(left_scores)
            example_deltas.append({"split": split, "canonical_id": canonical_id, "case_id": left.get("case_id"), "question": left["question"], "original_response": left["candidate_response"], "control_response": right["candidate_response"], "original_scores": left_scores, "control_scores": right_scores, "control_minus_original": round(delta, 4)})
    examples = []
    if example_deltas:
        examples.append(max(example_deltas, key=lambda row: row["control_minus_original"]))
        worst = min(example_deltas, key=lambda row: row["control_minus_original"])
        if worst["canonical_id"] != examples[0]["canonical_id"]:
            examples.append(worst)
    write_jsonl(ROOT / "results/representative_actual_cases.jsonl", examples)

    summary = {
        "version": VERSION,
        "status": "complete",
        "post_hoc_test_comparison": True,
        "generation": generation_manifest,
        "evaluation_counts": evaluation_contract["counts"],
        "api_usage": usage,
        "technical_invalid_new_candidates": len(generation_issues),
        "test_current_rules_reproduced_stored_surface_relaxed_v1": test_current_reproduced,
        "original_arm_all_four_policies_reproduced_prior_comparison": original_replay_matches_prior,
        "valid_mask_off_stored_policy_note": "VALID43 stored Mask_OFF_current is legacy strict; current_rules in this experiment is an explicitly labeled surface_relaxed_v1 replay.",
        "reproducibility_limitation": "Frozen historical original candidates and newly generated controls share the input/settings contract but not a bit-reproducible 4-bit inference run.",
        "headline": headline,
        "known_issues": known_issues,
        "manual_control_response_review": {
            "reviewed": len(manual_response_review),
            "issue_found": sum(row["status"] == "issue_found" for row in manual_response_review),
            "no_clear_defect_in_manual_read": sum(row["status"] != "issue_found" for row in manual_response_review),
        },
        "representative_cases": examples,
    }
    write_json(ROOT / "SUMMARY.json", summary)

    agreement = headline["pairwise_judge_consensus_agreement"]
    lines = [
        "# Original-prompt SFT adapter control",
        "",
        "## Status and scope",
        "",
        "Complete: 17/17 new candidates (VALID9 + TEST8), with no retraining and no replacement of stored responses. TEST is a post-hoc exploratory comparison.",
        "",
        "The original denoiser and control used the same frozen question, unsafe response, SFT draft, no-mask z_t, g/router/scorer outputs, timestep, rendered prompt, input token IDs, tokenizer, base-model checkpoint, decoding, and EOS contract. The intended generation change was only the active adapter: `proposed_mask_off/final` versus `sft/final`. No feedback_v1 text was supplied.",
        "",
        "Representative replay matched input length, EOS, and generation settings for both splits. Stored output was not bit-exact: repeated 4-bit greedy inference varied, including within one loaded model for TEST. The compared original candidates remain the frozen historical outputs, whereas controls were generated in this run, so residual run-level numerical variation is a limitation even though the input contract is identical.",
        "",
        "## Direct answers",
        "",
    ]
    for split in ("VALID", "TEST", "ALL"):
        gpt_delta = headline["raw_original_vs_control"]["GPT-4.1"][split]["mean_four_axis_delta_right_minus_left"]
        llama_delta = headline["raw_original_vs_control"]["Llama"][split]["mean_four_axis_delta_right_minus_left"]
        gpt_winner = "original denoiser" if gpt_delta < 0 else ("SFT-adapter control" if gpt_delta > 0 else "tie")
        llama_winner = "original denoiser" if llama_delta < 0 else ("SFT-adapter control" if llama_delta > 0 else "tie")
        lines.append(f"- Same-input candidate quality, {split}: GPT-4.1 favors {gpt_winner} (control-original {gpt_delta:+.4f}); Llama favors {llama_winner} ({llama_delta:+.4f}).")
    for split in ("VALID", "TEST"):
        gpt_final = headline["current_rules_final_original_vs_control"]["GPT-4.1"][split]["called"]
        llama_final = headline["current_rules_final_original_vs_control"]["Llama"][split]["called"]
        lines.append(f"- After current_rules, {split} called set: control-final minus original-final is {gpt_final['mean_four_axis_delta_right_minus_left']:+.4f} by GPT-4.1 and {llama_final['mean_four_axis_delta_right_minus_left']:+.4f} by Llama.")
    lines += [
        f"- Judge conclusion agreement is not assumed: exact two-order pairwise case consensus matches on {agreement['same']}/{agreement['N']} cases and differs on {agreement['different']}.",
        "- Improvement/tie/worsening counts versus SFT are judge-dependent and listed below for every split and arm; risk reduction or acceptance is not treated as quality improvement.",
        "",
        "## Candidate quality: original denoiser vs SFT-adapter control",
        "",
        "| Judge | Split | N | Control - original | Control improved / tied / worsened |",
        "|---|---:|---:|---:|---:|",
    ]
    for judge in ("GPT-4.1", "Llama"):
        for split in ("VALID", "TEST", "ALL"):
            row = headline["raw_original_vs_control"][judge][split]
            counts = row["right_improved_tie_worsened"]
            lines.append(f"| {judge} | {split} | {row['N']} | {row['mean_four_axis_delta_right_minus_left']:+.4f} | {counts['improved']} / {counts['tie']} / {counts['worsened']} |")
    lines += [
        "",
        "## Same-ID method comparison",
        "",
        "The tables below use only the frozen called IDs. Per-question questions, responses, and scores are in `results/valid9_six_arm_comparison.jsonl` and `results/test8_four_arm_comparison.jsonl`.",
    ]
    for scope in ("VALID9", "TEST8"):
        lines += [
            "",
            f"### {scope}",
            "",
            "| Judge | Method | N | Overall | Empathy | Specificity | Medical Advice | Four-axis mean |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in same_id_method_summaries:
            if row["scope"] != scope:
                continue
            means = row["means"]
            lines.append(
                f"| {row['judge']} | {row['method']} | {row['N']} | {means['overall_quality']:.4f} | "
                f"{means['empathy']:.4f} | {means['specificity']:.4f} | "
                f"{means['medical_advice']:.4f} | {row['four_axis_mean']:.4f} |"
            )
    lines += ["", "Two-order blind pairwise consensus:", ""]
    for row in pair_summaries:
        if row["split"] != "ALL":
            lines.append(f"- {row['judge']} {row['split']} (N={row['N']}): {json.dumps(row['consensus_counts'], ensure_ascii=False, sort_keys=True)}")
    lines.append(f"- GPT/Llama exact case-level pairwise consensus agreement: {agreement['same']}/{agreement['N']} (different {agreement['different']}).")
    lines += ["", "## SFT draft comparison", "", "Raw candidate four-axis changes versus the same SFT draft:", ""]
    for judge in ("GPT-4.1", "Llama"):
        for split in ("VALID", "TEST"):
            for arm in ARMS:
                row = headline["raw_vs_sft"][judge][split][arm]
                counts = row["right_improved_tie_worsened"]
                lines.append(f"- {judge} {split} {arm}: {counts['improved']} improved / {counts['tie']} tied / {counts['worsened']} worsened (N={row['N']}, mean delta {row['mean_four_axis_delta_right_minus_left']:+.4f}).")
    lines += ["", "## After current_rules selection", "", "`current_rules` here means the frozen `surface_relaxed_v1` rule. VALID's stored `Mask_OFF_current` remains labeled legacy strict and is not renamed.", "", "| Judge | Split/scope | N | Control-final - original-final | Control improved / tied / worsened |", "|---|---:|---:|---:|---:|"]
    for judge in ("GPT-4.1", "Llama"):
        for split in ("VALID", "TEST"):
            for scope in ("called", "full"):
                row = headline["current_rules_final_original_vs_control"][judge][split][scope]
                counts = row["right_improved_tie_worsened"]
                lines.append(f"| {judge} | {split} {scope} | {row['N']} | {row['mean_four_axis_delta_right_minus_left']:+.4f} | {counts['improved']} / {counts['tie']} / {counts['worsened']} |")
    lines += [
        "",
        "## Frozen-policy selection counts",
        "",
        "| Split | Candidate arm | Policy | Called | Accepted | SFT retained |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in policy_selection_summary:
        lines.append(
            f"| {row['split']} | {row['arm']} | {row['policy']} | {row['called']} | "
            f"{row['accepted']} | {row['sft_retained_called']} |"
        )
    lines += [
        "",
        "Called-set and full VALID43/TEST24 four-axis quality for every policy are in `results/policy_quality_summary.jsonl`.",
        "",
        "## Actual cases",
        "",
    ]
    for row in examples:
        label = row.get("case_id") or row["canonical_id"]
        lines += [
            f"### {row['split']} {label} ({row['canonical_id']})",
            "",
            f"GPT-4.1 control-minus-original composite delta: {row['control_minus_original']:+.2f}",
            "",
            f"Question: {row['question']}",
            "",
            f"Original denoiser: {row['original_response']}",
            "",
            f"SFT-adapter control: {row['control_response']}",
            "",
        ]
    lines += [
        "## Interpretation",
        "",
        "The answer to whether denoiser training improves revision quality must follow the measured GPT and Llama deltas above. Selection effects are reported separately from raw candidate quality; acceptance rate or risk reduction alone is not treated as quality improvement.",
        "",
        "GPT-4.1 served as both selector and one evaluator through separate prompts/calls, so selection-aligned bias remains possible. Llama is retained as an independent auxiliary judge.",
        "",
        "## Manual review of all 17 new responses",
        "",
        f"All 17 control responses were read in full. Clear defects were recorded for {sum(row['status'] == 'issue_found' for row in manual_response_review)} responses; the remaining {sum(row['status'] != 'issue_found' for row in manual_response_review)} had no clear defect in this manual pass. This is separate from judge scores.",
        "",
    ]
    for row in manual_response_review:
        for issue in row["issues"]:
            lines.append(f"- {row['split']} {row.get('case_id') or row['canonical_id']} `{row['canonical_id']}` [{issue['type']}]: \"{issue['evidence']}\" - {issue['note']}")
    lines += [
        "",
        "## Known issues retained",
        "",
    ]
    lines.extend(f"- {item['case']}: {item['issue']}" for item in known_issues)
    lines += [
        "",
        "## API usage",
        "",
        f"New paid GPT-4.1 calls: {usage['new_paid_calls']}; input tokens: {usage['input_tokens']}; output tokens: {usage['output_tokens']}; estimated cost: ${usage['estimated_cost_usd']:.6f}.",
        "",
        "All prompts, rendered generation inputs, hashes, raw decoded outputs, raw judge JSON, per-item scores, selectors, policy replays, manifests, and logs are included in the bundle.",
    ]
    report_path = ROOT / "ORIGINAL_PROMPT_SFT_CONTROL_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    code_dir = ROOT / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "prepare_original_prompt_sft_control.py",
        "run_original_prompt_sft_control_generation.py",
        "prepare_original_prompt_sft_control_evaluation.py",
        "run_llama_original_prompt_sft_control.py",
        "run_original_prompt_sft_control_evaluations.sh",
        "aggregate_original_prompt_sft_control.py",
        "selective_risk_refinement_utils.py",
        "run_gemma_selective_risk_refinement.py",
    ):
        shutil.copy2(WORKTREE / "scripts" / name, code_dir / name)

    source_contract_dir = ROOT / "source_contracts"
    source_contract_dir.mkdir(parents=True, exist_ok=True)
    source_contracts = {
        "valid_sft_manifest.json": VALID_ROOT / "sft_valid_outputs.manifest.json",
        "valid_proposed_manifest.json": VALID_ROOT / "proposed_mask_off_valid_outputs.manifest.json",
        "test24_sft_manifest.json": TEST_ROOT / "sft_test_outputs.manifest.json",
        "test24_proposed_manifest.json": TEST_ROOT / "proposed_mask_off_test_outputs.manifest.json",
        "test15_generation_manifest.json": TEST_ROOT.parent / "part15/proposed_missing_outputs.manifest.json",
        "test24_generation_manifest.json": TEST_ROOT / "proposed_missing_outputs.manifest.json",
        "pointwise_evaluation_contract.json": LOCAL_TEST / "evaluation_contract.json",
        "pairwise_evaluation_contract.json": FEEDBACK_ROOT / "experiment_contract.json",
    }
    for destination, source in source_contracts.items():
        shutil.copy2(source, source_contract_dir / destination)

    final_manifest_path = ROOT / "FINAL_MANIFEST.json"
    delivery_record_path = ROOT / "DELIVERY.json"
    zip_path = ROOT / "original_prompt_sft_control_research_bundle.zip"
    for stale_packaging_file in (final_manifest_path, delivery_record_path, zip_path):
        stale_packaging_file.unlink(missing_ok=True)

    manifest_entries = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file():
            manifest_entries.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    final_manifest = {"version": VERSION, "status": "complete", "files": manifest_entries, "file_count": len(manifest_entries)}
    write_json(final_manifest_path, final_manifest)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(ROOT.rglob("*")):
            if path.is_file() and path != zip_path:
                archive.write(path, path.relative_to(ROOT))
    delivery_dir = Path("/home/user/hsoh/original_prompt_sft_control_20261009_v1")
    delivery_dir.mkdir(parents=True, exist_ok=True)
    delivery_zip = delivery_dir / zip_path.name
    shutil.copy2(zip_path, delivery_zip)
    shutil.copy2(report_path, delivery_dir / report_path.name)
    shutil.copy2(ROOT / "SUMMARY.json", delivery_dir / "SUMMARY.json")
    write_json(
        delivery_record_path,
        {
            "zip": str(zip_path),
            "zip_sha256": sha256_file(zip_path),
            "delivery_copy": str(delivery_zip),
            "delivery_copy_sha256": sha256_file(delivery_zip),
        },
    )
    print(json.dumps({"status": "complete", "zip": str(zip_path), "delivery_copy": str(delivery_zip), "sha256": sha256_file(zip_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
