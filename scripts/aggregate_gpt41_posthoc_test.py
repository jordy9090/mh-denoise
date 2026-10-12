#!/usr/bin/env python3
"""Aggregate frozen TEST24 GPT-4.1 evaluation and selector-policy replay."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "gpt41-posthoc-test24-selector-v1-20261009"
ROOT = Path("/mnt/ssd00/hsoh/gpt41_posthoc_selection_20261009_v1")
SOURCE = Path(
    "/home/user/hsoh/mh-denoise/outputs/fullpaper_acl/"
    "arr_revision2_gemma_first_seed20260910/gemma/"
    "generation_1536_test_surface_relaxed_v1_20261008/part24"
)
LLAMA_ROOT = Path("/mnt/ssd00/hsoh/test24_local_four_axis_20261009")
AXES = ("overall_quality", "empathy", "specificity", "medical_advice")
METHODS = ("SFT", "DPO", "A_current_rules", "B_risk_only", "C_gpt_selector")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_hash(question: str, response: str) -> str:
    return hashlib.sha256((question + "\0" + response).encode("utf-8")).hexdigest()


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def technical_valid(row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    if not str(row.get("sft_response") or "").strip():
        reasons.append("empty_sft_response")
    if not str(row.get("denoiser_response") or "").strip():
        reasons.append("empty_candidate_response")
    return not reasons, reasons


def policy_reasons(row: dict[str, Any], policy: str) -> list[str]:
    valid, reasons = technical_valid(row)
    if not valid:
        return reasons
    sft_risk = row.get("sft_risk_score")
    candidate_risk = row.get("denoiser_risk_score")
    if policy == "accept_generated":
        return []
    if policy == "risk_only":
        if not finite(sft_risk) or not finite(candidate_risk):
            return ["invalid_risk"]
        return [] if float(sft_risk) - float(candidate_risk) >= 0.01 else ["risk_not_improved"]
    if policy not in {"current_rules", "no_min_length", "legacy_strict"}:
        raise ValueError(policy)
    reasons = []
    if policy != "no_min_length" and int(row.get("denoiser_word_count") or 0) < 20:
        reasons.append("short_denoiser_response")
    if not finite(sft_risk) or not finite(candidate_risk):
        reasons.append("invalid_risk")
    elif float(candidate_risk) > float(sft_risk) - 0.01:
        reasons.append("risk_not_improved")
    if int(row.get("denoiser_bad_safety_count") or 0) > int(row.get("sft_bad_safety_count") or 0):
        reasons.append("bad_safety_increased")
    if policy == "legacy_strict":
        if float(row.get("specificity_ratio") or 0.0) < 0.6:
            reasons.append("specificity_ratio_low")
        if int(row.get("denoiser_generic_count") or 0) > int(row.get("sft_generic_count") or 0):
            reasons.append("genericity_increased")
    minimum_overlap = max(0.0, float(row.get("sft_keyword_overlap") or 0.0) - 0.15)
    if float(row.get("denoiser_keyword_overlap") or 0.0) < minimum_overlap:
        reasons.append("question_keyword_overlap_low")
    return reasons


def latest_ok(path: Path) -> list[dict[str, Any]]:
    latest = {}
    for row in read_jsonl(path):
        latest[row["request_id"]] = row
    return [row for row in latest.values() if row.get("ok") is True]


def mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4)


def composite(scores: dict[str, int]) -> float:
    return statistics.mean(float(scores[axis]) for axis in AXES)


def classify(delta: float) -> str:
    return "win" if delta > 0 else ("loss" if delta < 0 else "tie")


def main() -> None:
    pointwise_rows = latest_ok(ROOT / "results/test_pointwise_gpt41.jsonl")
    candidate_pointwise_rows = latest_ok(ROOT / "results/test_candidate_pointwise_gpt41.jsonl")
    selector_rows = latest_ok(ROOT / "results/test_selector_gpt41.jsonl")
    if (len(pointwise_rows), len(candidate_pointwise_rows), len(selector_rows)) != (47, 4, 16):
        raise RuntimeError("TEST GPT queues are incomplete")
    score_map = {}
    for row in pointwise_rows + candidate_pointwise_rows:
        key = row["metadata"]["question_response_sha256"]
        if key in score_map and score_map[key] != row["normalized"]["scores"]:
            raise RuntimeError("Conflicting GPT score cache")
        score_map[key] = row["normalized"]["scores"]
    if len(score_map) != 51:
        raise RuntimeError(f"Expected 51 TEST question-response scores, got {len(score_map)}")

    selector_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in selector_rows:
        selector_by_id.setdefault(row["metadata"]["canonical_id"], []).append(row)
    selector_decisions = {}
    for canonical_id, rows in selector_by_id.items():
        if len(rows) != 2:
            raise RuntimeError(f"Selector order count changed for {canonical_id}")
        mapped = []
        for row in sorted(rows, key=lambda item: item["metadata"]["order_index"]):
            choice = row["normalized"]["choice"]
            if choice == "tie":
                mapped.append("tie")
            else:
                mapped.append(row["metadata"][f"variant_for_{choice}"])
        adopt = mapped == ["candidate", "candidate"]
        selector_decisions[canonical_id] = {
            "mapped_choices": mapped,
            "accepted": adopt,
            "reason": "candidate_selected_both_orders" if adopt else (
                "selector_tie" if "tie" in mapped else "selector_kept_sft_or_disagreed"
            ),
            "requests": rows,
        }

    sft_rows = read_jsonl(SOURCE / "sft_test_outputs.jsonl")
    dpo_rows = read_jsonl(SOURCE / "dpo_test_outputs.jsonl")
    proposed_rows = read_jsonl(SOURCE / "proposed_mask_off_test_outputs.jsonl")
    ids = [row["canonical_id"] for row in sft_rows]
    sft_by = {row["canonical_id"]: row for row in sft_rows}
    dpo_by = {row["canonical_id"]: row for row in dpo_rows}
    proposed_by = {row["canonical_id"]: row for row in proposed_rows}
    if set(ids) != set(dpo_by) or set(ids) != set(proposed_by):
        raise RuntimeError("TEST method membership mismatch")

    policies = (
        "current_rules",
        "no_min_length",
        "risk_only",
        "gpt_selector",
        "legacy_strict",
        "accept_generated",
    )
    replay_rows = []
    method_responses: dict[str, dict[str, str]] = {method: {} for method in METHODS}
    for index, canonical_id in enumerate(ids):
        sft = sft_by[canonical_id]["sft_response"]
        dpo = dpo_by[canonical_id]["sft_response"]
        proposed = proposed_by[canonical_id]
        method_responses["SFT"][canonical_id] = sft
        method_responses["DPO"][canonical_id] = dpo
        method_responses["A_current_rules"][canonical_id] = proposed["final_response"]
        for policy in policies:
            called = proposed.get("used_denoiser") is True
            candidate = proposed.get("denoiser_response")
            if not called:
                accepted = False
                reasons = ["gate_not_called"]
            elif policy == "gpt_selector":
                decision = selector_decisions.get(canonical_id)
                if decision is None:
                    accepted = False
                    reasons = ["selector_failure"]
                else:
                    accepted = decision["accepted"]
                    reasons = [] if accepted else [decision["reason"]]
            else:
                reasons = policy_reasons(proposed, policy)
                accepted = not reasons
            final_response = candidate if accepted else sft
            replay_rows.append(
                {
                    "canonical_id": canonical_id,
                    "case_id": f"TEST-{index + 1:02d}",
                    "policy": policy,
                    "gate_called": called,
                    "candidate_available": bool(str(candidate or "").strip()),
                    "accepted": accepted,
                    "reasons": reasons,
                    "final_source": "candidate" if accepted else "sft",
                    "final_response": final_response,
                    "stored_current_accepted": proposed.get("accepted_denoiser"),
                    "sft_risk_score": proposed.get("sft_risk_score"),
                    "candidate_risk_score": proposed.get("denoiser_risk_score"),
                    "risk_delta": (
                        float(proposed["sft_risk_score"]) - float(proposed["denoiser_risk_score"])
                        if called and finite(proposed.get("sft_risk_score")) and finite(proposed.get("denoiser_risk_score"))
                        else None
                    ),
                }
            )
            if policy == "risk_only":
                method_responses["B_risk_only"][canonical_id] = final_response
            elif policy == "gpt_selector":
                method_responses["C_gpt_selector"][canonical_id] = final_response

    current_replay = [row for row in replay_rows if row["policy"] == "current_rules"]
    if any(
        row["accepted"] != bool(proposed_by[row["canonical_id"]].get("accepted_denoiser"))
        or row["final_response"] != proposed_by[row["canonical_id"]]["final_response"]
        for row in current_replay
    ):
        raise RuntimeError("current_rules replay does not reproduce stored TEST output")

    scored_methods = {}
    per_question = []
    for method in METHODS:
        rows = []
        for index, canonical_id in enumerate(ids):
            question = sft_by[canonical_id]["question"]
            response = method_responses[method][canonical_id]
            scores = score_map.get(pair_hash(question, response))
            if scores is None:
                raise RuntimeError(f"Missing GPT pointwise score for {method} {canonical_id}")
            rows.append({"canonical_id": canonical_id, "case_id": f"TEST-{index + 1:02d}", "response": response, "scores": scores})
        scored_methods[method] = rows
    sft_scored = {row["canonical_id"]: row for row in scored_methods["SFT"]}
    method_summary = {}
    for method, rows in scored_methods.items():
        deltas = []
        counts = Counter()
        for row in rows:
            base = sft_scored[row["canonical_id"]]
            delta = round(composite(row["scores"]) - composite(base["scores"]), 4)
            outcome = classify(delta)
            counts[outcome] += 1
            deltas.append(delta)
            per_question.append(
                {
                    "method": method,
                    "canonical_id": row["canonical_id"],
                    "case_id": row["case_id"],
                    "scores": row["scores"],
                    "sft_scores": base["scores"],
                    "four_axis_delta_vs_sft": delta,
                    "outcome_vs_sft": outcome,
                    "response": row["response"],
                }
            )
        method_summary[method] = {
            "completed": len(rows),
            "means": {axis: mean([row["scores"][axis] for row in rows]) for axis in AXES},
            "four_axis_mean": mean([composite(row["scores"]) for row in rows]),
            "mean_delta_vs_sft": mean(deltas),
            "win_tie_loss_vs_sft": {key: counts[key] for key in ("win", "tie", "loss")},
        }

    pairwise_table = []
    for left, right in itertools.combinations(METHODS, 2):
        left_by = {row["canonical_id"]: row for row in scored_methods[left]}
        right_by = {row["canonical_id"]: row for row in scored_methods[right]}
        deltas = []
        outcomes = Counter()
        axis_deltas = {axis: [] for axis in AXES}
        for canonical_id in ids:
            lscore, rscore = left_by[canonical_id]["scores"], right_by[canonical_id]["scores"]
            delta = composite(rscore) - composite(lscore)
            outcomes[classify(delta)] += 1
            deltas.append(delta)
            for axis in AXES:
                axis_deltas[axis].append(rscore[axis] - lscore[axis])
        pairwise_table.append(
            {
                "left": left,
                "right": right,
                "delta_definition": "right_minus_left",
                "N": 24,
                "mean_four_axis_delta": mean(deltas),
                "mean_axis_deltas": {axis: mean(values) for axis, values in axis_deltas.items()},
                "right_win_tie_loss": {key: outcomes[key] for key in ("win", "tie", "loss")},
            }
        )

    policy_summary = {}
    for policy in policies:
        rows = [row for row in replay_rows if row["policy"] == policy]
        policy_summary[policy] = {
            "gate_called": sum(row["gate_called"] for row in rows),
            "accepted": sum(row["accepted"] for row in rows),
            "sft_retained": sum(not row["accepted"] for row in rows),
            "selector_or_technical_failures": sum("selector_failure" in row["reasons"] for row in rows),
            "reasons": dict(Counter(reason for row in rows for reason in row["reasons"])),
            "mean_risk_delta_called": mean([row["risk_delta"] for row in rows if row["risk_delta"] is not None]),
        }

    llama_rows = read_jsonl(LLAMA_ROOT / "judged_representatives.jsonl")
    llama_by = {row["question_response_sha256"]: row for row in llama_rows if row.get("judge_ok")}
    judge_comparison = []
    for row in pointwise_rows:
        key = row["metadata"]["question_response_sha256"]
        llama = llama_by[key]
        gpt = row["normalized"]["scores"]
        judge_comparison.append(
            {
                "question_response_sha256": key,
                "canonical_id": row["metadata"]["canonical_id"],
                "case_id": row["metadata"]["case_id"],
                "member_methods": row["metadata"]["member_methods"],
                "llama_scores": llama["judge_scores"],
                "gpt41_scores": gpt,
                "gpt41_minus_llama": {axis: gpt[axis] - llama["judge_scores"][axis] for axis in AXES},
                "all_four_scores_equal": gpt == llama["judge_scores"],
            }
        )

    called_cases = []
    for index, canonical_id in enumerate(ids):
        row = proposed_by[canonical_id]
        if not row.get("used_denoiser"):
            continue
        called_cases.append(
            {
                "canonical_id": canonical_id,
                "case_id": f"TEST-{index + 1:02d}",
                "question": row["question"],
                "sft_response": row["sft_response"],
                "candidate_response": row["denoiser_response"],
                "stored_current_final_response": row["final_response"],
                "stored_current_accepted": row["accepted_denoiser"],
                "selector": selector_decisions[canonical_id],
                "policy_decisions": {
                    policy: next(
                        item for item in replay_rows
                        if item["canonical_id"] == canonical_id and item["policy"] == policy
                    )
                    for policy in policies
                },
                "sft_gpt41_scores": score_map[pair_hash(row["question"], row["sft_response"])],
                "candidate_gpt41_scores": score_map[pair_hash(row["question"], row["denoiser_response"])],
            }
        )

    usage_manifests = [
        json.loads((ROOT / f"results/{name}_gpt41.manifest.json").read_text(encoding="utf-8"))
        for name in ("test_pointwise", "test_selector", "test_candidate_pointwise")
    ]
    usage = {
        "api_calls": sum(item["completed"] for item in usage_manifests),
        "input_tokens": sum(item["input_tokens"] for item in usage_manifests),
        "output_tokens": sum(item["output_tokens"] for item in usage_manifests),
        "estimated_cost_usd": round(sum(item["estimated_cost_usd"] for item in usage_manifests), 8),
        "model_ids": sorted({model for item in usage_manifests for model in item["returned_models"]}),
        "system_fingerprints": sorted({fp for item in usage_manifests for fp in item["system_fingerprints"]}),
        "pricing_assumption_usd_per_million": {"input": 2.0, "output": 8.0},
    }
    summary = {
        "version": VERSION,
        "status": "complete_test24_gpt41_interim",
        "post_hoc_exploratory": True,
        "method_results": method_summary,
        "policy_results": policy_summary,
        "current_rules_reproduced_stored_output": True,
        "pointwise_cache": {
            "original_unique_pairs": 47,
            "new_called_candidate_unique_pairs": 4,
            "total_unique_pairs": 51,
            "shared_method_assignments_original": 25,
        },
        "gpt_selector": {
            "called_cases": 8,
            "two_order_requests": 16,
            "accepted": sum(item["accepted"] for item in selector_decisions.values()),
            "failures": 0,
        },
        "gpt41_vs_llama_on_same_test47": {
            "N": len(judge_comparison),
            "all_four_scores_equal": sum(row["all_four_scores_equal"] for row in judge_comparison),
            "mean_axis_delta_gpt41_minus_llama": {
                axis: mean([row["gpt41_minus_llama"][axis] for row in judge_comparison])
                for axis in AXES
            },
        },
        "api_usage": usage,
        "known_issues_preserved": [
            "TEST-15 stored collapsed-spacing generation failure",
            "TEST-24 Proposed candidate says U.S. crisis number 989 instead of 988",
            "TEST-01 DPO uses the previously recovered stored final text, not the old **Final export fragment",
        ],
    }
    write_json(ROOT / "TEST24_GPT41_SUMMARY.json", summary)
    write_jsonl(ROOT / "test_policy_replay.jsonl", replay_rows)
    write_jsonl(ROOT / "test_method_pointwise_gpt41.jsonl", per_question)
    write_jsonl(ROOT / "test_all_method_pairwise_gpt41.jsonl", pairwise_table)
    write_jsonl(ROOT / "test_gpt41_vs_llama_same47.jsonl", judge_comparison)
    write_jsonl(ROOT / "test_called_cases_full.jsonl", called_cases)

    lines = [
        "# TEST24 GPT-4.1 post-hoc exploratory comparison",
        "",
        "Stored responses and candidates only. No response regeneration, retraining, gate change, or prompt iteration was performed.",
        "",
        "## GPT-4.1 pointwise results",
        "",
        "| Method | N | Overall | Empathy | Specificity | Medical Advice | Four-axis mean | W/T/L vs SFT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = method_summary[method]
        scores = item["means"]
        wtl = item["win_tie_loss_vs_sft"]
        lines.append(
            f"| {method} | {item['completed']} | {scores['overall_quality']:.4f} | {scores['empathy']:.4f} | "
            f"{scores['specificity']:.4f} | {scores['medical_advice']:.4f} | {item['four_axis_mean']:.4f} | "
            f"{wtl['win']}/{wtl['tie']}/{wtl['loss']} |"
        )
    lines += ["", "A=current surface_relaxed_v1, B=risk-only, C=two-order unanimous GPT-4.1 selector.", "", "## Selector policy replay", "", "| Policy | Called | Accepted | SFT retained | Failures |", "|---|---:|---:|---:|---:|"]
    for policy in policies:
        item = policy_summary[policy]
        lines.append(f"| {policy} | {item['gate_called']} | {item['accepted']} | {item['sft_retained']} | {item['selector_or_technical_failures']} |")
    lines += [
        "",
        "## Direct judge comparison",
        "",
        f"The same 47 frozen question-response pairs were scored by both judges. Exact four-score matches: {summary['gpt41_vs_llama_on_same_test47']['all_four_scores_equal']}/47.",
        "",
        "## Preserved known issues",
        "",
        "- TEST-15 collapsed-spacing generation failure remains present.",
        "- TEST-24 candidate contains the incorrect U.S. crisis number `989`; the correct number is `988`.",
        "- TEST-01 DPO evaluates the stored recovered response rather than the historical broken export fragment.",
        "",
        "## Called cases and actual responses",
        "",
    ]
    for row in called_cases:
        decisions = ", ".join(
            f"{policy}={'candidate' if item['accepted'] else 'SFT'}"
            for policy, item in row["policy_decisions"].items()
        )
        lines += [
            f"### {row['case_id']} / {row['canonical_id']}",
            "",
            f"Decisions: {decisions}",
            "",
            "**Question**",
            "",
            row["question"],
            "",
            "**SFT**",
            "",
            row["sft_response"],
            "",
            "**Stored denoiser candidate**",
            "",
            row["candidate_response"],
            "",
        ]
    report = ROOT / "TEST24_GPT41_INTERIM_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    bundle = ROOT / "TEST24_GPT41_posthoc_interim_research_bundle.zip"
    members = [
        "TEST24_GPT41_INTERIM_REPORT.md",
        "TEST24_GPT41_SUMMARY.json",
        "test_request_contract.json",
        "test_candidate_pointwise_contract.json",
        "test_called_candidates.jsonl",
        "test_policy_replay.jsonl",
        "test_method_pointwise_gpt41.jsonl",
        "test_all_method_pairwise_gpt41.jsonl",
        "test_gpt41_vs_llama_same47.jsonl",
        "test_called_cases_full.jsonl",
        "requests/test_pointwise.jsonl",
        "requests/test_candidate_pointwise.jsonl",
        "requests/test_selector.jsonl",
        "results/test_pointwise_gpt41.jsonl",
        "results/test_candidate_pointwise_gpt41.jsonl",
        "results/test_selector_gpt41.jsonl",
    ]
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in members:
            archive.write(ROOT / name, arcname=name)
    write_json(
        ROOT / "TEST24_GPT41_INTERIM_MANIFEST.json",
        {
            "status": "complete",
            "post_hoc_exploratory": True,
            "report": str(report),
            "bundle": str(bundle),
            "bundle_sha256": sha256_file(bundle),
            "members": members,
            "api_usage": usage,
        },
    )
    print(json.dumps({"status": "complete", "bundle": str(bundle), "methods": method_summary, "policies": policy_summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
