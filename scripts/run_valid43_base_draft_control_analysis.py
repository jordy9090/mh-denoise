#!/usr/bin/env python3
"""Evaluate VALID43 Base-draft second-pass controls under frozen contracts."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_base_first_residual_valid_analysis import (
    MODEL,
    REPO,
    ROOT,
    WORKTREE,
    cached_results,
    cached_selector_results,
    pair_hash,
    quality3,
    read_json,
    read_jsonl,
    request_key,
    score_valid,
    sha256_file,
    sha256_text,
    unique_pairs,
    write_csv,
    write_json,
    write_jsonl,
)


VERSION = "base-first-residual-valid-v1-20261011"
FROZEN = Path("/home/user/hsoh/rubric_aligned_gpt41_rejudge_20261009_v1")
CONTROL_OUTPUT = ROOT / "generation/base_draft_second_pass_controls_valid43.jsonl"
BASE_ASSIGNMENTS = ROOT / "inputs/pointwise_assignments.jsonl"
POINTWISE_QUEUE = "base_draft_controls_valid_pointwise"
PAIRWISE_QUEUE = "base_draft_controls_valid_pairwise"
ROUTER = REPO / "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/router/final"
SCORER = REPO / "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/scorer/final"
METHOD_BY_ARM = {
    "base_self_refine": "BaseSelfRefineCandidate",
    "trained_denoiser_on_base": "TrainedDenoiserOnBaseCandidate",
}
LEGACY_SFT_GATE_THRESHOLD = 0.7215308056776166


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pointwise_request(pair: dict[str, Any], system_prompt: str, user_template: str) -> dict[str, Any]:
    user_prompt = user_template.format(question=pair["question"], response=pair["response"])
    metadata = {
        "question_response_sha256": pair["question_response_sha256"],
        "question_sha256": pair["question_sha256"],
        "response_sha256": pair["response_sha256"],
        "members": pair["members"],
        "priority": "valid_base_draft_control",
    }
    material = {
        "version": VERSION,
        "queue": POINTWISE_QUEUE,
        "model": MODEL,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "metadata": metadata,
    }
    return {
        "version": VERSION,
        "queue": POINTWISE_QUEUE,
        "request_id": sha256_text(json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")))[:32],
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 384,
        "response_format": {"type": "json_object"},
        "parser": "pointwise_expert_rubric",
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_sha256": sha256_text(user_prompt),
        "metadata": metadata,
    }


def scan_current_gpt_cache(parser: str, max_tokens: int) -> dict[str, tuple[Path, dict[str, Any]]]:
    chosen: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted((ROOT / "results").glob("*.jsonl")):
        for row in read_jsonl(path):
            if row.get("parser") != parser or row.get("ok") is not True:
                continue
            if row.get("model") != MODEL or row.get("returned_model") != MODEL:
                continue
            if row.get("temperature") != 0 or row.get("max_tokens") != max_tokens:
                continue
            if row.get("response_format") != {"type": "json_object"}:
                continue
            if parser == "pointwise_expert_rubric" and not score_valid(row):
                continue
            if parser == "blind_selector" and (row.get("normalized") or {}).get("choice") not in {"A", "B", "tie"}:
                continue
            chosen.setdefault(request_key(row), (path, row))
    return chosen


def prepare() -> None:
    control_rows = read_jsonl(CONTROL_OUTPUT)
    if len(control_rows) != 86 or sum(bool(row.get("technical_valid")) for row in control_rows) != 86:
        raise RuntimeError("Expected 86 technically valid second-pass control rows")
    base_source = [row for row in read_jsonl(BASE_ASSIGNMENTS) if row["method"] == "BasePromptRewrite"]
    if len(base_source) != 43:
        raise RuntimeError("Expected 43 frozen BasePromptRewrite assignments")
    base_by_id = {row["canonical_id"]: row for row in base_source}
    control_by_key = {(row["canonical_id"], row["arm"]): row for row in control_rows}
    ids = [row["canonical_id"] for row in sorted(base_source, key=lambda row: row["source_order"])]
    assignments: list[dict[str, Any]] = []

    def add(source_order: int, canonical_id: str, method: str, question: str, response: str, source: Path, field: str) -> None:
        response = str(response or "").strip()
        assignments.append(
            {
                "scope": "VALID43_base_draft_control",
                "split": "VALID",
                "source_order": source_order,
                "canonical_id": canonical_id,
                "method": method,
                "question": question,
                "response": response,
                "question_sha256": sha256_text(question),
                "response_sha256": sha256_text(response),
                "question_response_sha256": pair_hash(question, response),
                "source_path": str(source.resolve()),
                "source_response_field": field,
            }
        )

    for canonical_id in ids:
        base = base_by_id[canonical_id]
        add(base["source_order"], canonical_id, "BasePromptRewrite", base["question"], base["response"], Path(base["source_path"]), base["source_response_field"])
        for arm, method in METHOD_BY_ARM.items():
            row = control_by_key[(canonical_id, arm)]
            if row["question"] != base["question"] or row["base_draft"] != base["response"]:
                raise RuntimeError(f"Frozen draft mismatch for {canonical_id}/{arm}")
            add(base["source_order"], canonical_id, method, base["question"], row["candidate_response"], CONTROL_OUTPUT, "candidate_response")
    pairs = unique_pairs(assignments)

    contract = read_json(FROZEN / "run_contract.json")
    prompts = contract["prompts"]
    point_system = prompts["pointwise_system"]
    point_template = prompts["pointwise_user_template"]
    selector_system = prompts["selector_system"]
    selector_template = prompts["selector_user_template"]
    if sha256_text(point_system) != prompts["pointwise_system_sha256"] or sha256_text(point_template) != prompts["pointwise_user_template_sha256"]:
        raise RuntimeError("Frozen pointwise prompt hash mismatch")
    if sha256_text(selector_system) != prompts["selector_system_sha256"] or sha256_text(selector_template) != prompts["selector_user_template_sha256"]:
        raise RuntimeError("Frozen selector prompt hash mismatch")

    point_requests = [pointwise_request(pair, point_system, point_template) for pair in pairs]
    old_cache, _ = cached_results()
    current_cache = scan_current_gpt_cache("pointwise_expert_rubric", 384)
    point_cache = {**old_cache, **current_cache}
    point_reused = []
    point_new = []
    for request in point_requests:
        cached = point_cache.get(request_key(request))
        if cached is None:
            point_new.append(request)
            continue
        path, result = cached
        point_reused.append(
            {
                "target_request_id": request["request_id"],
                "target_metadata": request["metadata"],
                "source_result_path": str(path),
                "source_request_id": result.get("request_id"),
                "normalized": result["normalized"],
                "raw_json_text": result.get("raw_json_text"),
                "returned_model": result.get("returned_model"),
            }
        )

    assignment_by_method_id = {(row["method"], row["canonical_id"]): row for row in assignments}
    comparisons = (
        ("BasePromptRewrite", "BaseSelfRefineCandidate"),
        ("BasePromptRewrite", "TrainedDenoiserOnBaseCandidate"),
        ("BaseSelfRefineCandidate", "TrainedDenoiserOnBaseCandidate"),
    )
    pair_requests: list[dict[str, Any]] = []
    for left_method, right_method in comparisons:
        for canonical_id in ids:
            left = assignment_by_method_id[(left_method, canonical_id)]
            right = assignment_by_method_id[(right_method, canonical_id)]
            comparison_id = f"{canonical_id}::{left_method}::{right_method}"
            for order in (0, 1):
                a, b = (left, right) if order == 0 else (right, left)
                user_prompt = selector_template.format(question=left["question"], response_a=a["response"], response_b=b["response"])
                metadata = {
                    "comparison_id": comparison_id,
                    "canonical_id": canonical_id,
                    "source_order": left["source_order"],
                    "left_method": left_method,
                    "right_method": right_method,
                    "presentation_order": order,
                    "response_a_method": a["method"],
                    "response_b_method": b["method"],
                    "response_a_sha256": a["response_sha256"],
                    "response_b_sha256": b["response_sha256"],
                }
                material = {
                    "version": VERSION,
                    "queue": PAIRWISE_QUEUE,
                    "model": MODEL,
                    "system_prompt": selector_system,
                    "user_prompt": user_prompt,
                    "metadata": metadata,
                }
                pair_requests.append(
                    {
                        "version": VERSION,
                        "queue": PAIRWISE_QUEUE,
                        "request_id": sha256_text(json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")))[:32],
                        "model": MODEL,
                        "temperature": 0,
                        "max_tokens": 256,
                        "response_format": {"type": "json_object"},
                        "parser": "blind_selector",
                        "system_prompt": selector_system,
                        "user_prompt": user_prompt,
                        "system_prompt_sha256": sha256_text(selector_system),
                        "user_prompt_sha256": sha256_text(user_prompt),
                        "metadata": metadata,
                    }
                )
    old_selector_cache = cached_selector_results()
    current_selector_cache = scan_current_gpt_cache("blind_selector", 256)
    selector_cache = {**old_selector_cache, **current_selector_cache}
    pair_reused = []
    pair_new = []
    for request in pair_requests:
        cached = selector_cache.get(request_key(request))
        if cached is None:
            pair_new.append(request)
            continue
        path, result = cached
        pair_reused.append(
            {
                "target_request_id": request["request_id"],
                "target_metadata": request["metadata"],
                "source_result_path": str(path),
                "source_request_id": result.get("request_id"),
                "normalized": result["normalized"],
                "raw_json_text": result.get("raw_json_text"),
                "returned_model": result.get("returned_model"),
            }
        )

    write_jsonl(ROOT / "inputs/second_pass_pointwise_assignments.jsonl", assignments)
    write_jsonl(ROOT / "inputs/second_pass_unique_pairs.jsonl", pairs)
    write_jsonl(ROOT / "requests/second_pass_expected_pointwise.jsonl", point_requests)
    write_jsonl(ROOT / "requests/second_pass_new_pointwise.jsonl", point_new)
    write_jsonl(ROOT / "results/second_pass_reused_pointwise.jsonl", point_reused)
    write_jsonl(ROOT / "requests/second_pass_expected_pairwise.jsonl", pair_requests)
    write_jsonl(ROOT / "requests/second_pass_new_pairwise.jsonl", pair_new)
    write_jsonl(ROOT / "results/second_pass_reused_pairwise.jsonl", pair_reused)
    write_json(
        ROOT / "second_pass_evaluation_contract.json",
        {
            "version": VERSION,
            "status": "prepared_before_api_execution",
            "created_at_utc": utc_now(),
            "development_only": True,
            "frozen_contract": {"path": str(FROZEN / "run_contract.json"), "sha256": sha256_file(FROZEN / "run_contract.json")},
            "pointwise": {"assignments": len(assignments), "unique": len(pairs), "reused": len(point_reused), "new": len(point_new)},
            "pairwise": {"comparisons": len(pair_requests) // 2, "requests": len(pair_requests), "reused": len(pair_reused), "new": len(pair_new)},
            "comparison_set": [list(value) for value in comparisons],
            "selector_consensus": "candidate/right method only when both presentation orders select it",
            "same_gpt_selector_and_evaluator_limitation": True,
        },
    )
    print(json.dumps(read_json(ROOT / "second_pass_evaluation_contract.json"), ensure_ascii=False, indent=2))


def score_risk() -> None:
    import torch

    sys.path.insert(0, str(WORKTREE / "scripts"))
    from fullpaper_risk_contract import require_fullpaper_axis_order
    from run_gemma_selective_risk_refinement import load_classifier, response_metrics
    from selective_risk_refinement_utils import DIMS, configure_risk_contract, score_candidate

    pairs = read_jsonl(ROOT / "inputs/second_pass_unique_pairs.jsonl")
    prior = {row["question_response_sha256"]: row for row in read_jsonl(ROOT / "risk/unique_pair_full_scorer_risk.jsonl")}
    output = ROOT / "risk/second_pass_unique_pair_risk.jsonl"
    existing = {row["question_response_sha256"]: row for row in read_jsonl(output)}
    for pair in pairs:
        key = pair["question_response_sha256"]
        if key in prior and key not in existing:
            existing[key] = {**prior[key], "risk_provenance": "exact_cache_reuse"}
    pending = [pair for pair in pairs if pair["question_response_sha256"] not in existing]
    configure_risk_contract("fullpaper_v1")
    focus_idx = DIMS.index("medical_boundary")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    router_tok, router = load_classifier(str(ROUTER), device)
    scorer_tok, scorer = load_classifier(str(SCORER), device)
    require_fullpaper_axis_order(router.config)
    require_fullpaper_axis_order(scorer.config)
    with torch.inference_mode():
        for index, pair in enumerate(pending, start=1):
            score = score_candidate(pair["question"], pair["response"], router, router_tok, scorer, scorer_tok, device, router_max_len=512, risk_max_len=384, component_mode="full")
            metrics = response_metrics(pair["question"], pair["response"], score, focus_idx)
            existing[pair["question_response_sha256"]] = {
                "question_response_sha256": pair["question_response_sha256"],
                "question_sha256": pair["question_sha256"],
                "response_sha256": pair["response_sha256"],
                "risk_score": metrics["risk_score"],
                "focus_risk_score": metrics["focus_risk_score"],
                "g": metrics["g"],
                "span_risks": metrics["span_risks"],
                "word_count": metrics["word_count"],
                "risk_contract": "fullpaper_v1",
                "component_mode": "full",
                "risk_provenance": "new_scoring",
            }
            if index % 10 == 0 or index == len(pending):
                write_jsonl(output, [existing[row["question_response_sha256"]] for row in pairs if row["question_response_sha256"] in existing])
                print(json.dumps({"risk_new_completed": index, "risk_new_total": len(pending)}), flush=True)
    write_jsonl(output, [existing[row["question_response_sha256"]] for row in pairs])


def result_maps() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    point = {}
    for row in read_jsonl(ROOT / "results/second_pass_reused_pointwise.jsonl"):
        point[row["target_request_id"]] = {"ok": True, "normalized": row["normalized"], "provenance": "reused_exact"}
    for row in read_jsonl(ROOT / "results/second_pass_new_pointwise.jsonl"):
        point[row["request_id"]] = {**row, "provenance": "new_api"}
    pair = {}
    for row in read_jsonl(ROOT / "results/second_pass_reused_pairwise.jsonl"):
        pair[row["target_request_id"]] = {"ok": True, "normalized": row["normalized"], "provenance": "reused_exact"}
    for row in read_jsonl(ROOT / "results/second_pass_new_pairwise.jsonl"):
        pair[row["request_id"]] = {**row, "provenance": "new_api"}
    return point, pair


def aggregate() -> None:
    assignments = read_jsonl(ROOT / "inputs/second_pass_pointwise_assignments.jsonl")
    point_requests = read_jsonl(ROOT / "requests/second_pass_expected_pointwise.jsonl")
    pair_requests = read_jsonl(ROOT / "requests/second_pass_expected_pairwise.jsonl")
    point_results, pair_results = result_maps()
    point_request_by_hash = {row["metadata"]["question_response_sha256"]: row for row in point_requests}
    scored = []
    failures = []
    for row in assignments:
        request = point_request_by_hash[row["question_response_sha256"]]
        result = point_results.get(request["request_id"])
        if result is None or result.get("ok") is not True or not score_valid(result):
            failures.append({"canonical_id": row["canonical_id"], "method": row["method"], "request_id": request["request_id"]})
            continue
        normalized = result["normalized"]
        scores = normalized["scores"]
        scored.append(
            {
                **row,
                "overall_quality": scores["overall_quality"],
                "empathy": scores["empathy"],
                "specificity": scores["specificity"],
                "quality_3axis_mean": quality3(normalized),
                "medical_advice": normalized["medical_advice"],
                "rationale": normalized.get("rationale", ""),
                "evaluation_provenance": result["provenance"],
            }
        )
    write_jsonl(ROOT / "results/second_pass_pointwise_scores.jsonl", scored)
    write_jsonl(ROOT / "results/second_pass_pointwise_failures.jsonl", failures)

    grouped_requests: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in pair_requests:
        grouped_requests[request["metadata"]["comparison_id"]].append(request)
    consensus = []
    for comparison_id, requests in grouped_requests.items():
        physical = []
        raw_choices = []
        rationales = []
        failed = False
        for request in sorted(requests, key=lambda row: row["metadata"]["presentation_order"]):
            result = pair_results.get(request["request_id"])
            if result is None or result.get("ok") is not True:
                failed = True
                raw_choices.append(None)
                continue
            choice = (result.get("normalized") or {}).get("choice")
            raw_choices.append(choice)
            rationales.append((result.get("normalized") or {}).get("rationale", ""))
            if choice == "tie":
                physical.append("tie")
            elif choice in {"A", "B"}:
                physical.append(request["metadata"]["response_a_method"] if choice == "A" else request["metadata"]["response_b_method"])
            else:
                failed = True
        metadata = requests[0]["metadata"]
        left = metadata["left_method"]
        right = metadata["right_method"]
        if failed:
            outcome = "technical_failure"
        elif physical == [right, right]:
            outcome = "right_wins_consensus"
        elif physical == [left, left]:
            outcome = "left_wins_consensus"
        elif physical == ["tie", "tie"]:
            outcome = "tie_consensus"
        else:
            outcome = "order_disagreement"
        consensus.append(
            {
                "comparison_id": comparison_id,
                "source_order": metadata["source_order"],
                "canonical_id": metadata["canonical_id"],
                "left_method": left,
                "right_method": right,
                "raw_choices_by_order": raw_choices,
                "physical_choices_by_order": physical,
                "outcome": outcome,
                "rationales_by_order": rationales,
            }
        )
    consensus.sort(key=lambda row: (row["source_order"], row["left_method"], row["right_method"]))
    write_jsonl(ROOT / "results/second_pass_pairwise_consensus.jsonl", consensus)

    summary = []
    for method in sorted({row["method"] for row in scored}):
        rows = [row for row in scored if row["method"] == method]
        medical = Counter(row["medical_advice"] for row in rows)
        summary.append(
            {
                "method": method,
                "N": len(rows),
                "overall_quality": statistics.mean(row["overall_quality"] for row in rows),
                "empathy": statistics.mean(row["empathy"] for row in rows),
                "specificity": statistics.mean(row["specificity"] for row in rows),
                "quality_3axis_mean": statistics.mean(row["quality_3axis_mean"] for row in rows),
                "medical_yes": medical["yes"],
                "medical_no": medical["no"],
                "medical_unsure": medical["unsure"],
                "mean_words": statistics.mean(len(row["response"].split()) for row in rows),
            }
        )
    write_jsonl(ROOT / "tables/second_pass_candidate_quality.jsonl", summary)
    write_csv(ROOT / "tables/second_pass_candidate_quality.csv", summary, list(summary[0]))

    pair_summary = []
    for left, right in sorted({(row["left_method"], row["right_method"]) for row in consensus}):
        rows = [row for row in consensus if row["left_method"] == left and row["right_method"] == right]
        counts = Counter(row["outcome"] for row in rows)
        pair_summary.append(
            {
                "comparison": f"{left}_vs_{right}",
                "N": len(rows),
                "left_wins_consensus": counts["left_wins_consensus"],
                "right_wins_consensus": counts["right_wins_consensus"],
                "tie_consensus": counts["tie_consensus"],
                "order_disagreement": counts["order_disagreement"],
                "technical_failure": counts["technical_failure"],
                "right_win_ids": [row["canonical_id"] for row in rows if row["outcome"] == "right_wins_consensus"],
            }
        )
    write_jsonl(ROOT / "tables/second_pass_pairwise_summary.jsonl", pair_summary)

    score_by_method_id = {(row["method"], row["canonical_id"]): row for row in scored}
    candidate_methods = ("BaseSelfRefineCandidate", "TrainedDenoiserOnBaseCandidate")
    generation_rows = read_jsonl(CONTROL_OUTPUT)
    generation_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in generation_rows:
        generation_by_id[row["canonical_id"]].append(row)
    prompt_hash_mismatch_ids = [
        canonical_id
        for canonical_id, rows in generation_by_id.items()
        if len(rows) != 2
        or len({row["prompt_provenance"]["rendered_prompt_sha256"] for row in rows}) != 1
    ]
    token_hash_mismatch_ids = [
        canonical_id
        for canonical_id, rows in generation_by_id.items()
        if len(rows) != 2
        or len({row["prompt_provenance"]["input_token_ids_used_sha256"] for row in rows}) != 1
    ]
    if prompt_hash_mismatch_ids or token_hash_mismatch_ids:
        raise RuntimeError(
            "Adapter-only input equivalence failed: "
            f"prompt={prompt_hash_mismatch_ids}, tokens={token_hash_mismatch_ids}"
        )
    input_equivalence = {
        "case_pairs": len(generation_by_id),
        "rendered_prompt_hash_mismatches": len(prompt_hash_mismatch_ids),
        "input_token_ids_hash_mismatches": len(token_hash_mismatch_ids),
        "prompt_hash_mismatch_ids": prompt_hash_mismatch_ids,
        "token_hash_mismatch_ids": token_hash_mismatch_ids,
    }
    generation_by_method_id = {
        (METHOD_BY_ARM[row["arm"]], row["canonical_id"]): row for row in generation_rows
    }
    final_rows = []
    for candidate_method in candidate_methods:
        decisions = {
            row["canonical_id"]: row
            for row in consensus
            if row["left_method"] == "BasePromptRewrite" and row["right_method"] == candidate_method
        }
        for policy in ("all_generated", "legacy_sft_threshold_transfer"):
            final_method = candidate_method.replace(
                "Candidate",
                "AllGeneratedGPTSelected" if policy == "all_generated" else "LegacyThresholdTransferGPTSelected",
            )
            for canonical_id in sorted(decisions, key=lambda value: decisions[value]["source_order"]):
                decision = decisions[canonical_id]
                base_risk = generation_by_method_id[(candidate_method, canonical_id)]["base_draft_risk_score"]
                gate_called = policy == "all_generated" or base_risk >= LEGACY_SFT_GATE_THRESHOLD
                accepted = gate_called and decision["outcome"] == "right_wins_consensus"
                chosen_method = candidate_method if accepted else "BasePromptRewrite"
                chosen = score_by_method_id[(chosen_method, canonical_id)]
                final_rows.append(
                    {
                        **chosen,
                        "method": final_method,
                        "source_method": chosen_method,
                        "candidate_accepted": accepted,
                        "candidate_generated_for_diagnostic": True,
                        "deployment_gate_called": gate_called,
                        "gate_policy": policy,
                        "gate_threshold": LEGACY_SFT_GATE_THRESHOLD if policy != "all_generated" else None,
                        "selector_outcome": decision["outcome"] if gate_called else "gate_off_not_considered",
                    }
                )
    for row in [value for value in scored if value["method"] == "BasePromptRewrite"]:
        final_rows.append(
            {
                **row,
                "candidate_accepted": False,
                "candidate_generated_for_diagnostic": False,
                "deployment_gate_called": False,
                "gate_policy": "none",
                "gate_threshold": None,
                "selector_outcome": "not_applicable",
            }
        )
    write_jsonl(ROOT / "results/second_pass_final_outputs.jsonl", final_rows)
    final_summary = []
    for method in sorted({row["method"] for row in final_rows}):
        rows = [row for row in final_rows if row["method"] == method]
        final_summary.append(
            {
                "method": method,
                "N": len(rows),
                "gate_called": sum(bool(row["deployment_gate_called"]) for row in rows),
                "accepted": sum(bool(row["candidate_accepted"]) for row in rows),
                "overall_quality": statistics.mean(row["overall_quality"] for row in rows),
                "empathy": statistics.mean(row["empathy"] for row in rows),
                "specificity": statistics.mean(row["specificity"] for row in rows),
                "quality_3axis_mean": statistics.mean(row["quality_3axis_mean"] for row in rows),
            }
        )
    write_jsonl(ROOT / "tables/second_pass_final_quality.jsonl", final_summary)
    write_csv(ROOT / "tables/second_pass_final_quality.csv", final_summary, list(final_summary[0]))

    base_by_id = {
        row["canonical_id"]: row for row in final_rows if row["method"] == "BasePromptRewrite"
    }
    final_comparisons = []
    for method in sorted({row["method"] for row in final_rows if row["method"] != "BasePromptRewrite"}):
        rows = [row for row in final_rows if row["method"] == method]
        wins = ties = losses = 0
        for row in rows:
            delta = row["quality_3axis_mean"] - base_by_id[row["canonical_id"]]["quality_3axis_mean"]
            if delta > 0:
                wins += 1
            elif delta < 0:
                losses += 1
            else:
                ties += 1
        final_comparisons.append(
            {
                "method": method,
                "N": len(rows),
                "wins_vs_base": wins,
                "ties_vs_base": ties,
                "losses_vs_base": losses,
                "accepted_ids": [row["canonical_id"] for row in rows if row["candidate_accepted"]],
            }
        )
    write_jsonl(ROOT / "tables/second_pass_final_wtl.jsonl", final_comparisons)

    refusal_pattern = re.compile(
        r"\b(?:"
        r"I (?:cannot|can't) (?:rewrite|refine|fulfill this request|comply with this request)"
        r"|I(?:'m| am) unable to (?:rewrite|refine|fulfill this request)"
        r"|unable to (?:rewrite|refine) the (?:provided )?(?:text|response)"
        r")\b",
        re.I,
    )
    meta_only_pattern = re.compile(
        r"^(?:Here is|Below is|I have rewritten|The rewritten response).{0,240}"
        r"(?:based on your request|let me know|assist you|full body|continue until)",
        re.I | re.S,
    )
    foreign_script_pattern = re.compile(
        r"[\u0400-\u04ff\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]"
    )
    placeholder_corruption_pattern = re.compile(
        r"\[(?:rest of (?:the )?(?:text|response)|the full body[^\]]*|continue[^\]]*|"
        r"full response[^\]]*|insert[^\]]*|[a-z]{1,3})\]",
        re.I,
    )
    malformed_surface_pattern = re.compile(
        r"\bvarsigma\b|\bend-oflife\b|['’]{2}s\b|\bdoctor['’]?sne\b|"
        r"\b(?:don|can|won|isn|aren|wasn|weren|hasn|haven|hadn|shouldn|wouldn|couldn)'s\b|"
        r"\\(?:mathbf|text)\{",
        re.I,
    )
    generation_failures = []
    for method in candidate_methods:
        rows = [row for row in generation_rows if METHOD_BY_ARM[row["arm"]] == method]
        refusal_ids = [
            row["canonical_id"]
            for row in rows
            if refusal_pattern.search(row["candidate_response"])
            or meta_only_pattern.search(row["candidate_response"])
        ]
        foreign_script_ids = [
            row["canonical_id"] for row in rows if foreign_script_pattern.search(row["candidate_response"])
        ]
        placeholder_corruption_ids = [
            row["canonical_id"]
            for row in rows
            if placeholder_corruption_pattern.search(row["candidate_response"])
        ]
        malformed_surface_ids = [
            row["canonical_id"]
            for row in rows
            if malformed_surface_pattern.search(row["candidate_response"])
        ]
        cleanup_ids = [row["canonical_id"] for row in rows if row.get("cleanup_recovery_applied")]
        generation_failures.append(
            {
                "method": method,
                "N": len(rows),
                "refusal_or_meta_refusal_count": len(refusal_ids),
                "refusal_or_meta_refusal_ids": refusal_ids,
                "unexpected_non_latin_script_count": len(foreign_script_ids),
                "unexpected_non_latin_script_ids": foreign_script_ids,
                "placeholder_or_bracket_corruption_count": len(placeholder_corruption_ids),
                "placeholder_or_bracket_corruption_ids": placeholder_corruption_ids,
                "malformed_surface_token_or_markup_count": len(malformed_surface_ids),
                "malformed_surface_token_or_markup_ids": malformed_surface_ids,
                "cleanup_recovery_count": len(cleanup_ids),
                "cleanup_recovery_ids": cleanup_ids,
            }
        )
    write_jsonl(ROOT / "tables/second_pass_generation_failure_signals.jsonl", generation_failures)

    risk_by_hash = {row["question_response_sha256"]: row for row in read_jsonl(ROOT / "risk/second_pass_unique_pair_risk.jsonl")}
    risk_summary = []
    for scope, rows_source in (("candidate", scored), ("final", final_rows)):
        for method in sorted({row["method"] for row in rows_source}):
            rows = [row for row in rows_source if row["method"] == method]
            risk_summary.append(
                {
                    "scope": scope,
                    "method": method,
                    "N": len(rows),
                    "mean_risk": statistics.mean(risk_by_hash[row["question_response_sha256"]]["risk_score"] for row in rows),
                    "mean_focus_risk": statistics.mean(risk_by_hash[row["question_response_sha256"]]["focus_risk_score"] for row in rows),
                }
            )
    write_jsonl(ROOT / "tables/second_pass_risk_summary.jsonl", risk_summary)
    write_csv(ROOT / "tables/second_pass_risk_summary.csv", risk_summary, list(risk_summary[0]))

    result = {
        "version": VERSION,
        "status": "complete" if not failures and all(row["technical_failure"] == 0 for row in pair_summary) else "complete_with_failures",
        "created_at_utc": utc_now(),
        "candidate_quality": summary,
        "pairwise": pair_summary,
        "selected_final_quality": final_summary,
        "selected_final_wtl": final_comparisons,
        "generation_failure_signals": generation_failures,
        "risk": risk_summary,
        "input_equivalence": input_equivalence,
        "limitations": [
            "VALID43 is development data.",
            "Candidates were always generated for diagnosis; this is not an always-on deployment claim.",
            "GPT-4.1 is used for both pairwise selection and pointwise evaluation.",
            "No TEST outputs were generated or used to adjust the second-pass prompt.",
        ],
    }
    write_json(ROOT / "VALID43_BASE_DRAFT_CONTROL_RESULTS.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "score-risk", "aggregate"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "score-risk":
        score_risk()
    else:
        aggregate()
