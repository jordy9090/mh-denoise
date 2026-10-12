#!/usr/bin/env python3
"""Prepare, score, aggregate, and package the TEST60 component matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "acl-component-ablation-matrix-v1-20261009"
ROOT = Path("/mnt/ssd00/hsoh/acl_component_ablation_matrix_20261009_v1")
REPO = Path("/home/user/hsoh/mh-denoise")
WORKTREE = Path("/home/user/hsoh/mh-denoise-surface-relaxed-v1-20261008")
FROZEN = Path("/home/user/hsoh/rubric_aligned_gpt41_rejudge_20261009_v1")
CORRECTED_PUBLICATION = Path("/home/user/hsoh/test60_corrected_gpt41_pointwise_publication_20261009_v1")
NO_SELECTOR = Path("/home/user/hsoh/test60_gated_denoiser_no_selector_ablation_20261009_v1")
CONTROL = Path("/mnt/ssd00/hsoh/original_prompt_sft_control_20261009_v1")
MODEL = "gpt-4.1-2025-04-14"
QUEUE = "acl_component_pointwise"
FINAL_OUTPUTS = ROOT / "results/final_method_outputs.jsonl"
CANDIDATE_OUTPUTS = ROOT / "results/candidate_quality_outputs.jsonl"
CACHE_ROOTS = (CORRECTED_PUBLICATION, NO_SELECTOR, FROZEN, CONTROL)
LOCAL_MODEL = Path(
    "/home/user/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/"
    "snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
ROUTER = REPO / "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/router/final"
SCORER = REPO / "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/scorer/final"
MAIN_METHODS = ("BasePromptRewrite", "SFT", "DPO", "Proposed")
ABLATION_METHODS = (
    "GatedNoSelector",
    "AlwaysOnNoSelector",
    "AlwaysOnGPTSelector",
    "OriginalPromptSFTControlGPTSelector",
    "MaskONGPTSelector",
    "WithoutRouterGPTSelector",
    "WithoutScorerGPTSelector",
)
POLICY_METHODS = ("LegacyStrict", "CurrentRules", "NoMinLength", "RiskOnly", "Proposed")
DIAGNOSTIC_METHODS = ("UnsafeResponse", "ReferenceResponse")


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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_hash(question: str, response: str) -> str:
    return sha256_text(json.dumps([question, response], ensure_ascii=False, separators=(",", ":")))


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


def score_valid(row: dict[str, Any]) -> bool:
    normalized = row.get("normalized") or {}
    scores = normalized.get("scores") or {}
    for axis in ("overall_quality", "empathy", "specificity"):
        value = scores.get(axis)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            return False
    medical = normalized.get("medical_advice")
    if medical not in {"yes", "no", "unsure"}:
        return False
    if medical in {"yes", "unsure"} and not str(normalized.get("medical_advice_evidence") or "").strip():
        return False
    return bool(str(normalized.get("medical_advice_rationale") or "").strip())


def load_assignments() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    finals = read_jsonl(FINAL_OUTPUTS)
    candidates = read_jsonl(CANDIDATE_OUTPUTS)
    final_methods = {row["method"] for row in finals}
    expected_methods = set(MAIN_METHODS + ABLATION_METHODS + POLICY_METHODS + DIAGNOSTIC_METHODS)
    if final_methods != expected_methods:
        raise RuntimeError(f"Final method coverage differs: {sorted(final_methods ^ expected_methods)}")
    if len(finals) != 60 * len(expected_methods):
        raise RuntimeError("Final method assignments are incomplete")
    assignments = finals + candidates
    for row in assignments:
        expected = pair_hash(row["question"], row["response"])
        if row["question_response_sha256"] != expected:
            raise RuntimeError(f"Question-response hash mismatch for {row['method']}/{row['canonical_id']}")
    representatives: dict[str, dict[str, Any]] = {}
    for row in assignments:
        key = row["question_response_sha256"]
        member = {
            "scope": row["scope"],
            "source_order": row["source_order"],
            "case_id": row["case_id"],
            "canonical_id": row["canonical_id"],
            "method": row["method"],
        }
        if key not in representatives:
            representatives[key] = {
                "question_response_sha256": key,
                "question": row["question"],
                "response": row["response"],
                "question_sha256": sha256_text(row["question"]),
                "response_sha256": sha256_text(row["response"]),
                "members": [],
            }
        elif representatives[key]["question"] != row["question"] or representatives[key]["response"] != row["response"]:
            raise RuntimeError(f"Hash collision for {key}")
        representatives[key]["members"].append(member)
    unique = sorted(
        representatives.values(),
        key=lambda row: (
            min(int(member["source_order"]) for member in row["members"]),
            row["question_response_sha256"],
        ),
    )
    return assignments, unique


def cached_gpt_results() -> tuple[dict[str, tuple[Path, dict[str, Any]]], list[dict[str, Any]]]:
    chosen: dict[str, tuple[Path, dict[str, Any]]] = {}
    conflicts = []
    seen_paths: set[Path] = set()
    for root in CACHE_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.jsonl")):
            in_results_dir = any(part.lower() == "results" for part in path.parts)
            if path in seen_paths or ("result" not in path.name.lower() and not in_results_dir):
                continue
            seen_paths.add(path)
            try:
                rows = read_jsonl(path)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            for row in rows:
                if row.get("parser") != "pointwise_expert_rubric" or row.get("ok") is not True:
                    continue
                if row.get("model") != MODEL or row.get("returned_model") != MODEL:
                    continue
                if row.get("temperature") != 0 or row.get("max_tokens") != 384:
                    continue
                if row.get("response_format") != {"type": "json_object"} or not score_valid(row):
                    continue
                key = request_key(row)
                prior = chosen.get(key)
                if prior is not None and prior[1]["normalized"] != row["normalized"]:
                    conflicts.append(
                        {
                            "request_key": key,
                            "kept_path": str(prior[0]),
                            "other_path": str(path),
                            "resolution": "kept_first_by_frozen_cache_priority",
                        }
                    )
                    continue
                if prior is None:
                    chosen[key] = (path, row)
    return chosen, conflicts


def make_request(pair: dict[str, Any], system_prompt: str, user_template: str) -> dict[str, Any]:
    user_prompt = user_template.format(question=pair["question"], response=pair["response"])
    metadata = {
        "question_response_sha256": pair["question_response_sha256"],
        "question_sha256": pair["question_sha256"],
        "response_sha256": pair["response_sha256"],
        "members": pair["members"],
        "priority": "acl_component_matrix",
    }
    request_id = sha256_text(
        json.dumps(
            {"version": VERSION, "queue": QUEUE, "user_prompt": user_prompt, "metadata": metadata},
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
        "max_tokens": 384,
        "response_format": {"type": "json_object"},
        "parser": "pointwise_expert_rubric",
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_sha256": sha256_text(user_prompt),
        "metadata": metadata,
    }


def prepare() -> None:
    assignments, unique = load_assignments()
    contract = read_json(FROZEN / "run_contract.json")
    settings = contract["model_settings"]
    prompts = contract["prompts"]
    if settings["model"] != MODEL or settings["temperature"] != 0 or settings["pointwise_max_tokens"] != 384:
        raise RuntimeError("Frozen pointwise settings changed")
    system_prompt = prompts["pointwise_system"]
    user_template = prompts["pointwise_user_template"]
    if sha256_text(system_prompt) != prompts["pointwise_system_sha256"]:
        raise RuntimeError("Frozen pointwise system prompt hash mismatch")
    if sha256_text(user_template) != prompts["pointwise_user_template_sha256"]:
        raise RuntimeError("Frozen pointwise template hash mismatch")

    requests = [make_request(pair, system_prompt, user_template) for pair in unique]
    cache, conflicts = cached_gpt_results()
    reused = []
    new = []
    for request in requests:
        cached = cache.get(request_key(request))
        if cached is None:
            new.append(request)
            continue
        source_path, result = cached
        reused.append(
            {
                "target_request_id": request["request_id"],
                "target_request_key": request_key(request),
                "question_response_sha256": request["metadata"]["question_response_sha256"],
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
    write_jsonl(ROOT / "inputs/all_pointwise_assignments.jsonl", assignments)
    write_jsonl(ROOT / "inputs/unique_question_response_pairs.jsonl", unique)
    write_jsonl(ROOT / "requests/expected_pointwise_requests.jsonl", requests)
    write_jsonl(ROOT / "requests/new_pointwise_requests.jsonl", new)
    write_jsonl(ROOT / "results/reused_pointwise_cache.jsonl", reused)
    write_jsonl(ROOT / "results/pointwise_cache_conflicts.jsonl", conflicts)
    pointwise_contract = {
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
            "max_tokens": 384,
            "response_format": {"type": "json_object"},
            "parser": "pointwise_expert_rubric",
        },
        "prompts": {
            "system_sha256": sha256_text(system_prompt),
            "user_template_sha256": sha256_text(user_template),
        },
        "privacy": {
            "receives": ["question", "anonymous response", "rubric"],
            "hidden": ["method", "risk", "selector decision", "reference status"],
        },
        "scope": {
            "final_methods": sorted({row["method"] for row in read_jsonl(FINAL_OUTPUTS)}),
            "candidate_methods": sorted({row["method"] for row in read_jsonl(CANDIDATE_OUTPUTS)}),
            "post_hoc_exploratory_test_comparison": True,
            "no_response_regeneration_for_evaluation": True,
            "medical_advice_not_converted_to_numeric_quality": True,
        },
        "counts": {
            "assignments": len(assignments),
            "final_assignments": len(read_jsonl(FINAL_OUTPUTS)),
            "candidate_assignments": len(read_jsonl(CANDIDATE_OUTPUTS)),
            "unique_question_response": len(unique),
            "reused_exact": len(reused),
            "new": len(new),
            "cache_conflicts_logged": len(conflicts),
        },
    }
    write_json(ROOT / "pointwise_contract.json", pointwise_contract)
    print(json.dumps(pointwise_contract["counts"], sort_keys=True))


def gpt_result_map() -> dict[str, dict[str, Any]]:
    mapped = {}
    for row in read_jsonl(ROOT / "results/reused_pointwise_cache.jsonl"):
        mapped[row["target_request_id"]] = {
            "ok": True,
            "normalized": row["normalized"],
            "provenance": "exact_cache_reuse",
            "source_result_path": row["source_result_path"],
        }
    for path in sorted((ROOT / "results").glob("new_pointwise*_results.jsonl")):
        for row in read_jsonl(path):
            mapped[row["request_id"]] = {**row, "provenance": "new_api"}
    return mapped


def llama_result_map() -> dict[str, dict[str, Any]]:
    mapped = {}
    for row in read_jsonl(ROOT / "results/reused_llama_pointwise_cache.jsonl"):
        mapped[row["target_request_id"]] = {
            "judge_ok": True,
            "judge_normalized": row["judge_normalized"],
            "provenance": "exact_cache_reuse",
            "source_result_path": row["source_result_path"],
        }
    for path in sorted((ROOT / "results").glob("new_llama_pointwise*_results.jsonl")):
        for row in read_jsonl(path):
            mapped[row["request_id"]] = {**row, "provenance": "new_local_generation"}
    return mapped


def scored_assignments(judge: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignments = read_jsonl(ROOT / "inputs/all_pointwise_assignments.jsonl")
    requests = read_jsonl(ROOT / "requests/expected_pointwise_requests.jsonl")
    request_by_pair = {
        row["metadata"]["question_response_sha256"]: row for row in requests
    }
    results = gpt_result_map() if judge == "GPT-4.1" else llama_result_map()
    scored = []
    failures = []
    for row in assignments:
        request = request_by_pair[row["question_response_sha256"]]
        result = results.get(request["request_id"])
        normalized = None
        if judge == "GPT-4.1" and result is not None and result.get("ok") is True:
            normalized = result.get("normalized")
        if judge == "Llama" and result is not None and result.get("judge_ok") is True:
            normalized = result.get("judge_normalized")
        output = {**row, "judge": judge, "evaluation_ok": normalized is not None}
        if normalized is None:
            failures.append(
                {
                    "judge": judge,
                    "request_id": request["request_id"],
                    "question_response_sha256": row["question_response_sha256"],
                    "method": row["method"],
                    "canonical_id": row["canonical_id"],
                    "reason": "missing_or_failed_evaluation",
                }
            )
        else:
            scores = normalized["scores"]
            output.update(
                {
                    "request_id": request["request_id"],
                    "overall_quality": scores["overall_quality"],
                    "empathy": scores["empathy"],
                    "specificity": scores["specificity"],
                    "quality_3axis_mean": statistics.mean(
                        [scores["overall_quality"], scores["empathy"], scores["specificity"]]
                    ),
                    "medical_advice": normalized["medical_advice"],
                    "rationale": normalized.get("rationale", ""),
                    "medical_advice_evidence": normalized.get("medical_advice_evidence", ""),
                    "medical_advice_rationale": normalized.get("medical_advice_rationale", ""),
                    "evaluation_provenance": result.get("provenance"),
                }
            )
        scored.append(output)
    return scored, failures


def quality_summary(rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["scope"] == scope]
    methods = sorted({row["method"] for row in selected})
    summary = []
    for method in methods:
        method_rows = [row for row in selected if row["method"] == method]
        valid = [row for row in method_rows if row["evaluation_ok"]]
        medical = Counter(row["medical_advice"] for row in valid)
        denominator = medical["yes"] + medical["no"]
        summary.append(
            {
                "method": method,
                "scope": scope,
                "N_expected": len(method_rows),
                "N_evaluated": len(valid),
                "overall_quality_mean": statistics.mean(row["overall_quality"] for row in valid) if valid else None,
                "empathy_mean": statistics.mean(row["empathy"] for row in valid) if valid else None,
                "specificity_mean": statistics.mean(row["specificity"] for row in valid) if valid else None,
                "quality_3axis_mean": statistics.mean(row["quality_3axis_mean"] for row in valid) if valid else None,
                "medical_advice_yes": medical["yes"],
                "medical_advice_no": medical["no"],
                "medical_advice_unsure": medical["unsure"],
                "medical_advice_yes_rate": medical["yes"] / denominator if denominator else None,
                "medical_advice_yes_rate_denominator": denominator,
                "technical_or_parse_failures": len(method_rows) - len(valid),
            }
        )
    return summary


def bootstrap_ci(values: list[float], seed: int = 20261009, samples: int = 10000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    import numpy as np

    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975], method="linear")
    return float(low), float(high)


def sign_test_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def paired_tables(rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["scope"] == scope and row["evaluation_ok"]]
    by_method: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        by_method[row["method"]][row["canonical_id"]] = row
    methods = sorted(by_method)
    output = []
    for index, left in enumerate(methods):
        for right in methods[index + 1 :]:
            ids = sorted(set(by_method[left]) & set(by_method[right]))
            deltas = [
                by_method[right][row_id]["quality_3axis_mean"]
                - by_method[left][row_id]["quality_3axis_mean"]
                for row_id in ids
            ]
            wins = sum(value > 1e-12 for value in deltas)
            losses = sum(value < -1e-12 for value in deltas)
            ties = len(deltas) - wins - losses
            low, high = bootstrap_ci(deltas, seed=20261009 + index)
            output.append(
                {
                    "scope": scope,
                    "left": left,
                    "right": right,
                    "delta_definition": "right_minus_left",
                    "N": len(ids),
                    "right_win": wins,
                    "tie": ties,
                    "right_loss": losses,
                    "quality_3axis_mean_delta": statistics.mean(deltas) if deltas else None,
                    "bootstrap_95ci_low": low,
                    "bootstrap_95ci_high": high,
                    "two_sided_sign_test_p": sign_test_p(wins, losses),
                }
            )
    return output


def aggregate() -> None:
    for judge in ("GPT-4.1", "Llama"):
        scored, failures = scored_assignments(judge)
        slug = "gpt41" if judge == "GPT-4.1" else "llama"
        write_jsonl(ROOT / f"results/{slug}_assignment_scores.jsonl", scored)
        write_jsonl(ROOT / f"results/{slug}_pointwise_failures.jsonl", failures)
        final_summary = quality_summary(scored, "TEST60_final_policy_output")
        candidate_summary = quality_summary(scored, "called_candidate_quality")
        write_jsonl(ROOT / f"tables/{slug}_final_quality_summary.jsonl", final_summary)
        write_csv(
            ROOT / f"tables/{slug}_final_quality_summary.csv",
            final_summary,
            list(final_summary[0].keys()),
        )
        write_jsonl(ROOT / f"tables/{slug}_candidate_quality_summary.jsonl", candidate_summary)
        write_csv(
            ROOT / f"tables/{slug}_candidate_quality_summary.csv",
            candidate_summary,
            list(candidate_summary[0].keys()),
        )
        final_pairs = paired_tables(scored, "TEST60_final_policy_output")
        candidate_pairs = paired_tables(scored, "called_candidate_quality")
        write_jsonl(ROOT / f"tables/{slug}_final_all_pairs.jsonl", final_pairs)
        write_jsonl(ROOT / f"tables/{slug}_candidate_all_pairs.jsonl", candidate_pairs)

    gpt = read_jsonl(ROOT / "results/gpt41_assignment_scores.jsonl")
    llama = read_jsonl(ROOT / "results/llama_assignment_scores.jsonl")
    gpt_by_key = {
        (row["scope"], row["method"], row["canonical_id"]): row
        for row in gpt
        if row["evaluation_ok"]
    }
    llama_by_key = {
        (row["scope"], row["method"], row["canonical_id"]): row
        for row in llama
        if row["evaluation_ok"]
    }
    common = sorted(set(gpt_by_key) & set(llama_by_key))
    agreement_rows = []
    for key in common:
        left, right = gpt_by_key[key], llama_by_key[key]
        agreement_rows.append(
            {
                "scope": key[0],
                "method": key[1],
                "canonical_id": key[2],
                "gpt41_quality_3axis_mean": left["quality_3axis_mean"],
                "llama_quality_3axis_mean": right["quality_3axis_mean"],
                "llama_minus_gpt41": right["quality_3axis_mean"] - left["quality_3axis_mean"],
                "gpt41_medical_advice": left["medical_advice"],
                "llama_medical_advice": right["medical_advice"],
                "medical_label_agrees": left["medical_advice"] == right["medical_advice"],
            }
        )
    write_jsonl(ROOT / "tables/judge_direct_comparison.jsonl", agreement_rows)

    final_outputs = read_jsonl(FINAL_OUTPUTS)
    selector_summary = read_jsonl(ROOT / "results/selector_summary.jsonl")
    pipeline = []
    for method in sorted({row["method"] for row in final_outputs}):
        rows = [row for row in final_outputs if row["method"] == method]
        pipeline.append(
            {
                "method": method,
                "N": len(rows),
                "candidate_accepted": sum(bool(row["candidate_accepted"]) for row in rows),
                "text_changed_vs_sft": sum(bool(row["text_differs_from_sft"]) for row in rows),
            }
        )
    write_jsonl(ROOT / "tables/final_pipeline_counts.jsonl", pipeline)
    status = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "gpt41": {
            "assignments": len(gpt),
            "failed": sum(not row["evaluation_ok"] for row in gpt),
        },
        "llama": {
            "assignments": len(llama),
            "failed": sum(not row["evaluation_ok"] for row in llama),
        },
        "judge_common_assignments": len(common),
        "selector_summary": selector_summary,
    }
    write_json(ROOT / "results/evaluation_completion.json", status)
    print(json.dumps(status, sort_keys=True))


def score_risk() -> None:
    import torch

    sys.path.insert(0, str(WORKTREE / "scripts"))
    from fullpaper_risk_contract import require_fullpaper_axis_order
    from run_gemma_selective_risk_refinement import load_classifier, response_metrics
    from selective_risk_refinement_utils import DIMS, configure_risk_contract, score_candidate

    pairs = read_jsonl(ROOT / "inputs/unique_question_response_pairs.jsonl")
    output = ROOT / "risk/unique_pair_full_scorer_risk.jsonl"
    existing = {row["question_response_sha256"]: row for row in read_jsonl(output)}
    prior_path = CORRECTED_PUBLICATION / "risk/unique_pair_scorer_risk.jsonl"
    prior_reused = 0
    for row in read_jsonl(prior_path):
        key = row["question_response_sha256"]
        if key in {pair["question_response_sha256"] for pair in pairs} and key not in existing:
            existing[key] = {**row, "risk_provenance": "exact_cache_reuse", "source_path": str(prior_path)}
            prior_reused += 1
    pending = [pair for pair in pairs if pair["question_response_sha256"] not in existing]
    configure_risk_contract("fullpaper_v1")
    focus_idx = DIMS.index("medical_boundary")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    router_tok, router = load_classifier(str(ROUTER), device)
    risk_tok, risk_model = load_classifier(str(SCORER), device)
    require_fullpaper_axis_order(router.config)
    require_fullpaper_axis_order(risk_model.config)
    with torch.inference_mode():
        for index, pair in enumerate(pending, start=1):
            score = score_candidate(
                pair["question"],
                pair["response"],
                router,
                router_tok,
                risk_model,
                risk_tok,
                device,
                router_max_len=512,
                risk_max_len=384,
                component_mode="full",
            )
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
    ordered = [existing[row["question_response_sha256"]] for row in pairs]
    write_jsonl(output, ordered)
    assignments = read_jsonl(ROOT / "inputs/all_pointwise_assignments.jsonl")
    assignment_risk = [
        {
            "scope": row["scope"],
            "source_order": row["source_order"],
            "canonical_id": row["canonical_id"],
            "case_id": row["case_id"],
            "method": row["method"],
            "question_response_sha256": row["question_response_sha256"],
            "risk_score": existing[row["question_response_sha256"]]["risk_score"],
            "focus_risk_score": existing[row["question_response_sha256"]]["focus_risk_score"],
        }
        for row in assignments
    ]
    write_jsonl(ROOT / "risk/assignment_full_scorer_risk.jsonl", assignment_risk)
    summaries = []
    for scope in sorted({row["scope"] for row in assignment_risk}):
        for method in sorted({row["method"] for row in assignment_risk if row["scope"] == scope}):
            rows = [row for row in assignment_risk if row["scope"] == scope and row["method"] == method]
            summaries.append(
                {
                    "scope": scope,
                    "method": method,
                    "N": len(rows),
                    "mean_risk": statistics.mean(row["risk_score"] for row in rows),
                    "mean_focus_risk": statistics.mean(row["focus_risk_score"] for row in rows),
                }
            )
    write_jsonl(ROOT / "tables/full_scorer_risk_summary.jsonl", summaries)
    write_csv(ROOT / "tables/full_scorer_risk_summary.csv", summaries, list(summaries[0].keys()))
    manifest = {
        "status": "complete",
        "unique_pairs": len(ordered),
        "cache_reused_this_run": prior_reused,
        "new_scored_this_run": len(pending),
        "settings": {
            "risk_contract": "fullpaper_v1",
            "component_mode": "full",
            "router_max_len": 512,
            "risk_max_len": 384,
            "focus_axis": "medical_boundary",
        },
        "output_sha256": sha256_file(output),
    }
    write_json(ROOT / "risk/risk_scoring_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


def latex_escape(value: str) -> str:
    replacements = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(char, char) for char in value)


def build_tables_and_package() -> None:
    gpt = {row["method"]: row for row in read_jsonl(ROOT / "tables/gpt41_final_quality_summary.jsonl")}
    llama = {row["method"]: row for row in read_jsonl(ROOT / "tables/llama_final_quality_summary.jsonl")}
    risk = {
        row["method"]: row
        for row in read_jsonl(ROOT / "tables/full_scorer_risk_summary.jsonl")
        if row["scope"] == "TEST60_final_policy_output"
    }
    pipeline = {row["method"]: row for row in read_jsonl(ROOT / "tables/final_pipeline_counts.jsonl")}
    integrity = read_json(ROOT / "generation/generation_integrity_audit.json")
    selector_by_arm = {row["arm"]: row for row in read_jsonl(ROOT / "results/selector_summary.jsonl")}
    display = {
        "BasePromptRewrite": "Prompt rewrite (base)",
        "SFT": "SFT",
        "DPO": "DPO",
        "Proposed": "Proposed",
        "GatedNoSelector": "w/o selector",
        "AlwaysOnNoSelector": "always-on, w/o selector",
        "AlwaysOnGPTSelector": "always-on + selector",
        "OriginalPromptSFTControlGPTSelector": "SFT-adapter control + selector",
        "MaskONGPTSelector": "Mask ON + selector",
        "WithoutRouterGPTSelector": "w/o Router + selector",
        "WithoutScorerGPTSelector": "w/o Scorer + selector",
        "LegacyStrict": "legacy strict",
        "CurrentRules": "surface-relaxed rules",
        "NoMinLength": "w/o min-length veto",
        "RiskOnly": "risk-only",
        "UnsafeResponse": "unsafe input",
        "ReferenceResponse": "stored reference",
    }

    def format_row(method: str, judge_rows: dict[str, dict[str, Any]]) -> str:
        row = judge_rows[method]
        medical = f"{row['medical_advice_yes']}/{row['medical_advice_no']}/{row['medical_advice_unsure']}"
        return (
            f"{latex_escape(display[method])} & {row['N_evaluated']} & {row['overall_quality_mean']:.3f} & "
            f"{row['empathy_mean']:.3f} & {row['specificity_mean']:.3f} & "
            f"{row['quality_3axis_mean']:.3f} & {medical} \\\\"
        )

    main_order = list(MAIN_METHODS)
    ablation_order = ["SFT", *ABLATION_METHODS, "Proposed"]
    policy_order = ["SFT", "LegacyStrict", "CurrentRules", "NoMinLength", "RiskOnly", "GatedNoSelector", "Proposed"]
    for name, order, caption in (
        ("main_baselines", main_order, "Post-hoc GPT-4.1 TEST60 main baselines under the expert-aligned rubric."),
        ("component_ablations", ablation_order, "Post-hoc GPT-4.1 TEST60 component ablations under the expert-aligned rubric."),
        ("policy_sensitivity", policy_order, "Post-hoc GPT-4.1 TEST60 frozen-candidate policy sensitivity under the expert-aligned rubric."),
    ):
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrrrrl}",
            r"\toprule",
            r"Method & $N$ & Overall & Empathy & Specificity & Quality-3 & Med. Y/N/U \\",
            r"\midrule",
        ]
        lines.extend(format_row(method, gpt) for method in order)
        lines.extend([r"\bottomrule", r"\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{tab:{name}}}", r"\end{table}"])
        (ROOT / f"publication/{name}_gpt41.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def llama_table(name: str, order: list[str], caption: str) -> str:
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrrrrll}",
            r"\toprule",
            r"Method & Scored & Overall & Empathy & Specificity & Quality-3 & Med. Y/N/U & Fail. \\",
            r"\midrule",
        ]
        for method in order:
            row = llama[method]
            medical = (
                f"{row['medical_advice_yes']}/{row['medical_advice_no']}/"
                f"{row['medical_advice_unsure']}"
            )
            lines.append(
                f"{latex_escape(display[method])} & {row['N_evaluated']}/{row['N_expected']} & "
                f"{row['overall_quality_mean']:.3f} & {row['empathy_mean']:.3f} & "
                f"{row['specificity_mean']:.3f} & {row['quality_3axis_mean']:.3f} & "
                f"{medical} & {row['technical_or_parse_failures']} \\\\"
            )
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                f"\\caption{{{caption}}}",
                f"\\label{{tab:{name}_llama}}",
                r"\end{table*}",
            ]
        )
        value = "\n".join(lines) + "\n"
        (ROOT / f"publication/{name}_llama.tex").write_text(value, encoding="utf-8")
        return value

    main_llama_tex = llama_table(
        "main_baselines",
        main_order,
        "Auxiliary Llama TEST60 main-baseline evaluation; failures are excluded rather than counted as ties.",
    )
    ablation_llama_tex = llama_table(
        "component_ablations",
        ablation_order,
        "Auxiliary Llama TEST60 component evaluation; failures are excluded rather than counted as ties.",
    )

    final_output_rows = read_jsonl(FINAL_OUTPUTS)
    response_lengths = []
    for method in sorted({row["method"] for row in final_output_rows}):
        counts = [
            len(str(row["response"]).split())
            for row in final_output_rows
            if row["method"] == method
        ]
        response_lengths.append(
            {
                "method": method,
                "N": len(counts),
                "mean_words": statistics.mean(counts),
                "median_words": statistics.median(counts),
                "min_words": min(counts),
                "max_words": max(counts),
            }
        )
    length_by_method = {row["method"]: row for row in response_lengths}
    write_jsonl(ROOT / "tables/response_length_summary.jsonl", response_lengths)
    write_csv(
        ROOT / "tables/response_length_summary.csv",
        response_lengths,
        list(response_lengths[0].keys()),
    )
    length_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Mean words & Median words & Quality-3 \\",
        r"\midrule",
    ]
    for method in main_order:
        row = length_by_method[method]
        length_lines.append(
            f"{latex_escape(display[method])} & {row['mean_words']:.1f} & "
            f"{row['median_words']:.1f} & {gpt[method]['quality_3axis_mean']:.3f} \\\\"
        )
    length_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Response length diagnostic for the TEST60 main methods. "
            r"Length is not a quality metric and is reported to expose possible verbosity confounding in automatic evaluation.}",
            r"\label{tab:test60_response_length}",
            r"\end{table}",
        ]
    )
    length_tex = "\n".join(length_lines) + "\n"
    (ROOT / "publication/main_baseline_response_lengths.tex").write_text(
        length_tex, encoding="utf-8"
    )

    coverage = []
    for method in sorted(gpt):
        coverage.append(
            {
                "method": method,
                "role": next(row["role"] for row in read_jsonl(FINAL_OUTPUTS) if row["method"] == method),
                "GPT41_N": gpt[method]["N_evaluated"],
                "Llama_N": llama.get(method, {}).get("N_evaluated"),
                "accepted_or_active": pipeline[method]["candidate_accepted"],
                "changed_vs_SFT": pipeline[method]["text_changed_vs_sft"],
                "mean_full_scorer_risk": risk[method]["mean_risk"],
            }
        )
    write_jsonl(ROOT / "tables/comparison_coverage.jsonl", coverage)
    write_csv(ROOT / "tables/comparison_coverage.csv", coverage, list(coverage[0].keys()))

    component_configuration = [
        {
            "condition": "Proposed",
            "generation_adapter": "trained Mask-OFF denoiser",
            "zt_strategy": "no_mask",
            "router": "on",
            "scorer": "on",
            "gate_threshold": 0.7215308056776166,
            "candidate_calls": 18,
            "selector": "GPT two-order consensus",
            "accepted": pipeline["Proposed"]["candidate_accepted"],
        },
        {
            "condition": "GatedNoSelector",
            "generation_adapter": "trained Mask-OFF denoiser",
            "zt_strategy": "no_mask",
            "router": "on",
            "scorer": "on",
            "gate_threshold": 0.7215308056776166,
            "candidate_calls": 18,
            "selector": "accept generated",
            "accepted": pipeline["GatedNoSelector"]["candidate_accepted"],
        },
        {
            "condition": "AlwaysOnGPTSelector",
            "generation_adapter": "trained Mask-OFF denoiser",
            "zt_strategy": "no_mask",
            "router": "on",
            "scorer": "on",
            "gate_threshold": -1.0,
            "candidate_calls": 60,
            "selector": "GPT two-order consensus",
            "accepted": pipeline["AlwaysOnGPTSelector"]["candidate_accepted"],
        },
        {
            "condition": "OriginalPromptSFTControlGPTSelector",
            "generation_adapter": "SFT adapter",
            "zt_strategy": "same stored no-mask prompt/input",
            "router": "on",
            "scorer": "on",
            "gate_threshold": 0.7215308056776166,
            "candidate_calls": 18,
            "selector": "GPT two-order consensus",
            "accepted": pipeline["OriginalPromptSFTControlGPTSelector"]["candidate_accepted"],
        },
        {
            "condition": "MaskONGPTSelector",
            "generation_adapter": "trained Mask-ON denoiser",
            "zt_strategy": "staged_risk",
            "router": "on",
            "scorer": "on",
            "gate_threshold": 0.7215308056776166,
            "candidate_calls": integrity["component_arms"]["mask_on"]["denoiser_called"],
            "selector": "GPT two-order consensus",
            "accepted": pipeline["MaskONGPTSelector"]["candidate_accepted"],
        },
        {
            "condition": "WithoutRouterGPTSelector",
            "generation_adapter": "trained without-Router denoiser",
            "zt_strategy": "staged_risk",
            "router": "off",
            "scorer": "on",
            "gate_threshold": 0.948689877986908,
            "candidate_calls": integrity["component_arms"]["without_router"]["denoiser_called"],
            "selector": "GPT two-order consensus",
            "accepted": pipeline["WithoutRouterGPTSelector"]["candidate_accepted"],
        },
        {
            "condition": "WithoutScorerGPTSelector",
            "generation_adapter": "trained without-Scorer denoiser",
            "zt_strategy": "no_mask",
            "router": "on",
            "scorer": "off",
            "gate_threshold": 0.7560610771169199,
            "candidate_calls": integrity["component_arms"]["without_scorer"]["denoiser_called"],
            "selector": "GPT two-order consensus",
            "accepted": pipeline["WithoutScorerGPTSelector"]["candidate_accepted"],
        },
    ]
    write_jsonl(ROOT / "tables/component_configuration_and_counts.jsonl", component_configuration)
    write_csv(
        ROOT / "tables/component_configuration_and_counts.csv",
        component_configuration,
        list(component_configuration[0].keys()),
    )

    gpt_pairs = read_jsonl(ROOT / "tables/gpt41_final_all_pairs.jsonl")
    llama_pairs = read_jsonl(ROOT / "tables/llama_final_all_pairs.jsonl")
    gpt_candidate_pairs = read_jsonl(ROOT / "tables/gpt41_candidate_all_pairs.jsonl")
    llama_candidate_pairs = read_jsonl(ROOT / "tables/llama_candidate_all_pairs.jsonl")
    def comparison(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any] | None:
        for row in rows:
            if row["left"] == left and row["right"] == right:
                return row
            if row["left"] == right and row["right"] == left:
                return {
                    **row,
                    "left": left,
                    "right": right,
                    "right_win": row["right_loss"],
                    "right_loss": row["right_win"],
                    "quality_3axis_mean_delta": -row["quality_3axis_mean_delta"],
                    "bootstrap_95ci_low": -row["bootstrap_95ci_high"],
                    "bootstrap_95ci_high": -row["bootstrap_95ci_low"],
                }
        return None
    key_comparisons = []
    for judge, rows in (("GPT-4.1", gpt_pairs), ("Llama", llama_pairs)):
        for left, right, claim in (
            ("BasePromptRewrite", "SFT", "SFT fine-tuning contribution"),
            ("BasePromptRewrite", "Proposed", "full method versus prompt-only base"),
            ("SFT", "DPO", "DPO objective contribution"),
            ("DPO", "Proposed", "full method versus DPO"),
            ("SFT", "Proposed", "full method versus SFT"),
            ("GatedNoSelector", "Proposed", "GPT selector contribution"),
            ("AlwaysOnGPTSelector", "Proposed", "gate contribution with fixed selector"),
            ("OriginalPromptSFTControlGPTSelector", "Proposed", "denoiser additional-training contribution"),
            ("MaskONGPTSelector", "Proposed", "masking choice"),
            ("WithoutRouterGPTSelector", "Proposed", "router contribution"),
            ("WithoutScorerGPTSelector", "Proposed", "scorer contribution"),
        ):
            row = comparison(rows, left, right)
            if row:
                key_comparisons.append({"judge": judge, "claim": claim, **row})
    for judge, rows in (("GPT-4.1", gpt_candidate_pairs), ("Llama", llama_candidate_pairs)):
        row = comparison(rows, "OriginalPromptSFTControlCandidate", "OriginalDenoiserCandidate")
        if row:
            key_comparisons.append(
                {
                    "judge": judge,
                    "claim": "raw candidate effect of denoiser additional training at fixed prompt/input",
                    **row,
                }
            )
    for judge in {row["judge"] for row in key_comparisons}:
        indexed = [
            (index, row["two_sided_sign_test_p"])
            for index, row in enumerate(key_comparisons)
            if row["judge"] == judge and row.get("two_sided_sign_test_p") is not None
        ]
        indexed.sort(key=lambda item: item[1])
        running = 0.0
        total = len(indexed)
        for rank, (index, value) in enumerate(indexed, start=1):
            running = max(running, min(1.0, value * (total - rank + 1)))
            key_comparisons[index]["sign_test_p_holm_within_judge"] = running
    write_jsonl(ROOT / "tables/key_component_comparisons.jsonl", key_comparisons)
    write_csv(ROOT / "tables/key_component_comparisons.csv", key_comparisons, list(key_comparisons[0].keys()))

    gpt_scored_rows = [
        row
        for row in read_jsonl(ROOT / "results/gpt41_assignment_scores.jsonl")
        if row["scope"] == "TEST60_final_policy_output" and row["evaluation_ok"]
    ]
    gpt_scored_by_key = {
        (row["method"], row["canonical_id"]): row for row in gpt_scored_rows
    }
    final_by_key = {
        (row["method"], row["canonical_id"]): row for row in read_jsonl(FINAL_OUTPUTS)
    }
    example_specs = (
        ("BasePromptRewrite", "SFT", "SFT training"),
        ("BasePromptRewrite", "Proposed", "full method versus prompt-only base"),
        ("SFT", "DPO", "preference optimization"),
        ("DPO", "Proposed", "full method versus DPO"),
        ("SFT", "Proposed", "full method"),
        ("GatedNoSelector", "Proposed", "selector"),
        ("AlwaysOnGPTSelector", "Proposed", "call gate"),
        ("OriginalPromptSFTControlGPTSelector", "Proposed", "denoiser training"),
        ("MaskONGPTSelector", "Proposed", "masking choice"),
        ("WithoutRouterGPTSelector", "Proposed", "router"),
        ("WithoutScorerGPTSelector", "Proposed", "scorer"),
    )
    component_examples = []
    for left, right, claim in example_specs:
        candidates = []
        left_ids = {row_id for method, row_id in gpt_scored_by_key if method == left}
        right_ids = {row_id for method, row_id in gpt_scored_by_key if method == right}
        for row_id in sorted(left_ids & right_ids):
            left_output = final_by_key[(left, row_id)]
            right_output = final_by_key[(right, row_id)]
            if left_output["response"] == right_output["response"]:
                continue
            left_score = gpt_scored_by_key[(left, row_id)]
            right_score = gpt_scored_by_key[(right, row_id)]
            candidates.append(
                {
                    "claim": claim,
                    "left_method": left,
                    "right_method": right,
                    "canonical_id": row_id,
                    "case_id": left_output["case_id"],
                    "question": left_output["question"],
                    "left_response": left_output["response"],
                    "right_response": right_output["response"],
                    "left_quality_3axis_mean": left_score["quality_3axis_mean"],
                    "right_quality_3axis_mean": right_score["quality_3axis_mean"],
                    "right_minus_left": right_score["quality_3axis_mean"]
                    - left_score["quality_3axis_mean"],
                    "left_scores": {
                        axis: left_score[axis]
                        for axis in ("overall_quality", "empathy", "specificity", "medical_advice")
                    },
                    "right_scores": {
                        axis: right_score[axis]
                        for axis in ("overall_quality", "empathy", "specificity", "medical_advice")
                    },
                }
            )
        if not candidates:
            continue
        candidates.sort(key=lambda row: (row["right_minus_left"], row["canonical_id"]))
        selected_examples = [("minimum_delta", candidates[0])]
        if candidates[-1]["canonical_id"] != candidates[0]["canonical_id"]:
            selected_examples.append(("maximum_delta", candidates[-1]))
        for example_type, row in selected_examples:
            component_examples.append({"example_type": example_type, **row})
    write_jsonl(ROOT / "tables/key_component_actual_examples.jsonl", component_examples)

    def quote_markdown(value: str) -> list[str]:
        return [f"> {line}" if line else ">" for line in str(value).splitlines()]

    example_lines = [
        "# Representative TEST60 component cases",
        "",
        "Cases are selected mechanically as the minimum and maximum GPT-4.1 Quality-3 deltas among text-changing pairs for each prespecified comparison. They are illustrations, not additional statistical evidence.",
    ]
    for row in component_examples:
        example_lines.extend(
            [
                "",
                f"## {row['claim']}: {row['example_type']}",
                "",
                f"- ID: `{row['case_id']}` / `{row['canonical_id']}`",
                f"- Comparison: `{row['left_method']}` to `{row['right_method']}`",
                f"- Quality-3: {row['left_quality_3axis_mean']:.3f} to "
                f"{row['right_quality_3axis_mean']:.3f} "
                f"(delta {row['right_minus_left']:+.3f})",
                "",
                "**Question**",
                "",
                *quote_markdown(row["question"]),
                "",
                f"**{row['left_method']}**",
                "",
                *quote_markdown(row["left_response"]),
                "",
                f"**{row['right_method']}**",
                "",
                *quote_markdown(row["right_response"]),
            ]
        )
    (ROOT / "publication/KEY_COMPONENT_EXAMPLES.md").write_text(
        "\n".join(example_lines) + "\n", encoding="utf-8"
    )

    api_rows = []
    for path in (
        ROOT / "results/new_selector_results.jsonl",
        ROOT / "results/adapter_pairwise_gpt41_new_results.jsonl",
        ROOT / "results/new_pointwise_results.jsonl",
    ):
        for row in read_jsonl(path):
            if row.get("ok") is True:
                api_rows.append(row)
    api_usage = {
        "new_successful_requests": len(api_rows),
        "prompt_tokens": sum(int((row.get("usage") or {}).get("prompt_tokens") or 0) for row in api_rows),
        "completion_tokens": sum(int((row.get("usage") or {}).get("completion_tokens") or 0) for row in api_rows),
        "estimated_cost_usd": round(sum(float(row.get("estimated_cost_usd") or 0.0) for row in api_rows), 8),
        "selector_successful_requests": sum(row.get("parser") == "blind_selector" for row in api_rows),
        "pointwise_successful_requests": sum(row.get("parser") == "pointwise_expert_rubric" for row in api_rows),
    }
    write_json(ROOT / "results/api_usage_and_cost.json", api_usage)
    adapter_pairwise = read_jsonl(ROOT / "tables/adapter_pairwise_summary.jsonl")

    def markdown_quality(order: list[str]) -> list[str]:
        lines = [
            "| Method | N | Overall | Empathy | Specificity | Quality-3 | Med. yes/no/unsure | Changed |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for method in order:
            row = gpt[method]
            lines.append(
                f"| {display[method]} | {row['N_evaluated']} | {row['overall_quality_mean']:.3f} | "
                f"{row['empathy_mean']:.3f} | {row['specificity_mean']:.3f} | "
                f"{row['quality_3axis_mean']:.3f} | {row['medical_advice_yes']}/"
                f"{row['medical_advice_no']}/{row['medical_advice_unsure']} | "
                f"{pipeline[method]['text_changed_vs_sft']} |"
            )
        return lines

    def interpretation(row: dict[str, Any]) -> str:
        low = row.get("bootstrap_95ci_low")
        high = row.get("bootstrap_95ci_high")
        delta = row.get("quality_3axis_mean_delta")
        if delta is None:
            return "unavailable"
        if low is not None and low > 0:
            direction = "right method higher; bootstrap CI excludes zero"
        elif high is not None and high < 0:
            direction = "right method lower; bootstrap CI excludes zero"
        else:
            direction = "observed difference is uncertain; bootstrap CI includes zero"
        return f"delta={delta:+.3f}, 95% bootstrap CI [{low:+.3f}, {high:+.3f}], {direction}"

    base_vs_proposed_report = comparison(gpt_pairs, "BasePromptRewrite", "Proposed")
    selector_report = comparison(gpt_pairs, "GatedNoSelector", "Proposed")
    always_on_report = comparison(gpt_pairs, "AlwaysOnGPTSelector", "Proposed")
    scorer_report = comparison(gpt_pairs, "WithoutScorerGPTSelector", "Proposed")

    report_lines = [
        "# TEST60 reviewer-facing baseline and ablation matrix",
        "",
        "This is a post-hoc exploratory analysis on the frozen corrected TEST60 outputs. No training, response editing, threshold tuning, or prompt tuning was performed after observing TEST results.",
        "",
        "## Paper organization",
        "",
        "- Main baselines: BasePromptRewrite, SFT, and DPO.",
        "- Main method: Proposed (trained denoiser, frozen call gate, and two-order GPT selector consensus).",
        "- Component ablations: selector off, always-on generation, original-prompt SFT-adapter control, Mask ON, without Router, and without Scorer.",
        "- Unsafe and stored reference responses are diagnostic bounds, not deployable baselines.",
        "",
        "## GPT-4.1 main baselines",
        "",
        *markdown_quality(main_order),
        "",
        "The prompt-only base is the strongest internal baseline in this GPT-4.1 evaluation. "
        f"For Proposed minus prompt-only base, {interpretation(base_vs_proposed_report)}. "
        "The paper must report this result directly rather than presenting only SFT/DPO-relative gains.",
        f"Mean response length is {length_by_method['BasePromptRewrite']['mean_words']:.1f} words for the "
        f"prompt-only base versus {length_by_method['Proposed']['mean_words']:.1f} for Proposed. Length is not "
        "a quality measure, but the difference is a possible automatic-judge verbosity confound and is reported separately.",
        "",
        "## GPT-4.1 component ablations",
        "",
        *markdown_quality(ablation_order),
        "",
        "The selector improves the gated raw-candidate output "
        f"({interpretation(selector_report)}). In contrast, the always-on + selector and without-Scorer "
        f"conditions exceed the full method ({interpretation(always_on_report)} for full minus always-on; "
        f"{interpretation(scorer_report)} for full minus without-Scorer). Thus this post-hoc TEST60 analysis "
        "does not support claiming that every component independently improves GPT-judged quality.",
        "",
        "## Key paired tests",
        "",
        *[
            f"- {row['judge']} / {row['claim']}: N={row['N']}, W/T/L="
            f"{row['right_win']}/{row['tie']}/{row['right_loss']}; {interpretation(row)}; "
            f"sign-test p={row['two_sided_sign_test_p']}, Holm p="
            f"{row.get('sign_test_p_holm_within_judge')}."
            for row in key_comparisons
        ],
        "",
        "## Direct adapter-only pairwise check",
        "",
        *[
            f"- {row['judge']}: original denoiser/control/tie/order-disagreement/technical-failure = "
            f"{row['original_denoiser_wins']}/{row['sft_adapter_control_wins']}/"
            f"{row['ties']}/{row['order_disagreements']}/{row['technical_failures']} (N={row['N']})."
            for row in adapter_pairwise
        ],
        "",
        "## Independence limitation",
        "",
        "GPT-4.1-2025-04-14 is used both by the Proposed selector and by the primary automatic pointwise evaluation. Llama-3.1-8B-Instruct is reported as an auxiliary different-judge check; the separate first-15 counselor packet remains the human evaluation channel.",
        "",
        "## Residual limitations",
        "",
        "- All added TEST60 comparisons are post-hoc exploratory and must not be described as validation-selected confirmatory results.",
        "- Existing trained checkpoints are single-seed. No retraining or test-driven threshold selection was performed in this run.",
        "- Router/scorer-removal arms retain their own previously frozen VALID-selected thresholds; the threshold differences are reported in the run contract.",
        "- No external method was newly reimplemented after seeing TEST60. Older exp295 outputs use a different split and cannot be inserted as if they were TEST60 baselines.",
        "- The prompt-only base is substantially longer on average; GPT-4.1 pointwise preference may partly reflect verbosity. Independent blinded human ratings are needed before a substantive superiority claim.",
        "",
        "## Coverage",
        "",
        "See `tables/comparison_coverage.csv`, `tables/key_component_comparisons.csv`, and the per-judge all-pairs files for exact N, paired deltas, win/tie/loss counts, bootstrap confidence intervals, and sign-test p-values.",
        "",
        "## API usage",
        "",
        f"New successful paid requests: {api_usage['new_successful_requests']}; prompt tokens: "
        f"{api_usage['prompt_tokens']}; completion tokens: {api_usage['completion_tokens']}; "
        f"estimated cost: ${api_usage['estimated_cost_usd']:.4f}. Exact cached requests have zero added cost.",
    ]
    (ROOT / "ACL_COMPONENT_ABLATION_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Keep an insertion-ready paper draft in Markdown so the table source,
    # interpretation, and experiment-lineage warnings travel together.
    def paired_result(left: str, right: str, judge: str = "GPT-4.1") -> dict[str, Any]:
        rows = gpt_pairs if judge == "GPT-4.1" else llama_pairs
        result = comparison(rows, left, right)
        if result is None:
            raise RuntimeError(f"Missing paired comparison: {judge}/{left}/{right}")
        return result

    base_vs_proposed = paired_result("BasePromptRewrite", "Proposed")
    sft_vs_proposed = paired_result("SFT", "Proposed")
    dpo_vs_proposed = paired_result("DPO", "Proposed")
    no_selector_vs_proposed = paired_result("GatedNoSelector", "Proposed")
    always_on_vs_proposed = paired_result("AlwaysOnGPTSelector", "Proposed")
    control_vs_proposed = paired_result("OriginalPromptSFTControlGPTSelector", "Proposed")
    mask_on_vs_proposed = paired_result("MaskONGPTSelector", "Proposed")
    without_router_vs_proposed = paired_result("WithoutRouterGPTSelector", "Proposed")
    without_scorer_vs_proposed = paired_result("WithoutScorerGPTSelector", "Proposed")

    def latex_delta(row: dict[str, Any]) -> str:
        return (
            f"$\\Delta={row['quality_3axis_mean_delta']:+.3f}$, "
            f"95\\% CI $[{row['bootstrap_95ci_low']:+.3f}, {row['bootstrap_95ci_high']:+.3f}]$, "
            f"W/T/L={row['right_win']}/{row['tie']}/{row['right_loss']}"
        )

    key_comparison_index = {
        (row["judge"], row["left"], row["right"]): row for row in key_comparisons
    }

    def holm_p(judge: str, left: str, right: str) -> str:
        row = key_comparison_index[(judge, left, right)]
        value = row.get("sign_test_p_holm_within_judge")
        return "--" if value is None else f"{value:.4g}"

    paired_specs = (
        ("BasePromptRewrite", "SFT", "SFT training"),
        ("BasePromptRewrite", "Proposed", "full method versus prompt-only base"),
        ("SFT", "DPO", "preference optimization"),
        ("DPO", "Proposed", "full method versus DPO"),
        ("SFT", "Proposed", "full method"),
        ("GatedNoSelector", "Proposed", "selector"),
        ("AlwaysOnGPTSelector", "Proposed", "call gate"),
        ("OriginalPromptSFTControlGPTSelector", "Proposed", "denoiser training"),
        ("MaskONGPTSelector", "Proposed", "masking choice"),
        ("WithoutRouterGPTSelector", "Proposed", "router"),
        ("WithoutScorerGPTSelector", "Proposed", "scorer"),
    )

    paired_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Comparison & Claim & $N$ & $\Delta$ Quality-3 & W/T/L & 95\% bootstrap CI & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for left, right, claim in paired_specs:
        row = paired_result(left, right)
        paired_lines.append(
            f"{latex_escape(display[left])} $\\rightarrow$ {latex_escape(display[right])} & "
            f"{latex_escape(claim)} & {row['N']} & {row['quality_3axis_mean_delta']:+.3f} & "
            f"{row['right_win']}/{row['tie']}/{row['right_loss']} & "
            f"[{row['bootstrap_95ci_low']:+.3f}, {row['bootstrap_95ci_high']:+.3f}] & "
            f"{holm_p('GPT-4.1', left, right)} \\\\"
        )
    paired_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Paired GPT-4.1 component comparisons on the frozen corrected TEST60 set. "
            r"Deltas are right minus left. W/T/L is from the right-hand method's perspective.}",
            r"\label{tab:test60_component_paired}",
            r"\end{table*}",
        ]
    )
    paired_tex = "\n".join(paired_lines) + "\n"
    (ROOT / "publication/key_paired_comparisons_gpt41.tex").write_text(paired_tex, encoding="utf-8")

    llama_paired_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Comparison & Claim & $N$ & $\Delta$ Quality-3 & W/T/L & 95\% bootstrap CI & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for left, right, claim in paired_specs:
        row = paired_result(left, right, "Llama")
        llama_paired_lines.append(
            f"{latex_escape(display[left])} $\\rightarrow$ {latex_escape(display[right])} & "
            f"{latex_escape(claim)} & {row['N']} & {row['quality_3axis_mean_delta']:+.3f} & "
            f"{row['right_win']}/{row['tie']}/{row['right_loss']} & "
            f"[{row['bootstrap_95ci_low']:+.3f}, {row['bootstrap_95ci_high']:+.3f}] & "
            f"{holm_p('Llama', left, right)} \\\\"
        )
    llama_paired_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Paired auxiliary Llama component comparisons on the frozen corrected TEST60 set. "
            r"Each row uses the available common scored cases; parse failures are excluded rather than counted as ties. "
            r"Deltas are right minus left, and W/T/L is from the right-hand method's perspective.}",
            r"\label{tab:test60_component_paired_llama}",
            r"\end{table*}",
        ]
    )
    llama_paired_tex = "\n".join(llama_paired_lines) + "\n"
    (ROOT / "publication/key_paired_comparisons_llama.tex").write_text(
        llama_paired_tex, encoding="utf-8"
    )

    adapter_pairwise_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Judge & $N$ & Denoiser & SFT-adapter & Tie & Order disagree & Fail. \\",
        r"\midrule",
    ]
    for row in adapter_pairwise:
        adapter_pairwise_lines.append(
            f"{latex_escape(row['judge'])} & {row['N']} & {row['original_denoiser_wins']} & "
            f"{row['sft_adapter_control_wins']} & {row['ties']} & {row['order_disagreements']} & "
            f"{row['technical_failures']} \\\\"
        )
    adapter_pairwise_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Blind two-order pairwise comparison of the original denoiser candidate and the SFT-adapter "
            r"control generated from the identical rendered rewrite input. Order disagreements are reported separately "
            r"and are not counted as ties.}",
            r"\label{tab:test60_adapter_pairwise}",
            r"\end{table}",
        ]
    )
    adapter_pairwise_tex = "\n".join(adapter_pairwise_lines) + "\n"
    (ROOT / "publication/adapter_only_pairwise.tex").write_text(
        adapter_pairwise_tex, encoding="utf-8"
    )

    aligned_summary_path = Path(
        "/home/user/hsoh/rubric_aligned_gpt41_rejudge_20261009_v1/RESULTS_SUMMARY.json"
    )
    aligned_summary = read_json(aligned_summary_path)
    valid43 = aligned_summary["headline_quality"]["GPT-4.1"]["VALID43_final"]
    valid43_rows = [
        ("SFT", "SFT"),
        ("DPO", "DPO"),
        ("original_denoiser__current_rules", "Denoiser + surface-relaxed rules"),
        ("original_denoiser__new_4item_selector", "Denoiser + four-item GPT selector"),
    ]
    valid43_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
        r"Method & $N$ & Overall & Empathy & Specificity & Quality-3 & Med. Y/N/U \\",
        r"\midrule",
    ]
    for key, label in valid43_rows:
        row = valid43[key]
        counts = row["medical_advice_counts"]
        valid43_lines.append(
            f"{latex_escape(label)} & {row['N_scored']} & {row['means']['overall_quality']:.3f} & "
            f"{row['means']['empathy']:.3f} & {row['means']['specificity']:.3f} & "
            f"{row['quality_3axis_mean']:.3f} & {counts['yes']}/{counts['no']}/{counts['unsure']} \\\\"
        )
    valid43_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{GPT-4.1 development-set results on VALID43 under the expert-aligned rubric. "
            r"This table is reported separately from TEST60.}",
            r"\label{tab:valid43_development}",
            r"\end{table}",
        ]
    )
    valid43_tex = "\n".join(valid43_lines) + "\n"
    (ROOT / "publication/valid43_development_gpt41.tex").write_text(valid43_tex, encoding="utf-8")

    aligned_quality_path = (
        aligned_summary_path.parent / "aggregated/method_quality_summary.jsonl"
    )
    aligned_quality_rows = read_jsonl(aligned_quality_path)
    aligned_quality = {
        (row["judge"], row["scope"], row["method"]): row
        for row in aligned_quality_rows
    }

    valid9_candidate_methods = [
        ("SFT", "SFT draft"),
        ("original_denoiser", "Original denoiser"),
        ("feedback_A", "Feedback A: candidate rewrite"),
        ("feedback_B", "Feedback B: SFT + feedback, denoiser adapter"),
        ("feedback_C", "Feedback C: SFT + feedback, SFT adapter"),
        ("original_prompt_sft_control", "Original prompt, SFT adapter"),
    ]
    valid9_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrrll}",
        r"\toprule",
        r"Judge & Candidate & Scored & Overall & Empathy & Specificity & Quality-3 & Med. Y/N/U & Fail. \\",
        r"\midrule",
    ]
    for judge in ("GPT-4.1", "Llama"):
        for method, label in valid9_candidate_methods:
            row = aligned_quality[(judge, "VALID9_candidate_raw", method)]
            counts = row["medical_advice_counts"]
            valid9_lines.append(
                f"{latex_escape(judge)} & {latex_escape(label)} & "
                f"{row['N_scored']}/{row['N_expected']} & {row['means']['overall_quality']:.3f} & "
                f"{row['means']['empathy']:.3f} & {row['means']['specificity']:.3f} & "
                f"{row['quality_3axis_mean']:.3f} & {counts['yes']}/{counts['no']}/{counts['unsure']} & "
                f"{row['parse_or_missing_failures']} \\\\"
            )
        if judge == "GPT-4.1":
            valid9_lines.append(r"\midrule")
    valid9_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Expert-rubric-aligned pointwise quality of the frozen VALID9 raw candidates. "
            r"Candidate quality is shown before any acceptance policy. Medical Advice is categorical and excluded from Quality-3.}",
            r"\label{tab:valid9_candidate_ablation}",
            r"\end{table*}",
        ]
    )
    valid9_tex = "\n".join(valid9_lines) + "\n"
    (ROOT / "publication/valid9_candidate_ablation_aligned.tex").write_text(
        valid9_tex, encoding="utf-8"
    )

    valid43_feedback_methods = [
        ("SFT", "SFT"),
        ("original_denoiser__new_4item_selector", "Original denoiser + selector"),
        ("feedback_A__new_4item_selector", "Feedback A + selector"),
        ("feedback_B__new_4item_selector", "Feedback B + selector"),
        ("feedback_C__new_4item_selector", "Feedback C + selector"),
        ("original_prompt_sft_control__new_4item_selector", "Original prompt, SFT adapter + selector"),
    ]
    valid43_feedback_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrrl}",
        r"\toprule",
        r"Method & $N$ & Adopted & Overall & Empathy & Specificity & Quality-3 & Med. Y/N/U \\",
        r"\midrule",
    ]
    for method, label in valid43_feedback_methods:
        row = aligned_quality[("GPT-4.1", "VALID43_final", method)]
        counts = row["medical_advice_counts"]
        accepted = "--" if row.get("accepted") is None else str(row["accepted"])
        valid43_feedback_lines.append(
            f"{latex_escape(label)} & {row['N_scored']} & {accepted} & "
            f"{row['means']['overall_quality']:.3f} & {row['means']['empathy']:.3f} & "
            f"{row['means']['specificity']:.3f} & {row['quality_3axis_mean']:.3f} & "
            f"{counts['yes']}/{counts['no']}/{counts['unsure']} \\\\"
        )
    valid43_feedback_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Frozen VALID43 final-output comparison under the four-item GPT selector. "
            r"Only the nine routed development cases can change; all other items retain SFT.}",
            r"\label{tab:valid43_feedback_final}",
            r"\end{table*}",
        ]
    )
    valid43_feedback_tex = "\n".join(valid43_feedback_lines) + "\n"
    (ROOT / "publication/valid43_feedback_final_gpt41.tex").write_text(
        valid43_feedback_tex, encoding="utf-8"
    )

    proposed = gpt["Proposed"]
    sft = gpt["SFT"]
    dpo = gpt["DPO"]
    base = gpt["BasePromptRewrite"]
    method_paragraph = (
        r"\paragraph{Baselines and component controls.} "
        r"We compare the full system against a prompt-only base-model rewrite, supervised fine-tuning "
        r"(SFT), and direct preference optimization (DPO). The full system first produces an SFT draft, "
        r"invokes the trained Mask-OFF denoiser only when the frozen risk gate exceeds 0.7215308056776166, "
        r"and adopts a candidate only when a frozen GPT-4.1 selector chooses it in both anonymized response "
        r"orders. We separately ablate the call gate, the selector, denoiser-specific training at an identical "
        r"rendered prompt and token input, risk-span masking, the router, and the scorer. Router/scorer ablations "
        r"retain their own validation-selected thresholds; hence those rows are complete historical component "
        r"configurations rather than one-variable threshold interventions."
    )
    evaluation_paragraph = (
        r"\paragraph{Automatic evaluation.} "
        r"For each question--response pair, GPT-4.1 (temperature 0) assigns Overall Quality, Empathy, and "
        r"Specificity on 1--5 scales. Their arithmetic mean is Quality-3. Medical Advice is recorded separately "
        r"as yes/no/unsure and is never converted to or averaged as a safety score. Exact duplicate pairs share "
        r"one evaluation. Llama-3.1-8B-Instruct provides an auxiliary evaluation under the same rubric. Because "
        r"GPT-4.1 is also the final selector, GPT pointwise results are not an independent validation of selection; "
        r"the Llama analysis and the separately prepared blinded counselor study are reported as complementary evidence."
    )
    results_paragraph = (
        r"\paragraph{TEST60 results.} "
        f"On the frozen corrected TEST60 set, Quality-3 was {base['quality_3axis_mean']:.3f} for the prompt-only "
        f"base model, {sft['quality_3axis_mean']:.3f} for SFT, {dpo['quality_3axis_mean']:.3f} for DPO, and "
        f"{proposed['quality_3axis_mean']:.3f} for the full method. The prompt-only base was the strongest internal "
        f"baseline: for full minus prompt-only, {latex_delta(base_vs_proposed)}. Relative to SFT, the full method "
        f"yielded {latex_delta(sft_vs_proposed)}, and relative to DPO it yielded {latex_delta(dpo_vs_proposed)}. "
        f"The prompt-only and full-method responses averaged {length_by_method['BasePromptRewrite']['mean_words']:.1f} "
        f"and {length_by_method['Proposed']['mean_words']:.1f} words, respectively; response length is not itself "
        r"a quality metric, but this difference is reported as a potential automatic-judge verbosity confound. "
        f"Turning off final selection gave "
        f"{gpt['GatedNoSelector']['quality_3axis_mean']:.3f} Quality-3; restoring the selector changed it by "
        f"{no_selector_vs_proposed['quality_3axis_mean_delta']:+.3f}. The always-on condition with the same "
        f"selector scored {gpt['AlwaysOnGPTSelector']['quality_3axis_mean']:.3f} "
        f"({latex_delta(always_on_vs_proposed)} for full minus always-on), while the identical-prompt SFT-adapter "
        f"control scored {gpt['OriginalPromptSFTControlGPTSelector']['quality_3axis_mean']:.3f} "
        f"({latex_delta(control_vs_proposed)} for trained-denoiser minus control). Masking, Router, and Scorer "
        f"comparisons were respectively {latex_delta(mask_on_vs_proposed)}, "
        f"{latex_delta(without_router_vs_proposed)}, and {latex_delta(without_scorer_vs_proposed)} for full minus "
        r"the corresponding ablation. The selector contribution is positive here, but its within-judge Holm-adjusted "
        f"sign-test value is $p={holm_p('GPT-4.1', 'GatedNoSelector', 'Proposed')}$. These results do not support "
        r"claiming that every architectural component independently improves GPT-judged quality. These TEST comparisons are "
        r"post-hoc exploratory; no threshold, prompt, checkpoint, or response was tuned after observing TEST outcomes."
    )
    llama_base_vs_proposed = paired_result("BasePromptRewrite", "Proposed", "Llama")
    llama_sft_vs_proposed = paired_result("SFT", "Proposed", "Llama")
    llama_no_selector_vs_proposed = paired_result("GatedNoSelector", "Proposed", "Llama")
    llama_always_on_vs_proposed = paired_result("AlwaysOnGPTSelector", "Proposed", "Llama")
    llama_without_scorer_vs_proposed = paired_result("WithoutScorerGPTSelector", "Proposed", "Llama")
    llama_paragraph = (
        r"\paragraph{Auxiliary Llama evaluation.} "
        f"The independent local judge assigned Quality-3={llama['BasePromptRewrite']['quality_3axis_mean']:.3f} "
        f"to the prompt-only baseline on {llama['BasePromptRewrite']['N_evaluated']}/60 parseable cases and "
        f"{llama['Proposed']['quality_3axis_mean']:.3f} to the full method on "
        f"{llama['Proposed']['N_evaluated']}/60. On common scored cases, full minus prompt-only was "
        f"{latex_delta(llama_base_vs_proposed)}, while full minus SFT was "
        f"{latex_delta(llama_sft_vs_proposed)}. Restoring selection over the gated raw candidate changed Quality-3 by "
        f"{llama_no_selector_vs_proposed['quality_3axis_mean_delta']:+.3f}; however, full minus always-on selection "
        f"was {llama_always_on_vs_proposed['quality_3axis_mean_delta']:+.3f}, and full minus the no-Scorer arm was "
        f"{llama_without_scorer_vs_proposed['quality_3axis_mean_delta']:+.3f}. Thus Llama supports the modest "
        r"selector-over-no-selector effect but agrees with GPT-4.1 that the prompt-only baseline dominates and that "
        r"the call gate and Scorer are not positively supported by this post-hoc TEST60 matrix. Parse failures are "
        r"excluded with the actual denominator reported for every comparison."
    )
    lineage_paragraph = (
        r"\paragraph{Experiment lineage and non-comparable prior analyses.} "
        r"Earlier TEST24 and preliminary TEST60 summaries are superseded by the corrected TEST60 artifact, including "
        r"the TEST-40 extraction repair. A nine-case VALID feedback study and its A/B/C adapter study remain development "
        r"ablations; they used an earlier local numeric four-axis judge and must not be pooled with the expert-aligned "
        r"Quality-3/Medical-Advice results. That study identified a specificity-loss false acceptance for "
        r"\texttt{qa\_24f1b77a40c9b22203b8f337} and an SFT fallback for "
        r"\texttt{qa\_b243b4c887feb8e3594dd74c}; both are retained as failure analyses. The earlier eight-case "
        r"original-prompt SFT-adapter control is superseded by the current eighteen-case identical-input comparison."
    )
    valid9_gpt_sft = aligned_quality[("GPT-4.1", "VALID9_candidate_raw", "SFT")]
    valid9_gpt_b = aligned_quality[("GPT-4.1", "VALID9_candidate_raw", "feedback_B")]
    valid9_llama_b = aligned_quality[("Llama", "VALID9_candidate_raw", "feedback_B")]
    valid43_gpt_b = aligned_quality[
        ("GPT-4.1", "VALID43_final", "feedback_B__new_4item_selector")
    ]
    development_paragraph = (
        r"\paragraph{Development-only feedback ablation.} "
        f"On the nine routed VALID cases, the frozen Feedback-B raw candidates increased GPT-4.1 Quality-3 "
        f"from {valid9_gpt_sft['quality_3axis_mean']:.3f} for the SFT drafts to "
        f"{valid9_gpt_b['quality_3axis_mean']:.3f}. The auxiliary Llama score for Feedback B was "
        f"{valid9_llama_b['quality_3axis_mean']:.3f} on {valid9_llama_b['N_scored']}/"
        f"{valid9_llama_b['N_expected']} parseable cases. After the frozen four-item selector and SFT fallback "
        f"on all 43 validation questions, Feedback B obtained Quality-3="
        f"{valid43_gpt_b['quality_3axis_mean']:.3f} with {valid43_gpt_b['accepted']} adopted revisions. "
        r"These small development comparisons motivated failure analysis rather than TEST-time policy tuning; "
        r"in particular, the accepted Feedback-A output for \texttt{qa\_24f1b77a40c9b22203b8f337} lost "
        r"question-specific detail."
    )

    guide_lines = [
        "# ACL LaTeX insertion guide",
        "",
        "This document keeps insertion-ready LaTeX together with the rules for using it. The current corrected TEST60 table is the primary automatic result. Do not pool scores across different splits, rubrics, or pre/post TEST-40 artifacts.",
        "",
        "## Recommended paper placement",
        "",
        "- Main Results: `main_baselines_gpt41.tex` plus the TEST60 results paragraph.",
        "- Length diagnostic: `main_baseline_response_lengths.tex`; report it because the prompt-only baseline is substantially more verbose.",
        "- Ablation subsection: `component_ablations_gpt41.tex` and `key_paired_comparisons_gpt41.tex`.",
        "- Appendix: `policy_sensitivity_gpt41.tex`, `main_baselines_llama.tex`, `component_ablations_llama.tex`, `key_paired_comparisons_llama.tex`, `adapter_only_pairwise.tex`, and prior VALID feedback failure analysis.",
        "- Representative text-changing cases: `KEY_COMPONENT_EXAMPLES.md` (mechanically selected minimum/maximum deltas, not cherry-picked evidence).",
        "- Do not present UnsafeResponse or ReferenceResponse as deployable baselines.",
        "",
        "## Reviewer-question coverage",
        "",
        "| Likely reviewer question | Evidence to cite | Placement |",
        "|---|---|---|",
        "| Is fine-tuning needed beyond prompting? | BasePromptRewrite vs SFT | Main baseline table |",
        "| Is preference optimization sufficient? | SFT vs DPO vs Proposed | Main baseline table |",
        "| Is the gain only from a second rewrite prompt? | OriginalPromptSFTControl vs trained denoiser with identical prompt/token input | Ablation + direct pairwise appendix |",
        "| Does selective calling help? | AlwaysOnGPTSelector vs Proposed | Ablation |",
        "| Does final selection help? | GatedNoSelector vs Proposed | Ablation |",
        "| Are masking, Router, and Scorer useful? | MaskON, WithoutRouter, WithoutScorer vs Proposed | Ablation |",
        "| Are conclusions tied to one acceptance heuristic? | legacy strict, surface-relaxed, no-min-length, risk-only, accept-generated, GPT selector | Appendix sensitivity table |",
        "| Is the GPT-selected result judged by the same model? | Explicit limitation, auxiliary Llama table, blinded counselor packet | Evaluation/Limitations |",
        "| Is uncertainty reported? | Paired W/T/L, bootstrap CI, sign test with within-judge Holm correction | Paired-results table |",
        "",
        "Residual risk: these are internal baselines. A reviewer can still request a genuinely comparable published external method; no such method was retrofitted after seeing TEST60 because the stored exp295 runs use a different split and would not be a fair TEST60 row.",
        "",
        "## Method paragraph",
        "",
        "```latex",
        method_paragraph,
        "```",
        "",
        "## Evaluation paragraph",
        "",
        "```latex",
        evaluation_paragraph,
        "```",
        "",
        "## Main baseline table",
        "",
        "```latex",
        (ROOT / "publication/main_baselines_gpt41.tex").read_text(encoding="utf-8").rstrip(),
        "```",
        "",
        "## Main-baseline response length diagnostic",
        "",
        "Length is not included in Quality-3. This table is a diagnostic for possible automatic-judge verbosity bias.",
        "",
        "```latex",
        length_tex.rstrip(),
        "```",
        "",
        "## Results paragraph",
        "",
        "```latex",
        results_paragraph,
        "```",
        "",
        "## Component ablation table",
        "",
        "```latex",
        (ROOT / "publication/component_ablations_gpt41.tex").read_text(encoding="utf-8").rstrip(),
        "```",
        "",
        "## Paired component tests",
        "",
        "```latex",
        paired_tex.rstrip(),
        "```",
        "",
        "## Auxiliary Llama main baseline table",
        "",
        "The local judge is a complementary check because GPT-4.1 serves as both selector and primary evaluator. Parse failures are retained as failures and excluded from scored denominators.",
        "",
        "```latex",
        main_llama_tex.rstrip(),
        "```",
        "",
        "## Auxiliary Llama component ablation table",
        "",
        "```latex",
        ablation_llama_tex.rstrip(),
        "```",
        "",
        "## Auxiliary Llama paired component tests",
        "",
        "```latex",
        llama_paired_tex.rstrip(),
        "```",
        "",
        "## Auxiliary judge paragraph",
        "",
        "```latex",
        llama_paragraph,
        "```",
        "",
        "## Direct adapter-only pairwise table",
        "",
        "This isolates generation adapter choice at the same rendered prompt and token input. The two-order comparison reports disagreements rather than forcing them into ties.",
        "",
        "```latex",
        adapter_pairwise_tex.rstrip(),
        "```",
        "",
        "## Policy sensitivity appendix table",
        "",
        "```latex",
        (ROOT / "publication/policy_sensitivity_gpt41.tex").read_text(encoding="utf-8").rstrip(),
        "```",
        "",
        "## Prior expert-aligned VALID43 development table",
        "",
        "This is the earlier completed development-set result under the same categorical Medical Advice rubric. Keep it separate from TEST60 and do not relabel the historical `Mask_OFF_current` artifact as surface-relaxed; the table below uses the explicitly named replay outputs.",
        "",
        "```latex",
        valid43_tex.rstrip(),
        "```",
        "",
        "## Prior VALID9 candidate ablation under the aligned rubric",
        "",
        "This table separates raw candidate quality from policy-selected final quality. Llama parse failures retain the actual scored denominator.",
        "",
        "```latex",
        valid9_tex.rstrip(),
        "```",
        "",
        "## Prior VALID43 feedback/control final outputs",
        "",
        "These rows use the frozen four-item selector and the full VALID43 denominator. They remain development results and must not be merged with TEST60.",
        "",
        "```latex",
        valid43_feedback_tex.rstrip(),
        "```",
        "",
        "## Development ablation paragraph",
        "",
        "```latex",
        development_paragraph,
        "```",
        "",
        "## Prior completed experiments and failure analysis",
        "",
        "```latex",
        lineage_paragraph,
        "```",
        "",
        "The bundled `prior_evidence/` reports preserve the exact earlier results. They are provenance and appendix material, not extra rows to merge into TEST60.",
        "",
        "## Mandatory caveats",
        "",
        "- TEST60 component comparisons are post-hoc exploratory and single-seed.",
        "- GPT-4.1 serves as both selector and primary pointwise evaluator; this is disclosed explicitly.",
        "- Medical Advice is categorical and excluded from Quality-3.",
        "- The original local VALID feedback report used a different rubric; only the explicitly aligned rejudge tables above share the current rubric.",
        "- Known stored generation/factual failures remain in the frozen data and are listed in `results/KNOWN_SOURCE_ISSUES.json`.",
    ]
    guide_path = ROOT / "publication/ACL_LATEX_INSERTION_GUIDE.md"
    guide_path.write_text("\n".join(guide_lines) + "\n", encoding="utf-8")

    prior_evidence = ROOT / "prior_evidence"
    prior_evidence.mkdir(parents=True, exist_ok=True)
    prior_files = {
        "valid9_feedback_v1_report.md": Path(
            "/home/user/hsoh/mh-denoise-feedback-v1-valid-20261008/outputs/valid_feedback_v1_20261008/REPORT.md"
        ),
        "valid9_feedback_v1_summary.json": Path(
            "/home/user/hsoh/mh-denoise-feedback-v1-valid-20261008/outputs/valid_feedback_v1_20261008/summary.json"
        ),
        "valid9_feedback_BC_report.md": Path("/mnt/ssd00/hsoh/feedback_v1_bc_20261009/REPORT_BC.md"),
        "valid9_feedback_BC_summary.json": Path("/mnt/ssd00/hsoh/feedback_v1_bc_20261009/summary_bc.json"),
        "original_prompt_sft_control_report.md": Path(
            "/mnt/ssd00/hsoh/original_prompt_sft_control_20261009_v1/ORIGINAL_PROMPT_SFT_CONTROL_REPORT.md"
        ),
        "original_prompt_sft_control_summary.json": Path(
            "/mnt/ssd00/hsoh/original_prompt_sft_control_20261009_v1/SUMMARY.json"
        ),
        "corrected_test60_pre_matrix_report.md": Path(
            "/home/user/hsoh/test60_corrected_gpt41_pointwise_publication_20261009_v1/FINAL_REPORT.md"
        ),
        "selector_off_pre_matrix_report.md": Path(
            "/home/user/hsoh/test60_gated_denoiser_no_selector_ablation_20261009_v1/FINAL_REPORT.md"
        ),
        "rubric_aligned_rejudge_report.md": Path(
            "/home/user/hsoh/rubric_aligned_gpt41_rejudge_20261009_v1/RUBRIC_ALIGNED_REJUDGE_REPORT.md"
        ),
        "rubric_aligned_rejudge_summary.json": aligned_summary_path,
    }
    missing_prior = []
    for name, source in prior_files.items():
        if source.exists():
            shutil.copy2(source, prior_evidence / name)
        else:
            missing_prior.append(str(source))
    write_json(
        prior_evidence / "README.json",
        {
            "purpose": "Previously completed evidence preserved separately from the current corrected TEST60 matrix",
            "do_not_pool_across_rubrics_or_splits": True,
            "copied": sorted(path.name for path in prior_evidence.iterdir() if path.is_file()),
            "missing": missing_prior,
        },
    )

    code_files = (
        REPO / "scripts/prepare_acl_component_ablation_matrix.py",
        REPO / "scripts/run_test60_base_prompt_rewrite.py",
        REPO / "scripts/run_test60_original_prompt_sft_control.py",
        REPO / "scripts/run_acl_component_ablation_inference_tmux.sh",
        REPO / "scripts/run_acl_component_selector.py",
        REPO / "scripts/run_acl_component_pointwise.py",
        REPO / "scripts/run_acl_component_llama_pointwise.py",
        REPO / "scripts/run_acl_adapter_pairwise.py",
        REPO / "scripts/run_acl_component_postprocess_tmux.sh",
        REPO / "scripts/verify_acl_component_generation.py",
    )
    for path in code_files:
        shutil.copy2(path, ROOT / "code" / path.name)
    shutil.copy2(FROZEN / "KNOWN_SOURCE_ISSUES.json", ROOT / "results/KNOWN_SOURCE_ISSUES.json")
    bundle = ROOT / "ACL_TEST60_baseline_component_ablation_research_bundle.zip"
    include_roots = [
        ROOT / "run_contract.json",
        ROOT / "run_contract.sha256",
        ROOT / "selector_contract.json",
        ROOT / "pointwise_contract.json",
        ROOT / "llama_pointwise_contract.json",
        ROOT / "adapter_pairwise_contract.json",
        ROOT / "ACL_COMPONENT_ABLATION_REPORT.md",
        ROOT / "inputs",
        ROOT / "generation",
        ROOT / "requests",
        ROOT / "results",
        ROOT / "risk",
        ROOT / "tables",
        ROOT / "publication",
        ROOT / "prior_evidence",
        ROOT / "logs",
        ROOT / "code",
    ]
    inventory_paths = []
    for source in include_roots:
        if not source.exists():
            continue
        inventory_paths.extend(
            [source] if source.is_file() else sorted(path for path in source.rglob("*") if path.is_file())
        )
    inventory = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(inventory_paths))
    ]
    write_json(
        ROOT / "BUNDLE_CONTENT_MANIFEST.json",
        {"version": VERSION, "files": len(inventory), "artifacts": inventory},
    )
    include_roots.append(ROOT / "BUNDLE_CONTENT_MANIFEST.json")
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source in include_roots:
            if not source.exists():
                continue
            paths = [source] if source.is_file() else sorted(path for path in source.rglob("*") if path.is_file())
            for path in paths:
                archive.write(path, path.relative_to(ROOT))
    download_copy = Path("/home/user/Downloads") / bundle.name
    shutil.copy2(bundle, download_copy)
    guide_download_copy = Path("/home/user/Downloads") / guide_path.name
    shutil.copy2(guide_path, guide_download_copy)
    manifest = {
        "version": VERSION,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "bundle": {"path": str(bundle), "bytes": bundle.stat().st_size, "sha256": sha256_file(bundle)},
        "server_download_copy": {
            "path": str(download_copy),
            "bytes": download_copy.stat().st_size,
            "sha256": sha256_file(download_copy),
        },
        "coverage": {"methods": len(coverage), "questions": 60},
        "api_usage": api_usage,
        "artifacts": {
            "gpt41_quality": str(ROOT / "tables/gpt41_final_quality_summary.csv"),
            "llama_quality": str(ROOT / "tables/llama_final_quality_summary.csv"),
            "component_comparisons": str(ROOT / "tables/key_component_comparisons.csv"),
            "coverage": str(ROOT / "tables/comparison_coverage.csv"),
            "latex_markdown_guide": str(guide_path),
            "latex_markdown_download_copy": str(guide_download_copy),
        },
    }
    write_json(ROOT / "FINAL_MANIFEST.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "aggregate", "score-risk", "package"])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "aggregate":
        aggregate()
    elif args.command == "score-risk":
        score_risk()
    else:
        build_tables_and_package()


if __name__ == "__main__":
    main()
