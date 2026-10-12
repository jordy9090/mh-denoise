#!/usr/bin/env python3
"""Preserved request preparation and aggregation for the frozen TEST60 matrix.

Research-only risk rescoring, example export, and archive packaging are omitted.
No model or API invocation is performed by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "acl-component-ablation-matrix-v1-20261009"
ROOT = Path("outputs/acl_component_matrix")
FROZEN = Path("outputs/rubric_aligned_evaluation")
MODEL = "gpt-4.1-2025-04-14"
QUEUE = "acl_component_pointwise"
FINAL_OUTPUTS = ROOT / "results/final_method_outputs.jsonl"
CANDIDATE_OUTPUTS = ROOT / "results/candidate_quality_outputs.jsonl"
CACHE_ROOTS: tuple[Path, ...] = ()
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


def main() -> None:
    global ROOT, FROZEN, CACHE_ROOTS, FINAL_OUTPUTS, CANDIDATE_OUTPUTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "aggregate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--frozen-contract-dir", type=Path)
    parser.add_argument("--cache-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.command == "prepare" and args.frozen_contract_dir is None:
        parser.error("prepare requires --frozen-contract-dir")
    ROOT = args.root
    if args.frozen_contract_dir is not None:
        FROZEN = args.frozen_contract_dir
    CACHE_ROOTS = tuple(args.cache_root)
    FINAL_OUTPUTS = ROOT / "results/final_method_outputs.jsonl"
    CANDIDATE_OUTPUTS = ROOT / "results/candidate_quality_outputs.jsonl"
    {"prepare": prepare, "aggregate": aggregate}[args.command]()


if __name__ == "__main__":
    main()
