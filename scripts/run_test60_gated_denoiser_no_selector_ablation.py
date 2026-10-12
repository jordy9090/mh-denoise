#!/usr/bin/env python3
"""Build and evaluate the frozen TEST60 gated-denoiser/no-selector ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/home/user/hsoh/mh-denoise")
sys.path.insert(0, str(REPO / "scripts"))
import run_corrected_test60_pointwise_publication as common


VERSION = "test60-gated-denoiser-no-selector-ablation-v1-20261009"
QUEUE = "test60_gated_denoiser_no_selector_missing"
MODEL = "gpt-4.1-2025-04-14"
METHOD = "GatedDenoiser_NoSelector"
SOURCE_ROOT = Path("/home/user/hsoh/test60_rubric_aligned_final_selection_20261009_v2_test40_fix")
SOURCE = SOURCE_ROOT / "test60_final_outputs.jsonl"
PRIOR = Path("/home/user/hsoh/test60_corrected_gpt41_pointwise_publication_20261009_v1")
FROZEN = Path("/home/user/hsoh/rubric_aligned_gpt41_rejudge_20261009_v1")
ROOT = Path("/home/user/hsoh/test60_gated_denoiser_no_selector_ablation_20261009_v1")


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


def file_meta(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = rows
    return result


def request_id(system_prompt: str, user_prompt: str, metadata: dict[str, Any]) -> str:
    material = json.dumps(
        {
            "version": VERSION,
            "queue": QUEUE,
            "model": MODEL,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "metadata": metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(material)[:32]


def make_request(
    pair: dict[str, Any], system_prompt: str, user_prompt: str, max_tokens: int
) -> dict[str, Any]:
    metadata = {
        "question_response_sha256": pair["question_response_sha256"],
        "question_sha256": pair["question_sha256"],
        "response_sha256": pair["response_sha256"],
        "members": pair["members"],
        "priority": "TEST60_no_selector_ablation",
    }
    return {
        "version": VERSION,
        "queue": QUEUE,
        "request_id": request_id(system_prompt, user_prompt, metadata),
        "model": MODEL,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "parser": "pointwise_expert_rubric",
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_sha256": sha256_text(user_prompt),
        "metadata": metadata,
    }


def frozen_source() -> list[dict[str, Any]]:
    rows = sorted(read_jsonl(SOURCE), key=lambda row: int(row["source_order"]))
    if len(rows) != 60 or [int(row["source_order"]) for row in rows] != list(range(60)):
        raise RuntimeError("Corrected TEST60 source is not the frozen ordered 60-row set")
    if len({row["canonical_id"] for row in rows}) != 60:
        raise RuntimeError("Corrected TEST60 contains duplicate canonical IDs")
    return rows


def build_arm(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outputs: list[dict[str, Any]] = []
    pairs: dict[str, dict[str, Any]] = {}
    for row in rows:
        called = bool(row["denoiser_called"])
        if called and not str(row.get("denoiser_response") or "").strip():
            raise RuntimeError(f"Called case lacks a frozen candidate: {row['canonical_id']}")
        response = str(row["denoiser_response"] if called else row["sft_response"])
        key = common.pair_hash(str(row["question"]), response)
        output = {
            "source_order": int(row["source_order"]),
            "canonical_id": row["canonical_id"],
            "case_id": row["case_id"],
            "question": row["question"],
            "method": METHOD,
            "gate_called": called,
            "final_source": "frozen_denoiser_candidate" if called else "SFT_gate_off",
            "final_response": response,
            "question_sha256": sha256_text(str(row["question"])),
            "response_sha256": sha256_text(response),
            "question_response_sha256": key,
            "sft_response_sha256": sha256_text(str(row["sft_response"])),
            "candidate_response_sha256": sha256_text(str(row.get("denoiser_response") or "")) if called else None,
            "final_text_differs_from_sft": response != str(row["sft_response"]),
            "risk_score": row["denoiser_risk_score"] if called else row["sft_risk_score"],
            "sft_risk_score": row["sft_risk_score"],
        }
        outputs.append(output)
        member = {
            "split": "TEST",
            "scope": "corrected_TEST60_gated_denoiser_no_selector",
            "source_order": int(row["source_order"]),
            "canonical_id": row["canonical_id"],
            "case_id": row["case_id"],
            "condition": METHOD,
        }
        pairs[key] = {
            "question_response_sha256": key,
            "question": row["question"],
            "response": response,
            "question_sha256": sha256_text(str(row["question"])),
            "response_sha256": sha256_text(response),
            "members": [member],
        }
    return outputs, list(pairs.values())


def cache_files() -> tuple[list[Path], list[Path]]:
    request_files = [
        FROZEN / "requests/expert15_pointwise.jsonl",
        FROZEN / "requests/remaining_pointwise.jsonl",
        PRIOR / "requests/reused_pointwise_requests.jsonl",
        PRIOR / "requests/new_pointwise_requests.jsonl",
    ]
    result_files = [
        FROZEN / "results/expert15_pointwise_gpt41.jsonl",
        FROZEN / "results/remaining_pointwise_gpt41.jsonl",
        PRIOR / "results/reused_pointwise_results.jsonl",
        PRIOR / "results/new_pointwise_results.jsonl",
    ]
    return request_files, result_files


def prepare() -> None:
    rows = frozen_source()
    outputs, pairs = build_arm(rows)
    ROOT.mkdir(parents=True, exist_ok=True)
    write_jsonl(ROOT / "inputs/no_selector_final_outputs.jsonl", outputs)
    write_jsonl(ROOT / "inputs/no_selector_unique_pairs.jsonl", pairs)

    contract = read_json(FROZEN / "run_contract.json")
    system_prompt = contract["prompts"]["pointwise_system"]
    user_template = contract["prompts"]["pointwise_user_template"]
    max_tokens = int(contract["model_settings"]["pointwise_max_tokens"])
    if contract["model_settings"]["model"] != MODEL or max_tokens != 384:
        raise RuntimeError("Frozen pointwise contract differs from the corrected TEST60 run")

    request_files, result_files = cache_files()
    requests_by_pair: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path in request_files:
        for request in read_jsonl(path):
            key = str((request.get("metadata") or {}).get("question_response_sha256") or "")
            if key:
                requests_by_pair[key].append((path, request))
    results_by_id: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in result_files:
        for result in read_jsonl(path):
            results_by_id[str(result.get("request_id"))] = (path, result)

    reused_requests: list[dict[str, Any]] = []
    reused_results: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for pair in pairs:
        expected_user = user_template.format(question=pair["question"], response=pair["response"])
        selected = None
        for request_path, request in requests_by_pair.get(pair["question_response_sha256"], []):
            matches, _ = common.request_matches_contract(
                request, pair, system_prompt, expected_user, max_tokens
            )
            if not matches:
                continue
            result_record = results_by_id.get(str(request["request_id"]))
            if result_record is None:
                continue
            result_path, result = result_record
            valid, _ = common.result_is_valid(result, request, pair["response"])
            if valid:
                selected = (request_path, request, result_path, result)
                break
        if selected is None:
            missing.append(pair)
            continue
        request_path, request, result_path, result = selected
        reused_requests.append(request)
        reused_results.append(result)
        provenance.append(
            {
                "question_response_sha256": pair["question_response_sha256"],
                "request_id": request["request_id"],
                "source_request_file": str(request_path),
                "source_result_file": str(result_path),
            }
        )

    new_requests = []
    for pair in missing:
        user_prompt = user_template.format(question=pair["question"], response=pair["response"])
        new_requests.append(make_request(pair, system_prompt, user_prompt, max_tokens))
    new_requests.sort(key=lambda row: min(member["source_order"] for member in row["metadata"]["members"]))
    reused_requests.sort(key=lambda row: row["request_id"])
    reused_results.sort(key=lambda row: row["request_id"])
    provenance.sort(key=lambda row: row["question_response_sha256"])
    write_jsonl(ROOT / "requests/reused_pointwise_requests.jsonl", reused_requests)
    write_jsonl(ROOT / "results/reused_pointwise_results.jsonl", reused_results)
    write_jsonl(ROOT / "cache/reuse_provenance.jsonl", provenance)
    write_jsonl(ROOT / "requests/new_pointwise_requests.jsonl", new_requests)

    counts = {
        "questions": 60,
        "gate_called": sum(row["gate_called"] for row in outputs),
        "gate_off_sft": sum(not row["gate_called"] for row in outputs),
        "final_text_changes_vs_sft": sum(row["final_text_differs_from_sft"] for row in outputs),
        "reused_exact_evaluations": len(reused_results),
        "new_evaluations": len(new_requests),
    }
    expected = {
        "questions": 60,
        "gate_called": 18,
        "gate_off_sft": 42,
        "final_text_changes_vs_sft": 18,
        "reused_exact_evaluations": 54,
        "new_evaluations": 6,
    }
    if counts != expected:
        raise RuntimeError(f"Unexpected frozen no-selector coverage: {counts}")

    run_contract = {
        "version": VERSION,
        "status": "frozen_before_api_execution",
        "created_at_utc": utc_now(),
        "method_name": METHOD,
        "definition": (
            "Keep the corrected frozen call gate. On the 18 called cases, use the stored technically valid "
            "original trained-denoiser candidate without GPT final selection. On 42 gate-off cases, retain SFT."
        ),
        "classification": "claim-aligned selector ablation, not an external baseline and not always-on generation",
        "scope": {
            "questions": 60,
            "no_generation": True,
            "no_training": True,
            "no_gate_change": True,
            "no_candidate_change": True,
            "no_selector_call": True,
            "no_expert_packet_change": True,
        },
        "pointwise_evaluation": {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "parser": "pointwise_expert_rubric",
            "system_prompt": system_prompt,
            "user_template": user_template,
            "system_prompt_sha256": sha256_text(system_prompt),
            "user_template_sha256": sha256_text(user_template),
            "api_input": "question and anonymous response only",
        },
        "sources": {
            "corrected_test60": file_meta(SOURCE, 60),
            "corrected_test60_contract": file_meta(SOURCE_ROOT / "run_contract.json"),
            "frozen_pointwise_contract": file_meta(FROZEN / "run_contract.json"),
            "corrected_pointwise_bundle": file_meta(PRIOR / "completion_summary.json"),
        },
        "frozen_outputs": file_meta(ROOT / "inputs/no_selector_final_outputs.jsonl", 60),
        "request_queue": file_meta(ROOT / "requests/new_pointwise_requests.jsonl", len(new_requests)),
        "counts": counts,
        "paid_api_authorized_under_existing_gpt41_evaluation_scope": True,
        "post_hoc_exploratory": True,
    }
    contract_path = ROOT / "run_contract.json"
    if contract_path.exists():
        existing = read_json(contract_path)
        for key in ("version", "method_name", "definition", "pointwise_evaluation", "frozen_outputs", "counts"):
            if existing.get(key) != run_contract.get(key):
                raise RuntimeError(f"Existing contract differs at {key}")
    else:
        write_json(contract_path, run_contract)
        (ROOT / "run_contract.sha256").write_text(
            sha256_file(contract_path) + "  run_contract.json\n", encoding="utf-8"
        )
    print(json.dumps({"status": "prepared", **counts}, sort_keys=True))


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def aggregate() -> None:
    source = frozen_source()
    outputs = read_jsonl(ROOT / "inputs/no_selector_final_outputs.jsonl")
    pairs = read_jsonl(ROOT / "inputs/no_selector_unique_pairs.jsonl")
    requests = read_jsonl(ROOT / "requests/reused_pointwise_requests.jsonl") + read_jsonl(
        ROOT / "requests/new_pointwise_requests.jsonl"
    )
    results = read_jsonl(ROOT / "results/reused_pointwise_results.jsonl") + read_jsonl(
        ROOT / "results/new_pointwise_results.jsonl"
    )
    if len(outputs) != 60 or len(pairs) != 60:
        raise RuntimeError("No-selector frozen outputs are incomplete")

    frozen_contract = read_json(FROZEN / "run_contract.json")
    system_prompt = frozen_contract["prompts"]["pointwise_system"]
    user_template = frozen_contract["prompts"]["pointwise_user_template"]
    max_tokens = int(frozen_contract["model_settings"]["pointwise_max_tokens"])
    pair_map = {row["question_response_sha256"]: row for row in pairs}
    request_map = {row["request_id"]: row for row in requests}
    result_map = {row["request_id"]: row for row in results}
    score_map: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for request_id, request in request_map.items():
        key = request["metadata"]["question_response_sha256"]
        pair = pair_map[key]
        expected_user = user_template.format(question=pair["question"], response=pair["response"])
        matches, reason = common.request_matches_contract(
            request, pair, system_prompt, expected_user, max_tokens
        )
        if not matches:
            failures.append({"request_id": request_id, "pair": key, "reason": reason})
            continue
        result = result_map.get(request_id)
        if result is None:
            failures.append({"request_id": request_id, "pair": key, "reason": "missing_result"})
            continue
        valid, reason = common.result_is_valid(
            result, request, pair["response"], require_verbatim_evidence=False
        )
        if not valid:
            failures.append({"request_id": request_id, "pair": key, "reason": reason})
            continue
        normalized = result["normalized"]
        evidence = str(normalized.get("medical_advice_evidence") or "").strip()
        if normalized["medical_advice"] in {"yes", "unsure"} and evidence not in pair["response"]:
            warnings.append(
                {
                    "request_id": request_id,
                    "pair": key,
                    "reason": "nonverbatim_medical_advice_evidence",
                    "evidence": evidence,
                }
            )
        score_map[key] = result
    write_jsonl(ROOT / "results/pointwise_failures.jsonl", failures)
    write_jsonl(ROOT / "results/evidence_warnings.jsonl", warnings)
    if len(score_map) != 60:
        raise RuntimeError(f"Only {len(score_map)}/60 no-selector evaluations are valid")

    scored_arm: list[dict[str, Any]] = []
    output_by_id = {row["canonical_id"]: row for row in outputs}
    for row in outputs:
        result = score_map[row["question_response_sha256"]]
        normalized = result["normalized"]
        scores = normalized["scores"]
        scored_arm.append(
            {
                **row,
                "evaluation_ok": True,
                "request_id": result["request_id"],
                "overall_quality": scores["overall_quality"],
                "empathy": scores["empathy"],
                "specificity": scores["specificity"],
                "quality_3axis_mean": mean(list(scores.values())),
                "medical_advice": normalized["medical_advice"],
                "rationale": normalized["rationale"],
                "medical_advice_evidence": normalized["medical_advice_evidence"],
                "medical_advice_rationale": normalized["medical_advice_rationale"],
            }
        )
    write_jsonl(ROOT / "results/no_selector_assignment_scores.jsonl", scored_arm)

    prior_scores = read_jsonl(PRIOR / "results/method_assignment_scores.jsonl")
    combined = [row for row in prior_scores if row["method"] in {"SFT", "DPO", "Proposed"}]
    for row in scored_arm:
        combined.append(
            {
                "split": "TEST",
                "scope": "corrected_TEST60_final",
                "source_order": row["source_order"],
                "canonical_id": row["canonical_id"],
                "case_id": row["case_id"],
                "method": METHOD,
                "question": row["question"],
                "response": row["final_response"],
                "question_sha256": row["question_sha256"],
                "response_sha256": row["response_sha256"],
                "question_response_sha256": row["question_response_sha256"],
                "evaluation_ok": True,
                "request_id": row["request_id"],
                "overall_quality": row["overall_quality"],
                "empathy": row["empathy"],
                "specificity": row["specificity"],
                "quality_3axis_mean": row["quality_3axis_mean"],
                "medical_advice": row["medical_advice"],
                "rationale": row["rationale"],
                "medical_advice_evidence": row["medical_advice_evidence"],
                "medical_advice_rationale": row["medical_advice_rationale"],
            }
        )
    method_order = ["SFT", "DPO", METHOD, "Proposed"]
    combined.sort(key=lambda row: (int(row["source_order"]), method_order.index(row["method"])))
    write_jsonl(ROOT / "results/four_method_assignment_scores.jsonl", combined)

    summaries: list[dict[str, Any]] = []
    for method in method_order:
        rows = [row for row in combined if row["method"] == method]
        medical = Counter(row["medical_advice"] for row in rows)
        denominator = medical["yes"] + medical["no"]
        summaries.append(
            {
                "method": method,
                "N": len(rows),
                "overall_quality_mean": mean([row["overall_quality"] for row in rows]),
                "empathy_mean": mean([row["empathy"] for row in rows]),
                "specificity_mean": mean([row["specificity"] for row in rows]),
                "quality_3axis_mean": mean([row["quality_3axis_mean"] for row in rows]),
                "medical_advice_yes": medical["yes"],
                "medical_advice_no": medical["no"],
                "medical_advice_unsure": medical["unsure"],
                "medical_advice_yes_rate": medical["yes"] / denominator if denominator else None,
                "medical_advice_yes_rate_denominator": denominator,
                "technical_or_parse_failures": 0,
            }
        )
    summary_fields = list(summaries[0])
    write_json(ROOT / "tables/test60_four_method_quality.json", summaries)
    write_csv(ROOT / "tables/test60_four_method_quality.csv", summaries, summary_fields)

    by_key = {(row["canonical_id"], row["method"]): row for row in combined}
    comparisons = []
    comparison_summaries = []
    for left, right in ((METHOD, "SFT"), ("Proposed", METHOD), ("Proposed", "SFT")):
        counts = Counter()
        deltas = []
        for row in source:
            left_row = by_key[(row["canonical_id"], left)]
            right_row = by_key[(row["canonical_id"], right)]
            delta = left_row["quality_3axis_mean"] - right_row["quality_3axis_mean"]
            outcome = "win" if delta > 1e-12 else "loss" if delta < -1e-12 else "tie"
            counts[outcome] += 1
            deltas.append(delta)
            comparisons.append(
                {
                    "source_order": row["source_order"],
                    "canonical_id": row["canonical_id"],
                    "case_id": row["case_id"],
                    "left_method": left,
                    "right_method": right,
                    "left_quality_3axis_mean": left_row["quality_3axis_mean"],
                    "right_quality_3axis_mean": right_row["quality_3axis_mean"],
                    "left_minus_right": delta,
                    "outcome_for_left": outcome,
                }
            )
        comparison_summaries.append(
            {
                "left_method": left,
                "right_method": right,
                "N": 60,
                "mean_delta": mean(deltas),
                "win": counts["win"],
                "tie": counts["tie"],
                "loss": counts["loss"],
            }
        )
    write_jsonl(ROOT / "tables/paired_quality_comparisons.jsonl", comparisons)
    write_json(ROOT / "tables/paired_quality_summary.json", comparison_summaries)

    prior_risk = read_json(PRIOR / "tables/test60_pipeline_and_risk.json")
    prior_risk_by_method = {row["method"]: row for row in prior_risk}
    no_selector_risks = [float(row["risk_score"]) for row in outputs]
    no_selector_reductions = [float(row["sft_risk_score"]) - float(row["risk_score"]) for row in outputs]
    pipeline = []
    for method in method_order:
        if method == METHOD:
            pipeline.append(
                {
                    "method": METHOD,
                    "N": 60,
                    "mean_scorer_risk": mean(no_selector_risks),
                    "mean_risk_reduction_vs_sft": mean(no_selector_reductions),
                    "denoiser_calls": 18,
                    "candidate_generation_success": 18,
                    "candidate_generation_failure": 0,
                    "final_candidate_adoptions": 18,
                    "final_text_changes_vs_sft": sum(row["final_text_differs_from_sft"] for row in outputs),
                }
            )
        else:
            prior = prior_risk_by_method[method]
            pipeline.append(
                {
                    "method": method,
                    "N": prior["N"],
                    "mean_scorer_risk": prior["mean_scorer_risk"],
                    "mean_risk_reduction_vs_sft": prior["mean_risk_reduction_vs_sft"],
                    "denoiser_calls": prior["denoiser_calls"],
                    "candidate_generation_success": prior["candidate_generation_success"],
                    "candidate_generation_failure": prior["candidate_generation_failure"],
                    "final_candidate_adoptions": prior["gpt_selector_adoptions"],
                    "final_text_changes_vs_sft": prior["final_text_changes_vs_sft"],
                }
            )
    write_json(ROOT / "tables/test60_four_method_pipeline_risk.json", pipeline)
    write_csv(ROOT / "tables/test60_four_method_pipeline_risk.csv", pipeline, list(pipeline[0]))

    publication = ROOT / "publication"
    publication.mkdir(parents=True, exist_ok=True)
    table = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{GPT-4.1 evaluation on corrected TEST60. Gated Denoiser (no selector) is a selector ablation, not an external baseline. Medical Advice is categorical and excluded from Quality-3.}",
        r"\label{tab:test60-four-method}",
        r"\small",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Method & N & Overall & Empathy & Specificity & Quality-3 & Med. y/n/u & Yes rate (denom.) \\",
        r"\midrule",
    ]
    display = {
        "SFT": "SFT",
        "DPO": "DPO",
        METHOD: "Gated Denoiser (no selector)",
        "Proposed": "Proposed",
    }
    for row in summaries:
        table.append(
            f"{display[row['method']]} & {row['N']} & {row['overall_quality_mean']:.3f} & "
            f"{row['empathy_mean']:.3f} & {row['specificity_mean']:.3f} & "
            f"{row['quality_3axis_mean']:.3f} & {row['medical_advice_yes']}/{row['medical_advice_no']}/{row['medical_advice_unsure']} & "
            f"{row['medical_advice_yes_rate']:.3f} ({row['medical_advice_yes_rate_denominator']}) \\\\"
        )
    table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (publication / "test60_four_method_quality_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )

    summary_by_method = {row["method"]: row for row in summaries}
    ns = summary_by_method[METHOD]
    proposed = summary_by_method["Proposed"]
    ns_vs_sft = next(row for row in comparison_summaries if row["left_method"] == METHOD)
    proposed_vs_ns = next(
        row for row in comparison_summaries if row["left_method"] == "Proposed" and row["right_method"] == METHOD
    )
    text = rf"""% Insertion-ready ablation text.
\paragraph{{Selector ablation.}}
To isolate final selection from candidate generation, we replayed a gated-denoiser/no-selector condition on the corrected frozen TEST60 set. The call gate and all stored model outputs were unchanged. The ablation used the stored trained-denoiser candidate on all 18 called cases and retained SFT on the 42 gate-off cases; it is therefore not an always-on generation baseline. No responses were regenerated. Under the same GPT-4.1 pointwise rubric, the no-selector condition obtained a Quality-3 mean of {ns['quality_3axis_mean']:.3f}, compared with {proposed['quality_3axis_mean']:.3f} for Proposed. Relative to SFT, the no-selector condition had {ns_vs_sft['win']} wins, {ns_vs_sft['tie']} ties, and {ns_vs_sft['loss']} losses. Proposed versus the no-selector condition had {proposed_vs_ns['win']} wins, {proposed_vs_ns['tie']} ties, and {proposed_vs_ns['loss']} losses. The no-selector condition's mean scorer risk was {mean(no_selector_risks):.3f}, with a mean reduction of {mean(no_selector_reductions):.3f} versus SFT over all 60 questions. This post-hoc ablation uses GPT-4.1 as the pointwise evaluator, while Proposed also used GPT-4.1 for final selection, so the comparison is not independent-judge validation.
"""
    (publication / "selector_ablation_text.tex").write_text(text, encoding="utf-8")

    new_request_ids = {row["request_id"] for row in read_jsonl(ROOT / "requests/new_pointwise_requests.jsonl")}
    new_results = {row["request_id"]: row for row in read_jsonl(ROOT / "results/new_pointwise_results.jsonl")}
    new_ok = [new_results[rid] for rid in new_request_ids if new_results.get(rid, {}).get("ok") is True]
    usage = {
        "reused_exact_evaluations": len(read_jsonl(ROOT / "results/reused_pointwise_results.jsonl")),
        "new_api_requests": len(new_request_ids),
        "new_api_success": len(new_ok),
        "new_api_failures": len(new_request_ids) - len(new_ok),
        "input_tokens": sum(int((row.get("usage") or {}).get("prompt_tokens") or 0) for row in new_ok),
        "output_tokens": sum(int((row.get("usage") or {}).get("completion_tokens") or 0) for row in new_ok),
        "estimated_cost_usd": round(sum(float(row.get("estimated_cost_usd") or 0) for row in new_ok), 8),
    }
    write_json(ROOT / "api_usage.json", usage)

    report = [
        "# TEST60 Gated Denoiser Without Selector Ablation",
        "",
        "Status: complete",
        "",
        "This is a selector ablation, not an external baseline. It keeps the frozen gate and uses every stored trained-denoiser candidate on the 18 called cases; the other 42 cases retain SFT.",
        "",
        "## Four-Method Result",
        "",
        "| Method | Overall | Empathy | Specificity | Quality-3 | Medical yes/no/unsure | Yes rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        report.append(
            f"| {display[row['method']]} | {row['overall_quality_mean']:.3f} | {row['empathy_mean']:.3f} | "
            f"{row['specificity_mean']:.3f} | {row['quality_3axis_mean']:.3f} | "
            f"{row['medical_advice_yes']}/{row['medical_advice_no']}/{row['medical_advice_unsure']} | "
            f"{row['medical_advice_yes_rate']:.3f} (N={row['medical_advice_yes_rate_denominator']}) |"
        )
    report.extend(
        [
            "",
            "## Paired Quality-3",
            "",
        ]
    )
    for row in comparison_summaries:
        report.append(
            f"- {display[row['left_method']]} vs {display[row['right_method']]}: "
            f"mean delta {row['mean_delta']:+.3f}; W/T/L {row['win']}/{row['tie']}/{row['loss']} (N=60)."
        )
    report.extend(
        [
            "",
            "## Execution",
            "",
            f"- Exact evaluations reused: {usage['reused_exact_evaluations']}.",
            f"- New GPT-4.1 evaluations: {usage['new_api_success']}; failures: {usage['new_api_failures']}.",
            f"- New estimated API cost: ${usage['estimated_cost_usd']:.6f}.",
            "- New response generation: 0; selector calls: 0.",
            "- The existing 15x3 expert packet was not changed. In its first 15 questions this arm adds only one response text not already represented, so a 15x4 packet would add substantial duplicate rating work.",
            "",
        ]
    )
    (ROOT / "FINAL_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    completion = {
        "version": VERSION,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "definition": read_json(ROOT / "run_contract.json")["definition"],
        "quality": summaries,
        "paired_comparisons": comparison_summaries,
        "pipeline_risk": pipeline,
        "api_usage": usage,
        "pointwise_failures": len(failures),
        "evidence_warnings": len(warnings),
    }
    write_json(ROOT / "completion_summary.json", completion)

    code_dir = ROOT / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
    shutil.copy2(REPO / "scripts/run_gpt41_checkpoint_queue.py", code_dir / "run_gpt41_checkpoint_queue.py")
    bundle = ROOT / "TEST60_four_method_selector_ablation_research_bundle.zip"
    excluded = {bundle, ROOT / "FINAL_MANIFEST.json", ROOT / "DELIVERY.json"}
    files = {
        str(path.relative_to(ROOT)): file_meta(path)
        for path in sorted(ROOT.rglob("*"))
        if path.is_file() and path not in excluded
    }
    write_json(
        ROOT / "FINAL_MANIFEST.json",
        {"version": VERSION, "status": "complete", "summary": completion, "files": files},
    )
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            if path.is_file() and path not in {bundle, ROOT / "DELIVERY.json"}:
                archive.write(path, path.relative_to(ROOT))
    write_json(
        ROOT / "DELIVERY.json",
        {
            "status": "complete",
            "bundle": file_meta(bundle),
            "report": file_meta(ROOT / "FINAL_REPORT.md"),
            "quality_csv": file_meta(ROOT / "tables/test60_four_method_quality.csv"),
            "latex_table": file_meta(publication / "test60_four_method_quality_table.tex"),
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "reused": usage["reused_exact_evaluations"],
                "new": usage["new_api_success"],
                "failed": usage["new_api_failures"],
                "bundle": str(bundle),
            },
            sort_keys=True,
        )
    )


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
