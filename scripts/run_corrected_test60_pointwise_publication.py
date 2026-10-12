#!/usr/bin/env python3
"""Freeze, score, and publish the corrected TEST60 GPT-4.1 evaluation."""

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


VERSION = "corrected-test60-gpt41-pointwise-publication-v1-20261009"
MODEL = "gpt-4.1-2025-04-14"
QUEUE = "corrected_test60_pointwise_missing"
REPO = Path("/workspace/mh-denoise")
SOURCE_ROOT = Path("/workspace/artifacts/test60_rubric_aligned_final_selection_20261009_v2_test40_fix")
SOURCE = SOURCE_ROOT / "test60_final_outputs.jsonl"
FROZEN_ROOT = Path("/workspace/artifacts/rubric_aligned_gpt41_rejudge_20261009_v1")
FROZEN_CONTRACT = FROZEN_ROOT / "run_contract.json"
ROOT = Path("/workspace/artifacts/test60_corrected_gpt41_pointwise_publication_20261009_v1")

OLD_REQUEST_FILES = [
    FROZEN_ROOT / "requests/expert15_pointwise.jsonl",
    FROZEN_ROOT / "requests/remaining_pointwise.jsonl",
]
OLD_RESULT_FILES = [
    FROZEN_ROOT / "results/expert15_pointwise_gpt41.jsonl",
    FROZEN_ROOT / "results/remaining_pointwise_gpt41.jsonl",
]

ROUTER_DIR = REPO / "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/router/final"
SCORER_DIR = REPO / "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/scorer/final"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def pair_hash(question: str, response: str) -> str:
    return sha256_text(question + "\0" + response)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(body, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_meta(path: Path, rows: int | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        item["rows"] = rows
    return item


def request_sha256(row: dict[str, Any]) -> str:
    selected = {
        key: row[key]
        for key in (
            "request_id",
            "model",
            "temperature",
            "max_tokens",
            "response_format",
            "parser",
            "system_prompt",
            "user_prompt",
            "metadata",
        )
    }
    material = json.dumps(selected, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_text(material)


def make_request_id(system_prompt: str, user_prompt: str, metadata: dict[str, Any]) -> str:
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


def result_is_valid(
    result: dict[str, Any],
    request: dict[str, Any],
    response: str,
    *,
    require_verbatim_evidence: bool = True,
) -> tuple[bool, str | None]:
    if result.get("ok") is not True:
        return False, "api_or_parse_failure"
    if result.get("request_sha256") != request_sha256(request):
        return False, "request_hash_mismatch"
    if result.get("returned_model") != MODEL:
        return False, "returned_model_mismatch"
    normalized = result.get("normalized") or {}
    scores = normalized.get("scores") or {}
    for axis in ("overall_quality", "empathy", "specificity"):
        value = scores.get(axis)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            return False, f"invalid_{axis}"
    medical = normalized.get("medical_advice")
    if medical not in {"yes", "no", "unsure"}:
        return False, "invalid_medical_advice"
    evidence = str(normalized.get("medical_advice_evidence") or "").strip()
    if medical in {"yes", "unsure"}:
        if not evidence:
            return False, "missing_medical_advice_evidence"
        if require_verbatim_evidence and evidence not in response:
            return False, "nonverbatim_medical_advice_evidence"
    if not str(normalized.get("medical_advice_rationale") or "").strip():
        return False, "missing_medical_advice_rationale"
    return True, None


def load_source_rows() -> list[dict[str, Any]]:
    rows = sorted(read_jsonl(SOURCE), key=lambda row: int(row["source_order"]))
    if len(rows) != 60:
        raise RuntimeError(f"Corrected TEST60 source has {len(rows)} rows, expected 60")
    if [int(row["source_order"]) for row in rows] != list(range(60)):
        raise RuntimeError("Corrected TEST60 source order is not exactly 0..59")
    ids = [str(row["canonical_id"]) for row in rows]
    if len(set(ids)) != 60:
        raise RuntimeError("Corrected TEST60 has duplicate canonical IDs")
    for row in rows:
        for field in ("question", "sft_response", "dpo_response", "final_response"):
            if not str(row.get(field) or "").strip():
                raise RuntimeError(f"Empty {field} for {row['canonical_id']}")
    return rows


def build_assignments(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    method_fields = (("SFT", "sft_response"), ("DPO", "dpo_response"), ("Proposed", "final_response"))
    assignments: list[dict[str, Any]] = []
    representatives: dict[str, dict[str, Any]] = {}
    for row in rows:
        question = str(row["question"])
        for method, field in method_fields:
            response = str(row[field])
            key = pair_hash(question, response)
            assignment = {
                "split": "TEST",
                "scope": "corrected_TEST60_final",
                "source_order": int(row["source_order"]),
                "canonical_id": row["canonical_id"],
                "case_id": row["case_id"],
                "method": method,
                "question": question,
                "response": response,
                "question_sha256": sha256_text(question),
                "response_sha256": sha256_text(response),
                "question_response_sha256": key,
            }
            assignments.append(assignment)
            member = {
                "split": "TEST",
                "scope": "corrected_TEST60_final",
                "source_order": int(row["source_order"]),
                "canonical_id": row["canonical_id"],
                "case_id": row["case_id"],
                "condition": method,
            }
            if key not in representatives:
                representatives[key] = {
                    "question_response_sha256": key,
                    "question": question,
                    "response": response,
                    "question_sha256": sha256_text(question),
                    "response_sha256": sha256_text(response),
                    "members": [],
                }
            else:
                prior = representatives[key]
                if prior["question"] != question or prior["response"] != response:
                    raise RuntimeError(f"Question-response hash collision: {key}")
            representatives[key]["members"].append(member)
    unique = sorted(
        representatives.values(),
        key=lambda item: (
            min(member["source_order"] for member in item["members"]),
            item["question_response_sha256"],
        ),
    )
    return assignments, unique


def request_matches_contract(
    request: dict[str, Any], pair: dict[str, Any], system_prompt: str, user_prompt: str, max_tokens: int
) -> tuple[bool, str | None]:
    expected = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "parser": "pointwise_expert_rubric",
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    for key, value in expected.items():
        if request.get(key) != value:
            return False, f"request_{key}_mismatch"
    metadata = request.get("metadata") or {}
    if metadata.get("question_response_sha256") != pair["question_response_sha256"]:
        return False, "request_pair_hash_mismatch"
    if request.get("system_prompt_sha256") != sha256_text(system_prompt):
        return False, "system_prompt_hash_mismatch"
    if request.get("user_prompt_sha256") != sha256_text(user_prompt):
        return False, "user_prompt_hash_mismatch"
    return True, None


def build_new_request(
    pair: dict[str, Any], system_prompt: str, user_prompt: str, max_tokens: int
) -> dict[str, Any]:
    metadata = {
        "question_response_sha256": pair["question_response_sha256"],
        "question_sha256": pair["question_sha256"],
        "response_sha256": pair["response_sha256"],
        "members": pair["members"],
        "priority": "corrected_TEST60",
    }
    request_id = make_request_id(system_prompt, user_prompt, metadata)
    return {
        "version": VERSION,
        "queue": QUEUE,
        "request_id": request_id,
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


def prepare() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_source_rows()
    assignments, unique = build_assignments(rows)
    contract = read_json(FROZEN_CONTRACT)
    settings = contract["model_settings"]
    prompts = contract["prompts"]
    if settings["model"] != MODEL or settings["temperature"] != 0:
        raise RuntimeError("Frozen pointwise model settings changed")
    max_tokens = int(settings["pointwise_max_tokens"])
    if max_tokens != 384:
        raise RuntimeError(f"Unexpected frozen pointwise max_tokens: {max_tokens}")
    system_prompt = prompts["pointwise_system"]
    user_template = prompts["pointwise_user_template"]
    if sha256_text(system_prompt) != prompts["pointwise_system_sha256"]:
        raise RuntimeError("Frozen pointwise system prompt hash mismatch")
    if sha256_text(user_template) != prompts["pointwise_user_template_sha256"]:
        raise RuntimeError("Frozen pointwise user template hash mismatch")

    inputs = ROOT / "inputs"
    write_jsonl(inputs / "corrected_test60_assignments.jsonl", assignments)
    write_jsonl(inputs / "unique_question_response_pairs.jsonl", unique)
    ids = [row["canonical_id"] for row in rows]
    (inputs / "test60_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    frozen_hashes = [
        {
            "source_order": row["source_order"],
            "canonical_id": row["canonical_id"],
            "case_id": row["case_id"],
            "question_sha256": sha256_text(str(row["question"])),
            "sft_response_sha256": sha256_text(str(row["sft_response"])),
            "dpo_response_sha256": sha256_text(str(row["dpo_response"])),
            "proposed_response_sha256": sha256_text(str(row["final_response"])),
        }
        for row in rows
    ]
    write_jsonl(inputs / "frozen_test60_response_hashes.jsonl", frozen_hashes)
    write_csv(
        inputs / "frozen_test60_response_hashes.csv",
        frozen_hashes,
        [
            "source_order",
            "canonical_id",
            "case_id",
            "question_sha256",
            "sft_response_sha256",
            "dpo_response_sha256",
            "proposed_response_sha256",
        ],
    )

    old_requests: list[tuple[Path, dict[str, Any]]] = []
    for path in OLD_REQUEST_FILES:
        old_requests.extend((path, row) for row in read_jsonl(path))
    old_results_by_id: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in OLD_RESULT_FILES:
        for result in read_jsonl(path):
            old_results_by_id[str(result.get("request_id"))] = (path, result)
    requests_by_pair: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path, request in old_requests:
        key = str((request.get("metadata") or {}).get("question_response_sha256") or "")
        if key:
            requests_by_pair[key].append((path, request))

    reused_requests: list[dict[str, Any]] = []
    reused_results: list[dict[str, Any]] = []
    reuse_provenance: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    missing_pairs: list[dict[str, Any]] = []
    exact_old_request_pairs = 0
    for pair in unique:
        key = pair["question_response_sha256"]
        user_prompt = user_template.format(question=pair["question"], response=pair["response"])
        found_exact = False
        selected = None
        candidate_reasons: list[dict[str, Any]] = []
        for request_path, request in requests_by_pair.get(key, []):
            matches, reason = request_matches_contract(
                request, pair, system_prompt, user_prompt, max_tokens
            )
            if not matches:
                candidate_reasons.append({"request_id": request.get("request_id"), "reason": reason})
                continue
            found_exact = True
            result_record = old_results_by_id.get(str(request.get("request_id")))
            if result_record is None:
                candidate_reasons.append({"request_id": request.get("request_id"), "reason": "missing_result"})
                continue
            result_path, result = result_record
            valid, reason = result_is_valid(result, request, pair["response"])
            if not valid:
                candidate_reasons.append({"request_id": request.get("request_id"), "reason": reason})
                continue
            selected = (request_path, request, result_path, result)
            break
        if found_exact:
            exact_old_request_pairs += 1
        if selected is None:
            missing_pairs.append(pair)
            if found_exact or candidate_reasons:
                exclusions.append(
                    {
                        "question_response_sha256": key,
                        "members": pair["members"],
                        "reasons": candidate_reasons or [{"reason": "no_valid_old_result"}],
                    }
                )
            continue
        request_path, request, result_path, result = selected
        reused_requests.append(request)
        reused_results.append(result)
        reuse_provenance.append(
            {
                "question_response_sha256": key,
                "request_id": request["request_id"],
                "source_request_file": str(request_path),
                "source_result_file": str(result_path),
                "request_sha256": result["request_sha256"],
            }
        )

    new_requests = []
    for pair in missing_pairs:
        user_prompt = user_template.format(question=pair["question"], response=pair["response"])
        new_requests.append(build_new_request(pair, system_prompt, user_prompt, max_tokens))
    new_requests.sort(
        key=lambda request: (
            min(member["source_order"] for member in request["metadata"]["members"]),
            request["request_id"],
        )
    )
    reused_requests.sort(key=lambda row: row["request_id"])
    reused_results.sort(key=lambda row: row["request_id"])
    reuse_provenance.sort(key=lambda row: row["question_response_sha256"])

    write_jsonl(ROOT / "requests/reused_pointwise_requests.jsonl", reused_requests)
    write_jsonl(ROOT / "results/reused_pointwise_results.jsonl", reused_results)
    write_jsonl(ROOT / "cache/reuse_provenance.jsonl", reuse_provenance)
    write_jsonl(ROOT / "cache/reuse_exclusions.jsonl", exclusions)
    write_jsonl(ROOT / "requests/new_pointwise_requests.jsonl", new_requests)

    counts = {
        "method_assignments": len(assignments),
        "unique_question_response_pairs": len(unique),
        "exact_old_request_pairs": exact_old_request_pairs,
        "reused_valid_pairs": len(reused_results),
        "new_unique_requests": len(new_requests),
        "excluded_old_pairs": len(exclusions),
    }
    if counts != {
        "method_assignments": 180,
        "unique_question_response_pairs": 109,
        "exact_old_request_pairs": 45,
        "reused_valid_pairs": 44,
        "new_unique_requests": 65,
        "excluded_old_pairs": 1,
    }:
        raise RuntimeError(f"Unexpected reuse audit counts: {counts}")

    run_contract = {
        "version": VERSION,
        "status": "frozen_before_api_execution",
        "created_at_utc": utc_now(),
        "purpose": "Pointwise GPT-4.1 evaluation and publication tables for corrected frozen TEST60",
        "scope": {
            "split": "TEST",
            "questions": 60,
            "methods": ["SFT", "DPO", "Proposed"],
            "assignments": 180,
            "no_response_generation": True,
            "no_selector_change": True,
            "no_training": True,
            "additional_ablations": False,
        },
        "model_settings": {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "parser": "pointwise_expert_rubric",
            "paid_api_authorized": True,
        },
        "prompts": {
            "system_prompt": system_prompt,
            "user_template": user_template,
            "system_prompt_sha256": sha256_text(system_prompt),
            "user_template_sha256": sha256_text(user_template),
        },
        "privacy": {
            "api_receives": ["question", "anonymous response", "rubric"],
            "api_hidden": ["method", "risk", "selector decision"],
        },
        "deduplication": "identical question and response share one pointwise evaluation",
        "cache_validation": {
            "required_exact": [
                "question",
                "response",
                "rubric prompt",
                "model version",
                "temperature",
                "max_tokens",
                "response format",
                "parser",
                "request hash",
            ],
            "medical_yes_or_unsure_evidence_must_be_verbatim": True,
            "evidence_repair_prompt_results_reused": False,
        },
        "risk_scoring": {
            "contract": "fullpaper_v1",
            "component_mode": "full",
            "router_dir": file_meta(ROUTER_DIR / "model.safetensors"),
            "scorer_dir": file_meta(SCORER_DIR / "model.safetensors"),
            "router_max_len": 512,
            "risk_max_len": 384,
            "focus_axis": "medical_boundary",
            "purpose": "separate fixed-response scorer-risk summary; not part of GPT quality score",
        },
        "sources": {
            "corrected_test60": file_meta(SOURCE, 60),
            "corrected_test60_contract": file_meta(SOURCE_ROOT / "run_contract.json"),
            "frozen_pointwise_contract": file_meta(FROZEN_CONTRACT),
            "old_request_files": [file_meta(path, len(read_jsonl(path))) for path in OLD_REQUEST_FILES],
            "old_result_files": [file_meta(path, len(read_jsonl(path))) for path in OLD_RESULT_FILES],
        },
        "frozen_inputs": {
            "ordered_ids_newline_sha256": sha256_text("\n".join(ids) + "\n"),
            "assignments": file_meta(inputs / "corrected_test60_assignments.jsonl", len(assignments)),
            "unique_pairs": file_meta(inputs / "unique_question_response_pairs.jsonl", len(unique)),
            "response_hashes": file_meta(inputs / "frozen_test60_response_hashes.jsonl", 60),
        },
        "reuse": counts,
        "request_queue": file_meta(ROOT / "requests/new_pointwise_requests.jsonl", len(new_requests)),
        "independence_disclosure": (
            "The Proposed selector and automatic pointwise evaluator both use GPT-4.1-2025-04-14. "
            "This automatic evaluation is not independent expert or different-judge validation."
        ),
        "manuscript_source": {
            "canonical_tex_found_in_project_outside_outputs": False,
            "action": "produce insertion-ready LaTeX tables and replacement paragraphs without editing a manuscript",
        },
    }
    contract_path = ROOT / "run_contract.json"
    if contract_path.exists():
        existing = read_json(contract_path)
        for key in ("version", "model_settings", "prompts", "frozen_inputs", "reuse", "request_queue"):
            if existing.get(key) != run_contract.get(key):
                raise RuntimeError(f"Existing run contract differs at {key}")
    else:
        write_json(contract_path, run_contract)
        (ROOT / "run_contract.sha256").write_text(sha256_file(contract_path) + "  run_contract.json\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "root": str(ROOT), **counts}, sort_keys=True))


def score_risk() -> None:
    import torch

    sys.path.insert(0, str(REPO / "scripts"))
    from fullpaper_risk_contract import require_fullpaper_axis_order
    from run_gemma_selective_risk_refinement import load_classifier, response_metrics
    from selective_risk_refinement_utils import DIMS, configure_risk_contract, score_candidate

    pairs = read_jsonl(ROOT / "inputs/unique_question_response_pairs.jsonl")
    if len(pairs) != 109:
        raise RuntimeError("Run prepare before score-risk")
    output = ROOT / "risk/unique_pair_scorer_risk.jsonl"
    existing = {row["question_response_sha256"]: row for row in read_jsonl(output)}
    configure_risk_contract("fullpaper_v1")
    if "medical_boundary" not in DIMS:
        raise RuntimeError(f"medical_boundary is absent from fullpaper axes: {DIMS}")
    focus_idx = DIMS.index("medical_boundary")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    router_tok, router = load_classifier(str(ROUTER_DIR), device)
    risk_tok, risk_model = load_classifier(str(SCORER_DIR), device)
    require_fullpaper_axis_order(router.config)
    require_fullpaper_axis_order(risk_model.config)
    scored = dict(existing)
    with torch.inference_mode():
        for index, pair in enumerate(pairs, start=1):
            key = pair["question_response_sha256"]
            if key in scored:
                continue
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
            scored[key] = {
                "question_response_sha256": key,
                "question_sha256": pair["question_sha256"],
                "response_sha256": pair["response_sha256"],
                "risk_score": metrics["risk_score"],
                "focus_risk_score": metrics["focus_risk_score"],
                "g": metrics["g"],
                "span_risks": metrics["span_risks"],
                "word_count": metrics["word_count"],
                "risk_contract": "fullpaper_v1",
                "component_mode": "full",
            }
            if index % 10 == 0 or index == len(pairs):
                ordered = [scored[row["question_response_sha256"]] for row in pairs if row["question_response_sha256"] in scored]
                write_jsonl(output, ordered)
                print(json.dumps({"risk_scored": len(ordered), "risk_total": len(pairs)}), flush=True)
    ordered = [scored[row["question_response_sha256"]] for row in pairs]
    write_jsonl(output, ordered)
    assignments = read_jsonl(ROOT / "inputs/corrected_test60_assignments.jsonl")
    assignment_rows = [
        {
            "source_order": row["source_order"],
            "canonical_id": row["canonical_id"],
            "case_id": row["case_id"],
            "method": row["method"],
            "question_response_sha256": row["question_response_sha256"],
            "risk_score": scored[row["question_response_sha256"]]["risk_score"],
            "focus_risk_score": scored[row["question_response_sha256"]]["focus_risk_score"],
        }
        for row in assignments
    ]
    write_jsonl(ROOT / "risk/method_assignment_scorer_risk.jsonl", assignment_rows)
    manifest = {
        "status": "complete",
        "completed_at_utc": utc_now(),
        "device": device,
        "unique_pairs": len(ordered),
        "method_assignments": len(assignment_rows),
        "settings": {
            "risk_contract": "fullpaper_v1",
            "component_mode": "full",
            "router_max_len": 512,
            "risk_max_len": 384,
            "focus_axis": "medical_boundary",
            "router_dir": str(ROUTER_DIR),
            "scorer_dir": str(SCORER_DIR),
        },
        "output": file_meta(output, len(ordered)),
    }
    write_json(ROOT / "risk/risk_scoring_manifest.json", manifest)
    del router, risk_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps({"status": "risk_complete", "unique_pairs": len(ordered)}, sort_keys=True))


def latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def format_float(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def aggregate() -> None:
    source_rows = load_source_rows()
    assignments = read_jsonl(ROOT / "inputs/corrected_test60_assignments.jsonl")
    pairs = read_jsonl(ROOT / "inputs/unique_question_response_pairs.jsonl")
    if len(assignments) != 180 or len(pairs) != 109:
        raise RuntimeError("Frozen evaluation inputs are incomplete")
    contract = read_json(FROZEN_CONTRACT)
    system_prompt = contract["prompts"]["pointwise_system"]
    user_template = contract["prompts"]["pointwise_user_template"]
    max_tokens = int(contract["model_settings"]["pointwise_max_tokens"])
    pair_by_hash = {row["question_response_sha256"]: row for row in pairs}

    request_rows = read_jsonl(ROOT / "requests/reused_pointwise_requests.jsonl") + read_jsonl(
        ROOT / "requests/new_pointwise_requests.jsonl"
    )
    request_by_id = {row["request_id"]: row for row in request_rows}
    result_paths = [ROOT / "results/reused_pointwise_results.jsonl", ROOT / "results/new_pointwise_results.jsonl"]
    result_paths.extend(sorted((ROOT / "results").glob("new_pointwise_retry*_results.jsonl")))
    latest: dict[str, dict[str, Any]] = {}
    for path in result_paths:
        for row in read_jsonl(path):
            latest[row["request_id"]] = row

    valid_by_pair: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    evidence_warnings: list[dict[str, Any]] = []
    for request_id, request in request_by_id.items():
        key = request["metadata"]["question_response_sha256"]
        pair = pair_by_hash[key]
        expected_user = user_template.format(question=pair["question"], response=pair["response"])
        matches, reason = request_matches_contract(request, pair, system_prompt, expected_user, max_tokens)
        if not matches:
            failures.append({"request_id": request_id, "question_response_sha256": key, "reason": reason})
            continue
        result = latest.get(request_id)
        if result is None:
            failures.append({"request_id": request_id, "question_response_sha256": key, "reason": "missing_result"})
            continue
        valid, reason = result_is_valid(
            result,
            request,
            pair["response"],
            require_verbatim_evidence=False,
        )
        if not valid:
            failures.append({"request_id": request_id, "question_response_sha256": key, "reason": reason})
            continue
        normalized = result["normalized"]
        evidence = str(normalized.get("medical_advice_evidence") or "").strip()
        if normalized.get("medical_advice") in {"yes", "unsure"} and evidence not in pair["response"]:
            evidence_warnings.append(
                {
                    "request_id": request_id,
                    "question_response_sha256": key,
                    "medical_advice": normalized["medical_advice"],
                    "returned_evidence": evidence,
                    "reason": "nonverbatim_medical_advice_evidence",
                    "members": pair["members"],
                    "note": "The score and label passed the frozen parser; only evidence-string conformance failed.",
                }
            )
        valid_by_pair[key] = result
    missing_pair_hashes = sorted(set(pair_by_hash) - set(valid_by_pair))
    write_jsonl(ROOT / "results/pointwise_failures.jsonl", failures)
    write_jsonl(ROOT / "results/medical_advice_evidence_warnings.jsonl", evidence_warnings)
    if missing_pair_hashes:
        write_json(
            ROOT / "results/incomplete_pairs.json",
            {"count": len(missing_pair_hashes), "question_response_sha256": missing_pair_hashes},
        )

    unique_scores: list[dict[str, Any]] = []
    for pair in pairs:
        result = valid_by_pair.get(pair["question_response_sha256"])
        row = {
            "question_response_sha256": pair["question_response_sha256"],
            "question_sha256": pair["question_sha256"],
            "response_sha256": pair["response_sha256"],
            "members": pair["members"],
            "evaluation_ok": result is not None,
        }
        if result is not None:
            normalized = result["normalized"]
            row.update(
                {
                    "request_id": result["request_id"],
                    "overall_quality": normalized["scores"]["overall_quality"],
                    "empathy": normalized["scores"]["empathy"],
                    "specificity": normalized["scores"]["specificity"],
                    "quality_3axis_mean": statistics.mean(normalized["scores"].values()),
                    "medical_advice": normalized["medical_advice"],
                    "rationale": normalized["rationale"],
                    "medical_advice_evidence": normalized["medical_advice_evidence"],
                    "medical_advice_rationale": normalized["medical_advice_rationale"],
                    "returned_model": result["returned_model"],
                    "system_fingerprint": result.get("system_fingerprint"),
                }
            )
        unique_scores.append(row)
    write_jsonl(ROOT / "results/unique_pair_pointwise_scores.jsonl", unique_scores)

    score_by_pair = {row["question_response_sha256"]: row for row in unique_scores if row["evaluation_ok"]}
    assignment_scores: list[dict[str, Any]] = []
    for assignment in assignments:
        score = score_by_pair.get(assignment["question_response_sha256"])
        item = {**assignment, "evaluation_ok": score is not None}
        if score:
            for key in (
                "request_id",
                "overall_quality",
                "empathy",
                "specificity",
                "quality_3axis_mean",
                "medical_advice",
                "rationale",
                "medical_advice_evidence",
                "medical_advice_rationale",
            ):
                item[key] = score[key]
        assignment_scores.append(item)
    write_jsonl(ROOT / "results/method_assignment_scores.jsonl", assignment_scores)

    summary_rows: list[dict[str, Any]] = []
    for method in ("SFT", "DPO", "Proposed"):
        method_rows = [row for row in assignment_scores if row["method"] == method]
        ok_rows = [row for row in method_rows if row["evaluation_ok"]]
        medical = Counter(row["medical_advice"] for row in ok_rows)
        denominator = medical["yes"] + medical["no"]
        summary_rows.append(
            {
                "method": method,
                "n_questions": len(method_rows),
                "n_evaluated": len(ok_rows),
                "overall_quality_mean": mean_or_none([row["overall_quality"] for row in ok_rows]),
                "empathy_mean": mean_or_none([row["empathy"] for row in ok_rows]),
                "specificity_mean": mean_or_none([row["specificity"] for row in ok_rows]),
                "quality_3axis_mean": mean_or_none([row["quality_3axis_mean"] for row in ok_rows]),
                "medical_advice_yes": medical["yes"],
                "medical_advice_no": medical["no"],
                "medical_advice_unsure": medical["unsure"],
                "medical_advice_yes_rate": medical["yes"] / denominator if denominator else None,
                "medical_advice_yes_rate_denominator": denominator,
                "technical_or_parse_failures": len(method_rows) - len(ok_rows),
            }
        )
    summary_fields = [
        "method",
        "n_questions",
        "n_evaluated",
        "overall_quality_mean",
        "empathy_mean",
        "specificity_mean",
        "quality_3axis_mean",
        "medical_advice_yes",
        "medical_advice_no",
        "medical_advice_unsure",
        "medical_advice_yes_rate",
        "medical_advice_yes_rate_denominator",
        "technical_or_parse_failures",
    ]
    write_csv(ROOT / "tables/test60_method_summary.csv", summary_rows, summary_fields)
    write_json(ROOT / "tables/test60_method_summary.json", summary_rows)

    score_index = {(row["canonical_id"], row["method"]): row for row in assignment_scores}
    wtl_rows: list[dict[str, Any]] = []
    wtl = Counter()
    for source in source_rows:
        canonical_id = source["canonical_id"]
        sft = score_index[(canonical_id, "SFT")]
        proposed = score_index[(canonical_id, "Proposed")]
        item = {
            "source_order": source["source_order"],
            "canonical_id": canonical_id,
            "case_id": source["case_id"],
            "sft_evaluation_ok": sft["evaluation_ok"],
            "proposed_evaluation_ok": proposed["evaluation_ok"],
        }
        if sft["evaluation_ok"] and proposed["evaluation_ok"]:
            delta = proposed["quality_3axis_mean"] - sft["quality_3axis_mean"]
            outcome = "win" if delta > 1e-12 else "loss" if delta < -1e-12 else "tie"
            item.update(
                {
                    "sft_quality_3axis_mean": sft["quality_3axis_mean"],
                    "proposed_quality_3axis_mean": proposed["quality_3axis_mean"],
                    "proposed_minus_sft": delta,
                    "outcome_for_proposed": outcome,
                    "response_text_identical": source["sft_response"] == source["final_response"],
                }
            )
            wtl[outcome] += 1
        else:
            item["outcome_for_proposed"] = "unavailable"
            wtl["unavailable"] += 1
        wtl_rows.append(item)
    write_jsonl(ROOT / "tables/test60_sft_vs_proposed_per_item.jsonl", wtl_rows)
    write_csv(
        ROOT / "tables/test60_sft_vs_proposed_per_item.csv",
        wtl_rows,
        [
            "source_order",
            "canonical_id",
            "case_id",
            "sft_evaluation_ok",
            "proposed_evaluation_ok",
            "sft_quality_3axis_mean",
            "proposed_quality_3axis_mean",
            "proposed_minus_sft",
            "outcome_for_proposed",
            "response_text_identical",
        ],
    )
    wtl_summary = {
        "comparison": "Proposed versus SFT on per-question quality_3axis_mean",
        "N": len(source_rows),
        "win": wtl["win"],
        "tie": wtl["tie"],
        "loss": wtl["loss"],
        "unavailable": wtl["unavailable"],
    }
    write_json(ROOT / "tables/test60_sft_vs_proposed_wtl.json", wtl_summary)

    risk_rows = read_jsonl(ROOT / "risk/method_assignment_scorer_risk.jsonl")
    if len(risk_rows) != 180:
        raise RuntimeError("Risk scoring is incomplete; run score-risk before aggregate")
    risk_index = {(row["canonical_id"], row["method"]): row for row in risk_rows}
    pipeline_rows: list[dict[str, Any]] = []
    sft_risks = {row["canonical_id"]: risk_index[(row["canonical_id"], "SFT")]["risk_score"] for row in source_rows}
    for method in ("SFT", "DPO", "Proposed"):
        values = [risk_index[(row["canonical_id"], method)]["risk_score"] for row in source_rows]
        decreases = [sft_risks[row["canonical_id"]] - risk_index[(row["canonical_id"], method)]["risk_score"] for row in source_rows]
        pipeline_rows.append(
            {
                "method": method,
                "N": 60,
                "mean_scorer_risk": statistics.mean(values),
                "mean_risk_reduction_vs_sft": statistics.mean(decreases),
                "denoiser_calls": sum(bool(row["denoiser_called"]) for row in source_rows) if method == "Proposed" else "not_applicable",
                "candidate_generation_success": sum(bool(row["denoiser_called"] and row.get("denoiser_response")) for row in source_rows) if method == "Proposed" else "not_applicable",
                "candidate_generation_failure": sum(bool(row["denoiser_called"] and not row.get("denoiser_response")) for row in source_rows) if method == "Proposed" else "not_applicable",
                "gpt_selector_adoptions": sum(bool(row["gpt_selector_accepted_candidate"]) for row in source_rows) if method == "Proposed" else "not_applicable",
                "final_text_changes_vs_sft": sum(row["final_response"] != row["sft_response"] for row in source_rows) if method == "Proposed" else (0 if method == "SFT" else "not_applicable"),
            }
        )
    pipeline_fields = [
        "method",
        "N",
        "mean_scorer_risk",
        "mean_risk_reduction_vs_sft",
        "denoiser_calls",
        "candidate_generation_success",
        "candidate_generation_failure",
        "gpt_selector_adoptions",
        "final_text_changes_vs_sft",
    ]
    write_csv(ROOT / "tables/test60_pipeline_and_risk.csv", pipeline_rows, pipeline_fields)
    write_json(ROOT / "tables/test60_pipeline_and_risk.json", pipeline_rows)

    quality_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{GPT-4.1 pointwise evaluation on the corrected frozen TEST60 set. Medical Advice is categorical and is not included in the quality mean.}",
        r"\label{tab:corrected-test60-gpt41}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Method & N & Overall & Empathy & Specificity & Quality-3 & Med. yes/no/unsure & Yes rate (denom.) \\",
        r"\midrule",
    ]
    for row in summary_rows:
        quality_lines.append(
            f"{latex_escape(row['method'])} & {row['n_evaluated']} & {format_float(row['overall_quality_mean'])} & "
            f"{format_float(row['empathy_mean'])} & {format_float(row['specificity_mean'])} & "
            f"{format_float(row['quality_3axis_mean'])} & "
            f"{row['medical_advice_yes']}/{row['medical_advice_no']}/{row['medical_advice_unsure']} & "
            f"{format_float(row['medical_advice_yes_rate'])} ({row['medical_advice_yes_rate_denominator']}) \\\\"
        )
    quality_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    publication = ROOT / "publication"
    publication.mkdir(parents=True, exist_ok=True)
    (publication / "test60_quality_table.tex").write_text("\n".join(quality_lines) + "\n", encoding="utf-8")

    pipeline_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Pipeline activity and frozen scorer risk on TEST60. Calls and adoptions apply only to Proposed.}",
        r"\label{tab:corrected-test60-pipeline}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & Mean risk & Risk reduction & Calls & Adoptions & Changed \\",
        r"\midrule",
    ]
    for row in pipeline_rows:
        calls = "--" if row["denoiser_calls"] == "not_applicable" else row["denoiser_calls"]
        adoptions = "--" if row["gpt_selector_adoptions"] == "not_applicable" else row["gpt_selector_adoptions"]
        changes = "--" if row["final_text_changes_vs_sft"] == "not_applicable" else row["final_text_changes_vs_sft"]
        pipeline_lines.append(
            f"{latex_escape(row['method'])} & {format_float(row['mean_scorer_risk'])} & "
            f"{format_float(row['mean_risk_reduction_vs_sft'])} & {calls} & "
            f"{adoptions} & {changes} \\\\"
        )
    pipeline_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (publication / "test60_pipeline_risk_table.tex").write_text("\n".join(pipeline_lines) + "\n", encoding="utf-8")

    summary_by_method = {row["method"]: row for row in summary_rows}
    proposed = summary_by_method["Proposed"]
    proposed_pipeline = next(row for row in pipeline_rows if row["method"] == "Proposed")
    replacement = rf"""% Insertion-ready replacement text; no canonical manuscript .tex was found in the project.
\paragraph{{Method.}}
We froze the corrected TEST60 outputs before evaluation: 60 fixed questions with one SFT, DPO, and Proposed final response per question. Proposed used the frozen learned-denoiser call gate followed by a GPT-4.1 selector (\texttt{{gpt-4.1-2025-04-14}}, temperature 0). The selector viewed only the question and anonymous SFT/candidate responses in both presentation orders; a candidate was adopted only when both orders selected it. Ties, order disagreement, technical failure, missing candidates, and gate-off cases retained SFT. No response was regenerated and no selector threshold or rule was adjusted after observing TEST results.

\paragraph{{Evaluation.}}
We evaluated anonymous question-response pairs pointwise with \texttt{{gpt-4.1-2025-04-14}} at temperature 0 using the frozen expert-aligned rubric. Overall Quality, Empathy, and Specificity were integer scores from 1 to 5; their unweighted mean is reported as \textit{{quality\_3axis\_mean}}. Medical Advice was labeled yes, no, or unsure and was not converted to a numeric score. Exact duplicate question-response pairs shared one evaluation. The selector and automatic evaluator used the same GPT model, so this analysis is not an independent-judge validation and may reflect correlated model preferences; prior expert and Llama evaluations were not mixed into this corrected TEST60 table.

\paragraph{{Results.}}
Across all 60 questions, Proposed obtained means of {proposed['overall_quality_mean']:.3f} for Overall Quality, {proposed['empathy_mean']:.3f} for Empathy, {proposed['specificity_mean']:.3f} for Specificity, and {proposed['quality_3axis_mean']:.3f} for the three-axis quality mean. Medical Advice labels were {proposed['medical_advice_yes']} yes, {proposed['medical_advice_no']} no, and {proposed['medical_advice_unsure']} unsure; the yes rate was {proposed['medical_advice_yes_rate']:.3f} over a denominator of {proposed['medical_advice_yes_rate_denominator']} non-unsure judgments. Relative to SFT, Proposed yielded {wtl['win']} wins, {wtl['tie']} ties, and {wtl['loss']} losses on per-question quality\_3axis\_mean. The denoiser was called for {proposed_pipeline['denoiser_calls']} of 60 questions, and the two-order selector adopted {proposed_pipeline['gpt_selector_adoptions']} candidates, producing {proposed_pipeline['final_text_changes_vs_sft']} final responses that differed textually from SFT. Mean frozen scorer risk was {proposed_pipeline['mean_scorer_risk']:.3f}, a mean reduction of {proposed_pipeline['mean_risk_reduction_vs_sft']:.3f} versus SFT over the fixed 60-question denominator. These TEST60 comparisons are post-hoc exploratory.
"""
    (publication / "manuscript_replacement_text.tex").write_text(replacement, encoding="utf-8")
    (publication / "README.md").write_text(
        "# Corrected TEST60 publication inserts\n\n"
        "No canonical manuscript `.tex` source was found in the project outside generated output paths, so no manuscript was edited. "
        "The two table files require `booktabs`; `manuscript_replacement_text.tex` is insertion-ready text aligned to the frozen run contract.\n",
        encoding="utf-8",
    )

    new_results = read_jsonl(ROOT / "results/new_pointwise_results.jsonl")
    new_request_ids = {row["request_id"] for row in read_jsonl(ROOT / "requests/new_pointwise_requests.jsonl")}
    new_latest = {}
    for row in new_results:
        if row.get("request_id") in new_request_ids:
            new_latest[row["request_id"]] = row
    new_ok = [row for row in new_latest.values() if row.get("ok") is True]
    usage = {
        "reused_unique_evaluations": len(read_jsonl(ROOT / "results/reused_pointwise_results.jsonl")),
        "new_unique_requests": len(new_request_ids),
        "new_successful_api_results": len(new_ok),
        "new_failed_latest_api_results": len(new_request_ids) - len(new_ok),
        "final_valid_unique_evaluations": len(valid_by_pair),
        "final_invalid_or_missing_unique_evaluations": len(missing_pair_hashes),
        "medical_advice_evidence_conformance_warnings": len(evidence_warnings),
        "new_input_tokens": sum(int((row.get("usage") or {}).get("prompt_tokens") or 0) for row in new_ok),
        "new_output_tokens": sum(int((row.get("usage") or {}).get("completion_tokens") or 0) for row in new_ok),
        "new_estimated_cost_usd": round(sum(float(row.get("estimated_cost_usd") or 0) for row in new_ok), 8),
        "model": MODEL,
    }
    write_json(ROOT / "api_usage.json", usage)

    independent = {
        "corrected_TEST60_expert_evaluation": "not_run_in_this_task",
        "corrected_TEST60_llama_evaluation": "not_run_in_this_task",
        "prior_rubric_aligned_expert_packet": {
            "status": "existing_first15_packet_only",
            "path": str(FROZEN_ROOT / "counselor_expert_packet_TEST15x3_rubric_aligned.zip"),
            "used_in_corrected_TEST60_automatic_table": False,
        },
        "prior_rubric_aligned_llama_manifest": {
            "status": read_json(FROZEN_ROOT / "results/llama_rubric_aligned_manifest.json").get("status", "unknown"),
            "path": str(FROZEN_ROOT / "results/llama_rubric_aligned_manifest.json"),
            "used_in_corrected_TEST60_automatic_table": False,
        },
        "disclosure": "GPT-4.1 served as both Proposed selector and automatic pointwise evaluator; this is not independent validation.",
    }
    write_json(ROOT / "independent_validation_status.json", independent)

    coverage = [
        {
            "dataset": "corrected_TEST60",
            "judge": MODEL,
            "methods": "SFT/DPO/Proposed",
            "questions": 60,
            "method_assignments": 180,
            "unique_question_response_pairs": 109,
            "reused_unique": usage["reused_unique_evaluations"],
            "new_unique": usage["new_successful_api_results"],
            "failed_unique": usage["final_invalid_or_missing_unique_evaluations"],
            "included_in_publication_table": True,
        },
        {
            "dataset": "corrected_TEST60",
            "judge": "Llama",
            "methods": "SFT/DPO/Proposed",
            "questions": 0,
            "method_assignments": 0,
            "unique_question_response_pairs": 0,
            "reused_unique": 0,
            "new_unique": 0,
            "failed_unique": 0,
            "included_in_publication_table": False,
        },
        {
            "dataset": "corrected_TEST60",
            "judge": "human expert",
            "methods": "SFT/DPO/Proposed",
            "questions": 0,
            "method_assignments": 0,
            "unique_question_response_pairs": 0,
            "reused_unique": 0,
            "new_unique": 0,
            "failed_unique": 0,
            "included_in_publication_table": False,
        },
    ]
    write_csv(
        ROOT / "tables/comparison_coverage.csv",
        coverage,
        [
            "dataset",
            "judge",
            "methods",
            "questions",
            "method_assignments",
            "unique_question_response_pairs",
            "reused_unique",
            "new_unique",
            "failed_unique",
            "included_in_publication_table",
        ],
    )

    report_lines = [
        "# Corrected TEST60 GPT-4.1 Pointwise Evaluation",
        "",
        f"Status: {'complete' if not missing_pair_hashes else 'partial'}",
        "",
        "## Coverage",
        "",
        f"- Frozen questions: 60; method assignments: 180; unique question-response pairs: 109.",
        f"- Reused exact valid evaluations: {usage['reused_unique_evaluations']}.",
        f"- New successful evaluations: {usage['new_successful_api_results']}.",
        f"- Invalid or missing final evaluations: {usage['final_invalid_or_missing_unique_evaluations']}.",
        f"- Medical Advice evidence-string conformance warnings: {usage['medical_advice_evidence_conformance_warnings']} (scores and labels passed the frozen parser).",
        f"- New GPT usage: {usage['new_input_tokens']} input tokens, {usage['new_output_tokens']} output tokens, estimated ${usage['new_estimated_cost_usd']:.6f}.",
        "",
        "## Quality",
        "",
        "| Method | N | Overall | Empathy | Specificity | Quality-3 | Medical yes/no/unsure | Yes rate | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| {row['method']} | {row['n_evaluated']}/60 | {format_float(row['overall_quality_mean'])} | "
            f"{format_float(row['empathy_mean'])} | {format_float(row['specificity_mean'])} | "
            f"{format_float(row['quality_3axis_mean'])} | {row['medical_advice_yes']}/{row['medical_advice_no']}/{row['medical_advice_unsure']} | "
            f"{format_float(row['medical_advice_yes_rate'])} (N={row['medical_advice_yes_rate_denominator']}) | "
            f"{row['technical_or_parse_failures']} |"
        )
    report_lines.extend(
        [
            "",
            "Medical Advice is categorical and is not included in Quality-3.",
            "The one evidence-string warning is the frozen TEST-15 SFT response: its source text has collapsed word spacing, while GPT returned the supporting phrase with normal spaces. Its score and label passed the frozen parser.",
            "",
            "## Paired Result",
            "",
            f"Proposed versus SFT Quality-3 W/T/L: {wtl['win']}/{wtl['tie']}/{wtl['loss']} (unavailable {wtl['unavailable']}, N=60).",
            "",
            "## Pipeline And Risk",
            "",
            "| Method | N | Mean scorer risk | Mean reduction vs SFT | Calls | Adoptions | Final text changes |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pipeline_rows:
        report_lines.append(
            f"| {row['method']} | 60 | {format_float(row['mean_scorer_risk'])} | "
            f"{format_float(row['mean_risk_reduction_vs_sft'])} | {row['denoiser_calls']} | "
            f"{row['gpt_selector_adoptions']} | {row['final_text_changes_vs_sft']} |"
        )
    report_lines.extend(
        [
            "",
            "The frozen Proposed pipeline called the denoiser on 18 questions. The two-order GPT selector adopted a candidate only when both presentation orders selected it. The selector and evaluator both used GPT-4.1, so these automatic results are not independent validation.",
            "",
            "No canonical manuscript `.tex` source was found outside generated output paths. Insertion-ready LaTeX tables and replacement text are under `publication/`.",
            "TEST354, VALID43, and earlier numeric-rubric results are not included in these aggregates.",
            "",
        ]
    )
    (ROOT / "FINAL_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    completion = {
        "version": VERSION,
        "status": "complete" if not missing_pair_hashes else "partial",
        "completed_at_utc": utc_now(),
        "coverage": usage,
        "quality_summary": summary_rows,
        "sft_vs_proposed_wtl": wtl_summary,
        "pipeline_and_risk": pipeline_rows,
        "source": file_meta(SOURCE, 60),
        "frozen_contract": file_meta(ROOT / "run_contract.json"),
    }
    write_json(ROOT / "completion_summary.json", completion)

    code_dir = ROOT / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
    shutil.copy2(REPO / "scripts/run_gpt41_checkpoint_queue.py", code_dir / "run_gpt41_checkpoint_queue.py")
    bundle = ROOT / "corrected_TEST60_GPT41_pointwise_publication_research_bundle.zip"
    files = {}
    excluded = {bundle, ROOT / "FINAL_MANIFEST.json", ROOT / "DELIVERY.json"}
    for path in sorted(path for path in ROOT.rglob("*") if path.is_file() and path not in excluded):
        files[str(path.relative_to(ROOT))] = file_meta(path)
    manifest = {
        "version": VERSION,
        "status": completion["status"],
        "summary": completion,
        "files": files,
    }
    write_json(ROOT / "FINAL_MANIFEST.json", manifest)
    include_paths = [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path not in {bundle, ROOT / "DELIVERY.json"}
    ]
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(include_paths):
            archive.write(path, path.relative_to(ROOT))
    write_json(
        ROOT / "DELIVERY.json",
        {
            "status": completion["status"],
            "research_bundle": file_meta(bundle),
            "report": file_meta(ROOT / "FINAL_REPORT.md"),
            "quality_csv": file_meta(ROOT / "tables/test60_method_summary.csv"),
            "latex_quality_table": file_meta(publication / "test60_quality_table.tex"),
            "latex_replacement_text": file_meta(publication / "manuscript_replacement_text.tex"),
        },
    )
    print(
        json.dumps(
            {
                "status": completion["status"],
                "reused": usage["reused_unique_evaluations"],
                "new": usage["new_successful_api_results"],
                "failed": usage["final_invalid_or_missing_unique_evaluations"],
                "bundle": str(bundle),
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "score-risk", "aggregate"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "score-risk":
        score_risk()
    else:
        aggregate()


if __name__ == "__main__":
    main()
