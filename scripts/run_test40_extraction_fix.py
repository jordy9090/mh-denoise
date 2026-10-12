#!/usr/bin/env python3
"""Repair the saved TEST-40 extraction and rebuild only its final selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


VERSION = "test60-rubric-aligned-final-selection-v2-test40-fix-20261009"
CASE_ID = "TEST-40"
CANONICAL_ID = "qa_a4697bfe6a1ce131b9bd754f"
SOURCE_ORDER = 39
MODEL = "gpt-4.1-2025-04-14"
QUEUE = "test40_fixed_extraction_selector"
REPO = Path("/home/user/hsoh/mh-denoise")
WORKTREE = Path("/home/user/hsoh/mh-denoise-surface-relaxed-v1-20261008")
V1 = Path("/home/user/hsoh/test60_rubric_aligned_final_selection_20261009_v1")
FROZEN_SELECTOR = Path("/home/user/hsoh/rubric_aligned_gpt41_rejudge_20261009_v1")
OUTPUT = Path("/home/user/hsoh/test60_rubric_aligned_final_selection_20261009_v2_test40_fix")
PART60 = REPO / (
    "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/gemma/"
    "generation_1536_test_surface_relaxed_v1_20261008/part60"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".tmp.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_frozen(path: Path, value: bytes) -> None:
    if path.exists():
        if path.read_bytes() != value:
            raise RuntimeError(f"Refusing to overwrite a different frozen artifact: {path}")
        return
    atomic_write(path, value)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write(path, jsonl_bytes(rows))


def file_record(path: Path, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = rows
    return record


def canonical_bytes(row: dict[str, Any]) -> bytes:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def row_hash(row: dict[str, Any]) -> str:
    return sha256_bytes(canonical_bytes(row))


def target_row(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row.get("canonical_id") or row.get("id")) == CANONICAL_ID]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one {label} target row, got {len(selected)}")
    row = selected[0]
    if rows.index(row) != SOURCE_ORDER:
        raise RuntimeError(f"{label} target is not at frozen source order {SOURCE_ORDER}")
    return row


def index_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        canonical_id = str(row.get("canonical_id") or row.get("id") or "")
        if not canonical_id or canonical_id in result:
            raise RuntimeError(f"Invalid or duplicate ID in {label}: {canonical_id}")
        result[canonical_id] = row
    return result


def replace_target(rows: list[dict[str, Any]], replacement: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    replaced = 0
    for row in rows:
        if str(row.get("canonical_id") or row.get("id")) == CANONICAL_ID:
            result.append(replacement)
            replaced += 1
        else:
            result.append(row)
    if replaced != 1:
        raise RuntimeError(f"Replacement count is {replaced}, expected one")
    return result


def load_cleanup_response():
    sys.path.insert(0, str(WORKTREE / "scripts"))
    from selective_risk_refinement_utils import cleanup_response

    return cleanup_response


def raw_body_after_label(raw: str) -> tuple[str, str]:
    match = re.search(
        r"(?im)^[ \t]*\*{0,2}[ \t]*final[ \t]+response[ \t]*:[ \t]*\*{0,2}[ \t]*$",
        raw,
    )
    if not match:
        raise RuntimeError("Saved raw output has no standalone Final Response label")
    preface = raw[: match.start()].strip()
    body_with_boundary_quotes = raw[match.end() :].strip()
    return preface, body_with_boundary_quotes


def prepare() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract_path = OUTPUT / "run_contract.json"
    if contract_path.exists():
        contract = read_json(contract_path)
        if contract.get("target", {}).get("canonical_id") != CANONICAL_ID:
            raise RuntimeError("Existing v2 contract targets a different case")
        print(json.dumps({"status": "reused_prepared_contract", "path": str(contract_path)}, sort_keys=True))
        return

    source_paths = {
        "sft60": PART60 / "sft_test_outputs.jsonl",
        "dpo60": PART60 / "dpo_test_outputs.jsonl",
        "proposed60": PART60 / "proposed_mask_off_test_outputs.jsonl",
        "v1_final": V1 / "test60_final_outputs.jsonl",
        "v1_decisions": V1 / "selector_decisions.jsonl",
        "v1_candidates": V1 / "inputs/test60_called_original_denoiser_candidates.jsonl",
        "v1_contract": V1 / "run_contract.json",
        "selector_contract": FROZEN_SELECTOR / "run_contract.json",
    }
    for path in source_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    sft_rows = read_jsonl(source_paths["sft60"])
    dpo_rows = read_jsonl(source_paths["dpo60"])
    proposed_rows = read_jsonl(source_paths["proposed60"])
    if not (len(sft_rows) == len(dpo_rows) == len(proposed_rows) == 60):
        raise RuntimeError("Frozen TEST60 generation coverage changed")
    sft = target_row(sft_rows, "SFT")
    dpo = target_row(dpo_rows, "DPO")
    old_proposed = target_row(proposed_rows, "Proposed")
    if sft["question"] != dpo["question"] or sft["question"] != old_proposed["question"]:
        raise RuntimeError("TEST-40 method questions differ")

    cleanup_response = load_cleanup_response()
    restored_sft = cleanup_response(str(sft.get("sft_response_raw") or ""))
    restored_dpo = cleanup_response(str(dpo.get("sft_response_raw") or ""))
    if not restored_sft or restored_sft != restored_dpo:
        raise RuntimeError("SFT/DPO raw restoration is empty or differs")
    sft_preface, sft_body_quoted = raw_body_after_label(sft["sft_response_raw"])
    dpo_preface, dpo_body_quoted = raw_body_after_label(dpo["sft_response_raw"])
    for label, quoted in (("SFT", sft_body_quoted), ("DPO", dpo_body_quoted)):
        if len(quoted) < 2 or quoted[0] != quoted[-1] or quoted[0] not in {"'", '"'}:
            raise RuntimeError(f"{label} saved body is not wholly enclosed by boundary quotes")
        if quoted[1:-1].strip() != restored_sft:
            raise RuntimeError(f"{label} restored text omits or changes saved body content")

    correction = {
        "version": VERSION,
        "case_id": CASE_ID,
        "canonical_id": CANONICAL_ID,
        "source_order": SOURCE_ORDER,
        "cause": (
            "cleanup_response compared labels case-sensitively. It did not recognize the standalone "
            "Markdown '**Final Response:**' label after a rewrite preface; the later 'Response:' rule "
            "then kept the nonempty prefix and discarded the answer body."
        ),
        "boundary_rule": (
            "Use the standalone Final Response label as the answer boundary, discard only the preceding "
            "rewrite preface and the matching quotes enclosing the whole body, and preserve every character "
            "inside those quotes."
        ),
        "SFT": {
            "raw": sft["sft_response_raw"],
            "raw_sha256": sha256_text(sft["sft_response_raw"]),
            "preface": sft_preface,
            "body_with_boundary_quotes": sft_body_quoted,
            "before": sft["sft_response"],
            "before_sha256": sha256_text(sft["sft_response"]),
            "restored": restored_sft,
            "restored_sha256": sha256_text(restored_sft),
        },
        "DPO": {
            "raw": dpo["sft_response_raw"],
            "raw_sha256": sha256_text(dpo["sft_response_raw"]),
            "preface": dpo_preface,
            "body_with_boundary_quotes": dpo_body_quoted,
            "before": dpo["sft_response"],
            "before_sha256": sha256_text(dpo["sft_response"]),
            "restored": restored_dpo,
            "restored_sha256": sha256_text(restored_dpo),
        },
        "restored_sft_and_dpo_text_identical": True,
        "body_omission_check": "passed_exact_inner_quote_comparison",
    }
    write_frozen(OUTPUT / "extraction_audit.json", json_bytes(correction))

    fixed_sft = dict(sft)
    fixed_sft["sft_response_before_extraction_fix"] = sft["sft_response"]
    fixed_sft["sft_response"] = restored_sft
    fixed_sft["extraction_fix"] = {
        "version": VERSION,
        "source": "saved_sft_response_raw",
        "audit": str(OUTPUT / "extraction_audit.json"),
    }
    fixed_dpo = dict(dpo)
    fixed_dpo["sft_response_before_extraction_fix"] = dpo["sft_response"]
    fixed_dpo["sft_response"] = restored_dpo
    fixed_dpo["extraction_fix"] = {
        "version": VERSION,
        "source": "saved_sft_response_raw",
        "audit": str(OUTPUT / "extraction_audit.json"),
    }
    fixed_sft_rows = replace_target(sft_rows, fixed_sft)
    fixed_dpo_rows = replace_target(dpo_rows, fixed_dpo)
    write_frozen(OUTPUT / "generation/sft_test_outputs.jsonl", jsonl_bytes(fixed_sft_rows))
    write_frozen(OUTPUT / "generation/dpo_test_outputs.jsonl", jsonl_bytes(fixed_dpo_rows))
    write_frozen(OUTPUT / "inputs/test40_proposed_input.jsonl", jsonl_bytes([fixed_sft]))

    v1_final = target_row(read_jsonl(source_paths["v1_final"]), "v1 final")
    v1_decision = target_row(read_jsonl(source_paths["v1_decisions"]), "v1 decision")
    v1_candidate = next(
        row for row in read_jsonl(source_paths["v1_candidates"]) if row["canonical_id"] == CANONICAL_ID
    )
    history = {
        "sft_source_row": sft,
        "dpo_source_row": dpo,
        "proposed_source_row_from_truncated_sft": old_proposed,
        "v1_final_row": v1_final,
        "v1_selector_decision": v1_decision,
        "v1_called_candidate": v1_candidate,
    }
    write_frozen(OUTPUT / "history/test40_v1_history.json", json_bytes(history))

    extractor_path = WORKTREE / "scripts/selective_risk_refinement_utils.py"
    extractor_patch = subprocess.check_output(
        ["git", "diff", "--", "scripts/selective_risk_refinement_utils.py"],
        cwd=WORKTREE,
    )
    write_frozen(OUTPUT / "code/selective_risk_refinement_utils.patch", extractor_patch)
    frozen_selector = read_json(source_paths["selector_contract"])
    old_proposed_manifest = read_json(PART60 / "proposed_missing_outputs.manifest.json")
    contract = {
        "version": VERSION,
        "status": "frozen_before_one_case_risk_gate_recomputation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": {"case_id": CASE_ID, "canonical_id": CANONICAL_ID, "source_order": SOURCE_ORDER},
        "scope": {
            "changed_cases": 1,
            "unchanged_cases_required": 59,
            "no_sft_or_dpo_generation": True,
            "no_training": True,
            "no_quality_evaluation": True,
            "no_rubric_change": True,
        },
        "sources": {name: file_record(path) for name, path in source_paths.items()},
        "extraction": {
            "audit": file_record(OUTPUT / "extraction_audit.json"),
            "extractor": file_record(extractor_path),
            "extractor_patch": file_record(OUTPUT / "code/selective_risk_refinement_utils.patch"),
        },
        "restored_inputs": {
            "SFT60": file_record(OUTPUT / "generation/sft_test_outputs.jsonl", 60),
            "DPO60": file_record(OUTPUT / "generation/dpo_test_outputs.jsonl", 60),
            "TEST40_proposed_input": file_record(OUTPUT / "inputs/test40_proposed_input.jsonl", 1),
        },
        "risk_and_generation": {
            "source_manifest": file_record(PART60 / "proposed_missing_outputs.manifest.json"),
            "frozen_settings": old_proposed_manifest["settings"],
            "acceptance_policy": "surface_relaxed_v1",
        },
        "selector": {
            "frozen_contract": file_record(source_paths["selector_contract"]),
            "model_settings": frozen_selector["model_settings"],
            "selector_rule": frozen_selector["selector_rule"],
            "system_prompt_sha256": frozen_selector["prompts"]["selector_system_sha256"],
            "user_template_sha256": frozen_selector["prompts"]["selector_user_template_sha256"],
            "old_TEST40_requests_reusable": False,
            "reason": "The restored SFT changes both anonymous comparison prompts and request hashes.",
        },
    }
    write_frozen(contract_path, json_bytes(contract))
    write_frozen(OUTPUT / "run_contract.sha256", (sha256_file(contract_path) + "  run_contract.json\n").encode("ascii"))
    print(
        json.dumps(
            {
                "status": "prepared",
                "restored_sha256": sha256_text(restored_sft),
                "old_sft_sha256": sha256_text(sft["sft_response"]),
                "old_dpo_sha256": sha256_text(dpo["sft_response"]),
            },
            sort_keys=True,
        )
    )


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
    return sha256_text(json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def request_id(version: str, system: str, user: str, metadata: dict[str, Any]) -> str:
    material = {
        "version": version,
        "queue": QUEUE,
        "model": MODEL,
        "system_prompt": system,
        "user_prompt": user,
        "metadata": metadata,
    }
    return sha256_text(json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")))[:32]


def build_request(version: str, system: str, user: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": version,
        "queue": QUEUE,
        "request_id": request_id(version, system, user, metadata),
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
        "parser": "blind_selector",
        "system_prompt": system,
        "user_prompt": user,
        "system_prompt_sha256": sha256_text(system),
        "user_prompt_sha256": sha256_text(user),
        "metadata": metadata,
    }


def prepare_selector() -> None:
    risk_path = OUTPUT / "generation/test40_proposed_recomputed.jsonl"
    rows = read_jsonl(risk_path)
    if len(rows) != 1 or str(rows[0].get("canonical_id")) != CANONICAL_ID:
        raise RuntimeError("One-case risk output is missing or invalid")
    row = rows[0]
    old = read_json(OUTPUT / "history/test40_v1_history.json")["proposed_source_row_from_truncated_sft"]
    gate = {
        "case_id": CASE_ID,
        "canonical_id": CANONICAL_ID,
        "before": {
            "sft_response": old["sft_response"],
            "sft_response_sha256": sha256_text(old["sft_response"]),
            "sft_risk_score": old["sft_risk_score"],
            "sft_focus_risk_score": old["sft_focus_risk_score"],
            "sft_word_count": old["sft_word_count"],
            "used_denoiser": old["used_denoiser"],
            "denoiser_call_reason": old["denoiser_call_reason"],
        },
        "after": {
            "sft_response": row["sft_response"],
            "sft_response_sha256": sha256_text(row["sft_response"]),
            "sft_risk_score": row["sft_risk_score"],
            "sft_focus_risk_score": row["sft_focus_risk_score"],
            "sft_word_count": row["sft_word_count"],
            "used_denoiser": row["used_denoiser"],
            "denoiser_call_reason": row["denoiser_call_reason"],
        },
        "frozen_gate_settings": read_json(PART60 / "proposed_missing_outputs.manifest.json")["settings"],
    }
    write_frozen(OUTPUT / "gate_recomputation.json", json_bytes(gate))

    selector_contract = read_json(FROZEN_SELECTOR / "run_contract.json")
    prompts = selector_contract["prompts"]
    requests: list[dict[str, Any]] = []
    if bool(row["used_denoiser"]):
        question = str(row["question"])
        sft_response = str(row["sft_response"])
        candidate = str(row.get("denoiser_response") or "")
        if not candidate.strip():
            raise RuntimeError("Gate fired but one-case denoiser candidate is empty")
        for order_index in (0, 1):
            if order_index == 0:
                response_a, response_b = sft_response, candidate
                variant_a, variant_b = "sft", "candidate"
            else:
                response_a, response_b = candidate, sft_response
                variant_a, variant_b = "candidate", "sft"
            metadata = {
                "split": "TEST",
                "canonical_id": CANONICAL_ID,
                "case_id": CASE_ID,
                "source_order": SOURCE_ORDER,
                "candidate_type": "original_denoiser_after_test40_extraction_fix",
                "order_index": order_index,
                "variant_for_A": variant_a,
                "variant_for_B": variant_b,
                "question_sha256": sha256_text(question),
                "response_A_sha256": sha256_text(response_a),
                "response_B_sha256": sha256_text(response_b),
            }
            user = prompts["selector_user_template"].format(
                question=question,
                response_a=response_a,
                response_b=response_b,
            )
            requests.append(
                build_request(
                    str(selector_contract["version"]),
                    prompts["selector_system"],
                    user,
                    metadata,
                )
            )
    write_frozen(OUTPUT / "requests/test40_selector_requests.jsonl", jsonl_bytes(requests))
    request_contract = {
        "version": VERSION,
        "status": "frozen_before_api" if requests else "selector_skipped_gate_off",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_recomputation": file_record(OUTPUT / "gate_recomputation.json"),
        "risk_output": file_record(risk_path, 1),
        "selector_contract": file_record(FROZEN_SELECTOR / "run_contract.json"),
        "request_file": file_record(OUTPUT / "requests/test40_selector_requests.jsonl", len(requests)),
        "new_api_requests": len(requests),
        "past_selector_results_reused": 0,
        "selector_skip_reason": None if requests else "corrected_sft_gate_off",
    }
    write_frozen(OUTPUT / "selector_request_contract.json", json_bytes(request_contract))
    print(
        json.dumps(
            {
                "gate_on": bool(row["used_denoiser"]),
                "gate_reason": row["denoiser_call_reason"],
                "sft_risk_score": row["sft_risk_score"],
                "new_api_requests": len(requests),
            },
            sort_keys=True,
        )
    )


def latest_results(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        result[str(row.get("request_id") or "")] = row
    return result


def finalize() -> None:
    fixed_sft_rows = read_jsonl(OUTPUT / "generation/sft_test_outputs.jsonl")
    fixed_dpo_rows = read_jsonl(OUTPUT / "generation/dpo_test_outputs.jsonl")
    original_proposed_rows = read_jsonl(PART60 / "proposed_mask_off_test_outputs.jsonl")
    risk_row = read_jsonl(OUTPUT / "generation/test40_proposed_recomputed.jsonl")[0]
    fixed_proposed_rows = replace_target(original_proposed_rows, risk_row)
    write_jsonl(OUTPUT / "generation/proposed_mask_off_test_outputs.jsonl", fixed_proposed_rows)

    requests = read_jsonl(OUTPUT / "requests/test40_selector_requests.jsonl")
    results_path = OUTPUT / "results/test40_selector_results.jsonl"
    results = latest_results(results_path)
    orders: list[dict[str, Any]] = []
    accepted = False
    if not risk_row["used_denoiser"]:
        selection_reason = "gate_not_called_after_sft_restoration"
    elif not str(risk_row.get("denoiser_response") or "").strip():
        selection_reason = "candidate_generation_failure"
    else:
        mapped: list[str] = []
        failures: list[str] = []
        for request in sorted(requests, key=lambda item: int(item["metadata"]["order_index"])):
            result = results.get(request["request_id"])
            error = None
            if result is None:
                error = "missing_result"
            elif result.get("request_sha256") != request_sha256(request):
                error = "request_hash_mismatch"
            elif result.get("ok") is not True:
                error = str(result.get("error_type") or "api_or_parse_failure")
            elif (result.get("normalized") or {}).get("choice") not in {"A", "B", "tie"}:
                error = "invalid_normalized_choice"
            item: dict[str, Any] = {
                "order_index": request["metadata"]["order_index"],
                "request_id": request["request_id"],
                "ok": error is None,
                "error": error,
                "request_source": "new_api_request_after_test40_extraction_fix",
            }
            if error is None and result is not None:
                raw_choice = result["normalized"]["choice"]
                choice = "tie" if raw_choice == "tie" else request["metadata"][f"variant_for_{raw_choice}"]
                mapped.append(choice)
                item.update(
                    {
                        "raw_choice": raw_choice,
                        "mapped_choice": choice,
                        "rationale": result["normalized"].get("rationale", ""),
                    }
                )
            else:
                failures.append(str(error))
            orders.append(item)
        if failures or len(orders) != 2:
            selection_reason = "selector_technical_failure"
        elif mapped == ["candidate", "candidate"]:
            accepted = True
            selection_reason = "candidate_selected_both_orders"
        elif "tie" in mapped:
            selection_reason = "selector_tie"
        elif len(set(mapped)) > 1:
            selection_reason = "selector_order_disagreement"
        elif mapped == ["sft", "sft"]:
            selection_reason = "sft_selected_both_orders"
        else:
            selection_reason = "selector_technical_failure"

    final_response = risk_row["denoiser_response"] if accepted else risk_row["sft_response"]
    final_source = "original_denoiser_candidate" if accepted else "SFT"
    v1_final_rows = read_jsonl(V1 / "test60_final_outputs.jsonl")
    v1_decisions = read_jsonl(V1 / "selector_decisions.jsonl")
    old_final = target_row(v1_final_rows, "v1 final")
    old_decision = target_row(v1_decisions, "v1 decision")
    fixed_sft = target_row(fixed_sft_rows, "fixed SFT")
    fixed_dpo = target_row(fixed_dpo_rows, "fixed DPO")
    new_final = {
        **old_final,
        "sft_response": fixed_sft["sft_response"],
        "dpo_response": fixed_dpo["sft_response"],
        "denoiser_called": bool(risk_row["used_denoiser"]),
        "denoiser_call_reason": risk_row["denoiser_call_reason"],
        "denoiser_response_raw": risk_row.get("denoiser_response_raw", ""),
        "denoiser_response": risk_row.get("denoiser_response", ""),
        "denoiser_generation": risk_row.get("denoiser_generation"),
        "sft_risk_score": risk_row["sft_risk_score"],
        "denoiser_risk_score": risk_row.get("denoiser_risk_score"),
        "current_rules_accepted_denoiser": bool(risk_row["accepted_denoiser"]),
        "current_rules_reject_reason": risk_row.get("reject_reason"),
        "current_rules_final_response": risk_row["final_response"],
        "gpt_selector_orders": orders,
        "gpt_selector_accepted_candidate": accepted,
        "gpt_selector_selection_reason": selection_reason,
        "final_response": final_response,
        "final_response_source": final_source,
        "test40_extraction_fix": {
            "version": VERSION,
            "previous_final_response_sha256": sha256_text(old_final["final_response"]),
            "extraction_audit": str(OUTPUT / "extraction_audit.json"),
        },
    }
    new_decision = {
        **old_decision,
        "denoiser_called": bool(risk_row["used_denoiser"]),
        "candidate_generation_success": bool(
            risk_row["used_denoiser"] and str(risk_row.get("denoiser_response") or "").strip()
        ),
        "selector_orders": orders,
        "accepted_candidate": accepted,
        "selection_reason": selection_reason,
        "final_source": final_source,
        "sft_response_sha256": sha256_text(fixed_sft["sft_response"]),
        "candidate_response_sha256": sha256_text(str(risk_row.get("denoiser_response") or "")),
        "final_response_sha256": sha256_text(final_response),
        "final_differs_from_sft": final_response != fixed_sft["sft_response"],
        "test40_extraction_fix_version": VERSION,
    }
    final_rows = replace_target(v1_final_rows, new_final)
    decisions = replace_target(v1_decisions, new_decision)
    write_jsonl(OUTPUT / "test60_final_outputs.jsonl", final_rows)
    write_jsonl(OUTPUT / "selector_decisions.jsonl", decisions)

    source_sets = {
        "SFT_generation": (read_jsonl(PART60 / "sft_test_outputs.jsonl"), fixed_sft_rows),
        "DPO_generation": (read_jsonl(PART60 / "dpo_test_outputs.jsonl"), fixed_dpo_rows),
        "Proposed_generation": (original_proposed_rows, fixed_proposed_rows),
        "final_outputs": (v1_final_rows, final_rows),
        "selector_decisions": (v1_decisions, decisions),
    }
    verification: dict[str, Any] = {
        "version": VERSION,
        "required_unchanged_count": 59,
        "target": CANONICAL_ID,
        "artifacts": {},
    }
    for name, (before_rows, after_rows) in source_sets.items():
        before = index_rows(before_rows, name + " before")
        after = index_rows(after_rows, name + " after")
        unchanged_ids = [canonical_id for canonical_id in before if canonical_id != CANONICAL_ID]
        mismatches = [canonical_id for canonical_id in unchanged_ids if before[canonical_id] != after[canonical_id]]
        if len(unchanged_ids) != 59 or mismatches:
            raise RuntimeError(f"{name} changed outside TEST-40: {mismatches}")
        verification["artifacts"][name] = {
            "unchanged_count": len(unchanged_ids),
            "mismatches": mismatches,
            "unchanged_row_hashes": {
                canonical_id: row_hash(before[canonical_id]) for canonical_id in unchanged_ids
            },
            "target_before_sha256": row_hash(before[CANONICAL_ID]),
            "target_after_sha256": row_hash(after[CANONICAL_ID]),
        }
    write_json(OUTPUT / "unchanged_59_verification.json", verification)

    differing = [
        row["canonical_id"]
        for row in final_rows
        if str(row["final_response"]) != str(row["sft_response"])
    ]
    called = sum(bool(row.get("used_denoiser")) for row in fixed_proposed_rows)
    candidate_success = sum(
        bool(row.get("used_denoiser")) and bool(str(row.get("denoiser_response") or "").strip())
        for row in fixed_proposed_rows
    )
    reason_counts: dict[str, int] = {}
    for decision in decisions:
        reason = str(decision["selection_reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    result_latest = latest_results(results_path)
    api_usage = {
        "new_requests": len(requests),
        "new_successful_requests": sum(row.get("ok") is True for row in result_latest.values()),
        "input_tokens": sum(
            int((row.get("usage") or {}).get("prompt_tokens") or 0)
            for row in result_latest.values()
            if row.get("ok") is True
        ),
        "output_tokens": sum(
            int((row.get("usage") or {}).get("completion_tokens") or 0)
            for row in result_latest.values()
            if row.get("ok") is True
        ),
        "estimated_cost_usd": round(
            sum(float(row.get("estimated_cost_usd") or 0.0) for row in result_latest.values() if row.get("ok") is True),
            8,
        ),
        "past_changed_input_selector_results_reused": 0,
    }
    write_json(OUTPUT / "api_usage.json", api_usage)
    gate = read_json(OUTPUT / "gate_recomputation.json")
    summary = {
        "version": VERSION,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": {"case_id": CASE_ID, "canonical_id": CANONICAL_ID},
        "restored": {
            "SFT": fixed_sft["sft_response"],
            "DPO": fixed_dpo["sft_response"],
            "SFT_sha256": sha256_text(fixed_sft["sft_response"]),
            "DPO_sha256": sha256_text(fixed_dpo["sft_response"]),
        },
        "before": {
            "gate": gate["before"],
            "gpt_selector_accepted_candidate": old_final["gpt_selector_accepted_candidate"],
            "selection_reason": old_final["gpt_selector_selection_reason"],
            "final_response": old_final["final_response"],
            "final_response_sha256": sha256_text(old_final["final_response"]),
        },
        "after": {
            "gate": gate["after"],
            "gpt_selector_accepted_candidate": accepted,
            "selection_reason": selection_reason,
            "final_response": final_response,
            "final_response_sha256": sha256_text(final_response),
        },
        "new_api": api_usage,
        "TEST60": {
            "SFT_complete": sum(bool(str(row.get("sft_response") or "").strip()) for row in fixed_sft_rows),
            "DPO_complete": sum(bool(str(row.get("sft_response") or "").strip()) for row in fixed_dpo_rows),
            "denoiser_called": called,
            "candidate_generation_success": candidate_success,
            "candidate_generation_failure": called - candidate_success,
            "GPT_final_adopted": sum(bool(row["accepted_candidate"]) for row in decisions),
            "Proposed_final_differs_from_SFT": len(differing),
            "Proposed_final_differs_from_SFT_ids": differing,
            "selection_reason_counts": dict(sorted(reason_counts.items())),
        },
        "unchanged_59_verified": True,
    }
    write_json(OUTPUT / "completion_summary.json", summary)

    report = f"""# TEST-40 Extraction Correction

Status: complete. Only `{CASE_ID}` (`{CANONICAL_ID}`) changed; all other 59 generation, final-response, and selector-decision row hashes were verified unchanged.

## Restored saved outputs

- SFT: {fixed_sft['sft_response']}
- DPO: {fixed_dpo['sft_response']}
- Restored text SHA-256: `{sha256_text(fixed_sft['sft_response'])}`

The saved raw generations were intact. The old extractor missed the standalone capitalized Markdown `**Final Response:**` label after a rewrite preface, then treated its inner `Response:` substring as the label and retained the preface instead of the answer body.

## Gate and selection

- Before: risk={gate['before']['sft_risk_score']}, words={gate['before']['sft_word_count']}, gate={gate['before']['denoiser_call_reason']}, selector={old_final['gpt_selector_selection_reason']}
- After: risk={gate['after']['sft_risk_score']}, words={gate['after']['sft_word_count']}, gate={gate['after']['denoiser_call_reason']}, selector={selection_reason}
- New GPT API calls: {api_usage['new_requests']}
- Final response source: {final_source}

## Updated TEST60

- Denoiser called: {called}
- GPT candidate adoptions: {summary['TEST60']['GPT_final_adopted']}
- Proposed differs from SFT: {len(differing)}
- IDs: {', '.join(differing) if differing else 'none'}
"""
    atomic_write(OUTPUT / "FINAL_REPORT.md", report.encode("utf-8"))

    key_files = [
        OUTPUT / "run_contract.json",
        OUTPUT / "run_contract.sha256",
        OUTPUT / "extraction_audit.json",
        OUTPUT / "code/selective_risk_refinement_utils.patch",
        OUTPUT / "history/test40_v1_history.json",
        OUTPUT / "inputs/test40_proposed_input.jsonl",
        OUTPUT / "generation/test40_proposed_recomputed.jsonl",
        OUTPUT / "generation/test40_proposed_recomputed.manifest.json",
        OUTPUT / "generation/sft_test_outputs.jsonl",
        OUTPUT / "generation/dpo_test_outputs.jsonl",
        OUTPUT / "generation/proposed_mask_off_test_outputs.jsonl",
        OUTPUT / "gate_recomputation.json",
        OUTPUT / "selector_request_contract.json",
        OUTPUT / "requests/test40_selector_requests.jsonl",
        results_path,
        OUTPUT / "api_usage.json",
        OUTPUT / "selector_decisions.jsonl",
        OUTPUT / "test60_final_outputs.jsonl",
        OUTPUT / "unchanged_59_verification.json",
        OUTPUT / "completion_summary.json",
        OUTPUT / "FINAL_REPORT.md",
    ]
    for path in key_files:
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = {
        "version": VERSION,
        "status": "complete",
        "target": {"case_id": CASE_ID, "canonical_id": CANONICAL_ID},
        "summary": summary["TEST60"],
        "files": {str(path.relative_to(OUTPUT)): file_record(path) for path in key_files},
    }
    write_json(OUTPUT / "FINAL_MANIFEST.json", manifest)
    bundle = OUTPUT / "TEST60_rubric_aligned_final_selection_v2_TEST40_fix_research_bundle.zip"
    temporary = bundle.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in key_files + [OUTPUT / "FINAL_MANIFEST.json"]:
            archive.write(path, str(path.relative_to(OUTPUT)))
    os.replace(temporary, bundle)
    write_json(
        OUTPUT / "DELIVERY.json",
        {
            "status": "ready",
            "bundle": file_record(bundle),
            "final_outputs": file_record(OUTPUT / "test60_final_outputs.jsonl", 60),
            "report": file_record(OUTPUT / "FINAL_REPORT.md"),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "prepare-selector", "finalize"))
    return parser.parse_args()


def main() -> None:
    command = parse_args().command
    if command == "prepare":
        prepare()
    elif command == "prepare-selector":
        prepare_selector()
    else:
        finalize()


if __name__ == "__main__":
    main()
