#!/usr/bin/env python3
"""Freeze, extend, and finalize the rubric-aligned GPT selector for TEST60."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


VERSION = "test60-rubric-aligned-final-selection-v1-20261009"
EXPECTED_WORKTREE_COMMIT = "c0007a5965ddc52c1779159d35b4c8184169f3cb"
MODEL = "gpt-4.1-2025-04-14"
TARGET_COUNT = 60
REPO = Path("/home/user/hsoh/mh-denoise")
WORKTREE = Path("/home/user/hsoh/mh-denoise-surface-relaxed-v1-20261008")
HANDOFF = Path("/home/user/hsoh/revision2_test_live_handoff_20261008")
PRIOR_SELECTOR = Path("/home/user/hsoh/rubric_aligned_gpt41_rejudge_20261009_v1")
OUTPUT = Path("/home/user/hsoh/test60_rubric_aligned_final_selection_20261009_v1")
RUN_ROOT = REPO / "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910"
GENERATION_ROOT = RUN_ROOT / "gemma/generation_1536_test_surface_relaxed_v1_20261008"
PART24 = GENERATION_ROOT / "part24"
PART60 = GENERATION_ROOT / "part60"
PARTIAL60 = REPO / "data/fullpaper_acl_pipeline/frozen_revision2_test_partial60_surface_relaxed_v1_20261008"
QUEUE = "test60_new4_selector_missing"


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
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


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


def row_id(row: dict[str, Any]) -> str:
    value = str(row.get("canonical_id") or row.get("id") or "")
    if not value:
        raise ValueError("Row has no canonical_id/id")
    return value


def file_record(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = rows
    return result


def index_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        canonical_id = row_id(row)
        if canonical_id in indexed:
            raise RuntimeError(f"Duplicate canonical ID in {label}: {canonical_id}")
        indexed[canonical_id] = row
    return indexed


def ordered_ids(path: Path) -> list[str]:
    payload = read_json(path)
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"No rows list: {path}")
    result = [row_id(row) for row in rows]
    if len(result) != len(set(result)):
        raise RuntimeError(f"Duplicate IDs in {path}")
    return result


def latest_snapshot() -> tuple[dict[str, Any], Path, dict[str, Any]]:
    manifest_path = HANDOFF / "manifest.json"
    manifest = read_json(manifest_path)
    if int(manifest.get("accepted_count") or 0) != TARGET_COUNT:
        raise RuntimeError(f"Latest handoff is not accepted {TARGET_COUNT}: {manifest.get('accepted_count')}")
    snapshot_path = HANDOFF / str(manifest["current_snapshot_file"])
    snapshot = read_json(snapshot_path)
    if snapshot.get("snapshot_id") != manifest.get("current_snapshot_id"):
        raise RuntimeError("Handoff manifest and snapshot ID differ")
    for filename, expected_hash in manifest.get("published_files", {}).items():
        path = HANDOFF / "snapshots" / filename
        if not path.exists() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Published snapshot artifact hash mismatch: {path}")
    return manifest, snapshot_path, snapshot


def git_output(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def source_settings(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    keep = {
        key: payload[key]
        for key in (
            "status",
            "base_model",
            "adapter_dir",
            "sft_adapter_dir",
            "risk_adapter_dir",
            "router_dir",
            "risk_scorer_dir",
            "risk_contract",
            "component_mode",
            "settings",
        )
        if key in payload
    }
    return {"source": file_record(path), "frozen_values": keep}


def collect_existing(method_filename: str, input_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    files = sorted(GENERATION_ROOT.glob(f"part*/{method_filename}"))
    for path in files:
        for row in read_jsonl(path):
            canonical_id = row_id(row)
            if canonical_id not in input_by_id:
                continue
            if row.get("question") != input_by_id[canonical_id].get("question"):
                raise RuntimeError(f"Existing output question mismatch for {canonical_id}: {path}")
            if canonical_id in by_id and by_id[canonical_id] != row:
                raise RuntimeError(f"Conflicting existing output for {canonical_id}: {sources[canonical_id]} vs {path}")
            by_id[canonical_id] = row
            sources[canonical_id] = str(path.resolve())
    return {
        "rows": by_id,
        "sources": sources,
        "files": [file_record(path, len(read_jsonl(path))) for path in files],
    }


def prepare_contract() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract_path = OUTPUT / "run_contract.json"
    if contract_path.exists():
        contract = read_json(contract_path)
        manifest, _, _ = latest_snapshot()
        if contract.get("handoff", {}).get("snapshot_id") != manifest.get("current_snapshot_id"):
            raise RuntimeError("Existing run contract points to a different handoff snapshot")
        if sha256_file(PRIOR_SELECTOR / "run_contract.json") != contract["selector"]["frozen_contract_sha256"]:
            raise RuntimeError("Frozen selector contract changed after TEST60 contract creation")
        print(json.dumps({"status": "reused_frozen_run_contract", "path": str(contract_path)}, sort_keys=True))
        return

    manifest, snapshot_path, snapshot = latest_snapshot()
    ids = list(manifest["accepted_ids_frozen_selection_order"])
    if len(ids) != TARGET_COUNT or len(set(ids)) != TARGET_COUNT:
        raise RuntimeError("Latest handoff does not contain 60 unique ordered IDs")
    if ids != snapshot.get("accepted_ids_frozen_selection_order"):
        raise RuntimeError("Handoff manifest and snapshot ordered IDs differ")

    partial_selection = PARTIAL60 / "partial_selection.json"
    input_path = PARTIAL60 / "sft_test.jsonl"
    partial_manifest = PARTIAL60 / "manifest.json"
    for path in (partial_selection, input_path, partial_manifest):
        if not path.exists():
            raise FileNotFoundError(f"Preflight partial export missing: {path}")
    if ordered_ids(partial_selection) != ids:
        raise RuntimeError("Partial60 order differs from latest handoff accepted order")
    input_rows = read_jsonl(input_path)
    if [row_id(row) for row in input_rows] != ids:
        raise RuntimeError("Partial60 SFT input order differs from latest handoff")
    input_by_id = index_rows(input_rows, "partial60 input")

    method_files = {
        "SFT": "sft_test_outputs.jsonl",
        "DPO": "dpo_test_outputs.jsonl",
        "Proposed_current_rules": "proposed_mask_off_test_outputs.jsonl",
    }
    existing: dict[str, Any] = {}
    for method, filename in method_files.items():
        state = collect_existing(filename, input_by_id)
        existing_ids = [canonical_id for canonical_id in ids if canonical_id in state["rows"]]
        existing[method] = {
            "reused_count": len(existing_ids),
            "new_count": TARGET_COUNT - len(existing_ids),
            "reused_ids": existing_ids,
            "source_files": state["files"],
        }
    counts = {value["reused_count"] for value in existing.values()}
    if counts != {24}:
        raise RuntimeError(f"Expected exactly 24 reusable rows per method before TEST60, got {sorted(counts)}")
    for method, value in existing.items():
        if value["reused_ids"] != ids[:24]:
            raise RuntimeError(f"{method} reusable IDs are not the frozen TEST24 prefix")

    dpo24 = read_jsonl(PART24 / "dpo_test_outputs.jsonl")
    if dpo24[0].get("sft_response", "").strip() == "**Final":
        raise RuntimeError("Recovered DPO TEST-01 response is not present in latest part24")
    if not str(dpo24[0].get("sft_response") or "").strip():
        raise RuntimeError("Recovered DPO TEST-01 sft_response is empty")

    selector_contract_path = PRIOR_SELECTOR / "run_contract.json"
    selector_contract = read_json(selector_contract_path)
    prompts = selector_contract["prompts"]
    model_settings = selector_contract["model_settings"]
    if model_settings != {
        "model": MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "pointwise_max_tokens": 384,
        "selector_max_tokens": 256,
    }:
        raise RuntimeError("Frozen selector model settings differ from expected completed contract")
    if sha256_text(prompts["selector_system"]) != prompts["selector_system_sha256"]:
        raise RuntimeError("Frozen selector system prompt hash mismatch")
    if sha256_text(prompts["selector_user_template"]) != prompts["selector_user_template_sha256"]:
        raise RuntimeError("Frozen selector user template hash mismatch")
    runner = REPO / "scripts/run_gpt41_checkpoint_queue.py"
    frozen_runner = PRIOR_SELECTOR / "code/run_gpt41_checkpoint_queue.py"
    if sha256_file(runner) != sha256_file(frozen_runner):
        raise RuntimeError("Current GPT checkpoint runner differs from the completed frozen runner")

    commit = git_output(["git", "rev-parse", "HEAD"], WORKTREE)
    if commit != EXPECTED_WORKTREE_COMMIT:
        raise RuntimeError(f"Inference worktree commit changed: {commit}")

    id_text = "".join(f"{canonical_id}\n" for canonical_id in ids)
    id_hash_rows = [
        {
            "source_order": index,
            "case_id": f"TEST-{index + 1:02d}",
            "canonical_id": canonical_id,
            "question_sha256": sha256_text(str(input_by_id[canonical_id]["question"])),
        }
        for index, canonical_id in enumerate(ids)
    ]
    write_frozen(OUTPUT / "test60_ids.txt", id_text.encode("utf-8"))
    write_frozen(OUTPUT / "input_id_question_hashes.jsonl", jsonl_bytes(id_hash_rows))

    generation_manifests = {
        "SFT": source_settings(PART24 / "sft_missing_outputs.manifest.json"),
        "DPO": source_settings(PART24 / "dpo_missing_outputs.manifest.json"),
        "Proposed": source_settings(PART24 / "proposed_missing_outputs.manifest.json"),
    }
    contract = {
        "version": VERSION,
        "status": "frozen_before_new_generation_or_api_calls",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": {"split": "TEST", "ordered_count": TARGET_COUNT},
        "handoff": {
            "manifest": file_record(HANDOFF / "manifest.json"),
            "snapshot": file_record(snapshot_path),
            "snapshot_id": manifest["current_snapshot_id"],
            "accepted_ids_sha256": sha256_text(id_text),
            "published_files": manifest["published_files"],
        },
        "inputs": {
            "partial_selection": file_record(partial_selection),
            "partial_manifest": file_record(partial_manifest),
            "sft_test": file_record(input_path, TARGET_COUNT),
            "id_question_hashes": file_record(OUTPUT / "input_id_question_hashes.jsonl", TARGET_COUNT),
        },
        "reuse_and_new": existing,
        "dpo_response_field": "sft_response",
        "dpo_test01_recovery_preserved": True,
        "inference": {
            "worktree": str(WORKTREE),
            "commit": commit,
            "required_commit": EXPECTED_WORKTREE_COMMIT,
            "generation_manifests_from_completed_test24": generation_manifests,
            "live_runner": file_record(WORKTREE / "scripts/run_h100_test_surface_relaxed_live_tmux.sh"),
            "generation_utility": file_record(WORKTREE / "scripts/h100_test_surface_relaxed_utils.py"),
            "acceptance_policy_for_candidate_generation": "surface_relaxed_v1",
            "adapter_control_or_feedback_arms_included": False,
        },
        "selector": {
            "frozen_contract": str(selector_contract_path),
            "frozen_contract_sha256": sha256_file(selector_contract_path),
            "model_settings": model_settings,
            "prompts": {
                "selector_system_sha256": prompts["selector_system_sha256"],
                "selector_user_template_sha256": prompts["selector_user_template_sha256"],
            },
            "rule": selector_contract["selector_rule"],
            "parser": "blind_selector",
            "runner": file_record(runner),
            "api_execution": {
                "initial_concurrency": 4,
                "max_concurrency": 8,
                "max_retries": 5,
                "timeout_seconds": 120,
            },
            "paid_api_authorized": True,
            "prior_five_item_selector_results_allowed": False,
        },
        "scope_guards": {
            "no_training": True,
            "no_response_regeneration_for_completed_rows": True,
            "no_quality_re_evaluation": True,
            "no_threshold_or_decoding_change": True,
            "no_first15_mapping_change": True,
        },
    }
    write_frozen(contract_path, json_bytes(contract))
    write_frozen(OUTPUT / "run_contract.sha256", (sha256_file(contract_path) + "  run_contract.json\n").encode("ascii"))
    print(
        json.dumps(
            {
                "status": "frozen_run_contract_created",
                "path": str(contract_path),
                "snapshot_id": manifest["current_snapshot_id"],
                "reused_per_method": 24,
                "new_per_method": 36,
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
    material = json.dumps(selected, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_text(material)


def request_id(
    version: str,
    queue: str,
    system_prompt: str,
    user_prompt: str,
    metadata: dict[str, Any],
) -> str:
    material = json.dumps(
        {
            "version": version,
            "queue": queue,
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


def build_request(
    *,
    version: str,
    queue: str,
    system_prompt: str,
    user_prompt: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": version,
        "queue": queue,
        "request_id": request_id(version, queue, system_prompt, user_prompt, metadata),
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
        "parser": "blind_selector",
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_sha256": sha256_text(user_prompt),
        "metadata": metadata,
    }


def load_and_validate_outputs() -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ids = [line.strip() for line in (OUTPUT / "test60_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != TARGET_COUNT or len(set(ids)) != TARGET_COUNT:
        raise RuntimeError("Frozen TEST60 ID file is invalid")
    paths = {
        "SFT": PART60 / "sft_test_outputs.jsonl",
        "DPO": PART60 / "dpo_test_outputs.jsonl",
        "Proposed": PART60 / "proposed_mask_off_test_outputs.jsonl",
    }
    loaded = {name: read_jsonl(path) for name, path in paths.items()}
    for method, rows in loaded.items():
        if [row_id(row) for row in rows] != ids:
            raise RuntimeError(f"{method} output order or membership differs from frozen TEST60")
        if len(rows) != TARGET_COUNT:
            raise RuntimeError(f"{method} output has {len(rows)} rows, expected {TARGET_COUNT}")
    for index, canonical_id in enumerate(ids):
        questions = {str(loaded[method][index].get("question") or "") for method in loaded}
        if len(questions) != 1 or not next(iter(questions)).strip():
            raise RuntimeError(f"Method question mismatch or empty question: {canonical_id}")
        if loaded["SFT"][index].get("sft_response") != loaded["Proposed"][index].get("sft_response"):
            raise RuntimeError(f"Proposed SFT draft differs from SFT output: {canonical_id}")
        policy = loaded["Proposed"][index].get("acceptance_policy")
        if policy != "surface_relaxed_v1":
            raise RuntimeError(f"Unexpected Proposed acceptance policy for {canonical_id}: {policy}")
    for method, filename in (
        ("SFT", "sft_test_outputs.jsonl"),
        ("DPO", "dpo_test_outputs.jsonl"),
        ("Proposed", "proposed_mask_off_test_outputs.jsonl"),
    ):
        previous = read_jsonl(PART24 / filename)
        if loaded[method][:24] != previous:
            raise RuntimeError(f"{method} TEST24 prefix was not reused byte-for-record")
    if loaded["DPO"][0].get("sft_response", "").strip() == "**Final":
        raise RuntimeError("DPO TEST-01 regressed to the unrecovered output")
    return ids, loaded["SFT"], loaded["DPO"], loaded["Proposed"]


def old_selector_artifacts() -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, dict[str, Any]]]:
    requests: dict[tuple[str, int], dict[str, Any]] = {}
    for path in (
        PRIOR_SELECTOR / "requests/expert15_original_selector.jsonl",
        PRIOR_SELECTOR / "requests/remaining_selector.jsonl",
    ):
        for row in read_jsonl(path):
            metadata = row.get("metadata") or {}
            if metadata.get("split") != "TEST" or metadata.get("candidate_type") != "original_denoiser":
                continue
            key = (str(metadata["canonical_id"]), int(metadata["order_index"]))
            if key in requests:
                raise RuntimeError(f"Duplicate prior selector request: {key}")
            requests[key] = row
    results: dict[str, dict[str, Any]] = {}
    for path in (
        PRIOR_SELECTOR / "results/expert15_original_selector_gpt41.jsonl",
        PRIOR_SELECTOR / "results/remaining_selector_gpt41.jsonl",
    ):
        for row in read_jsonl(path):
            metadata = row.get("metadata") or {}
            if metadata.get("split") == "TEST" and metadata.get("candidate_type") == "original_denoiser":
                results[str(row["request_id"])] = row
    return requests, results


def prepare_selector() -> None:
    contract = read_json(OUTPUT / "run_contract.json")
    selector_contract = read_json(PRIOR_SELECTOR / "run_contract.json")
    if sha256_file(PRIOR_SELECTOR / "run_contract.json") != contract["selector"]["frozen_contract_sha256"]:
        raise RuntimeError("Frozen selector contract hash changed")
    request_contract_path = OUTPUT / "selector_request_contract.json"
    if request_contract_path.exists():
        request_contract = read_json(request_contract_path)
        expected_generation = {
            "SFT": PART60 / "sft_test_outputs.jsonl",
            "DPO": PART60 / "dpo_test_outputs.jsonl",
            "Proposed_current_rules": PART60 / "proposed_mask_off_test_outputs.jsonl",
        }
        for name, path in expected_generation.items():
            if sha256_file(path) != request_contract["generation_outputs"][name]["sha256"]:
                raise RuntimeError(f"Generation output changed after selector request freeze: {name}")
        for record in request_contract["files"].values():
            path = Path(record["path"])
            if not path.exists() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"Frozen selector artifact changed or is missing: {path}")
        print(json.dumps({"status": "reused_frozen_selector_requests", **request_contract["counts"]}, sort_keys=True))
        return
    prompts = selector_contract["prompts"]
    system_prompt = prompts["selector_system"]
    user_template = prompts["selector_user_template"]
    ids, sft_rows, _, proposed_rows = load_and_validate_outputs()
    prior_requests, prior_results = old_selector_artifacts()

    candidates: list[dict[str, Any]] = []
    reused_requests: list[dict[str, Any]] = []
    reused_results: list[dict[str, Any]] = []
    new_requests: list[dict[str, Any]] = []
    reuse_failures: list[dict[str, Any]] = []
    expected_request_rows: list[dict[str, Any]] = []

    for source_order, (canonical_id, sft_row, proposed_row) in enumerate(zip(ids, sft_rows, proposed_rows, strict=True)):
        if not bool(proposed_row.get("used_denoiser")):
            continue
        question = str(sft_row.get("question") or "")
        sft_response = str(sft_row.get("sft_response") or "")
        candidate_response = str(proposed_row.get("denoiser_response") or "")
        candidate_raw = str(proposed_row.get("denoiser_response_raw") or "")
        technical_valid = bool(question.strip() and sft_response.strip() and candidate_response.strip())
        candidate = {
            "split": "TEST",
            "canonical_id": canonical_id,
            "case_id": f"TEST-{source_order + 1:02d}",
            "source_order": source_order,
            "candidate_type": "original_denoiser",
            "question": question,
            "question_sha256": sha256_text(question),
            "sft_response": sft_response,
            "sft_response_sha256": sha256_text(sft_response),
            "candidate_response": candidate_response,
            "candidate_response_sha256": sha256_text(candidate_response),
            "candidate_response_raw": candidate_raw,
            "candidate_response_raw_sha256": sha256_text(candidate_raw),
            "technical_valid": technical_valid,
            "sft_risk_score": proposed_row.get("sft_risk_score"),
            "candidate_risk_score": proposed_row.get("denoiser_risk_score"),
            "denoiser_call_reason": proposed_row.get("denoiser_call_reason"),
            "candidate_generation": proposed_row.get("denoiser_generation"),
            "current_rules_accepted": bool(proposed_row.get("accepted_denoiser")),
            "current_rules_reject_reason": proposed_row.get("reject_reason"),
            "current_rules_final_response_sha256": sha256_text(str(proposed_row.get("final_response") or "")),
            "source_path": str((PART60 / "proposed_mask_off_test_outputs.jsonl").resolve()),
        }
        candidates.append(candidate)
        if not technical_valid:
            continue

        for order_index in (0, 1):
            if order_index == 0:
                response_a, response_b = sft_response, candidate_response
                variant_a, variant_b = "sft", "candidate"
            else:
                response_a, response_b = candidate_response, sft_response
                variant_a, variant_b = "candidate", "sft"
            metadata = {
                "split": "TEST",
                "canonical_id": canonical_id,
                "case_id": candidate["case_id"],
                "source_order": source_order,
                "candidate_type": "original_denoiser",
                "order_index": order_index,
                "variant_for_A": variant_a,
                "variant_for_B": variant_b,
                "question_sha256": sha256_text(question),
                "response_A_sha256": sha256_text(response_a),
                "response_B_sha256": sha256_text(response_b),
            }
            user_prompt = user_template.format(question=question, response_a=response_a, response_b=response_b)
            old = prior_requests.get((canonical_id, order_index))
            can_reuse = bool(
                old
                and old.get("model") == MODEL
                and old.get("temperature") == 0
                and old.get("max_tokens") == 256
                and old.get("response_format") == {"type": "json_object"}
                and old.get("parser") == "blind_selector"
                and old.get("system_prompt") == system_prompt
                and old.get("user_prompt") == user_prompt
                and old.get("metadata") == metadata
            )
            if can_reuse:
                result = prior_results.get(str(old["request_id"]))
                can_reuse = bool(
                    result
                    and result.get("ok") is True
                    and result.get("request_sha256") == request_sha256(old)
                    and result.get("returned_model") == MODEL
                )
            if can_reuse:
                reused_requests.append(old)
                reused_results.append(result)
                expected_request_rows.append(
                    {
                        "canonical_id": canonical_id,
                        "order_index": order_index,
                        "request_id": old["request_id"],
                        "source": "reused_exact_frozen_request",
                    }
                )
                continue

            if old is not None:
                reuse_failures.append(
                    {
                        "canonical_id": canonical_id,
                        "order_index": order_index,
                        "old_request_id": old.get("request_id"),
                        "reason": "prior_request_or_result_not_exactly_reusable",
                    }
                )
            request = build_request(
                version=str(selector_contract["version"]),
                queue=QUEUE,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                metadata=metadata,
            )
            new_requests.append(request)
            expected_request_rows.append(
                {
                    "canonical_id": canonical_id,
                    "order_index": order_index,
                    "request_id": request["request_id"],
                    "source": "new_api_request",
                }
            )

    if len({(row["canonical_id"], row["order_index"]) for row in expected_request_rows}) != len(expected_request_rows):
        raise RuntimeError("Duplicate expected selector request key")
    if len({row["request_id"] for row in reused_requests + new_requests}) != len(reused_requests) + len(new_requests):
        raise RuntimeError("Duplicate selector request ID")

    request_dir = OUTPUT / "requests"
    result_dir = OUTPUT / "results"
    input_dir = OUTPUT / "inputs"
    write_frozen(input_dir / "test60_called_original_denoiser_candidates.jsonl", jsonl_bytes(candidates))
    write_frozen(request_dir / "reused_selector_requests.jsonl", jsonl_bytes(reused_requests))
    write_frozen(result_dir / "reused_selector_results.jsonl", jsonl_bytes(reused_results))
    write_frozen(request_dir / "new_selector_requests.jsonl", jsonl_bytes(new_requests))
    write_frozen(request_dir / "expected_selector_requests.jsonl", jsonl_bytes(expected_request_rows))

    technical_valid_count = sum(bool(row["technical_valid"]) for row in candidates)
    request_contract = {
        "version": VERSION,
        "status": "frozen_before_new_selector_api_calls",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_contract": file_record(OUTPUT / "run_contract.json"),
        "generation_outputs": {
            "SFT": file_record(PART60 / "sft_test_outputs.jsonl", TARGET_COUNT),
            "DPO": file_record(PART60 / "dpo_test_outputs.jsonl", TARGET_COUNT),
            "Proposed_current_rules": file_record(PART60 / "proposed_mask_off_test_outputs.jsonl", TARGET_COUNT),
        },
        "selector_contract": file_record(PRIOR_SELECTOR / "run_contract.json"),
        "counts": {
            "denoiser_called": len(candidates),
            "candidate_generation_success": technical_valid_count,
            "candidate_generation_failure": len(candidates) - technical_valid_count,
            "expected_selector_requests": technical_valid_count * 2,
            "reused_exact_requests": len(reused_requests),
            "new_api_requests": len(new_requests),
            "reuse_failures": len(reuse_failures),
        },
        "files": {
            "candidates": file_record(input_dir / "test60_called_original_denoiser_candidates.jsonl", len(candidates)),
            "reused_requests": file_record(request_dir / "reused_selector_requests.jsonl", len(reused_requests)),
            "reused_results": file_record(result_dir / "reused_selector_results.jsonl", len(reused_results)),
            "new_requests": file_record(request_dir / "new_selector_requests.jsonl", len(new_requests)),
            "expected_requests": file_record(request_dir / "expected_selector_requests.jsonl", len(expected_request_rows)),
        },
        "reuse_failures": reuse_failures,
    }
    write_frozen(request_contract_path, json_bytes(request_contract))
    print(json.dumps(request_contract["counts"], sort_keys=True))


def latest_results(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            result[str(row.get("request_id") or "")] = row
    return result


def result_status(request: dict[str, Any], result: dict[str, Any] | None) -> tuple[bool, str | None]:
    if result is None:
        return False, "missing_result"
    if result.get("request_sha256") != request_sha256(request):
        return False, "request_hash_mismatch"
    if result.get("ok") is not True:
        return False, str(result.get("error_type") or "api_or_parse_failure")
    normalized = result.get("normalized") or {}
    if normalized.get("choice") not in {"A", "B", "tie"}:
        return False, "invalid_normalized_choice"
    return True, None


def finalize() -> None:
    ids, sft_rows, dpo_rows, proposed_rows = load_and_validate_outputs()
    candidates = read_jsonl(OUTPUT / "inputs/test60_called_original_denoiser_candidates.jsonl")
    candidates_by_id = index_rows(candidates, "called candidates")
    reused_requests = read_jsonl(OUTPUT / "requests/reused_selector_requests.jsonl")
    new_requests = read_jsonl(OUTPUT / "requests/new_selector_requests.jsonl")
    requests = {str(row["request_id"]): row for row in reused_requests + new_requests}
    results = latest_results(
        (
            OUTPUT / "results/reused_selector_results.jsonl",
            OUTPUT / "results/new_selector_results.jsonl",
        )
    )

    requests_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for request in requests.values():
        metadata = request["metadata"]
        key = (str(metadata["canonical_id"]), int(metadata["order_index"]))
        if key in requests_by_key:
            raise RuntimeError(f"Duplicate selector request key during finalization: {key}")
        requests_by_key[key] = request

    decisions: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for source_order, (canonical_id, sft_row, dpo_row, proposed_row) in enumerate(
        zip(ids, sft_rows, dpo_rows, proposed_rows, strict=True)
    ):
        sft_response = str(sft_row.get("sft_response") or "")
        dpo_response = str(dpo_row.get("sft_response") or "")
        called = bool(proposed_row.get("used_denoiser"))
        candidate_response = str(proposed_row.get("denoiser_response") or "")
        selector_orders: list[dict[str, Any]] = []
        accepted = False
        if not called:
            reason = "gate_not_called"
        elif not candidate_response.strip():
            reason = "candidate_generation_failure"
        else:
            mapped_choices: list[str] = []
            technical_errors: list[str] = []
            for order_index in (0, 1):
                request = requests_by_key.get((canonical_id, order_index))
                result = results.get(str(request["request_id"])) if request else None
                valid, error = result_status(request, result) if request else (False, "missing_request")
                item: dict[str, Any] = {
                    "order_index": order_index,
                    "request_id": request.get("request_id") if request else None,
                    "request_source": next(
                        (
                            row["source"]
                            for row in read_jsonl(OUTPUT / "requests/expected_selector_requests.jsonl")
                            if row["canonical_id"] == canonical_id and int(row["order_index"]) == order_index
                        ),
                        None,
                    ),
                    "ok": valid,
                    "error": error,
                }
                if valid and request and result:
                    raw_choice = result["normalized"]["choice"]
                    if raw_choice == "tie":
                        mapped = "tie"
                    else:
                        variant = request["metadata"][f"variant_for_{raw_choice}"]
                        mapped = str(variant)
                    mapped_choices.append(mapped)
                    item.update(
                        {
                            "raw_choice": raw_choice,
                            "mapped_choice": mapped,
                            "rationale": result["normalized"].get("rationale", ""),
                        }
                    )
                else:
                    technical_errors.append(str(error))
                selector_orders.append(item)
            if technical_errors:
                reason = "selector_technical_failure"
            elif mapped_choices == ["candidate", "candidate"]:
                accepted = True
                reason = "candidate_selected_both_orders"
            elif "tie" in mapped_choices:
                reason = "selector_tie"
            elif len(set(mapped_choices)) > 1:
                reason = "selector_order_disagreement"
            elif mapped_choices == ["sft", "sft"]:
                reason = "sft_selected_both_orders"
            else:
                reason = "selector_technical_failure"
        final_response = candidate_response if accepted else sft_response
        decision = {
            "canonical_id": canonical_id,
            "case_id": f"TEST-{source_order + 1:02d}",
            "source_order": source_order,
            "denoiser_called": called,
            "candidate_generation_success": bool(called and candidate_response.strip()),
            "selector_orders": selector_orders,
            "accepted_candidate": accepted,
            "selection_reason": reason,
            "final_source": "original_denoiser_candidate" if accepted else "SFT",
            "sft_response_sha256": sha256_text(sft_response),
            "candidate_response_sha256": sha256_text(candidate_response),
            "final_response_sha256": sha256_text(final_response),
            "final_differs_from_sft": final_response != sft_response,
        }
        decisions.append(decision)
        final_rows.append(
            {
                "split": "TEST",
                "canonical_id": canonical_id,
                "case_id": decision["case_id"],
                "source_order": source_order,
                "question": sft_row.get("question"),
                "sft_response": sft_response,
                "dpo_response": dpo_response,
                "dpo_source_response_field": "sft_response",
                "denoiser_called": called,
                "denoiser_call_reason": proposed_row.get("denoiser_call_reason"),
                "denoiser_response_raw": proposed_row.get("denoiser_response_raw", ""),
                "denoiser_response": candidate_response,
                "denoiser_generation": proposed_row.get("denoiser_generation"),
                "sft_risk_score": proposed_row.get("sft_risk_score"),
                "denoiser_risk_score": proposed_row.get("denoiser_risk_score"),
                "current_rules_accepted_denoiser": bool(proposed_row.get("accepted_denoiser")),
                "current_rules_reject_reason": proposed_row.get("reject_reason"),
                "current_rules_final_response": proposed_row.get("final_response"),
                "gpt_selector_orders": selector_orders,
                "gpt_selector_accepted_candidate": accepted,
                "gpt_selector_selection_reason": reason,
                "final_response": final_response,
                "final_response_source": decision["final_source"],
            }
        )

    reason_counts: dict[str, int] = {}
    for row in decisions:
        reason = str(row["selection_reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    differing = [row["canonical_id"] for row in decisions if row["final_differs_from_sft"]]
    candidate_failures = sum(
        row["denoiser_called"] and not row["candidate_generation_success"] for row in decisions
    )
    selector_technical = reason_counts.get("selector_technical_failure", 0)
    summary = {
        "version": VERSION,
        "status": "complete_test60_final_selection",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "ordered_questions": TARGET_COUNT,
            "SFT_inference_complete": sum(bool(row.get("sft_response", "").strip()) for row in sft_rows),
            "DPO_inference_complete": sum(bool(row.get("sft_response", "").strip()) for row in dpo_rows),
            "Proposed_generation_rows_complete": len(proposed_rows),
            "denoiser_called": sum(bool(row.get("used_denoiser")) for row in proposed_rows),
            "candidate_generation_success": sum(
                bool(row.get("used_denoiser")) and bool(str(row.get("denoiser_response") or "").strip())
                for row in proposed_rows
            ),
            "candidate_generation_failure": candidate_failures,
            "GPT_final_adopted": sum(bool(row["accepted_candidate"]) for row in decisions),
            "Proposed_final_differs_from_SFT": len(differing),
            "selector_tie": reason_counts.get("selector_tie", 0),
            "selector_order_disagreement": reason_counts.get("selector_order_disagreement", 0),
            "selector_technical_failure": selector_technical,
            "technical_failure_total_including_candidate_generation": selector_technical + candidate_failures,
        },
        "selection_reason_counts_all_60": dict(sorted(reason_counts.items())),
        "Proposed_final_differs_from_SFT_ids": differing,
        "source_reuse": {
            "existing_TEST24_per_method": 24,
            "new_TEST25_to_TEST60_per_method": 36,
        },
        "paths": {
            "generation_dir": str(PART60),
            "final_outputs": str(OUTPUT / "test60_final_outputs.jsonl"),
            "selector_decisions": str(OUTPUT / "selector_decisions.jsonl"),
        },
    }
    write_jsonl(OUTPUT / "selector_decisions.jsonl", decisions)
    write_jsonl(OUTPUT / "test60_final_outputs.jsonl", final_rows)
    write_json(OUTPUT / "completion_summary.json", summary)

    new_results = read_jsonl(OUTPUT / "results/new_selector_results.jsonl")
    new_ok_latest = latest_results((OUTPUT / "results/new_selector_results.jsonl",))
    api_usage = {
        "new_result_rows_written": len(new_results),
        "new_unique_request_results": len(new_ok_latest),
        "new_successful_requests": sum(row.get("ok") is True for row in new_ok_latest.values()),
        "new_input_tokens": sum(
            int((row.get("usage") or {}).get("prompt_tokens") or 0)
            for row in new_ok_latest.values()
            if row.get("ok") is True
        ),
        "new_output_tokens": sum(
            int((row.get("usage") or {}).get("completion_tokens") or 0)
            for row in new_ok_latest.values()
            if row.get("ok") is True
        ),
        "new_estimated_cost_usd": round(
            sum(float(row.get("estimated_cost_usd") or 0.0) for row in new_ok_latest.values() if row.get("ok") is True),
            8,
        ),
        "reused_prior_successful_requests": len(read_jsonl(OUTPUT / "results/reused_selector_results.jsonl")),
    }
    write_json(OUTPUT / "api_usage.json", api_usage)

    report = f"""# TEST60 Rubric-Aligned GPT-4.1 Final Selection

Status: complete. This run reused the frozen TEST24 outputs and generated only TEST25-TEST60. It did not run training, response regeneration for completed rows, or quality re-evaluation.

## Completion

- SFT inference complete: {summary['counts']['SFT_inference_complete']}/60
- DPO inference complete: {summary['counts']['DPO_inference_complete']}/60
- Denoiser called: {summary['counts']['denoiser_called']}/60
- Candidate generation success/failure: {summary['counts']['candidate_generation_success']}/{summary['counts']['candidate_generation_failure']}
- GPT-4.1 candidate adoptions: {summary['counts']['GPT_final_adopted']}
- Final Proposed text different from SFT: {summary['counts']['Proposed_final_differs_from_SFT']}
- Different IDs: {', '.join(differing) if differing else 'none'}

## Selector Fallbacks

- Tie: {summary['counts']['selector_tie']}
- Order disagreement: {summary['counts']['selector_order_disagreement']}
- Selector/API/parse technical failure: {summary['counts']['selector_technical_failure']}
- Candidate-generation technical failure: {summary['counts']['candidate_generation_failure']}

The final selector used the frozen four-item expert rubric contract, GPT-4.1 model version, two reversed orders, and candidate adoption only when both orders selected the candidate. Risk reduction and minimum length were not added to final selection.
"""
    atomic_write(OUTPUT / "FINAL_REPORT.md", report.encode("utf-8"))

    key_files = [
        OUTPUT / "run_contract.json",
        OUTPUT / "run_contract.sha256",
        OUTPUT / "test60_ids.txt",
        OUTPUT / "input_id_question_hashes.jsonl",
        OUTPUT / "selector_request_contract.json",
        OUTPUT / "inputs/test60_called_original_denoiser_candidates.jsonl",
        OUTPUT / "requests/reused_selector_requests.jsonl",
        OUTPUT / "requests/new_selector_requests.jsonl",
        OUTPUT / "requests/expected_selector_requests.jsonl",
        OUTPUT / "results/reused_selector_results.jsonl",
        OUTPUT / "results/new_selector_results.jsonl",
        OUTPUT / "selector_decisions.jsonl",
        OUTPUT / "test60_final_outputs.jsonl",
        OUTPUT / "completion_summary.json",
        OUTPUT / "api_usage.json",
        OUTPUT / "FINAL_REPORT.md",
        PART60 / "sft_test_outputs.jsonl",
        PART60 / "sft_test_outputs.manifest.json",
        PART60 / "dpo_test_outputs.jsonl",
        PART60 / "dpo_test_outputs.manifest.json",
        PART60 / "proposed_mask_off_test_outputs.jsonl",
        PART60 / "proposed_mask_off_test_outputs.manifest.json",
    ]
    missing = [str(path) for path in key_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot finalize manifest/bundle; files missing: {missing}")
    manifest = {
        "version": VERSION,
        "status": "complete",
        "summary": summary["counts"],
        "files": {
            str(path.relative_to(OUTPUT)) if path.is_relative_to(OUTPUT) else f"generation/{path.name}": file_record(path)
            for path in key_files
        },
    }
    write_json(OUTPUT / "FINAL_MANIFEST.json", manifest)

    bundle = OUTPUT / "TEST60_rubric_aligned_final_selection_research_bundle.zip"
    bundle_files = key_files + [OUTPUT / "FINAL_MANIFEST.json"]
    temporary = bundle.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in bundle_files:
            if path.is_relative_to(OUTPUT):
                arcname = str(path.relative_to(OUTPUT))
            else:
                arcname = f"generation/{path.name}"
            archive.write(path, arcname)
    os.replace(temporary, bundle)
    write_json(
        OUTPUT / "DELIVERY.json",
        {
            "status": "ready",
            "bundle": file_record(bundle),
            "report": file_record(OUTPUT / "FINAL_REPORT.md"),
            "final_outputs": file_record(OUTPUT / "test60_final_outputs.jsonl", TARGET_COUNT),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare-contract", "prepare-selector", "finalize"))
    return parser.parse_args()


def main() -> None:
    command = parse_args().command
    if command == "prepare-contract":
        prepare_contract()
    elif command == "prepare-selector":
        prepare_selector()
    else:
        finalize()


if __name__ == "__main__":
    main()
