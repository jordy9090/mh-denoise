#!/usr/bin/env python3
"""Prepare authorized Psych8k/PsyQA clean-QA candidates without model calls.

The command is deliberately data-only and fail-closed.  It accepts an explicit
dataset kind, validates the corresponding schema/content signature, checks a
structured authorization attestation, and writes a new directory containing
``clean_candidates.jsonl``, ``generation_plan.jsonl``, and ``manifest.json``.
It never imports or loads a tokenizer/model, calls an API, generates text, or
starts training.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

try:
    from fullpaper_acl_pipeline import (
        SPLIT_RATIOS,
        UnionFind,
        clean_display,
        combined_question,
        largest_remainder_counts,
        normalize_text,
        read_jsonl,
        sha256_file,
        sha256_text,
        stable_random_key,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:  # Support importing as ``scripts.<module>`` in tests.
    from scripts.fullpaper_acl_pipeline import (  # type: ignore[no-redef]
        SPLIT_RATIOS,
        UnionFind,
        clean_display,
        combined_question,
        largest_remainder_counts,
        normalize_text,
        read_jsonl,
        sha256_file,
        sha256_text,
        stable_random_key,
        write_json,
        write_jsonl,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "data/fullpaper_acl_pipeline/canonical_clean_qa.jsonl"
DEFAULT_ACCEPTED_652 = (
    ROOT
    / "data/fullpaper_acl_pipeline"
    / "production_run_train1000_valid200_local_qwen35_27b_v3r1_20260910"
    / "accepted.jsonl"
)
VERSION = "authorized-supplement-preparation-v1-20260927"
PLAN_VERSION = "authorized-supplement-generation-plan-v1-20260927"
SPLIT_SEED = 20260904
GENERATION_SEED = 20260910
FROZEN_CANONICAL_ROWS = 18_224
FROZEN_CANONICAL_SHA256 = "0394c6665eb47d59192cea418f66f7c30f7427485257cfd75c1931c63258f9ef"
FROZEN_ACCEPTED_ROWS = 652
FROZEN_ACCEPTED_SPLITS = {"train": 545, "valid": 107}
FROZEN_ACCEPTED_SHA256 = "656ccbd5473b4bc7e3d3cb40a2682236a6202274e999f76e5c35b867c2dbde9b"

AXES = (
    "overall_quality",
    "empathy",
    "specificity",
    "factual_consistency",
    "medical_boundary",
    "toxicity_or_harm",
)
AXIS_COUNT_RATIOS = {1: 0.50, 2: 0.35, 3: 0.15}
GENERATOR = {
    "repo": "Qwen/Qwen3.5-4B",
    "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "temperature": 0.75,
    "top_p": 0.9,
    "do_sample": True,
    "max_input_tokens": 4096,
    "max_new_tokens_policy": "min(max(current_response_tokens + 256, 768), 1536)",
    "truncation_retry_ceiling": 2048,
    "max_semantic_attempts_per_stage": 2,
    "max_infrastructure_attempts": 4,
    "deterministic_seed": GENERATION_SEED,
    "enable_thinking": False,
    "execution_status": "planned_only_not_run",
}
QC_CONTRACT = {
    "repo": "Qwen/Qwen3.5-27B",
    "revision": "fc05daec18b0a78c049392ed2e771dde82bdf654",
    "qc_version": "local-qwen35-27b-paired-qc-v3-20260910",
    "paid_api_enabled": False,
    "execution_status": "not_started_wait_for_current_gpu_qc",
}
FROZEN_PRODUCTION_CONTRACT_HASHES = {
    "generation_prompt": "104a9023a609b0255c7d20774610b95a0b9bd4a14b07e11d764d74707fa52292",
    "acceptance_rules": "360ff68b5b959faeb13bd6662f340d620ab981079a3ac3a8de41e186aef20205",
    "runner_implementation": "1225f05e5d17e43fdda29ff4b4808e768283bc31cdac77bb1e58e946b66820b2",
    "local_qc_contract": "1d1b947b8d978a2aa9946e1b448dfd43df834b17b4ea311f1c3d2ea0984d9440",
}

DATASETS: dict[str, dict[str, Any]] = {
    "psych8k": {
        "source": "Psych8k",
        "repo": "EmoCareAI/Psych8k",
        "revision": "091787feccbce3e0adfd03b1ea3063f3d938c32d",
        "license": "cc-by-nc-sa-4.0",
        "expected_top_level_records": 8_187,
        "expected_answer_records": 8_187,
        "expected_bytes": 6_575_030,
        "expected_git_blob_sha1": "b0a9f254223cb3accb871872feb91156b9d9d719",
    },
    "psyqa-sample": {
        "source": "PsyQA-sample",
        "repo": "thu-coai/PsyQA",
        "revision": "e224c7e518c98a0c3df11e2fc5e6698044d8e156",
        "license": "official sample; repository terms apply",
        "expected_top_level_records": 100,
        "expected_answer_records": 268,
        "expected_sha256": "af0e8322727f4ea1a6dd5aac9c116ea56c488a704e3065e1382138e89d8a6946",
    },
    "psyqa-full": {
        "source": "PsyQA-full",
        "repo": "thu-coai/PsyQA",
        "revision": "e224c7e518c98a0c3df11e2fc5e6698044d8e156",
        "license": "approval-controlled PsyQA user agreement",
        "enabled": False,
        "disabled_reason": "register exact approved artifact hash and verified schema before use",
    },
}

PSYCH8K_INSTRUCTION_TOKENS = (
    ("counsellor", "counselor"),
    ("answer",),
    ("question",),
    ("patient", "client"),
)
PSYCH8K_EMBEDDED_PROMPT_RE = re.compile(
    r"^\s*if\s+you\s+are\s+a\s+counsell?or\s*,?\s*"
    r"please\s+answer\s+the\s+questions?\s+based\s+on\s+the\s+"
    r"description\s+of\s+the\s+patient\s*\.?\s*(?:description\s*:\s*)?",
    flags=re.IGNORECASE,
)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def deterministic_generation_seed(
    base_seed: int, canonical_id: str, stage_index: int, attempt: int
) -> int:
    material = f"{base_seed}:{canonical_id}:{stage_index}:{attempt}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF


def psych8k_instruction_ok(value: Any) -> bool:
    normalized = normalize_text(value)
    return bool(normalized) and all(any(token in normalized for token in alternatives) for alternatives in PSYCH8K_INSTRUCTION_TOKENS)


def role_name(value: Any) -> str:
    role = normalize_text(value)
    aliases = {
        "human": "human",
        "user": "human",
        "client": "human",
        "gpt": "assistant",
        "assistant": "assistant",
        "counsellor": "assistant",
        "counselor": "assistant",
    }
    return aliases.get(role, role)


def psych8k_user_question(value: Any) -> tuple[str, bool]:
    """Remove the known training instruction only when embedded in a user turn."""
    original = str(value or "").strip()
    stripped = PSYCH8K_EMBEDDED_PROMPT_RE.sub("", original, count=1).strip()
    return (stripped, True) if stripped and stripped != original else (original, False)


def load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Authorized source must be a non-empty JSON list")
    if not all(isinstance(row, dict) for row in payload):
        raise ValueError("Every authorized source record must be a JSON object")
    return payload


def git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def validate_official_artifact(
    path: Path, dataset_kind: str, *, enforce_official_identity: bool
) -> dict[str, Any]:
    spec = DATASETS[dataset_kind]
    observed = {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "git_blob_sha1": git_blob_sha1(path),
        "official_identity_enforced": enforce_official_identity,
    }
    if enforce_official_identity and dataset_kind == "psych8k":
        if (
            observed["bytes"] != spec["expected_bytes"]
            or observed["git_blob_sha1"] != spec["expected_git_blob_sha1"]
        ):
            raise ValueError(
                "Psych8k pinned artifact identity mismatch: expected "
                f"{spec['expected_bytes']} bytes / git blob {spec['expected_git_blob_sha1']}, "
                f"found {observed['bytes']} bytes / {observed['git_blob_sha1']}"
            )
    if (
        enforce_official_identity
        and dataset_kind == "psyqa-sample"
        and observed["sha256"] != spec["expected_sha256"]
    ):
        raise ValueError(
            "PsyQA sample artifact identity mismatch: expected SHA-256 "
            f"{spec['expected_sha256']}, found {observed['sha256']}"
        )
    return observed


def validate_authorization_attestation(
    path: Path, dataset_kind: str, *, as_of: date | None = None
) -> dict[str, Any]:
    """Validate an auditable authorization statement before opening source data."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PermissionError("Authorization record must be structured UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise PermissionError("Authorization attestation must be a JSON object")
    spec = DATASETS[dataset_kind]
    expected = {
        "dataset_kind": dataset_kind,
        "repo": spec["repo"],
        "revision": spec["revision"],
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise PermissionError(
                f"Authorization attestation {field} mismatch: expected {value!r}"
            )
    status = str(payload.get("status") or "").casefold()
    if status not in {"approved", "authorized"}:
        raise PermissionError("Authorization attestation status must be approved or authorized")
    identity = payload.get("authorized_identity")
    if not isinstance(identity, dict) or not all(
        clean_display(identity.get(field)) for field in ("name", "account")
    ):
        raise PermissionError(
            "Authorization attestation requires authorized_identity.name and .account"
        )
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict) or not all(
        clean_display(evidence.get(field)) for field in ("type", "reference", "sha256")
    ):
        raise PermissionError(
            "Authorization attestation requires evidence.type, .reference, and .sha256"
        )
    if re.fullmatch(r"[0-9a-f]{64}", str(evidence["sha256"]).casefold()) is None:
        raise PermissionError("Authorization evidence.sha256 must be 64 lowercase hex characters")
    allowed_uses = payload.get("allowed_uses")
    if (
        not isinstance(allowed_uses, list)
        or not all(isinstance(item, str) and item.strip() for item in allowed_uses)
        or "data_preparation" not in allowed_uses
    ):
        raise PermissionError(
            "Authorization attestation allowed_uses must include data_preparation"
        )
    try:
        valid_through = date.fromisoformat(str(payload.get("valid_through") or ""))
    except ValueError as exc:
        raise PermissionError("Authorization attestation needs ISO valid_through") from exc
    today = as_of or date.today()
    if valid_through < today:
        raise PermissionError(
            f"Authorization attestation expired on {valid_through.isoformat()}"
        )
    if payload.get("valid_from") is not None:
        try:
            valid_from = date.fromisoformat(str(payload["valid_from"]))
        except ValueError as exc:
            raise PermissionError("Authorization attestation valid_from must be ISO date") from exc
        if valid_from > today:
            raise PermissionError(
                f"Authorization attestation is not active until {valid_from.isoformat()}"
            )
    return {
        "dataset_kind": dataset_kind,
        "repo": spec["repo"],
        "revision": spec["revision"],
        "status": status,
        "authorized_identity": identity,
        "evidence": evidence,
        "allowed_uses": allowed_uses,
        "valid_from": payload.get("valid_from"),
        "valid_through": valid_through.isoformat(),
        "validated_as_of": today.isoformat(),
    }


def validate_official_cardinality(
    dataset_kind: str, *, top_level_records: int, answer_records: int
) -> None:
    spec = DATASETS[dataset_kind]
    expected_top = spec.get("expected_top_level_records")
    expected_answers = spec.get("expected_answer_records")
    minimum_top = spec.get("minimum_top_level_records")
    minimum_answers = spec.get("minimum_answer_records")
    if expected_top is not None and top_level_records != expected_top:
        raise ValueError(
            f"{dataset_kind} identity check failed: expected {expected_top} top-level records, "
            f"found {top_level_records}"
        )
    if expected_answers is not None and answer_records != expected_answers:
        raise ValueError(
            f"{dataset_kind} identity check failed: expected {expected_answers} answer records, "
            f"found {answer_records}"
        )
    if minimum_top is not None and top_level_records < minimum_top:
        raise ValueError(
            f"{dataset_kind} looks like the sample or a partial file: expected at least "
            f"{minimum_top} top-level records, found {top_level_records}"
        )
    if minimum_answers is not None and answer_records < minimum_answers:
        raise ValueError(
            f"{dataset_kind} looks like the sample or a partial file: expected at least "
            f"{minimum_answers} answers, found {answer_records}"
        )


def make_candidate(
    *,
    dataset_kind: str,
    detected_schema: str,
    raw_record: dict[str, Any],
    row_index: int,
    answer_index: int,
    question: str,
    answer: str,
    source_question_id: str,
    source_group_id: str,
    source_group_basis: str,
) -> dict[str, Any]:
    spec = DATASETS[dataset_kind]
    question = str(question).strip()
    answer = str(answer).strip()
    if not normalize_text(question):
        raise ValueError(f"Empty question after normalization at source row {row_index}")
    if not normalize_text(answer):
        raise ValueError(
            f"Empty original answer after normalization at source row {row_index}, answer {answer_index}"
        )
    question_exact = clean_display(question)
    response_exact = clean_display(answer)
    question_normalized = normalize_text(question)
    response_normalized = normalize_text(answer)
    identity = "\0".join(
        [
            "fullpaper-clean-qa-v1",
            spec["repo"],
            spec["revision"],
            detected_schema,
            str(row_index),
            source_group_id,
            sha256_text(answer),
        ]
    )
    canonical_id = "qa_" + sha256_text(identity)[:24]
    return {
        "schema_version": "canonical-clean-qa-v1",
        "supplement_preparation_version": VERSION,
        "canonical_id": canonical_id,
        "dataset_kind": dataset_kind,
        "source": spec["source"],
        "source_component": detected_schema,
        "source_repo": spec["repo"],
        "source_revision": spec["revision"],
        "source_row_index": row_index,
        "source_answer_index": answer_index,
        "source_question_id": source_question_id,
        "source_group_id": source_group_id,
        "source_group_basis": source_group_basis,
        "source_record_sha256": canonical_json_sha256(raw_record),
        "question": question,
        "clean_response": answer,
        "answer_role": "original_answer_from_authorized_source",
        "question_exact_sha256": sha256_text(question_exact),
        "question_normalized": question_normalized,
        "question_normalized_sha256": sha256_text(question_normalized),
        "response_exact_sha256": sha256_text(response_exact),
        "response_normalized": response_normalized,
        "response_normalized_sha256": sha256_text(response_normalized),
        "duplicate_cluster_id": "",
        "linked_existing_duplicate_cluster_ids": [],
        "split": "",
        "split_basis": "",
    }


def extract_psych8k(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    variants: Counter[str] = Counter()
    instruction_alias_rows = 0
    embedded_prompt_rows = 0
    candidates: list[dict[str, Any]] = []
    for row_index, raw in enumerate(rows):
        has_alpaca = {"input", "output"} <= set(raw) and bool(
            {"instruction", "instructions"} & set(raw)
        )
        has_sharegpt = isinstance(raw.get("conversations"), list)
        if has_alpaca == has_sharegpt:
            raise ValueError(
                f"Psych8k schema mismatch at row {row_index}: expected exactly one of "
                "instruction/input/output or ShareGPT conversations"
            )
        if has_alpaca:
            if "instruction" in raw and "instructions" in raw:
                raise ValueError(f"Ambiguous Psych8k instruction keys at row {row_index}")
            instruction_key = "instruction" if "instruction" in raw else "instructions"
            instruction_alias_rows += instruction_key == "instructions"
            if not psych8k_instruction_ok(raw.get(instruction_key)):
                raise ValueError(
                    f"Psych8k content signature mismatch in {instruction_key!r} at row {row_index}"
                )
            group_field = next(
                (
                    field
                    for field in ("conversation_id", "session_id", "dialogue_id", "source_id")
                    if clean_display(raw.get(field))
                ),
                None,
            )
            group_value = clean_display(raw.get(group_field)) if group_field else str(row_index)
            group_basis = group_field or "immutable source-row fallback; no released session ID"
            variants["psych8k_alpaca_instruction_input_output"] += 1
            candidates.append(
                make_candidate(
                    dataset_kind="psych8k",
                    detected_schema="psych8k_alpaca_instruction_input_output",
                    raw_record=raw,
                    row_index=row_index,
                    answer_index=0,
                    question=str(raw.get("input") or ""),
                    answer=str(raw.get("output") or ""),
                    source_question_id=f"psych8k:row:{row_index}",
                    source_group_id=f"psych8k:{group_basis}:{group_value}",
                    source_group_basis=group_basis,
                )
            )
            continue

        conversations = raw["conversations"]
        if len(conversations) != 2:
            raise ValueError(
                f"Psych8k ShareGPT row {row_index} must contain exactly one human/assistant pair"
            )
        if not all(isinstance(turn, dict) and {"from", "value"} <= set(turn) for turn in conversations):
            raise ValueError(f"Psych8k ShareGPT turn schema mismatch at row {row_index}")
        roles = [role_name(turn["from"]) for turn in conversations]
        expected = ["human" if index % 2 == 0 else "assistant" for index in range(len(roles))]
        if roles != expected:
            raise ValueError(
                f"Psych8k ShareGPT role/content order mismatch at row {row_index}: {roles}"
            )
        group_field = next(
            (
                field
                for field in ("conversation_id", "session_id", "dialogue_id", "source_id", "id")
                if clean_display(raw.get(field))
            ),
            None,
        )
        group_value = clean_display(raw.get(group_field)) if group_field else str(row_index)
        group_basis = group_field or "immutable source-row fallback; no released session ID"
        variants["psych8k_sharegpt_human_assistant"] += 1
        question, prompt_removed = psych8k_user_question(conversations[0]["value"])
        if not prompt_removed:
            raise ValueError(
                f"Psych8k ShareGPT content signature mismatch at row {row_index}: "
                "known counseling instruction prefix is required"
            )
        embedded_prompt_rows += 1
        candidates.append(
            make_candidate(
                dataset_kind="psych8k",
                detected_schema="psych8k_sharegpt_human_assistant",
                raw_record=raw,
                row_index=row_index,
                answer_index=0,
                question=question,
                answer=str(conversations[1]["value"]),
                source_question_id=f"psych8k:row:{row_index}:pair:0",
                source_group_id=f"psych8k:{group_basis}:{group_value}",
                source_group_basis=group_basis,
            )
        )
    if len(variants) != 1:
        raise ValueError(f"Psych8k file mixes incompatible top-level schemas: {dict(variants)}")
    return candidates, {
        "detected_schema": next(iter(variants)),
        "schema_variant_counts": dict(variants),
        "legacy_instructions_alias_rows": instruction_alias_rows,
        "sharegpt_embedded_instruction_removed": embedded_prompt_rows,
        "top_level_records": len(rows),
        "answer_records": len(candidates),
    }


def extract_psyqa(
    rows: list[dict[str, Any]], dataset_kind: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row_index, raw in enumerate(rows):
        required = {"questionID", "question", "description", "answers"}
        if not required <= set(raw) or not isinstance(raw.get("answers"), list):
            raise ValueError(
                f"PsyQA schema mismatch at row {row_index}: required keys are {sorted(required)}"
            )
        question_id = clean_display(raw.get("questionID"))
        if not question_id:
            raise ValueError(f"PsyQA questionID is empty at row {row_index}")
        answers = raw["answers"]
        if not answers:
            raise ValueError(f"PsyQA answers is empty at row {row_index}")
        question = combined_question(raw.get("question"), raw.get("description"))
        for answer_index, answer in enumerate(answers):
            if not isinstance(answer, dict) or "answer_text" not in answer:
                raise ValueError(
                    f"PsyQA answer schema mismatch at row {row_index}, answer {answer_index}"
                )
            answer_id = clean_display(answer.get("answerID")) or str(answer_index)
            candidates.append(
                make_candidate(
                    dataset_kind=dataset_kind,
                    detected_schema="psyqa_question_description_answers",
                    raw_record=raw,
                    row_index=row_index,
                    answer_index=answer_index,
                    question=question,
                    answer=str(answer.get("answer_text") or ""),
                    source_question_id=f"psyqa:questionID:{question_id}:answerID:{answer_id}",
                    source_group_id=f"psyqa:questionID:{question_id}",
                    source_group_basis="questionID; all original answers for one question stay together",
                )
            )
    return candidates, {
        "detected_schema": "psyqa_question_description_answers",
        "schema_variant_counts": {"psyqa_question_description_answers": len(rows)},
        "top_level_records": len(rows),
        "answer_records": len(candidates),
    }


def extract_candidates(
    rows: list[dict[str, Any]], dataset_kind: str, *, enforce_cardinality: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if dataset_kind == "psych8k":
        candidates, identity = extract_psych8k(rows)
    elif dataset_kind in {"psyqa-sample", "psyqa-full"}:
        candidates, identity = extract_psyqa(rows, dataset_kind)
    else:
        raise ValueError(f"Unsupported explicit dataset kind: {dataset_kind}")
    if enforce_cardinality:
        validate_official_cardinality(
            dataset_kind,
            top_level_records=identity["top_level_records"],
            answer_records=identity["answer_records"],
        )
    return candidates, identity


def validate_canonical(
    canonical_path: Path, canonical_rows: list[dict[str, Any]], *, enforce_frozen: bool
) -> dict[str, Any]:
    observed_hash = sha256_file(canonical_path)
    if enforce_frozen and (
        len(canonical_rows) != FROZEN_CANONICAL_ROWS or observed_hash != FROZEN_CANONICAL_SHA256
    ):
        raise RuntimeError(
            "Frozen canonical identity mismatch; refusing to prepare against a different corpus"
        )
    required = {
        "canonical_id",
        "split",
        "question_exact_sha256",
        "question_normalized_sha256",
        "response_exact_sha256",
        "response_normalized_sha256",
        "source_group_id",
        "duplicate_cluster_id",
    }
    for index, row in enumerate(canonical_rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Canonical row {index} is missing fields: {sorted(missing)}")
        if row["split"] not in SPLIT_RATIOS:
            raise ValueError(f"Canonical row {index} has invalid split: {row['split']!r}")
    return {
        "path": str(canonical_path),
        "rows": len(canonical_rows),
        "sha256": observed_hash,
        "frozen_identity_enforced": enforce_frozen,
    }


def validate_accepted_652(
    accepted_path: Path | None, *, enforce_frozen: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fingerprint the immutable merge anchor without changing or adapting it."""
    if accepted_path is None:
        if enforce_frozen:
            raise ValueError("The frozen accepted-652 merge anchor is required")
        return {
            "path": str(DEFAULT_ACCEPTED_652),
            "rows": FROZEN_ACCEPTED_ROWS,
            "split_counts": FROZEN_ACCEPTED_SPLITS,
            "sha256": FROZEN_ACCEPTED_SHA256,
            "validation": "skipped_only_for_synthetic_test_fixture",
        }, []
    accepted_path = accepted_path.resolve()
    if not accepted_path.is_file():
        raise FileNotFoundError(f"Missing accepted-652 merge anchor: {accepted_path}")
    rows = list(read_jsonl(accepted_path))
    digest = sha256_file(accepted_path)
    ids = [str(row.get("canonical_id") or "") for row in rows]
    if any(not row.get("question_normalized_sha256") for row in rows):
        raise RuntimeError("Accepted-652 merge anchor is missing normalized question hashes")
    split_counts = dict(sorted(Counter(str(row.get("split")) for row in rows).items()))
    if len(ids) != len(set(ids)) or not all(ids):
        raise RuntimeError("Accepted-652 merge anchor has missing or duplicate canonical IDs")
    if enforce_frozen and (
        len(rows) != FROZEN_ACCEPTED_ROWS
        or split_counts != FROZEN_ACCEPTED_SPLITS
        or digest != FROZEN_ACCEPTED_SHA256
    ):
        raise RuntimeError(
            "Frozen accepted-652 identity mismatch; refusing to plan against a different reuse set"
        )
    return {
        "path": str(accepted_path),
        "rows": len(rows),
        "unique_canonical_ids": len(set(ids)),
        "split_counts": split_counts,
        "sha256": digest,
        "frozen_identity_enforced": enforce_frozen,
        "merge_policy": "keep all existing IDs/text/splits immutable; append only prepared non-overlapping candidates later",
    }, rows


def canonical_indexes(
    canonical_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, set[str]]], dict[tuple[str, str], set[str]]]:
    fields = (
        "question_exact_sha256",
        "question_normalized_sha256",
        "response_exact_sha256",
        "response_normalized_sha256",
        "source_group_id",
    )
    split_index: dict[str, dict[str, set[str]]] = {
        field: defaultdict(set) for field in fields
    }
    clusters: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in canonical_rows:
        for field in fields:
            value = str(row[field])
            split_index[field][value].add(str(row["split"]))
            clusters[(field, value)].add(str(row["duplicate_cluster_id"]))
    conflicts = {
        field: [value for value, splits in values.items() if len(splits) > 1]
        for field, values in split_index.items()
    }
    if any(conflicts.values()):
        raise RuntimeError(
            "Existing canonical violates split isolation: "
            + repr({field: values[:3] for field, values in conflicts.items() if values})
        )
    return split_index, clusters


def deduplicate_pairs(
    candidates: list[dict[str, Any]],
    canonical_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    existing_pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in canonical_rows:
        existing_pairs[
            (
                str(row["question_normalized_sha256"]),
                str(row["response_normalized_sha256"]),
            )
        ].add(str(row["split"]))
    excluded: list[dict[str, Any]] = []
    new_pairs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        pair = (
            str(row["question_normalized_sha256"]),
            str(row["response_normalized_sha256"]),
        )
        if pair in existing_pairs:
            excluded.append(
                {
                    "canonical_id": row["canonical_id"],
                    "source_row_index": row["source_row_index"],
                    "source_answer_index": row["source_answer_index"],
                    "question_normalized_sha256": row["question_normalized_sha256"],
                    "response_normalized_sha256": row["response_normalized_sha256"],
                    "reason": "existing_canonical_normalized_question_response_pair",
                    "existing_splits": sorted(existing_pairs[pair]),
                }
            )
        else:
            new_pairs[pair].append(row)

    kept: list[dict[str, Any]] = []
    for pair in sorted(new_pairs):
        members = sorted(
            new_pairs[pair],
            key=lambda row: stable_random_key(GENERATION_SEED, row["canonical_id"]),
        )
        kept.append(members[0])
        for duplicate in members[1:]:
            excluded.append(
                {
                    "canonical_id": duplicate["canonical_id"],
                    "source_row_index": duplicate["source_row_index"],
                    "source_answer_index": duplicate["source_answer_index"],
                    "question_normalized_sha256": pair[0],
                    "response_normalized_sha256": pair[1],
                    "kept_canonical_id": members[0]["canonical_id"],
                    "reason": "within_source_normalized_question_response_pair_duplicate",
                }
            )
    kept.sort(key=lambda row: row["canonical_id"])
    return kept, excluded


def select_generation_candidates(
    rows: list[dict[str, Any]], *, covered_question_hashes: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select one deterministic original answer per normalized question."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["question_normalized_sha256"])].append(row)
    selected: list[dict[str, Any]] = []
    covered: list[dict[str, Any]] = []
    for question_hash in sorted(groups):
        members = sorted(
            groups[question_hash],
            key=lambda row: stable_random_key(GENERATION_SEED, row["canonical_id"]),
        )
        if question_hash in covered_question_hashes:
            covered.append(
                {
                    "question_normalized_sha256": question_hash,
                    "clean_candidate_ids": [row["canonical_id"] for row in members],
                    "reason": "covered_by_existing_accepted652_question",
                }
            )
            continue
        selected.append({**members[0], "plan_version": PLAN_VERSION})
    return selected, covered


def build_new_components(rows: list[dict[str, Any]]) -> list[list[int]]:
    union = UnionFind(len(rows))
    seen: dict[tuple[str, str], int] = {}
    fields = (
        "source_group_id",
        "question_exact_sha256",
        "question_normalized_sha256",
        "response_exact_sha256",
        "response_normalized_sha256",
    )
    for index, row in enumerate(rows):
        for field in fields:
            key = (field, str(row[field]))
            previous = seen.setdefault(key, index)
            union.union(index, previous)
    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        members[union.find(index)].append(index)
    return list(members.values())


def assign_components_and_splits(
    rows: list[dict[str, Any]],
    canonical_split_index: dict[str, dict[str, set[str]]],
    canonical_cluster_index: dict[tuple[str, str], set[str]],
) -> dict[str, Any]:
    components = build_new_components(rows)
    split_names = tuple(SPLIT_RATIOS)
    source_totals = Counter(row["source"] for row in rows)
    targets_source = {
        source: largest_remainder_counts(total, SPLIT_RATIOS)
        for source, total in source_totals.items()
    }
    targets_total = largest_remainder_counts(len(rows), SPLIT_RATIOS)
    current_total: Counter[str] = Counter()
    current_source: dict[str, Counter[str]] = {
        source: Counter() for source in source_totals
    }
    anchored_components = 0
    anchored_rows = 0
    pending: list[tuple[str, list[int], Counter[str]]] = []
    linkage_fields = (
        "question_exact_sha256",
        "question_normalized_sha256",
        "response_exact_sha256",
        "response_normalized_sha256",
        "source_group_id",
    )
    for indices in components:
        member_ids = sorted(rows[index]["canonical_id"] for index in indices)
        new_component_id = "dup_" + sha256_text("\n".join(member_ids))[:24]
        anchor_splits: set[str] = set()
        linked_clusters: set[str] = set()
        for index in indices:
            row = rows[index]
            for field in linkage_fields:
                value = str(row[field])
                anchor_splits.update(canonical_split_index[field].get(value, set()))
                linked_clusters.update(canonical_cluster_index.get((field, value), set()))
        if len(anchor_splits) > 1:
            raise RuntimeError(
                f"New connected component links multiple existing splits: {member_ids[:3]} -> "
                f"{sorted(anchor_splits)}"
            )
        if len(linked_clusters) > 1:
            raise RuntimeError(
                "New connected component bridges multiple immutable existing duplicate clusters: "
                f"{member_ids[:3]} -> {sorted(linked_clusters)}"
            )
        component_id = next(iter(linked_clusters)) if linked_clusters else new_component_id
        for index in indices:
            rows[index]["duplicate_cluster_id"] = component_id
            rows[index]["linked_existing_duplicate_cluster_ids"] = sorted(linked_clusters)
        unit_sources = Counter(rows[index]["source"] for index in indices)
        if anchor_splits:
            chosen = next(iter(anchor_splits))
            anchored_components += 1
            anchored_rows += len(indices)
            basis = "anchored_to_existing_canonical_linkage"
            for index in indices:
                rows[index]["split"] = chosen
                rows[index]["split_basis"] = basis
            current_total[chosen] += len(indices)
            for source, count in unit_sources.items():
                current_source[source][chosen] += count
        else:
            pending.append((component_id, indices, unit_sources))

    pending.sort(key=lambda item: (-len(item[1]), stable_random_key(SPLIT_SEED, item[0])))

    def loss(candidate: str, unit_sources: Counter[str]) -> int:
        result = 0
        for split in split_names:
            for source in source_totals:
                value = current_source[source][split]
                if split == candidate:
                    value += unit_sources[source]
                result += (value - targets_source[source][split]) ** 2
        return result

    for component_id, indices, unit_sources in pending:
        tie_order = sorted(
            split_names,
            key=lambda split: stable_random_key(SPLIT_SEED, f"{component_id}:{split}"),
        )
        chosen = min(tie_order, key=lambda split: loss(split, unit_sources))
        for index in indices:
            rows[index]["split"] = chosen
            rows[index]["split_basis"] = "deterministic_new_component_assignment"
        current_total[chosen] += len(indices)
        for source, count in unit_sources.items():
            current_source[source][chosen] += count

    return {
        "seed": SPLIT_SEED,
        "ratios": SPLIT_RATIOS,
        "targets": targets_total,
        "actual": {split: current_total[split] for split in split_names},
        "source_targets": targets_source,
        "source_actual": {
            source: {split: counts[split] for split in split_names}
            for source, counts in current_source.items()
        },
        "new_connected_components": len(components),
        "anchored_components": anchored_components,
        "anchored_rows": anchored_rows,
    }


def combined_split_invariants(
    canonical_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]
) -> dict[str, bool]:
    combined = canonical_rows + new_rows
    fields = (
        "question_exact_sha256",
        "question_normalized_sha256",
        "response_exact_sha256",
        "response_normalized_sha256",
        "source_group_id",
        "duplicate_cluster_id",
    )
    for field in fields:
        splits: dict[str, set[str]] = defaultdict(set)
        for row in combined:
            splits[str(row[field])].add(str(row["split"]))
        overlap = [value for value, assigned in splits.items() if len(assigned) > 1]
        if overlap:
            raise RuntimeError(f"Combined canonical/new split leakage for {field}: {overlap[:3]}")
    return {
        "zero_exact_question_overlap_across_splits": True,
        "zero_normalized_question_overlap_across_splits": True,
        "zero_exact_response_overlap_across_splits": True,
        "zero_normalized_response_overlap_across_splits": True,
        "zero_source_group_overlap_across_splits": True,
        "zero_duplicate_cluster_overlap_across_splits": True,
    }


def balanced_specs(total: int, seed: int) -> tuple[list[tuple[str, ...]], dict[str, Any]]:
    if total == 0:
        return [], {
            "axis_count_counts": {"1": 0, "2": 0, "3": 0},
            "marginal_counts": {axis: 0 for axis in AXES},
            "marginal_range": 0,
            "pairwise_counts": {
                "+".join(pair): 0 for pair in itertools.combinations(AXES, 2)
            },
            "pairwise_range": 0,
        }
    quotas = largest_remainder_counts(total, AXIS_COUNT_RATIOS)
    total_axis_slots = sum(k * count for k, count in quotas.items())
    total_pair_slots = sum(math.comb(k, 2) * count for k, count in quotas.items())
    axis_target = total_axis_slots / len(AXES)
    pair_target = total_pair_slots / math.comb(len(AXES), 2)
    marginal: Counter[str] = Counter()
    pairwise: Counter[tuple[str, str]] = Counter()
    combination_counts: dict[int, Counter[tuple[str, ...]]] = defaultdict(Counter)
    specs: list[tuple[str, ...]] = []
    for axis_count in sorted(quotas, reverse=True):
        choices = list(itertools.combinations(AXES, axis_count))
        for slot in range(quotas[axis_count]):
            def score(combo: tuple[str, ...]) -> tuple[float, float, int, str]:
                combo_pairs = set(itertools.combinations(combo, 2))
                marginal_loss = sum(
                    (marginal[axis] + (axis in combo) - axis_target) ** 2 for axis in AXES
                )
                pair_loss = sum(
                    (pairwise[pair] + (pair in combo_pairs) - pair_target) ** 2
                    for pair in itertools.combinations(AXES, 2)
                )
                return (
                    marginal_loss,
                    pair_loss,
                    combination_counts[axis_count][combo],
                    stable_random_key(seed, f"{axis_count}:{slot}:{'+'.join(combo)}"),
                )

            chosen = min(choices, key=score)
            specs.append(chosen)
            marginal.update(chosen)
            pairwise.update(itertools.combinations(chosen, 2))
            combination_counts[axis_count][chosen] += 1
    indexed = list(enumerate(specs))
    indexed.sort(
        key=lambda item: stable_random_key(
            seed + 2, f"{item[0]}:{'+'.join(item[1])}"
        )
    )
    specs = [combo for _, combo in indexed]
    pair_counts = [pairwise[pair] for pair in itertools.combinations(AXES, 2)]
    return specs, {
        "axis_count_counts": {str(key): value for key, value in sorted(quotas.items())},
        "marginal_counts": {axis: marginal[axis] for axis in AXES},
        "marginal_range": max(marginal.values()) - min(marginal.values()),
        "pairwise_counts": {
            "+".join(pair): pairwise[pair]
            for pair in itertools.combinations(AXES, 2)
        },
        "pairwise_range": max(pair_counts) - min(pair_counts),
    }


def attach_generation_settings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["split"] in {"train", "valid"}]
    eligible.sort(key=lambda row: stable_random_key(GENERATION_SEED + 3, row["canonical_id"]))
    specs, stats = balanced_specs(len(eligible), GENERATION_SEED)
    for row, axes in zip(eligible, specs, strict=True):
        ordered_axes = sorted(
            axes,
            key=lambda axis: stable_random_key(
                GENERATION_SEED, f"{row['canonical_id']}:{axis}"
            ),
        )
        row["generation"] = {
            "status": "planned_not_run",
            "generator": dict(GENERATOR),
            "intended_axes": ordered_axes,
            "axis_count": len(ordered_axes),
            "generation_seeds": [
                {
                    "stage_index": stage,
                    "axis": axis,
                    "attempt_1": deterministic_generation_seed(
                        GENERATION_SEED, row["canonical_id"], stage, 1
                    ),
                    "attempt_2": deterministic_generation_seed(
                        GENERATION_SEED, row["canonical_id"], stage, 2
                    ),
                }
                for stage, axis in enumerate(ordered_axes, 1)
            ],
        }
    for row in rows:
        if row["split"] == "test":
            row["generation"] = {
                "status": "reserved_test_no_generation",
                "generator": None,
                "intended_axes": [],
                "axis_count": 0,
                "generation_seeds": [],
            }
    split_order = {"valid": 0, "train": 1, "test": 2}
    rows.sort(
        key=lambda row: (
            split_order[row["split"]],
            stable_random_key(GENERATION_SEED + 4, row["canonical_id"]),
        )
    )
    return {
        "selection_seed": GENERATION_SEED,
        "eligible_splits": ["train", "valid"],
        "reserved_splits": ["test"],
        "ordering": "VALID first, then TRAIN, then reserved TEST",
        "axis_count_ratios": {str(key): value for key, value in AXIS_COUNT_RATIOS.items()},
        "axes": list(AXES),
        "assignment_stats": stats,
        "assignment_algorithm": (
            "existing balanced_specs algorithm including seed+2 specification reorder; "
            "eligible candidate ordering uses existing seed+3 rule"
        ),
        "generator": GENERATOR,
        "future_qc_contract": QC_CONTRACT,
        "frozen_production_contract_hashes": FROZEN_PRODUCTION_CONTRACT_HASHES,
        "paid_api_enabled": False,
        "model_execution_performed": False,
        "training_performed": False,
    }


def prepare_supplement(
    *,
    dataset_kind: str,
    input_json: Path,
    canonical_path: Path,
    accepted_652_path: Path | None,
    authorization_record: Path,
    authorization_confirmed: bool,
    output_dir: Path,
    enforce_cardinality: bool = True,
    enforce_frozen_canonical: bool = True,
    enforce_frozen_accepted: bool = True,
) -> dict[str, Any]:
    if dataset_kind not in DATASETS:
        raise ValueError(f"Unsupported explicit dataset kind: {dataset_kind}")
    if DATASETS[dataset_kind].get("enabled") is False:
        raise RuntimeError(
            f"{dataset_kind} is disabled: {DATASETS[dataset_kind]['disabled_reason']}"
        )
    input_json = input_json.resolve()
    canonical_path = canonical_path.resolve()
    accepted_652_path = accepted_652_path.resolve() if accepted_652_path is not None else None
    authorization_record = authorization_record.resolve()
    output_dir = output_dir.resolve()
    if not authorization_confirmed:
        raise PermissionError(
            "Explicit --authorization-confirmed assertion is required; a filename is not approval"
        )
    for name, path in (
        ("input JSON", input_json),
        ("canonical", canonical_path),
        ("authorization record", authorization_record),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    if output_dir in {input_json, canonical_path, authorization_record}:
        raise ValueError("Output directory cannot be an input file")
    if accepted_652_path is not None and output_dir == accepted_652_path:
        raise ValueError("Output directory cannot be the accepted-652 merge anchor")

    # Authorization is validated before the source artifact is opened.
    authorization = validate_authorization_attestation(
        authorization_record, dataset_kind
    )
    authorization_sha = sha256_file(authorization_record)
    artifact_identity = validate_official_artifact(
        input_json, dataset_kind, enforce_official_identity=enforce_cardinality
    )
    source_rows = load_json_list(input_json)
    extracted, identity = extract_candidates(
        source_rows, dataset_kind, enforce_cardinality=enforce_cardinality
    )
    canonical_rows = list(read_jsonl(canonical_path))
    canonical_meta = validate_canonical(
        canonical_path, canonical_rows, enforce_frozen=enforce_frozen_canonical
    )
    merge_anchor, accepted_rows = validate_accepted_652(
        accepted_652_path, enforce_frozen=enforce_frozen_accepted
    )
    split_index, cluster_index = canonical_indexes(canonical_rows)
    candidates, excluded = deduplicate_pairs(extracted, canonical_rows)
    if not candidates:
        raise RuntimeError("No distinct new normalized question-response pairs remain")
    split_stats = assign_components_and_splits(candidates, split_index, cluster_index)
    invariants = combined_split_invariants(canonical_rows, candidates)

    candidate_ids = [str(row["canonical_id"]) for row in candidates]
    canonical_ids = {str(row["canonical_id"]) for row in canonical_rows}
    accepted_ids = {str(row["canonical_id"]) for row in accepted_rows}
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("Supplement canonical IDs are not unique")
    if set(candidate_ids) & canonical_ids:
        raise RuntimeError("Supplement canonical IDs collide with frozen canonical IDs")
    if set(candidate_ids) & accepted_ids:
        raise RuntimeError("Supplement canonical IDs collide with accepted-652 IDs")

    input_sha = artifact_identity["sha256"]
    for row in candidates:
        row["source_file"] = str(input_json)
        row["source_artifact"] = {
            "sha256": input_sha,
            "bytes": artifact_identity["bytes"],
            "git_blob_sha1": artifact_identity["git_blob_sha1"],
            "authorization_record_sha256": authorization_sha,
        }
        row["provenance"] = {
            "dataset_kind": dataset_kind,
            "detected_schema": identity["detected_schema"],
            "source_row_index": row["source_row_index"],
            "source_answer_index": row["source_answer_index"],
            "source_record_sha256": row["source_record_sha256"],
            "source_artifact_sha256": input_sha,
            "authorization_attestation_sha256": authorization_sha,
        }
    candidates.sort(
        key=lambda row: (
            row["source"], row["source_component"], row["source_row_index"],
            row["source_answer_index"], row["canonical_id"],
        )
    )
    accepted_question_hashes = {
        str(row["question_normalized_sha256"]) for row in accepted_rows
    }
    generation_plan, generation_covered = select_generation_candidates(
        candidates, covered_question_hashes=accepted_question_hashes
    )
    generation_settings = attach_generation_settings(generation_plan)

    if accepted_652_path is not None and sha256_file(accepted_652_path) != merge_anchor["sha256"]:
        raise RuntimeError("Accepted-652 merge anchor changed during preparation")

    # All source/canonical/authorization checks and transformations finish before
    # the caller-supplied output directory is created.
    output_dir.mkdir(parents=True, exist_ok=False)
    clean_path = output_dir / "clean_candidates.jsonl"
    plan_path = output_dir / "generation_plan.jsonl"
    write_jsonl(clean_path, candidates)
    write_jsonl(plan_path, generation_plan)
    exclusion_counts = dict(Counter(item["reason"] for item in excluded))
    split_counts = dict(Counter(row["split"] for row in candidates))
    generation_split_counts = dict(Counter(row["split"] for row in generation_plan))
    generation_counts = dict(
        Counter(row["generation"]["status"] for row in generation_plan)
    )
    manifest = {
        "version": VERSION,
        "status": "complete_cpu_only_no_model_no_api_no_training",
        "dataset_identity": {
            "explicit_dataset_kind": dataset_kind,
            **DATASETS[dataset_kind],
            **identity,
            "artifact_identity": artifact_identity,
            "schema_and_content_validated": True,
            "filename_used_for_identity": False,
            "official_cardinality_enforced": enforce_cardinality,
        },
        "authorization": {
            "caller_asserted_authorized": True,
            "record_path": str(authorization_record),
            "record_sha256": authorization_sha,
            "record_bytes": authorization_record.stat().st_size,
            "attestation": authorization,
            "note": "Structured fields, scope, status, identity, evidence, and expiry were validated before source access.",
        },
        "inputs": {
            "authorized_source": {
                "path": str(input_json),
                **artifact_identity,
            },
            "frozen_canonical": canonical_meta,
            "accepted_652_merge_anchor": merge_anchor,
        },
        "policy": {
            "normalization": "fullpaper_acl_pipeline.normalize_text (HTML, NFKC, punctuation, casefold, whitespace)",
            "existing_dedup": "drop only normalized question+normalized response pairs already in frozen canonical",
            "within_source_dedup": "collapse only duplicate normalized question+normalized response pairs",
            "generation_selection": "after preserving all distinct clean pairs and split assignments, choose one deterministic original answer per normalized question with seed 20260910",
            "accepted652_generation_dedup": (
                "preserve supplement clean pairs, but omit normalized questions already covered by accepted-652 "
                "from the generation plan"
            ),
            "component_linkage": [
                "source_group_id",
                "question_exact_sha256",
                "question_normalized_sha256",
                "response_exact_sha256",
                "response_normalized_sha256",
            ],
            "existing_response_linkage": "retain but anchor the whole new connected component to the existing split",
            "split": "80/10/10 at connected-component/source-group level; existing rows and splits untouched",
            "split_seed": SPLIT_SEED,
            "merge_anchor": "append later to immutable accepted-652 reuse export; never rewrite existing IDs/text/splits",
            "downstream_bridge": (
                "Current production runner cannot consume this supplement directly. A future CPU-only bridge must "
                "combine frozen and supplement canonical rows, run the existing clean-QC contract when GPU work is "
                "allowed, and then render production inputs. Generation remains queued."
            ),
        },
        "assumptions_and_limits": {
            "psych8k_schema": (
                "The gated raw schema was not inspected while access was unavailable. Runtime therefore "
                "requires one homogeneous supported content signature, 8,187 rows/answers, and the pinned "
                "6,575,030-byte Git blob b0a9f254...; the filename is never trusted."
            ),
            "psyqa_identity": (
                "Only the exact official sample SHA-256 is registered. PsyQA-full is disabled until an exact "
                "approved artifact hash and verified schema are registered."
            ),
            "authorization": (
                "A structured, matching, non-expired attestation plus caller confirmation is required. This is "
                "an audit control and not an independent legal ruling."
            ),
            "no_source_quality_approval": (
                "Preparation preserves original answers but does not declare them clean-QC or training approved."
            ),
        },
        "counts": {
            "input_top_level_records": len(source_rows),
            "extracted_original_answer_pairs": len(extracted),
            "excluded_total": len(excluded),
            "excluded_by_reason": exclusion_counts,
            "clean_distinct_question_answer_candidates": len(candidates),
            "clean_unique_normalized_questions": len(
                {row["question_normalized_sha256"] for row in candidates}
            ),
            "clean_by_split": split_counts,
            "generation_plan_candidates": len(generation_plan),
            "covered_by_existing_accepted652_question": len(generation_covered),
            "generation_plan_by_split": generation_split_counts,
            "generation_status": generation_counts,
        },
        "excluded_records": excluded,
        "generation_plan_exclusions": generation_covered,
        "split_assignment": split_stats,
        "invariants": {
            **invariants,
            "existing_canonical_bytes_untouched": sha256_file(canonical_path) == canonical_meta["sha256"],
            "existing_accepted_652_bytes_untouched": (
                accepted_652_path is None
                or sha256_file(accepted_652_path) == merge_anchor["sha256"]
            ),
            "unique_normalized_question_response_pairs": len(candidates)
            == len({
                (row["question_normalized_sha256"], row["response_normalized_sha256"])
                for row in candidates
            }),
            "one_generation_plan_row_per_normalized_question": len(generation_plan)
            == len({row["question_normalized_sha256"] for row in generation_plan}),
            "supplement_canonical_ids_unique": len(candidate_ids) == len(set(candidate_ids)),
            "supplement_ids_disjoint_from_frozen_canonical": not (
                set(candidate_ids) & canonical_ids
            ),
            "supplement_ids_disjoint_from_accepted652": not (
                set(candidate_ids) & accepted_ids
            ),
            "output_directory_was_new": True,
        },
        "generation_settings": generation_settings,
        "outputs": {
            "clean_candidates.jsonl": {
                "role": "canonical-clean-qa-v1 rows preserving every distinct authorized original Q-answer pair and frozen split metadata",
                "rows": len(candidates),
                "sha256": sha256_file(clean_path),
            },
            "generation_plan.jsonl": {
                "role": "one deterministic candidate per normalized question; TRAIN/VALID queued and TEST reserved",
                "rows": len(generation_plan),
                "sha256": sha256_file(plan_path),
            }
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": str(output_dir / "manifest.json"),
                "clean_candidates": str(clean_path),
                "generation_plan": str(plan_path),
                "counts": manifest["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-kind", required=True, choices=sorted(DATASETS))
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--canonical", default=str(DEFAULT_CANONICAL))
    parser.add_argument("--accepted-652", default=str(DEFAULT_ACCEPTED_652))
    parser.add_argument("--authorization-record", required=True)
    parser.add_argument(
        "--authorization-confirmed",
        action="store_true",
        help="Caller assertion that this use is within the supplied authorization record",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_supplement(
        dataset_kind=args.dataset_kind,
        input_json=Path(args.input_json),
        canonical_path=Path(args.canonical),
        accepted_652_path=Path(args.accepted_652),
        authorization_record=Path(args.authorization_record),
        authorization_confirmed=args.authorization_confirmed,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
