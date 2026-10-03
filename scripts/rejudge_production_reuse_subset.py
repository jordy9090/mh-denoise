#!/usr/bin/env python3
"""Rejudge only unresolved warning rows from immutable production accepted pairs.

No text is generated.  The script runs the current local paired-QC contract on
existing question/clean/candidate triples, checkpoints every terminal result,
and can safely resume in the same output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from build_development_corruption_shard import ACCEPTANCE_RULES, candidate_a
from corruption_contract_v2 import parse_json_object
from fullpaper_acl_pipeline import read_jsonl, sha256_file, write_json
from local_qwen_production_qc_v4 import (
    clean_review_signals,
    content_disposition,
    paired_prompt,
    validate_and_map,
)
from run_fullpaper_corruption_production import LocalProductionJudge


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRODUCTION = ROOT / "data/fullpaper_acl_pipeline/production_run_train1000_valid200_local_qwen35_27b_v3r1_20260910"
DEFAULT_CODEX = ROOT / "data/fullpaper_acl_pipeline/codex_clean_candidate_review_batch20_v2_20260926/cumulative_ai_review_ledger.jsonl"
DEFAULT_PRIOR_LEDGER = Path(str(DEFAULT_PRODUCTION) + "_provisional_corrected_v2_20260926/review_ledger.jsonl")
DEFAULT_JUDGE = Path("/mnt/ssd00/user-qwen35-27b-hf/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654")
VERSION = "production-reuse-subset-rejudge-v3-20261001"


def atomic_append_jsonl(path: Path, row: dict[str, Any]) -> None:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def select_rows(
    accepted: list[dict[str, Any]], suspicious: list[dict[str, Any]],
    prior_reviews: list[dict[str, Any]], codex_reviews: list[dict[str, Any]],
    explicit_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    if explicit_ids is not None:
        accepted_ids = {row["canonical_id"] for row in accepted}
        unknown = explicit_ids - accepted_ids
        if unknown:
            raise ValueError(f"Explicit selection contains unknown IDs: {sorted(unknown)[:5]}")
        return [row for row in accepted if row["canonical_id"] in explicit_ids]
    warning_ids = {row["canonical_id"] for row in suspicious}
    held_ids = {
        row["canonical_id"] for row in prior_reviews
        if row.get("clean_disposition") == "held"
    }
    codex_ids = {row["canonical_id"] for row in codex_reviews}
    return [
        row for row in accepted
        if row["canonical_id"] in warning_ids
        and row["canonical_id"] not in held_ids
        and row["canonical_id"] not in codex_ids
    ]


def read_explicit_ids(path: Path) -> set[str]:
    """Read a newline, JSON, or JSONL canonical-ID selection without guessing IDs."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    ids: list[str] = []
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            rows = payload.get("rows", payload.get("canonical_ids", [])) if isinstance(payload, dict) else payload
            if not isinstance(rows, list):
                raise ValueError("Explicit ID JSON must be a list or contain rows/canonical_ids")
            ids = [str(row.get("canonical_id") if isinstance(row, dict) else row) for row in rows]
    if not ids:
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                row = json.loads(line)
                ids.append(str(row["canonical_id"]))
            else:
                ids.append(line)
    if not ids or any(not value or value == "None" for value in ids):
        raise ValueError("Explicit ID selection is empty or malformed")
    if len(ids) != len(set(ids)):
        raise ValueError("Explicit ID selection contains duplicates")
    return set(ids)


def adjudicate_grade(row: dict[str, Any], grade: dict[str, Any]) -> dict[str, Any]:
    """Separate clean eligibility, response-pair reuse, and local labels."""
    clean_signals = clean_review_signals(row["question"], row["clean_response"], grade)
    candidate_disposition, candidate_reason = content_disposition(
        row["question"], row["corrupted_response"], grade, clean=None
    )
    missing = [axis for axis in row["intended_axes"] if axis not in grade["realized_axes"]]
    if candidate_disposition == "hold":
        pair_status = "unresolved"
        pair_reason = candidate_reason
    elif candidate_disposition == "reject":
        pair_status = "regenerate_candidate"
        pair_reason = candidate_reason
    elif missing:
        pair_status = "regenerate_candidate"
        pair_reason = "missing_intended_axes:" + ",".join(missing)
    else:
        pair_status = "reusable"
        pair_reason = None
    if any(item.get("recommended_disposition") == "exclude_original_answer" for item in clean_signals):
        clean_status = "original_answer_issue"
    elif clean_signals:
        clean_status = "unresolved"
    else:
        clean_status = "suitable"
    return {
        "clean_status": clean_status,
        "clean_signals": clean_signals,
        "pair_status": pair_status,
        "pair_reason": pair_reason,
        "candidate_content_disposition": candidate_disposition,
        "candidate_content_reason": candidate_reason,
        "missing_intended_axes": missing,
    }


def paired_grade_local_many(
    judge: LocalProductionJudge, batch: list[dict[str, Any]], stage: int, attempt: int,
) -> list[tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]]:
    """Run the unchanged v6 prompt/schema in a batch, with bounded schema retries.

    Batching changes only device utilization.  Each member retains its own
    deterministic A/B assignment, validation feedback, raw-call record, usage,
    and the production retry ceiling.
    """
    states: list[dict[str, Any]] = []
    for row in batch:
        candidate_label = "A" if candidate_a(row["canonical_id"], stage, attempt) else "B"
        clean_label = "B" if candidate_label == "A" else "A"
        candidate = row["corrupted_response"]
        a = candidate if candidate_label == "A" else row["clean_response"]
        b = row["clean_response"] if candidate_label == "A" else candidate
        states.append({
            "row": row, "a": a, "b": b, "clean_label": clean_label,
            "candidate_label": candidate_label, "validation_error": None,
            "previous_output": None, "infrastructure": [], "best_quarantined": None,
            "grade": None, "usage": {"judge_calls": 0, "judge_prompt_tokens": 0,
                                      "judge_completion_tokens": 0, "judge_seconds": 0.0},
        })

    for infrastructure_attempt in range(1, ACCEPTANCE_RULES["max_infrastructure_attempts"] + 1):
        pending = [state for state in states if state["grade"] is None]
        if not pending:
            break
        prompts = [paired_prompt(
            question=state["row"]["question"], a=state["a"], b=state["b"],
            validation_error=state["validation_error"],
            previous_output=state["previous_output"],
        ) for state in pending]
        contexts = [{
            "canonical_id": state["row"]["canonical_id"],
            "call_purpose": "paired_stage_qc", "stage_index": stage,
            "semantic_attempt": attempt,
            "infrastructure_attempt": infrastructure_attempt,
            "validation_feedback": state["validation_error"],
        } for state in pending]
        started = time.monotonic()
        try:
            returned = judge.backend.call_many(prompts)
        except Exception as exc:
            elapsed = time.monotonic() - started
            for state, prompt, context in zip(pending, prompts, contexts):
                error = f"{type(exc).__name__}: {exc}"
                judge.context = context
                judge._record_call({
                    "call_kind": "judge", "context": context,
                    "status": "model_exception", "error": error,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "elapsed_seconds": elapsed,
                })
                state["validation_error"] = error
                state["usage"]["judge_calls"] += 1
                state["usage"]["judge_seconds"] += elapsed / len(pending)
                state["infrastructure"].append({
                    "kind": "local_paired_judge_parse_validation_or_timeout",
                    "attempt": infrastructure_attempt, "reason": error,
                    "feedback_added_to_next_attempt": infrastructure_attempt < ACCEPTANCE_RULES["max_infrastructure_attempts"],
                })
            continue

        for state, prompt, context, (raw, call_usage) in zip(pending, prompts, contexts, returned):
            judge.context = context
            judge._record_call({
                "call_kind": "judge", "context": context, "status": "returned",
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "raw_output": raw, "usage": call_usage,
            })
            state["usage"]["judge_calls"] += 1
            state["usage"]["judge_prompt_tokens"] += int(call_usage.get("input_tokens", 0))
            state["usage"]["judge_completion_tokens"] += int(call_usage.get("output_tokens", 0))
            state["usage"]["judge_seconds"] += float(call_usage.get("elapsed_seconds", 0.0))
            try:
                payload = parse_json_object(raw)
                grade, _ = validate_and_map(
                    payload, state["a"], state["b"], state["clean_label"],
                    state["candidate_label"], question=state["row"]["question"],
                )
                grade.update({
                    "raw_output": raw, "local_judge_usage": call_usage,
                    "response_a_sha256": hashlib.sha256(state["a"].encode()).hexdigest(),
                    "response_b_sha256": hashlib.sha256(state["b"].encode()).hexdigest(),
                    "response_a_chars": len(state["a"]), "response_b_chars": len(state["b"]),
                })
                grade["unintended_axes"] = [
                    axis for axis in grade["realized_axes"]
                    if axis not in state["row"]["intended_axes"]
                ]
                state["grade"] = grade
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                state["validation_error"] = error
                state["previous_output"] = raw
                try:
                    fallback_payload = parse_json_object(raw)
                    fallback_grade, fallback_invalid = validate_and_map(
                        fallback_payload, state["a"], state["b"], state["clean_label"],
                        state["candidate_label"], question=state["row"]["question"],
                        quarantine_invalid_evidence=True,
                    )
                    candidate_fallback = (len(fallback_invalid), fallback_grade, raw, call_usage)
                    if state["best_quarantined"] is None or candidate_fallback[0] < state["best_quarantined"][0]:
                        state["best_quarantined"] = candidate_fallback
                except Exception:
                    pass
                state["infrastructure"].append({
                    "kind": "local_paired_judge_parse_validation_or_timeout",
                    "attempt": infrastructure_attempt, "reason": error,
                    "feedback_added_to_next_attempt": infrastructure_attempt < ACCEPTANCE_RULES["max_infrastructure_attempts"],
                })

    results = []
    for state in states:
        grade = state["grade"]
        if grade is None and state["best_quarantined"] is not None:
            invalid_count, grade, raw, call_usage = state["best_quarantined"]
            grade.update({
                "raw_output": raw, "local_judge_usage": call_usage,
                "response_a_sha256": hashlib.sha256(state["a"].encode()).hexdigest(),
                "response_b_sha256": hashlib.sha256(state["b"].encode()).hexdigest(),
                "response_a_chars": len(state["a"]), "response_b_chars": len(state["b"]),
                "strict_validation_attempts_exhausted": True,
                "quarantined_invalid_evidence_count": invalid_count,
            })
            grade["unintended_axes"] = [
                axis for axis in grade["realized_axes"]
                if axis not in state["row"]["intended_axes"]
            ]
            state["infrastructure"].append({
                "kind": "local_paired_judge_non_verbatim_evidence_quarantined",
                "attempt": ACCEPTANCE_RULES["max_infrastructure_attempts"],
                "reason": f"strict validation exhausted; removed {invalid_count} non-verbatim evidence item(s) without creating local labels",
            })
        results.append((grade, state["infrastructure"], state["usage"]))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", default=str(DEFAULT_PRODUCTION))
    parser.add_argument("--prior-review-ledger", default=str(DEFAULT_PRIOR_LEDGER))
    parser.add_argument("--codex-review-ledger", default=str(DEFAULT_CODEX))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model-dir", default=str(DEFAULT_JUDGE))
    parser.add_argument("--judge-max-new-tokens", type=int, default=3200)
    parser.add_argument("--judge-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed-checkpoint", help="Optional completed checkpoint from the same frozen selection")
    parser.add_argument(
        "--explicit-ids-file",
        help="Optional newline/JSON/JSONL canonical-ID list; overrides warning-only selection.",
    )
    parser.add_argument("--max-inputs", type=int, default=1000000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    production = Path(args.production_dir).resolve()
    prior_path = Path(args.prior_review_ledger).resolve()
    codex_path = Path(args.codex_review_ledger).resolve()
    output = Path(args.output_dir).resolve()
    judge_path = Path(args.judge_model_dir).resolve()
    seed_checkpoint = Path(args.seed_checkpoint).resolve() if args.seed_checkpoint else None
    explicit_ids_path = Path(args.explicit_ids_file).resolve() if args.explicit_ids_file else None
    if not judge_path.is_dir():
        raise FileNotFoundError(f"Local judge snapshot is missing: {judge_path}")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if seed_checkpoint is not None and not seed_checkpoint.is_file():
        raise FileNotFoundError(f"Seed checkpoint is missing: {seed_checkpoint}")
    if explicit_ids_path is not None and not explicit_ids_path.is_file():
        raise FileNotFoundError(f"Explicit ID file is missing: {explicit_ids_path}")
    output.mkdir(parents=True, exist_ok=True)

    accepted = list(read_jsonl(production / "accepted.jsonl"))
    suspicious_path = production / "qwen27_clean_target_suspicious.jsonl"
    suspicious = list(read_jsonl(suspicious_path))
    prior_reviews = list(read_jsonl(prior_path))
    codex_reviews = list(read_jsonl(codex_path))
    explicit_ids = read_explicit_ids(explicit_ids_path) if explicit_ids_path is not None else None
    selected = select_rows(
        accepted, suspicious, prior_reviews, codex_reviews, explicit_ids=explicit_ids,
    )[: args.max_inputs]
    selection = [{
        "canonical_id": row["canonical_id"], "split": row["split"],
        "source": row["source"], "intended_axes": row["intended_axes"],
        "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
        "clean_response_sha256": hashlib.sha256(row["clean_response"].encode()).hexdigest(),
        "corrupted_response_sha256": hashlib.sha256(row["corrupted_response"].encode()).hexdigest(),
    } for row in selected]
    selection_hash = hashlib.sha256(
        json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    contract = {
        "version": VERSION,
        "production_dir": str(production),
        "input_sha256": {
            "accepted.jsonl": sha256_file(production / "accepted.jsonl"),
            "qwen27_clean_target_suspicious.jsonl": sha256_file(suspicious_path),
            "prior_review_ledger.jsonl": sha256_file(prior_path),
            "codex_review_ledger.jsonl": sha256_file(codex_path),
        },
        "selection_rule": (
            "explicit canonical ID list over immutable accepted rows"
            if explicit_ids is not None
            else "clean warning rows minus explicit prior holds minus detailed Codex reviews"
        ),
        "explicit_ids_file": (
            {"path": str(explicit_ids_path), "sha256": sha256_file(explicit_ids_path)}
            if explicit_ids_path is not None else None
        ),
        "selection_rows": len(selection),
        "selection_sha256": selection_hash,
        "judge_model_dir": str(judge_path),
        "judge_max_new_tokens": args.judge_max_new_tokens,
        "judge_timeout_seconds": args.judge_timeout_seconds,
        "batch_size": args.batch_size,
        "seed_checkpoint": (
            {"path": str(seed_checkpoint), "sha256": sha256_file(seed_checkpoint)}
            if seed_checkpoint is not None else None
        ),
        "paid_api_enabled": False,
        "generation_enabled": False,
    }
    selection_path, contract_path = output / "selection.json", output / "run_contract.json"
    if contract_path.exists():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != contract:
            raise RuntimeError("Existing run contract differs; refusing mixed resume")
    else:
        write_json(selection_path, {"rows": selection, "selection_sha256": selection_hash})
        write_json(contract_path, contract)
    if args.dry_run:
        write_json(output / "manifest.json", {**contract, "status": "dry_run"})
        print(json.dumps({**contract, "status": "dry_run"}, indent=2))
        return

    checkpoint = output / "rejudged_checkpoint.jsonl"
    if seed_checkpoint is not None and not checkpoint.exists():
        seeded = list(read_jsonl(seed_checkpoint))
        seeded_ids = [row["canonical_id"] for row in seeded]
        if len(seeded_ids) != len(set(seeded_ids)):
            raise RuntimeError("Seed checkpoint contains duplicate IDs")
        if not set(seeded_ids) <= {row["canonical_id"] for row in selected}:
            raise RuntimeError("Seed checkpoint contains IDs outside frozen selection")
        for row in seeded:
            atomic_append_jsonl(checkpoint, row)
    completed_rows = list(read_jsonl(checkpoint)) if checkpoint.exists() else []
    completed = {row["canonical_id"] for row in completed_rows}
    if not completed <= {row["canonical_id"] for row in selected}:
        raise RuntimeError("Checkpoint contains IDs outside frozen selection")
    remaining = [row for row in selected if row["canonical_id"] not in completed]
    judge = LocalProductionJudge(
        judge_path, args.judge_max_new_tokens, args.judge_timeout_seconds,
        output / "local_calls",
    )
    started = time.monotonic()
    try:
        for offset in range(0, len(remaining), args.batch_size):
            batch = remaining[offset : offset + args.batch_size]
            batch_started = time.monotonic()
            graded = paired_grade_local_many(judge, batch, 1, 1)
            batch_wall = time.monotonic() - batch_started
            for row, (grade, failures, usage) in zip(batch, graded):
                usage["wall_seconds_allocated"] = batch_wall / len(batch)
                if grade is None:
                    result = {
                        "canonical_id": row["canonical_id"], "split": row["split"],
                        "source": row["source"], "status": "technical_failure",
                        "clean_status": "unresolved", "pair_status": "unresolved",
                        "failure_reason": "paired_qc_infrastructure_exhausted",
                        "infrastructure_failures": failures, "usage": usage,
                    }
                else:
                    decision = adjudicate_grade(row, grade)
                    result = {
                        "canonical_id": row["canonical_id"], "split": row["split"],
                        "source": row["source"], "status": "complete",
                        "intended_axes": row["intended_axes"],
                        "legacy_realized_axes": row["realized_axes"],
                        "current_realized_axes": grade["realized_axes"],
                        **decision, "grade": grade,
                        "infrastructure_failures": failures, "usage": usage,
                    }
                atomic_append_jsonl(checkpoint, result)
                completed_rows.append(result)
            manifest = {
                **contract, "status": "running",
                "completed": len(completed_rows), "remaining": len(selection) - len(completed_rows),
                "result_counts": dict(Counter(row["status"] for row in completed_rows)),
                "clean_counts": dict(Counter(row["clean_status"] for row in completed_rows)),
                "pair_counts": dict(Counter(row["pair_status"] for row in completed_rows)),
            }
            write_json(output / "manifest.json", manifest)
    finally:
        judge.close()
    final = {
        **contract, "status": "complete", "completed": len(completed_rows), "remaining": 0,
        "result_counts": dict(Counter(row["status"] for row in completed_rows)),
        "clean_counts": dict(Counter(row["clean_status"] for row in completed_rows)),
        "pair_counts": dict(Counter(row["pair_status"] for row in completed_rows)),
        "elapsed_seconds_this_process": time.monotonic() - started,
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    write_json(output / "manifest.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
