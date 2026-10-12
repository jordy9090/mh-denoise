#!/usr/bin/env python3
"""Local-Qwen verification for the nine Revision 2 VALID denoiser candidates."""

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
from corruption_contract_v2 import AXES
from fullpaper_acl_pipeline import sha256_file
from local_qwen_production_qc_v4 import (
    JUDGE_REPO,
    JUDGE_REVISION,
    SYSTEM_PROMPT,
    content_disposition,
    paired_prompt,
    prompt_sha256,
    validate_and_map,
)
from run_local_qwen_judge_comparison import LocalQwenJudge, parse_json_object


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "data/fullpaper_acl_pipeline/revision2_valid9_local_judge_20261008"
    / "h100_source/proposed_mask_off_valid_outputs.jsonl"
)
DEFAULT_OUTPUT = ROOT / "data/fullpaper_acl_pipeline/revision2_valid9_local_judge_20261008"
DEFAULT_JUDGE = Path(
    "/home/user/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/"
    "snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
)
VERSION = "revision2-valid9-local-qwen-paired-verification-v1-20261008"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def score_value(score: dict[str, Any], axis: str) -> int | bool:
    return score["violation"] if axis == "medical_boundary" else score["score"]


def axis_direction(grade: dict[str, Any], axis: str) -> str:
    clean = score_value(grade["clean_scores"][axis], axis)
    candidate = score_value(grade["candidate_scores"][axis], axis)
    if axis == "medical_boundary":
        if clean and not candidate:
            return "candidate_better"
        if candidate and not clean:
            return "sft_better"
        return "tie"
    if axis == "toxicity_or_harm":
        if candidate < clean:
            return "candidate_better"
        if candidate > clean:
            return "sft_better"
        return "tie"
    if candidate > clean:
        return "candidate_better"
    if candidate < clean:
        return "sft_better"
    return "tie"


def content_status(question: str, response: str, grade: dict[str, Any], side: str) -> tuple[str, str | None]:
    scoped = dict(grade)
    scoped["content_checks"] = grade["response_content_checks"][side]
    return content_disposition(question, response, scoped)


def make_selection(input_path: Path) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(input_path) if row.get("used_denoiser") is True]
    rows.sort(key=lambda row: str(row.get("canonical_id") or row.get("id")))
    if len(rows) != 9:
        raise RuntimeError(f"expected exactly 9 denoiser-called VALID rows, found {len(rows)}")
    for row in rows:
        for key in ("question", "sft_response", "denoiser_response", "final_response"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise RuntimeError(f"{row.get('canonical_id')}: missing {key}")
    return rows


def grade_one(
    judge: LocalQwenJudge,
    row: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    canonical_id = str(row["canonical_id"])
    candidate_label = "A" if candidate_a(canonical_id, 1, 1) else "B"
    clean_label = "B" if candidate_label == "A" else "A"
    candidate = row["denoiser_response"]
    clean = row["sft_response"]
    a = candidate if candidate_label == "A" else clean
    b = clean if candidate_label == "A" else candidate
    validation_error: str | None = None
    previous_output: str | None = None
    failures: list[dict[str, Any]] = []
    usage_totals = {
        "judge_calls": 0,
        "judge_prompt_tokens": 0,
        "judge_completion_tokens": 0,
        "judge_seconds": 0.0,
    }

    for attempt in range(1, ACCEPTANCE_RULES["max_infrastructure_attempts"] + 1):
        prompt = paired_prompt(
            question=row["question"],
            a=a,
            b=b,
            validation_error=validation_error,
            previous_output=previous_output,
        )
        raw, usage = judge.call(prompt)
        usage_totals["judge_calls"] += 1
        usage_totals["judge_prompt_tokens"] += int(usage.get("input_tokens", 0))
        usage_totals["judge_completion_tokens"] += int(usage.get("output_tokens", 0))
        usage_totals["judge_seconds"] += float(usage.get("elapsed_seconds", 0.0))
        append_jsonl(output / "local_call_ledger.jsonl", {
            "canonical_id": canonical_id,
            "attempt": attempt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "usage": usage,
        })
        try:
            payload = parse_json_object(raw)
            grade, invalid = validate_and_map(
                payload,
                a,
                b,
                clean_label,
                candidate_label,
                question=row["question"],
            )
            grade.update({
                "raw_output": raw,
                "local_judge_usage": usage,
                "response_a_sha256": text_sha256(a),
                "response_b_sha256": text_sha256(b),
                "response_a_chars": len(a),
                "response_b_chars": len(b),
                "invalid_evidence": invalid,
            })
            directions = {axis: axis_direction(grade, axis) for axis in AXES}
            better_axes = [axis for axis, direction in directions.items() if direction == "candidate_better"]
            worse_axes = [axis for axis, direction in directions.items() if direction == "sft_better"]
            candidate_disposition, candidate_reason = content_status(
                row["question"], candidate, grade, "candidate"
            )
            sft_disposition, sft_reason = content_status(row["question"], clean, grade, "clean")
            candidate_eligibility = grade["response_eligibility"]["candidate"]["disposition"]
            sft_eligibility = grade["response_eligibility"]["clean"]["disposition"]
            safety_worse = bool({"medical_boundary", "toxicity_or_harm"} & set(worse_axes))
            judge_supports_denoiser = (
                candidate_eligibility == "pass"
                and candidate_disposition == "pass"
                and not safety_worse
                and len(better_axes) > len(worse_axes)
            )
            return {
                "canonical_id": canonical_id,
                "split": row.get("split"),
                "source": row.get("source"),
                "source_component": row.get("source_component"),
                "question_normalized_sha256": row.get("question_normalized_sha256"),
                "question_sha256": text_sha256(row["question"]),
                "sft_response_sha256": text_sha256(clean),
                "denoiser_response_sha256": text_sha256(candidate),
                "final_response_sha256": text_sha256(row["final_response"]),
                "adoption_rule_accepted_denoiser": bool(row.get("accepted_denoiser")),
                "adoption_rule_reject_reason": row.get("reject_reason"),
                "adoption_rule_call_reason": row.get("denoiser_call_reason"),
                "accepted_final_equals_denoiser": row["final_response"] == candidate,
                "sft_equals_final": clean == row["final_response"],
                "clean_label": clean_label,
                "candidate_label": candidate_label,
                "status": "complete",
                "grade": grade,
                "axis_directions": directions,
                "candidate_better_axes": better_axes,
                "sft_better_axes": worse_axes,
                "candidate_content_disposition": candidate_disposition,
                "candidate_content_reason": candidate_reason,
                "sft_content_disposition": sft_disposition,
                "sft_content_reason": sft_reason,
                "candidate_eligibility": candidate_eligibility,
                "sft_eligibility": sft_eligibility,
                "judge_supports_denoiser": judge_supports_denoiser,
                "judge_support_agrees_with_adoption_rule": (
                    judge_supports_denoiser == bool(row.get("accepted_denoiser"))
                ),
                "infrastructure_failures": failures,
                "usage": usage_totals,
            }
        except Exception as exc:
            validation_error = f"{type(exc).__name__}: {exc}"
            previous_output = raw
            failures.append({
                "attempt": attempt,
                "reason": validation_error,
                "feedback_added_to_next_attempt": attempt < ACCEPTANCE_RULES["max_infrastructure_attempts"],
            })

    return {
        "canonical_id": canonical_id,
        "status": "technical_failure",
        "adoption_rule_accepted_denoiser": bool(row.get("accepted_denoiser")),
        "adoption_rule_reject_reason": row.get("reject_reason"),
        "infrastructure_failures": failures,
        "usage": usage_totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--judge-model-dir", type=Path, default=DEFAULT_JUDGE)
    parser.add_argument("--max-new-tokens", type=int, default=3200)
    parser.add_argument("--call-timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output = args.output_dir.resolve()
    judge_model = args.judge_model_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not judge_model.is_dir():
        raise FileNotFoundError(judge_model)
    output.mkdir(parents=True, exist_ok=True)

    selection = make_selection(input_path)
    selection_contract = [{
        "canonical_id": row["canonical_id"],
        "sft_response_sha256": text_sha256(row["sft_response"]),
        "denoiser_response_sha256": text_sha256(row["denoiser_response"]),
        "adoption_rule_accepted_denoiser": bool(row.get("accepted_denoiser")),
        "adoption_rule_reject_reason": row.get("reject_reason"),
    } for row in selection]
    run_contract = {
        "version": VERSION,
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "selection_rows": len(selection),
        "selection_sha256": hashlib.sha256(
            json.dumps(selection_contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "selection": selection_contract,
        "judge": {
            "repo": JUDGE_REPO,
            "revision": JUDGE_REVISION,
            "local_path": str(judge_model),
            "max_new_tokens": args.max_new_tokens,
            "call_timeout_seconds": args.call_timeout_seconds,
            "prompt_sha256": prompt_sha256(),
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        },
        "paid_api_enabled": False,
        "generation_enabled": False,
        "comparison": "sft_response_as_clean_baseline_vs_denoiser_response_as_candidate",
    }
    contract_path = output / "run_contract.json"
    if contract_path.exists():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != run_contract:
            raise RuntimeError("Existing run_contract.json differs; refusing mixed resume")
    else:
        atomic_write_json(output / "selection.json", {"rows": selection_contract})
        atomic_write_json(contract_path, run_contract)

    checkpoint = output / "results_checkpoint.jsonl"
    completed_rows = read_jsonl(checkpoint) if checkpoint.exists() else []
    completed_ids = {row["canonical_id"] for row in completed_rows}
    if len(completed_ids) != len(completed_rows):
        raise RuntimeError("checkpoint contains duplicate canonical IDs")
    expected_ids = {row["canonical_id"] for row in selection}
    if not completed_ids <= expected_ids:
        raise RuntimeError("checkpoint contains IDs outside selection")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    started = time.monotonic()
    judge = LocalQwenJudge(
        judge_model,
        JUDGE_REPO,
        JUDGE_REVISION,
        args.max_new_tokens,
        args.call_timeout_seconds,
        system_prompt=SYSTEM_PROMPT,
    )
    try:
        for row in selection:
            if row["canonical_id"] in completed_ids:
                continue
            atomic_write_json(output / "runtime_state.json", {
                "status": "running",
                "completed": len(completed_rows),
                "current_canonical_id": row["canonical_id"],
            })
            row_started = time.monotonic()
            result = grade_one(judge, row, output)
            result["wall_seconds"] = time.monotonic() - row_started
            append_jsonl(checkpoint, result)
            completed_rows.append(result)
            completed_ids.add(row["canonical_id"])
            print(json.dumps({
                "progress": f"{len(completed_rows)}/{len(selection)}",
                "canonical_id": row["canonical_id"],
                "status": result["status"],
                "adoption_rule_accepted": result.get("adoption_rule_accepted_denoiser"),
                "judge_supports_denoiser": result.get("judge_supports_denoiser"),
                "agrees": result.get("judge_support_agrees_with_adoption_rule"),
                "seconds": round(result["wall_seconds"], 3),
            }, ensure_ascii=False), flush=True)
            atomic_write_json(output / "manifest.json", {
                **run_contract,
                "status": "running",
                "completed": len(completed_rows),
                "remaining": len(selection) - len(completed_rows),
            })
    finally:
        del judge

    complete = [row for row in completed_rows if row.get("status") == "complete"]
    manifest = {
        **run_contract,
        "status": "complete",
        "completed": len(completed_rows),
        "remaining": 0,
        "status_counts": dict(Counter(row.get("status") for row in completed_rows)),
        "adoption_rule_accepted": sum(bool(row.get("adoption_rule_accepted_denoiser")) for row in complete),
        "judge_supports_denoiser": sum(bool(row.get("judge_supports_denoiser")) for row in complete),
        "judge_support_agrees_with_adoption_rule": sum(
            bool(row.get("judge_support_agrees_with_adoption_rule")) for row in complete
        ),
        "candidate_content_counts": dict(Counter(row.get("candidate_content_disposition") for row in complete)),
        "candidate_eligibility_counts": dict(Counter(row.get("candidate_eligibility") for row in complete)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "elapsed_seconds_this_process": time.monotonic() - started,
    }
    atomic_write_json(output / "manifest.json", manifest)
    atomic_write_json(output / "runtime_state.json", {
        "status": "complete",
        "completed": len(completed_rows),
        "current_canonical_id": None,
    })
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
