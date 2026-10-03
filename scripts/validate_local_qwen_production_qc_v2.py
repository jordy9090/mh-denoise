#!/usr/bin/env python3
"""One frozen six-row validation of the revised local production QC boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from build_development_corruption_shard import candidate_a
from fullpaper_acl_pipeline import read_jsonl
from local_qwen_production_qc_v2 import (
    JUDGE_REPO, JUDGE_REVISION, PAIRED_PROMPT, SYSTEM_PROMPT, VERSION,
    content_disposition, prompt_sha256, validate_and_map,
)
from run_local_qwen_judge_comparison import LocalQwenJudge
from corruption_contract_v2 import parse_json_object


FROZEN_CASES = (
    ("qa_8ee574c74559c6dd2ec118a1", 1, 1, "grammatical repetition previously called not_fluent"),
    ("qa_552351ba742966adc648704e", 1, 1, "grammatical repetition previously called not_fluent"),
    ("qa_0e8928a8c8e7065a53037566", 1, 1, "grammatical repetition previously accepted"),
    ("qa_b9dd7dac1b39121247060822", 1, 2, "grammatical repetition previously accepted"),
    ("qa_de5ae960f73228743eec5aa6", 3, 1, "factual/medical defect previously flagged dangerous"),
    ("qa_51f0ddf0847ce37c5c43ebed", 2, 2, "nonclinical directiveness and legitimate missing-axis rejection"),
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp.replace(path)


def load_cases() -> list[dict]:
    root = Path("data/fullpaper_acl_pipeline/development_training_shard_120")
    source = []
    for path in sorted(root.glob("results_checkpoint.part*.jsonl")):
        source.extend(read_jsonl(path))
    by_id = {row["canonical_id"]: row for row in source}
    selected = []
    for index, (canonical_id, stage, attempt, purpose) in enumerate(FROZEN_CASES):
        row = by_id[canonical_id]
        match = next(
            item for item in row["stage_history"]
            if item["stage_index"] == stage and item["semantic_attempt"] == attempt
        )
        selected.append({
            "selection_index": index, "canonical_id": canonical_id,
            "stage_index": stage, "semantic_attempt": attempt, "purpose": purpose,
            "question": row["question"], "clean_response": row["clean_response"],
            "candidate_response": match["candidate_response"],
            "cumulative_intended_axes": row["intended_axes"][:stage],
            "old_stage_reason": match.get("reason"),
            "old_stage_accepted": match.get("accepted"),
            "clean_sha256": sha(row["clean_response"]),
            "candidate_sha256": sha(match["candidate_response"]),
        })
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=2400)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = load_cases()
    selection_payload = [{key: value for key, value in row.items() if key not in {"question", "clean_response", "candidate_response"}} for row in selected]
    identity = {
        "version": VERSION, "model": JUDGE_REPO, "revision": JUDGE_REVISION,
        "model_dir": str(Path(args.model_dir).resolve()), "prompt_sha256": prompt_sha256(),
        "max_new_tokens": args.max_new_tokens, "timeout_seconds": args.timeout_seconds,
        "enable_thinking": False, "do_sample": False, "use_cache": True,
        "selection": selection_payload,
    }
    identity_path = args.output_dir / "run_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("validation run identity mismatch")
    identity_path.write_text(json.dumps(identity, indent=2, ensure_ascii=False) + "\n")
    checkpoint = args.output_dir / "results_checkpoint.jsonl"
    results = list(read_jsonl(checkpoint)) if checkpoint.exists() else []
    if [row["canonical_id"] for row in results] != [row["canonical_id"] for row in selected[:len(results)]]:
        raise RuntimeError("validation checkpoint is not an exact selection prefix")
    judge = LocalQwenJudge(
        Path(args.model_dir), JUDGE_REPO, JUDGE_REVISION,
        args.max_new_tokens, args.timeout_seconds, system_prompt=SYSTEM_PROMPT,
    )
    try:
        for row in selected[len(results):]:
            candidate_label = "A" if candidate_a(row["canonical_id"], row["stage_index"], row["semantic_attempt"]) else "B"
            clean_label = "B" if candidate_label == "A" else "A"
            a = row["candidate_response"] if candidate_label == "A" else row["clean_response"]
            b = row["clean_response"] if candidate_label == "A" else row["candidate_response"]
            started = time.monotonic()
            raw, usage = judge.call(PAIRED_PROMPT.format(question=row["question"], a=a, b=b))
            result = {**selection_payload[row["selection_index"]], "raw_output": raw, "usage": usage}
            try:
                grade, invalid = validate_and_map(parse_json_object(raw), a, b, clean_label, candidate_label)
                disposition, reason = content_disposition(row["question"], row["candidate_response"], grade)
                missing = [axis for axis in row["cumulative_intended_axes"] if axis not in grade["realized_axes"]]
                result.update({
                    "json_valid": True, "schema_valid": True, "grade": grade,
                    "content_disposition": disposition, "content_reason": reason,
                    "missing_intended_axes": missing, "invalid_evidence": invalid,
                })
            except Exception as exc:
                result.update({"json_valid": False, "schema_valid": False, "error": f"{type(exc).__name__}: {exc}"})
            result["wall_seconds"] = time.monotonic() - started
            results.append(result)
            atomic_jsonl(checkpoint, results)
            print(json.dumps({
                "progress": f"{len(results)}/6", "canonical_id": row["canonical_id"],
                "schema_valid": result["schema_valid"], "usage": usage,
                "disposition": result.get("content_disposition"),
            }, ensure_ascii=False), flush=True)
    finally:
        del judge.model
    manifest = {
        **identity, "status": "complete", "processed": len(results),
        "schema_valid": sum(row.get("schema_valid", False) for row in results),
        "dispositions": {key: sum(row.get("content_disposition") == key for row in results) for key in ("pass", "reject", "hold")},
        "wall_seconds": sum(row["wall_seconds"] for row in results),
        "calls": sum(1 for _ in results),
        "input_tokens": sum(row["usage"]["input_tokens"] for row in results),
        "output_tokens": sum(row["usage"]["output_tokens"] for row in results),
    }
    (args.output_dir / "validation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
