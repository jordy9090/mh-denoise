#!/usr/bin/env python3
"""Guarded 30-row Qwen single-axis calibration with the pinned GPT-4.1 judge."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from corruption_contract_v2 import AXES
from fullpaper_acl_pipeline import DEFAULT_OUTPUT_DIR, read_jsonl, stable_random_key, write_json, write_jsonl
from run_paired_generator_diagnostic import (
    DEFAULT_REQUIRED_FREE_VRAM_MIB,
    ExternalJudge,
    GENERATORS,
    JUDGE_MODEL,
    PILOT_SEED,
    candidate_rows,
    choose_eligible,
    generation_prompt_sha256,
    gpu_status,
    load_repo_env,
    prompt_sha256,
    run_generator,
    selected_length_summary,
)


CALIBRATION_SEED = PILOT_SEED + 1
CALIBRATION_SIZE = 30


def single_axis_specs(seed: int) -> list[tuple[str, ...]]:
    specs = [(axis,) for axis in AXES for _ in range(5)]
    return sorted(specs, key=lambda spec: stable_random_key(seed, f"{spec[0]}:{specs.index(spec)}"))


def load_frozen_selection(path: Path, canonical: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["canonical_id"]: row for row in canonical}
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = []
    for slot, item in enumerate(payload["rows"]):
        clean = by_id[item["canonical_id"]]
        if clean["split"] != "train":
            raise AssertionError("single-axis calibration contains a non-TRAIN row")
        selected.append({"slot": slot, "clean": clean, "intended_axes": (item["intended_axis"],), "axis_count": 1, "eligibility": item["eligibility"]})
    if len(selected) != CALIBRATION_SIZE or Counter(item["intended_axes"][0] for item in selected) != {axis: 5 for axis in AXES}:
        raise AssertionError("frozen selection is not exactly five rows per axis")
    return selected


def write_selection(path: Path, selected: list[dict[str, Any]]) -> None:
    write_json(path, {"version": "qwen-single-axis-calibration-v1", "seed": CALIBRATION_SEED, "split": "train", "rows": [{"canonical_id": item["clean"]["canonical_id"], "intended_axis": item["intended_axes"][0], "eligibility": item["eligibility"]} for item in selected]})


def build_report(records: list[dict[str, Any]], failures: list[dict[str, Any]], metrics: dict[str, Any], judge_meta: dict[str, Any], path: Path) -> None:
    rows = records + failures
    lines = ["# Qwen single-axis calibration (30 TRAIN examples)", "", f"Generator: `Qwen/Qwen3.5-4B` at `{GENERATORS['qwen']['revision']}`", f"Judge: `{JUDGE_MODEL}`; prompt SHA-256 `{prompt_sha256()}`", "", "| Axis | First-attempt success | Final success | Go/no-go |", "|---|---:|---:|---|"]
    final_counts = Counter(r["intended_axes"][0] for r in records)
    for axis in AXES:
        axis_rows = [r for r in rows if r["intended_axes"] == [axis]]
        first = sum(any(event["accepted"] and event["attempt"] == 1 for event in row["stage_history"]) for row in axis_rows)
        final = final_counts[axis]
        lines.append(f"| `{axis}` | {first}/5 | {final}/5 | {'GO' if final >= 4 else 'NO-GO'} |")
    unintended = sum(len(set(row["realized_axes"]) - set(row["intended_axes"])) for row in records)
    available = len(records) * (len(AXES) - 1)
    events = [event for row in rows for event in row["stage_history"]]
    lines += ["", "## Realized axes and operational failures", "", f"- Accepted realized-axis sets: `{dict(Counter('+'.join(row['realized_axes']) for row in records))}`", f"- Unintended-axis events: {unintended}/{available} available unselected slots ({unintended / available if available else 0:.3f})", f"- Refusals: {sum(event['failure_reason'] == 'refusal' for event in events)}", f"- Genuine truncations: {sum(event['failure_reason'] == 'generation_truncation' for event in events)}", f"- Judge/parse/evidence failures: {sum(str(event['failure_reason']).startswith('judge_or_evidence_failure') for event in events)}", f"- Calls: {metrics['generator_calls']} generator + {metrics['judge_calls']} judge", f"- Tokens: {metrics['generator_prompt_tokens'] + metrics['judge_prompt_tokens']} prompt + {metrics['generator_completion_tokens'] + metrics['judge_completion_tokens']} completion", "", "## Concrete examples", "", "| Axis | Outcome | ID | Candidate excerpt | Reason |", "|---|---|---|---|---|"]
    for axis in AXES:
        axis_rows = [row for row in rows if row["intended_axes"] == [axis]]
        successes = [(row, event) for row in axis_rows for event in row["stage_history"] if event["accepted"]]
        rejected = [(row, event) for row in axis_rows for event in row["stage_history"] if not event["accepted"]]
        for outcome, examples in (("success", successes[:1]), ("failure", rejected[:1])):
            if not examples:
                lines.append(f"| `{axis}` | {outcome} | — | — | none |")
                continue
            row, event = examples[0]
            excerpt = event["candidate_response"][:280].replace("\n", " ").replace("|", "\\|")
            decision = event["judge"]["axes"][axis] if event["judge"] else None
            reason = decision["reason"] if decision else event["failure_reason"]
            lines.append(f"| `{axis}` | {outcome} | `{row['canonical_id']}` | {excerpt} | {str(reason).replace('|', '\\|')} |")
    below = [axis for axis in AXES if final_counts[axis] < 4]
    lines += ["", "## Gate", "", "GO to multi-axis calibration." if not below else "NO-GO. Patch and rerun only these below-threshold axes: " + ", ".join(f"`{axis}`" for axis in below), "", "No full corruption or model training was run.", "", "## Judge manifest", "", "```json", json.dumps(judge_meta, indent=2), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    load_repo_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", default=str(DEFAULT_OUTPUT_DIR / "canonical_clean_qa.jsonl"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "qwen_single_axis_calibration_30"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--selected-gpu", type=int, default=0)
    parser.add_argument("--required-free-vram-mib", type=int, default=DEFAULT_REQUIRED_FREE_VRAM_MIB)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    canonical_path = Path(args.canonical).resolve(); out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    canonical = list(read_jsonl(canonical_path)); train_candidates = candidate_rows(canonical_path, CALIBRATION_SEED)
    gpu = gpu_status(args.selected_gpu, args.required_free_vram_mib)
    preflight = {"size": 30, "split": "train", "axis_counts": {axis: 5 for axis in AXES}, "seed": CALIBRATION_SEED, "generator": GENERATORS["qwen"], "generation_prompt_sha256": generation_prompt_sha256(), "judge_model": JUDGE_MODEL, "judge_prompt_sha256": prompt_sha256(), "judge_available": bool(os.getenv("OPENAI_API_KEY")), "gpu": gpu}
    write_json(out / "preflight.json", preflight); print(json.dumps(preflight), flush=True)
    if args.preflight_only: return
    if not preflight["judge_available"]: raise RuntimeError("External GPT-4.1 judge unavailable; refusing to calibrate with self-QC")
    if not gpu["sufficient_free_vram"]: raise RuntimeError("insufficient free VRAM for Qwen calibration")
    judge = ExternalJudge(os.getenv("OPENAI_BASE_URL", "https://api.openai.com"))
    selection_path = out / "selection.json"
    if selection_path.exists(): selected = load_frozen_selection(selection_path, canonical)
    else:
        selected, ineligible = choose_eligible(judge, train_candidates, single_axis_specs(CALIBRATION_SEED))
        write_selection(selection_path, selected); write_jsonl(out / "source_ineligible.jsonl", ineligible)
    write_json(out / "adaptive_length_preflight.json", selected_length_summary(selected))
    records, failures, metrics = run_generator(
        "qwen", selected, judge, args, generation_seed_base=CALIBRATION_SEED
    )
    write_jsonl(out / "accepted.jsonl", records); write_jsonl(out / "failures.jsonl", failures)
    judge_meta = judge.metadata(); write_json(out / "judge_manifest.json", judge_meta); write_json(out / "metrics.json", metrics)
    build_report(records, failures, metrics, judge_meta, out / "report.md")
    print(json.dumps({"accepted": len(records), "failed": len(failures), "report": str(out / 'report.md')}), flush=True)


if __name__ == "__main__":
    main()
