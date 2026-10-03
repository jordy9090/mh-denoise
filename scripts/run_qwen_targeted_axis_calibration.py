#!/usr/bin/env python3
"""Targeted Qwen calibration for the five non-frozen corruption operators."""

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
    DEFAULT_REQUIRED_FREE_VRAM_MIB, ExternalJudge, GENERATORS, JUDGE_MODEL,
    generation_prompt_sha256, gpu_status, load_repo_env, prompt_sha256, run_generator,
)

TARGETED_SEED = 20260909
TARGET_AXES = ("overall_quality", "empathy", "specificity", "medical_boundary", "toxicity_or_harm")
FROZEN_AXIS = "factual_consistency"


def prior_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    for path in (
        root / "qwen_single_axis_calibration_30" / "selection.json",
        root / "paired_generator_diagnostic_24" / "selection.json",
    ):
        if path.exists():
            ids.update(row["canonical_id"] for row in json.loads(path.read_text())["rows"])
    return ids


def select_rows(canonical: list[dict[str, Any]], qc_rows: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    by_id = {row["canonical_id"]: row for row in canonical}
    excluded = prior_ids(root)
    clean_ids = {
        row["canonical_id"] for row in qc_rows
        if row.get("qc_ok") and row.get("baseline_degraded_axes") == []
        and row["canonical_id"] in by_id and by_id[row["canonical_id"]]["split"] == "train"
        and row["canonical_id"] not in excluded
    }
    candidates = sorted(clean_ids, key=lambda cid: stable_random_key(TARGETED_SEED, cid))
    if len(candidates) < 25:
        raise RuntimeError(f"need 25 fresh clean TRAIN checkpoints; found {len(candidates)}")
    selected = []
    specs = sorted([(axis, i) for axis in TARGET_AXES for i in range(5)], key=lambda x: stable_random_key(TARGETED_SEED, f"{x[0]}:{x[1]}"))
    for slot, ((axis, _), cid) in enumerate(zip(specs, candidates[:25], strict=True)):
        qc = next(row for row in qc_rows if row["canonical_id"] == cid and row.get("qc_ok"))
        selected.append({"slot": slot, "clean": by_id[cid], "intended_axes": (axis,), "axis_count": 1, "eligibility": qc})
    assert Counter(x["intended_axes"][0] for x in selected) == {axis: 5 for axis in TARGET_AXES}
    assert all(x["eligibility"]["baseline_degraded_axes"] == [] for x in selected)
    return selected


def build_report(records: list[dict[str, Any]], failures: list[dict[str, Any]], metrics: dict[str, Any], path: Path) -> None:
    rows = records + failures
    semantic = [r for r in failures if r.get("failure_class") == "semantic"]
    infrastructure = [r for r in failures if r.get("failure_class") == "infrastructure"]
    counts = Counter(r["intended_axes"][0] for r in records)
    lines = ["# Targeted Qwen five-axis calibration", "", f"Generator: `Qwen/Qwen3.5-4B` at `{GENERATORS['qwen']['revision']}`", f"Judge: `{JUDGE_MODEL}`; prompt SHA-256 `{prompt_sha256()}`", "", "| Axis | Valid judged | Intended-axis realized | Gate |", "|---|---:|---:|---|"]
    for axis in TARGET_AXES:
        valid = sum(r["intended_axes"] == [axis] for r in records + semantic)
        lines.append(f"| `{axis}` | {valid}/5 | {counts[axis]}/5 | {'GO' if counts[axis] >= 4 else 'NO-GO'} |")
    unintended = sum(len(r["unintended_axes"]) for r in records)
    slots = len(records) * (len(AXES) - 1)
    infra_events = [e for r in rows for e in r.get("infrastructure_failures", [])]
    lines += ["", "## Realized axes and infrastructure", "", f"- Realized-axis sets: `{dict(Counter('+'.join(r['realized_axes']) for r in records))}`", f"- Unintended-axis rate: {unintended}/{slots} ({unintended/slots if slots else 0:.3f})", f"- Infrastructure events: {len(infra_events)} `{dict(Counter(e['kind'] for e in infra_events))}`", f"- Infrastructure-exhausted examples: {len(infrastructure)}", f"- Generator calls: {metrics['generator_calls']}; judge calls: {metrics['judge_calls']}", "", "## Concrete accepted/rejected examples", "", "| Axis | Outcome | ID | Realized | Excerpt / reason |", "|---|---|---|---|---|"]
    for axis in TARGET_AXES:
        accepted = [r for r in records if r["intended_axes"] == [axis]][:1]
        rejected = [r for r in semantic if r["intended_axes"] == [axis]][:1]
        for outcome, examples in (("accepted", accepted), ("rejected", rejected)):
            if not examples:
                lines.append(f"| `{axis}` | {outcome} | — | — | none |")
                continue
            row = examples[0]
            if outcome == "accepted":
                detail = row["corrupted_response"][:300].replace("\n", " ").replace("|", "\\|")
                realized = ", ".join(row["realized_axes"])
            else:
                detail = str(row["failure_reason"]).replace("|", "\\|")
                realized = "—"
            lines.append(f"| `{axis}` | {outcome} | `{row['canonical_id']}` | {realized} | {detail} |")
    lines += ["", "No full corruption or training was run.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    load_repo_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", default=str(DEFAULT_OUTPUT_DIR / "canonical_clean_qa.jsonl"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "qwen_targeted_axis_calibration_25"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--selected-gpu", type=int, default=0)
    parser.add_argument("--required-free-vram-mib", type=int, default=DEFAULT_REQUIRED_FREE_VRAM_MIB)
    args = parser.parse_args()
    root = DEFAULT_OUTPUT_DIR
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    canonical = list(read_jsonl(Path(args.canonical).resolve()))
    qc_rows = list(read_jsonl(root / "clean_target_qc" / "corpus_qc.jsonl"))
    selected = select_rows(canonical, qc_rows, root)
    write_json(out / "selection.json", {"seed": TARGETED_SEED, "rows": [{"canonical_id": x["clean"]["canonical_id"], "intended_axis": x["intended_axes"][0], "baseline_degraded_axes": []} for x in selected]})
    gpu = gpu_status(args.selected_gpu, args.required_free_vram_mib)
    write_json(out / "preflight.json", {"seed": TARGETED_SEED, "axes": list(TARGET_AXES), "frozen_excluded_axis": FROZEN_AXIS, "generator": GENERATORS["qwen"], "generation_prompt_sha256": generation_prompt_sha256(), "judge": JUDGE_MODEL, "judge_prompt_sha256": prompt_sha256(), "gpu": gpu})
    if not gpu["sufficient_free_vram"]: raise RuntimeError("insufficient free VRAM")
    judge = ExternalJudge(os.getenv("OPENAI_BASE_URL", "https://api.openai.com"))
    records, failures, metrics = run_generator("qwen", selected, judge, args, generation_seed_base=TARGETED_SEED)
    write_jsonl(out / "accepted.jsonl", records); write_jsonl(out / "failures.jsonl", failures)
    write_json(out / "metrics.json", metrics); write_json(out / "judge_manifest.json", judge.metadata())
    build_report(records, failures, metrics, out / "report.md")
    print(json.dumps({"accepted": len(records), "semantic_failures": sum(r.get('failure_class') == 'semantic' for r in failures), "infrastructure_failures": sum(r.get('failure_class') == 'infrastructure' for r in failures)}), flush=True)


if __name__ == "__main__": main()
