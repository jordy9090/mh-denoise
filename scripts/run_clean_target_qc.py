#!/usr/bin/env python3
"""Automated GPT-4.1 clean-target QC calibration and optional corpus filtering."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
from email.utils import parsedate_to_datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from corruption_contract_v2 import AXES
from fullpaper_acl_pipeline import DEFAULT_OUTPUT_DIR, largest_remainder_counts, read_jsonl, sha256_file, stable_random_key, write_json, write_jsonl
from run_paired_generator_diagnostic import ExternalJudge, JUDGE_MODEL, load_repo_env
from source_integrity_contract import VERSION as SOURCE_INTEGRITY_VERSION, contract_hash as source_integrity_contract_hash


SAMPLE_SEED = 20260908
DEFAULT_SAMPLE_SIZE = 120


def stratified_sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(row["source"], row["split"])].append(row)
    quotas = largest_remainder_counts(size, {cell: len(items) / len(rows) for cell, items in cells.items()})
    selected = []
    for cell, quota in sorted(quotas.items()):
        candidates = sorted(cells[cell], key=lambda row: stable_random_key(seed, row["canonical_id"]))
        selected.extend(candidates[:quota])
    selected.sort(key=lambda row: stable_random_key(seed + 1, row["canonical_id"]))
    if len(selected) != size or len({row["canonical_id"] for row in selected}) != size:
        raise AssertionError("stratified clean-target sample has the wrong size")
    return selected


def audit_rows(rows: list[dict[str, Any]], judge: ExternalJudge, output: Path) -> list[dict[str, Any]]:
    results = []
    for index, row in enumerate(rows, 1):
        last_error = None
        for attempt in range(1, 4):
            try:
                eligibility = judge.eligibility(row)
                result = {"canonical_id": row["canonical_id"], "source": row["source"], "split": row["split"], "question": row["question"], "clean_response": row["clean_response"], "qc_ok": True, "qc_attempt": attempt, **eligibility}
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < 3:
                    time.sleep(2 ** (attempt - 1))
        else:
            result = {"canonical_id": row["canonical_id"], "source": row["source"], "split": row["split"], "question": row["question"], "clean_response": row["clean_response"], "qc_ok": False, "qc_attempt": 3, "qc_error": last_error, "eligible": False, "baseline_degraded_axes": []}
        results.append(result)
        write_jsonl(output, results)
        print(json.dumps({"clean_target_qc": f"{index}/{len(rows)}", "retained_so_far": sum(item.get("qc_ok") and item.get("eligible") for item in results), "qc_failures_so_far": sum(not item.get("qc_ok") for item in results)}), flush=True)
    return results


def retry_delay(exc: Exception, attempt: int) -> float:
    """Honor Retry-After, then add bounded jitter to exponential backoff."""
    exponential = min(5 * (2 ** (attempt - 1)), 120)
    retry_after = 0.0
    if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
        value = exc.headers.get("Retry-After") if exc.headers else None
        if value:
            try:
                retry_after = float(value)
            except ValueError:
                try:
                    retry_after = max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
                except (TypeError, ValueError, OverflowError):
                    pass
    base = max(exponential, retry_after)
    return base + random.uniform(0, min(5.0, base * 0.2))


def audit_one(row: dict[str, Any], judge: ExternalJudge) -> dict[str, Any]:
    last_error = None
    max_attempts = 7
    for attempt in range(1, max_attempts + 1):
        try:
            eligibility = judge.eligibility(row)
            return {"canonical_id": row["canonical_id"], "source": row["source"], "split": row["split"], "qc_ok": True, "qc_attempt": attempt, **eligibility}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts:
                time.sleep(retry_delay(exc, attempt))
    return {"canonical_id": row["canonical_id"], "source": row["source"], "split": row["split"], "qc_ok": False, "qc_attempt": max_attempts, "qc_error": last_error, "eligible": False, "baseline_degraded_axes": []}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def audit_corpus(rows: list[dict[str, Any]], judge: ExternalJudge, output: Path, workers: int) -> list[dict[str, Any]]:
    existing = list(read_jsonl(output)) if output.exists() else []
    done = {row["canonical_id"] for row in existing if row.get("qc_ok")}
    pending = [row for row in rows if row["canonical_id"] not in done]
    print(json.dumps({"corpus_total": len(rows), "already_complete": len(done), "pending": len(pending), "workers": workers}), flush=True)
    completed = len(done)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(audit_one, row, judge): row["canonical_id"] for row in pending}
        for future in as_completed(futures):
            result = future.result()
            if result["qc_ok"]:
                append_jsonl(output, result)
                done.add(result["canonical_id"])
            else:
                append_jsonl(output.with_suffix(".failures.jsonl"), result)
            completed += result["qc_ok"]
            if completed % 100 == 0 or not result["qc_ok"]:
                print(json.dumps({"corpus_qc_complete": completed, "total": len(rows), "latest_qc_ok": result["qc_ok"]}), flush=True)
    results = list(read_jsonl(output))
    by_id = {row["canonical_id"]: row for row in results if row.get("qc_ok")}
    return [by_id[row["canonical_id"]] for row in rows if row["canonical_id"] in by_id]


def audit_priority_component(
    rows: list[dict[str, Any]], judge: ExternalJudge, output: Path,
    source: str, component: str, target_eligible: int,
) -> None:
    existing = list(read_jsonl(output)) if output.exists() else []
    done = {row["canonical_id"] for row in existing if row.get("qc_ok")}
    eligible = {
        row["canonical_id"] for row in existing
        if row.get("qc_ok") and row.get("baseline_degraded_axes") == []
    }
    candidates = [
        row for row in rows
        if row["split"] == "train" and row["source"] == source
        and row["source_component"] == component and row["canonical_id"] not in done
    ]
    current = sum(
        row["canonical_id"] in eligible for row in rows
        if row["split"] == "train" and row["source"] == source
        and row["source_component"] == component
    )
    print(json.dumps({"priority_source": source, "priority_component": component, "eligible_complete": current, "target_eligible": target_eligible, "pending_candidates": len(candidates)}), flush=True)
    for row in candidates:
        if current >= target_eligible:
            break
        result = audit_one(row, judge)
        if result["qc_ok"]:
            append_jsonl(output, result)
            if result.get("baseline_degraded_axes") == []:
                current += 1
        else:
            append_jsonl(output.with_suffix(".failures.jsonl"), result)
        print(json.dumps({"priority_component": component, "eligible_complete": current, "target_eligible": target_eligible, "latest_qc_ok": result["qc_ok"]}), flush=True)
    if current < target_eligible:
        raise RuntimeError(f"priority clean QC exhausted before {target_eligible} eligible rows")


def materialize_filtered(rows: list[dict[str, Any]], qc_rows: list[dict[str, Any]], output_dir: Path, judge: ExternalJudge) -> None:
    qc_by_id = {row["canonical_id"]: row for row in qc_rows}
    if len(qc_by_id) != len(rows):
        raise RuntimeError(f"corpus QC incomplete: {len(qc_by_id)}/{len(rows)} successful")
    retained, excluded = [], []
    for row in rows:
        qc = qc_by_id[row["canonical_id"]]
        if qc["baseline_degraded_axes"] == []:
            retained.append({**row, "baseline_degraded_axes": [], "clean_target_qc_model": JUDGE_MODEL})
        else:
            excluded.append({"canonical_id": row["canonical_id"], "source": row["source"], "split": row["split"], "baseline_degraded_axes": qc["baseline_degraded_axes"], "axes": qc["axes"], "reason": qc["reason"]})
    retained_path = output_dir / "canonical_clean_targets_filtered.jsonl"
    excluded_path = output_dir / "clean_target_exclusions.jsonl"
    write_jsonl(retained_path, retained); write_jsonl(excluded_path, excluded)
    manifest = {"status": "complete", "policy": "retain iff baseline_degraded_axes == []", "input_rows": len(rows), "retained_rows": len(retained), "excluded_rows": len(excluded), "retention_rate": len(retained)/len(rows), "retained_by_source": dict(Counter(row["source"] for row in retained)), "excluded_by_source": dict(Counter(row["source"] for row in excluded)), "excluded_by_axis": dict(Counter(axis for row in excluded for axis in row["baseline_degraded_axes"])), "multi_axis_exclusions": sum(len(row["baseline_degraded_axes"]) > 1 for row in excluded), "judge": judge.metadata(), "source_integrity_contract_version": SOURCE_INTEGRITY_VERSION, "source_integrity_contract_sha256": source_integrity_contract_hash(), "retained_sha256": sha256_file(retained_path), "excluded_sha256": sha256_file(excluded_path)}
    write_json(output_dir / "corpus_manifest.json", manifest)


def report(results: list[dict[str, Any]], judge: ExternalJudge, path: Path) -> None:
    evaluated = [row for row in results if row["qc_ok"]]
    retained = [row for row in evaluated if row["eligible"] and row["baseline_degraded_axes"] == []]
    dropped = [row for row in evaluated if row not in retained]
    by_source = {}
    for source in sorted({row["source"] for row in evaluated}):
        subset = [row for row in evaluated if row["source"] == source]
        kept = sum(row in retained for row in subset)
        by_source[source] = (kept, len(subset))
    axis_drops = Counter(axis for row in dropped for axis in row["baseline_degraded_axes"])
    multi = sum(len(row["baseline_degraded_axes"]) > 1 for row in dropped)
    lines = ["# Clean-target automated QC calibration", "", f"Judge: `{JUDGE_MODEL}`. No counselor or expert annotation was used.", "", "## Retention", "", f"- Sampled: {len(results)}", f"- Successfully evaluated: {len(evaluated)}", f"- Retained (`baseline_degraded_axes == []`): {len(retained)}/{len(evaluated)} ({len(retained)/len(evaluated) if evaluated else 0:.3f})", f"- QC infrastructure failures: {len(results)-len(evaluated)}", "", "| Source | Retained | Evaluated | Rate |", "|---|---:|---:|---:|"]
    for source, (kept, total) in by_source.items():
        lines.append(f"| {source} | {kept} | {total} | {kept/total:.3f} |")
    lines += ["", "## Drops by clear material baseline axis", "", "| Axis | Drop count |", "|---|---:|"]
    for axis in AXES:
        lines.append(f"| `{axis}` | {axis_drops[axis]} |")
    lines += ["", f"Multi-axis dropped examples: {multi}", "", "## Representative retained examples", "", "| ID | Source | Question | Response excerpt |", "|---|---|---|---|"]
    for row in retained[:15]:
        question = row["question"][:180].replace("\n", " ").replace("|", "\\|")
        response = row["clean_response"][:300].replace("\n", " ").replace("|", "\\|")
        lines.append(f"| `{row['canonical_id']}` | {row['source']} | {question} | {response} |")
    lines += ["", "## Representative dropped examples", "", "| ID | Source | Axes | Evidence and reason |", "|---|---|---|---|"]
    for row in dropped[:15]:
        evidence = []
        for axis in row["baseline_degraded_axes"]:
            decision = row["axes"][axis]
            span = decision["evidence_span"] or "[whole response]"
            evidence.append(f"{axis}: “{span}” — {decision['reason']}")
        rendered = "; ".join(evidence).replace("\n", " ").replace("|", "\\|")
        lines.append(f"| `{row['canonical_id']}` | {row['source']} | {', '.join(row['baseline_degraded_axes'])} | {rendered} |")
    lines += ["", "## Judge usage", "", "```json", json.dumps(judge.metadata(), indent=2), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    load_repo_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", default=str(DEFAULT_OUTPUT_DIR / "canonical_clean_qa.jsonl"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "clean_target_qc"))
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--corpus-wide", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--priority-source")
    parser.add_argument("--priority-component")
    parser.add_argument("--stop-after-eligible", type=int)
    args = parser.parse_args()
    rows = list(read_jsonl(Path(args.canonical).resolve()))
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    judge = ExternalJudge(os.getenv("OPENAI_BASE_URL", "https://api.openai.com"))
    if args.priority_source or args.priority_component or args.stop_after_eligible is not None:
        if not (args.priority_source and args.priority_component and args.stop_after_eligible is not None):
            raise ValueError("priority source, component, and stop-after-eligible must be supplied together")
        audit_priority_component(rows, judge, out / "corpus_qc.jsonl", args.priority_source, args.priority_component, args.stop_after_eligible)
        return
    if args.corpus_wide:
        qc_rows = audit_corpus(rows, judge, out / "corpus_qc.jsonl", args.workers)
        materialize_filtered(rows, qc_rows, out, judge)
        return
    selected = stratified_sample(rows, args.sample_size, SAMPLE_SEED)
    output = out / "sample_qc.jsonl"
    if output.exists():
        output.unlink()
    results = audit_rows(selected, judge, output)
    report(results, judge, out / "sample_report.md")


if __name__ == "__main__":
    main()
