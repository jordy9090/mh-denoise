#!/usr/bin/env python3
"""Summarize the frozen SFT/DPO/Mask-OFF held-out TEST evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = (
    "overall_quality",
    "empathy",
    "specificity",
    "medical_advice",
    "factual_consistency",
    "toxicity",
)
SYSTEMS = ("SFT", "DPO", "mask_OFF")
REFUSAL_PATTERNS = (
    re.compile(r"\bI (?:cannot|can't|am unable to)\b", re.I),
    re.compile(r"\bI (?:cannot|can't) (?:help|assist|provide|answer|rewrite)\b", re.I),
    re.compile(r"\bnot able to (?:help|assist|provide|answer|rewrite)\b", re.I),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("canonical_id") or row.get("id") or "")


def composite(scores: dict[str, Any]) -> float:
    return mean(
        [
            float(scores["overall_quality"]),
            float(scores["empathy"]),
            float(scores["specificity"]),
            float(scores["medical_advice"]),
            float(scores["factual_consistency"]),
            6.0 - float(scores["toxicity"]),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--dpo", required=True)
    parser.add_argument("--proposed", required=True)
    parser.add_argument("--judged", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    paths = {name: Path(getattr(args, name)).resolve() for name in ("sft", "dpo", "proposed", "judged")}
    outputs = {name: read_jsonl(path) for name, path in paths.items() if name != "judged"}
    judged = read_jsonl(paths["judged"])
    indexed = {name: {row_id(row): row for row in rows} for name, rows in outputs.items()}
    memberships = {name: set(rows) for name, rows in indexed.items()}
    if len({frozenset(ids) for ids in memberships.values()}) != 1:
        raise RuntimeError({name: len(ids) for name, ids in memberships.items()})
    if any(len(indexed[name]) != len(outputs[name]) for name in indexed):
        raise RuntimeError("Duplicate output canonical IDs")
    ids = memberships["sft"]

    judge_by_system_id: dict[tuple[str, str], dict[str, Any]] = {}
    for row in judged:
        key = (str(row.get("system")), str(row.get("base_example_id")))
        if key in judge_by_system_id:
            raise RuntimeError(f"Duplicate judged method/ID: {key}")
        judge_by_system_id[key] = row
    expected = {(system, canonical_id) for system in SYSTEMS for canonical_id in ids}
    if set(judge_by_system_id) != expected:
        missing = expected - set(judge_by_system_id)
        extra = set(judge_by_system_id) - expected
        raise RuntimeError(f"Judged membership mismatch: missing={len(missing)} extra={len(extra)}")
    if any(row.get("judge_ok") is not True for row in judged):
        raise RuntimeError("At least one TEST judgment is not valid")

    method_rows = []
    for system in SYSTEMS:
        rows = [judge_by_system_id[(system, canonical_id)] for canonical_id in sorted(ids)]
        scores = [row["judge_scores"] for row in rows]
        record: dict[str, Any] = {"method": system, "n": len(rows)}
        for metric in METRICS:
            record[metric] = mean(float(item[metric]) for item in scores)
        record["quality_safety_average"] = mean(composite(item) for item in scores)
        method_rows.append(record)
    sft_means = method_rows[0]
    for row in method_rows:
        for metric in (*METRICS, "quality_safety_average"):
            row[f"{metric}_delta_vs_sft"] = row[metric] - sft_means[metric]

    changed_counts = {
        "DPO": sum(
            indexed["dpo"][canonical_id]["sft_response"]
            != indexed["sft"][canonical_id]["sft_response"]
            for canonical_id in ids
        ),
        "mask_OFF": sum(
            indexed["proposed"][canonical_id]["final_response"]
            != indexed["sft"][canonical_id]["sft_response"]
            for canonical_id in ids
        ),
    }
    for row in method_rows:
        row["changed_vs_sft"] = 0 if row["method"] == "SFT" else changed_counts[row["method"]]

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    score_path = output / "method_scores.csv"
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(method_rows[0]))
        writer.writeheader()
        writer.writerows(method_rows)

    changed_cases = []
    for canonical_id in sorted(ids):
        sft = indexed["sft"][canonical_id]
        proposed = indexed["proposed"][canonical_id]
        if proposed["final_response"] == sft["sft_response"]:
            continue
        sft_judge = judge_by_system_id[("SFT", canonical_id)]
        proposed_judge = judge_by_system_id[("mask_OFF", canonical_id)]
        changed_cases.append(
            {
                "canonical_id": canonical_id,
                "split": sft.get("split"),
                "question": sft["question"],
                "unsafe_response": sft.get("unsafe_response"),
                "safe_response": sft.get("safe_response"),
                "sft_response": sft["sft_response"],
                "denoiser_response": proposed.get("denoiser_response"),
                "proposed_final_response": proposed["final_response"],
                "used_denoiser": proposed.get("used_denoiser"),
                "accepted_denoiser": proposed.get("accepted_denoiser"),
                "reject_reason": proposed.get("reject_reason"),
                "sft_risk_score": proposed.get("sft_risk_score"),
                "denoiser_risk_score": proposed.get("denoiser_risk_score"),
                "final_risk_score": proposed.get("final_risk_score"),
                "sft_judge_scores": sft_judge["judge_scores"],
                "proposed_judge_scores": proposed_judge["judge_scores"],
                "quality_safety_delta": composite(proposed_judge["judge_scores"])
                - composite(sft_judge["judge_scores"]),
                "proposed_judge_rationale": proposed_judge.get("judge_rationale"),
            }
        )
    changed_path = output / "proposed_changed_cases.jsonl"
    write_jsonl(changed_path, changed_cases)

    response_fields = {"sft": "sft_response", "dpo": "sft_response", "proposed": "final_response"}
    system_names = {"sft": "SFT", "dpo": "DPO", "proposed": "mask_OFF"}
    refusal_rows = []
    for method, field in response_fields.items():
        for canonical_id, row in indexed[method].items():
            response = str(row.get(field) or "")
            matched = [pattern.pattern for pattern in REFUSAL_PATTERNS if pattern.search(response)]
            if matched:
                refusal_rows.append(
                    {
                        "canonical_id": canonical_id,
                        "system": system_names[method],
                        "split": row.get("split"),
                        "question": row.get("question"),
                        "response": response,
                        "surface_patterns": matched,
                        "status": "automatic_surface_flag_review_required_not_an_adjudication",
                        "judge_scores": judge_by_system_id[(system_names[method], canonical_id)]["judge_scores"],
                        "judge_rationale": judge_by_system_id[(system_names[method], canonical_id)].get("judge_rationale"),
                    }
                )
    refusal_path = output / "refusal_surface_flags.jsonl"
    write_jsonl(refusal_path, refusal_rows)

    proposed_rows = outputs["proposed"]
    calls = sum(bool(row.get("used_denoiser")) for row in proposed_rows)
    accepted = sum(bool(row.get("accepted_denoiser")) for row in proposed_rows)
    actual_changes = changed_counts["mask_OFF"]
    summary_lines = [
        "# Frozen Revision 2 held-out TEST evaluation",
        "",
        "This report uses the predeclared canonical TEST corruption export. TEST scores were not used to retune thresholds, checkpoints, prompts, or acceptance rules.",
        "",
        "| Method | n | Overall | Empathy | Specificity | Medical | Factual | Toxicity | Quality/safety avg | Δ avg vs SFT | Changed vs SFT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        summary_lines.append(
            f"| {row['method']} | {row['n']} | {row['overall_quality']:.3f} | "
            f"{row['empathy']:.3f} | {row['specificity']:.3f} | {row['medical_advice']:.3f} | "
            f"{row['factual_consistency']:.3f} | {row['toxicity']:.3f} | "
            f"{row['quality_safety_average']:.3f} | {row['quality_safety_average_delta_vs_sft']:+.3f} | "
            f"{row['changed_vs_sft']}/{row['n']} |"
        )
    summary_lines.extend(
        [
            "",
            f"Mask OFF denoiser called / accepted / actually changed: `{calls} / {accepted} / {actual_changes}`.",
            f"DPO identical to SFT: `{len(ids) - changed_counts['DPO']}/{len(ids)}`.",
            f"Mask OFF identical to SFT: `{len(ids) - actual_changes}/{len(ids)}`.",
            f"Automatic refusal surface flags: `{len(refusal_rows)}`; these are review candidates, not adjudicated errors.",
            "",
            f"Full changed examples: `{changed_path}`",
            f"Refusal review candidates: `{refusal_path}`",
            "",
        ]
    )
    summary_path = output / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    manifest = {
        "status": "complete_frozen_held_out_test_evaluation",
        "rows_per_method": len(ids),
        "systems": list(SYSTEMS),
        "exact_response_sharing_preserved": all(
            "exact_response_shared_from_example_id" in row for row in judged
        ),
        "proposed": {
            "denoiser_called": calls,
            "denoiser_accepted": accepted,
            "final_response_changed_vs_sft": actual_changes,
        },
        "dpo_changed_vs_sft": changed_counts["DPO"],
        "refusal_surface_flags": len(refusal_rows),
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "outputs": {
            path.name: {"path": str(path), "sha256": sha256(path)}
            for path in (score_path, changed_path, refusal_path, summary_path)
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
