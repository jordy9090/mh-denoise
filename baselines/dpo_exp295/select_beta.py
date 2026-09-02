#!/usr/bin/env python3
"""Select DPO beta on exp295 validation results under explicit safety constraints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

REQUIRED_METRICS = (
    "overall_quality_mean",
    "empathy_mean",
    "specificity_mean",
    "medical_advice_mean",
    "factual_consistency_mean",
    "toxicity_mean",
    "quality_safety_average_mean",
)


def read_csvs(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                converted: dict[str, Any] = dict(row)
                for key, value in list(converted.items()):
                    if key in {"system", "variant"} or value in (None, ""):
                        continue
                    try:
                        converted[key] = float(value)
                    except (TypeError, ValueError):
                        pass
                converted["_source_csv"] = str(path)
                rows.append(converted)
    return rows


def decode_beta(value: str) -> float:
    return float(value.replace("m", "-").replace("p", "."))


def parse_candidate(system: str, pattern: re.Pattern[str]) -> tuple[str, float, int] | None:
    match = pattern.fullmatch(system)
    if not match:
        return None
    return match.group("variant"), decode_beta(match.group("beta")), int(match.group("seed"))


def require_metrics(row: Mapping[str, Any], system: str) -> None:
    missing = [key for key in REQUIRED_METRICS if row.get(key) in (None, "")]
    if missing:
        raise ValueError(f"system={system!r} is missing metrics: {missing}")


def as_float(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric {key}={value}")
    return value


def write_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "system",
        "beta",
        "seed",
        "n",
        "medical_advice_mean",
        "medical_delta_from_sft",
        "toxicity_mean",
        "toxicity_delta_from_sft",
        "quality_safety_average_mean",
        "feasible",
        "rejection_reasons",
        "selected",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=Path, action="append", required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--baseline_system", default="sft_qd_s42")
    parser.add_argument(
        "--candidate_regex",
        default=r"(?P<variant>dpo_(?:minimal|hard_k4))_b(?P<beta>[0-9p]+)_s(?P<seed>[0-9]+)",
    )
    parser.add_argument("--sweep_seed", type=int, default=42)
    parser.add_argument("--expected_n", type=int, default=174)
    parser.add_argument("--medical_tolerance", type=float, default=0.0)
    parser.add_argument("--toxicity_tolerance", type=float, default=0.0)
    parser.add_argument(
        "--minimum_quality_delta",
        type=float,
        default=-0.05,
        help="Maximum allowed drop in the six-axis quality/safety mean.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pattern = re.compile(args.candidate_regex)
    rows = read_csvs(args.input_csv)
    by_system: dict[str, dict[str, Any]] = {}
    for row in rows:
        system = str(row.get("system", ""))
        if not system:
            continue
        if system in by_system:
            raise ValueError(f"duplicate system row across aggregate CSVs: {system}")
        by_system[system] = row

    if args.baseline_system not in by_system:
        raise ValueError(f"baseline system {args.baseline_system!r} not found")
    baseline = by_system[args.baseline_system]
    require_metrics(baseline, args.baseline_system)
    baseline_n = int(float(baseline.get("n", 0)))
    if baseline_n != args.expected_n:
        raise ValueError(
            f"baseline {args.baseline_system} has n={baseline_n}; expected validation n={args.expected_n}"
        )

    baseline_medical = as_float(baseline, "medical_advice_mean")
    baseline_toxicity = as_float(baseline, "toxicity_mean")
    baseline_quality = as_float(baseline, "quality_safety_average_mean")

    candidates: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for system, row in by_system.items():
        parsed = parse_candidate(system, pattern)
        if parsed is None:
            continue
        variant, beta, seed = parsed
        if seed != args.sweep_seed:
            continue
        require_metrics(row, system)
        n = int(float(row.get("n", 0)))
        medical = as_float(row, "medical_advice_mean")
        toxicity = as_float(row, "toxicity_mean")
        quality = as_float(row, "quality_safety_average_mean")
        medical_delta = medical - baseline_medical
        toxicity_delta = toxicity - baseline_toxicity
        quality_delta = quality - baseline_quality
        reasons: list[str] = []
        if n != args.expected_n:
            reasons.append(f"n={n}, expected={args.expected_n}")
        if medical_delta < -args.medical_tolerance:
            reasons.append(f"medical_delta={medical_delta:.6f} < {-args.medical_tolerance:.6f}")
        if toxicity_delta > args.toxicity_tolerance:
            reasons.append(f"toxicity_delta={toxicity_delta:.6f} > {args.toxicity_tolerance:.6f}")
        if quality_delta < args.minimum_quality_delta:
            reasons.append(f"quality_delta={quality_delta:.6f} < {args.minimum_quality_delta:.6f}")
        candidates[variant].append(
            {
                "variant": variant,
                "system": system,
                "beta": beta,
                "seed": seed,
                "n": n,
                "medical_advice_mean": medical,
                "medical_delta_from_sft": medical_delta,
                "toxicity_mean": toxicity,
                "toxicity_delta_from_sft": toxicity_delta,
                "quality_safety_average_mean": quality,
                "quality_delta_from_sft": quality_delta,
                "feasible": not reasons,
                "rejection_reasons": "; ".join(reasons),
                "selected": False,
            }
        )

    if not candidates:
        raise ValueError("no candidate systems matched candidate_regex and sweep_seed")

    selections: dict[str, Any] = {}
    flat: list[dict[str, Any]] = []
    for variant, items in sorted(candidates.items()):
        feasible = [item for item in items if item["feasible"]]
        if feasible:
            winner = max(
                feasible,
                key=lambda item: (
                    item["quality_safety_average_mean"],
                    item["medical_advice_mean"],
                    -item["toxicity_mean"],
                    -item["beta"],
                ),
            )
            winner["selected"] = True
            selections[variant] = {
                "status": "selected",
                "system": winner["system"],
                "beta": winner["beta"],
                "seed": winner["seed"],
                "selection_rule": (
                    "satisfy validation n and safety constraints; maximize quality_safety_average; "
                    "tie-break by medical boundary, toxicity, then smaller beta"
                ),
            }
        else:
            selections[variant] = {
                "status": "no_feasible_beta",
                "system": None,
                "beta": None,
                "seed": args.sweep_seed,
            }
        flat.extend(sorted(items, key=lambda item: item["beta"]))

    payload = {
        "baseline": {
            "system": args.baseline_system,
            "n": baseline_n,
            "medical_advice_mean": baseline_medical,
            "toxicity_mean": baseline_toxicity,
            "quality_safety_average_mean": baseline_quality,
        },
        "constraints": {
            "expected_n": args.expected_n,
            "medical_tolerance": args.medical_tolerance,
            "toxicity_tolerance": args.toxicity_tolerance,
            "minimum_quality_delta": args.minimum_quality_delta,
            "sweep_seed": args.sweep_seed,
        },
        "selections": selections,
        "candidates": flat,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    write_csv(flat, args.output_csv)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
