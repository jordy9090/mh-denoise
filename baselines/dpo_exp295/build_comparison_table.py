#!/usr/bin/env python3
"""Build paper-ready SFT/DPO/DPO+denoiser/Proposed comparison tables.

The script consumes a JSON catalog so every number remains traceable to a result
file. Missing values are rendered as an em dash; no result is fabricated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping

JUDGE_METRICS = (
    ("overall_quality_mean", "Overall$\\uparrow$"),
    ("empathy_mean", "Emp.$\\uparrow$"),
    ("specificity_mean", "Spec.$\\uparrow$"),
    ("medical_advice_mean", "Med.$\\uparrow$"),
    ("factual_consistency_mean", "Fact.$\\uparrow$"),
    ("toxicity_mean", "Tox.$\\downarrow$"),
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean_std(values: Iterable[float | None]) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return None, None
    return statistics.mean(clean), statistics.stdev(clean) if len(clean) > 1 else 0.0


def format_value(mean: float | None, std: float | None, digits: int = 3) -> str:
    if mean is None:
        return "—"
    if std is None or std == 0.0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def resolve_path(catalog_path: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (catalog_path.parent / path).resolve()


def judge_rows(spec: Mapping[str, Any], catalog_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    target_systems = set(str(x) for x in spec.get("judge_systems", []))
    csv_paths = list(spec.get("judge_csvs", []))
    if csv_paths and not target_systems:
        raise ValueError(f"{spec.get('key')}: judge_systems must be explicit when judge_csvs are provided")
    for raw_path in csv_paths:
        path = resolve_path(catalog_path, str(raw_path))
        for row in read_csv(path):
            if not target_systems or str(row.get("system", "")) in target_systems:
                rows.append(row)
    if target_systems:
        found = {str(row.get("system", "")) for row in rows}
        missing = sorted(target_systems - found)
        if missing:
            raise ValueError(f"{spec.get('key')}: judge systems missing from CSVs: {missing}")
    return rows


def efficiency_values(spec: Mapping[str, Any], catalog_path: Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {
        "latency_ms": [],
        "serve_vram_gib": [],
        "train_gpu_hours": [],
        "trainable_m": [],
    }
    for raw_path in spec.get("efficiency_jsons", []):
        payload = read_json(resolve_path(catalog_path, str(raw_path)))
        latency = finite_number((payload.get("latency_seconds") or {}).get("mean"))
        vram = finite_number(payload.get("peak_cuda_allocated_gib"))
        if latency is not None:
            values["latency_ms"].append(latency * 1000.0)
        if vram is not None:
            values["serve_vram_gib"].append(vram)

    for raw_path in spec.get("train_manifests", []):
        payload = read_json(resolve_path(catalog_path, str(raw_path)))
        runtime = finite_number(payload.get("runtime_seconds"))
        trainable = finite_number(
            (payload.get("reference_initialization") or {}).get("trainable_policy_parameters")
        )
        if runtime is not None:
            values["train_gpu_hours"].append(runtime / 3600.0)
        if trainable is not None:
            values["trainable_m"].append(trainable / 1_000_000.0)

    for key in values:
        manual = spec.get("manual", {}).get(key)
        if manual is not None:
            if isinstance(manual, list):
                values[key].extend(float(x) for x in manual)
            else:
                values[key].append(float(manual))
    return values


def gate_rates(spec: Mapping[str, Any], catalog_path: Path) -> tuple[float | None, float | None]:
    invoked: list[bool] = []
    accepted: list[bool] = []
    for raw_path in spec.get("selective_outputs", []):
        rows = read_jsonl(resolve_path(catalog_path, str(raw_path)))
        for row in rows:
            if "denoiser_invoked" in row:
                invoked.append(bool(row["denoiser_invoked"]))
            if "denoiser_accepted" in row:
                accepted.append(bool(row["denoiser_accepted"]))
    manual = spec.get("manual", {})
    invocation = finite_number(manual.get("invocation_rate"))
    acceptance = finite_number(manual.get("acceptance_rate"))
    if invocation is None and invoked:
        invocation = sum(invoked) / len(invoked)
    if acceptance is None and accepted:
        acceptance = sum(accepted) / len(accepted)
    return invocation, acceptance


def aggregate_system(spec: Mapping[str, Any], catalog_path: Path) -> dict[str, Any]:
    rows = judge_rows(spec, catalog_path)
    result: dict[str, Any] = {
        "key": spec["key"],
        "label": spec.get("label", spec["key"]),
        "group": spec.get("group", "main"),
        "n_seed_rows": len(rows),
    }
    for metric, _label in JUDGE_METRICS:
        result[metric], result[metric.replace("_mean", "_seed_std")] = mean_std(
            finite_number(row.get(metric)) for row in rows
        )

    efficiency = efficiency_values(spec, catalog_path)
    for key, values in efficiency.items():
        result[key], result[f"{key}_std"] = mean_std(values)
    invocation, acceptance = gate_rates(spec, catalog_path)
    result["invocation_rate"] = invocation
    result["acceptance_rate"] = acceptance
    return result


def markdown_table(rows: list[Mapping[str, Any]]) -> str:
    headers = [
        "System",
        *[label for _metric, label in JUDGE_METRICS],
        "Invoke (\\%)",
        "Accept (\\%)",
        "Latency (ms)",
        "Serve VRAM (GiB)",
        "Trainable (M)",
        "GPU h",
        "Seeds",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = [str(row["label"])]
        for metric, _label in JUDGE_METRICS:
            cells.append(format_value(row.get(metric), row.get(metric.replace("_mean", "_seed_std"))))
        invocation = row.get("invocation_rate")
        acceptance = row.get("acceptance_rate")
        cells.append("—" if invocation is None else f"{100.0 * invocation:.1f}")
        cells.append("—" if acceptance is None else f"{100.0 * acceptance:.1f}")
        cells.append(format_value(row.get("latency_ms"), row.get("latency_ms_std"), digits=1))
        cells.append(format_value(row.get("serve_vram_gib"), row.get("serve_vram_gib_std"), digits=2))
        cells.append(format_value(row.get("trainable_m"), row.get("trainable_m_std"), digits=2))
        cells.append(format_value(row.get("train_gpu_hours"), row.get("train_gpu_hours_std"), digits=2))
        cells.append(str(row.get("n_seed_rows", 0)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "key",
        "label",
        "n_seed_rows",
        *[metric for metric, _ in JUDGE_METRICS],
        *[metric.replace("_mean", "_seed_std") for metric, _ in JUDGE_METRICS],
        "invocation_rate",
        "acceptance_rate",
        "latency_ms",
        "latency_ms_std",
        "serve_vram_gib",
        "serve_vram_gib_std",
        "trainable_m",
        "trainable_m_std",
        "train_gpu_hours",
        "train_gpu_hours_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def template() -> dict[str, Any]:
    return {
        "systems": [
            {
                "key": "sft",
                "label": "SFT",
                "group": "main",
                "judge_csvs": ["outputs/analysis/exp295_comparison_by_system.csv"],
                "judge_systems": ["sft_qd_s42"],
                "efficiency_jsons": ["outputs/refinement/sft_qd_test.jsonl.metrics.json"],
                "train_manifests": [],
                "selective_outputs": [],
            },
            {
                "key": "dpo",
                "label": "DPO",
                "group": "main",
                "judge_csvs": ["outputs/analysis/exp295_comparison_by_system.csv"],
                "judge_systems": ["dpo_minimal_selected_s42", "dpo_minimal_selected_s43", "dpo_minimal_selected_s44"],
                "efficiency_jsons": [],
                "train_manifests": [],
                "selective_outputs": [],
            },
            {
                "key": "dpo_denoiser",
                "label": "DPO + selective denoiser",
                "group": "main",
                "judge_csvs": ["outputs/analysis/exp295_comparison_by_system.csv"],
                "judge_systems": [],
                "efficiency_jsons": [],
                "train_manifests": [],
                "selective_outputs": [],
            },
            {
                "key": "proposed",
                "label": "Proposed (SFT + selective denoiser)",
                "group": "main",
                "judge_csvs": ["outputs/analysis/exp295_comparison_by_system.csv"],
                "judge_systems": [],
                "efficiency_jsons": [],
                "train_manifests": [],
                "selective_outputs": [],
            },
        ]
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output_csv", type=Path)
    parser.add_argument("--output_md", type=Path)
    parser.add_argument("--group", default="main")
    parser.add_argument("--write_template", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_template:
        args.write_template.parent.mkdir(parents=True, exist_ok=True)
        with args.write_template.open("w", encoding="utf-8") as handle:
            json.dump(template(), handle, ensure_ascii=False, indent=2)
        print(f"wrote catalog template: {args.write_template}")
        return
    if not args.catalog or not args.output_csv or not args.output_md:
        raise ValueError("--catalog, --output_csv, and --output_md are required")

    payload = read_json(args.catalog)
    systems = payload.get("systems") if isinstance(payload, dict) else None
    if not isinstance(systems, list) or not systems:
        raise ValueError("catalog must contain a non-empty 'systems' list")
    rows = [aggregate_system(spec, args.catalog) for spec in systems]
    rows = [row for row in rows if row.get("group") == args.group]
    if not rows:
        raise ValueError(f"no systems belong to group={args.group!r}")

    write_csv(rows, args.output_csv)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown_table(rows), encoding="utf-8")
    print(markdown_table(rows))


if __name__ == "__main__":
    main()
