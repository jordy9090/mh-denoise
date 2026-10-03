#!/usr/bin/env python3
"""Freeze a deterministic source/axis-stratified bounded production selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from fullpaper_acl_pipeline import sha256_file, write_json


VERSION = "fullpaper-bounded-production-selection-v1"
SEED = 20260910
AXES = (
    "overall_quality", "empathy", "specificity", "factual_consistency",
    "medical_boundary", "toxicity_or_harm",
)


def stable_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{SEED}:{row['canonical_id']}".encode()).hexdigest()


def largest_remainder(counts: Counter[str], total: int) -> dict[str, int]:
    population = sum(counts.values())
    exact = {key: total * value / population for key, value in counts.items()}
    result = {key: math.floor(value) for key, value in exact.items()}
    remainder = total - sum(result.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:remainder]:
        result[key] += 1
    return result


def select(rows: list[dict[str, Any]], candidates: dict[str, dict[str, Any]], count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enriched = []
    for row in rows:
        candidate = candidates[row["canonical_id"]]
        enriched.append({**row, "source": candidate["source"], "source_component": candidate["source_component"]})
    if count > len(enriched):
        raise ValueError(f"Requested {count} from only {len(enriched)} eligible rows")

    source_available = Counter(f"{row['source']}:{row['source_component']}" for row in enriched)
    source_target = largest_remainder(source_available, count)
    axis_count_target = {1: count * 50 // 100, 2: count * 35 // 100}
    axis_count_target[3] = count - sum(axis_count_target.values())
    total_axis_labels = sum(key * value for key, value in axis_count_target.items())
    marginal_target = {axis: total_axis_labels // len(AXES) for axis in AXES}
    for axis in AXES[: total_axis_labels % len(AXES)]:
        marginal_target[axis] += 1

    remaining = sorted(enriched, key=stable_key)
    chosen: list[dict[str, Any]] = []
    source_used: Counter[str] = Counter()
    axis_count_used: Counter[int] = Counter()
    marginal_used: Counter[str] = Counter()
    while len(chosen) < count:
        eligible = [
            row for row in remaining
            if source_used[f"{row['source']}:{row['source_component']}"] < source_target[f"{row['source']}:{row['source_component']}"]
            and axis_count_used[row["axis_count"]] < axis_count_target[row["axis_count"]]
        ]
        if not eligible:
            raise RuntimeError("Unable to satisfy joint source and axis-count quotas")

        def score(row: dict[str, Any]) -> tuple[float, str]:
            gain = sum(max(marginal_target[axis] - marginal_used[axis], 0) for axis in row["intended_axes"])
            excess = sum(max(marginal_used[axis] + 1 - marginal_target[axis], 0) for axis in row["intended_axes"])
            return (-gain + excess * total_axis_labels, stable_key(row))

        picked = min(eligible, key=score)
        chosen.append(picked)
        remaining.remove(picked)
        source_used[f"{picked['source']}:{picked['source_component']}"] += 1
        axis_count_used[picked["axis_count"]] += 1
        marginal_used.update(picked["intended_axes"])

    return chosen, {
        "source_available": dict(sorted(source_available.items())),
        "source_target": dict(sorted(source_target.items())),
        "source_selected": dict(sorted(source_used.items())),
        "axis_count_target": {str(k): v for k, v in sorted(axis_count_target.items())},
        "axis_count_selected": {str(k): v for k, v in sorted(axis_count_used.items())},
        "marginal_target": marginal_target,
        "marginal_selected": {axis: marginal_used[axis] for axis in AXES},
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["rows"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--valid-count", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source = Path(args.source_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    candidates: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid"):
        path = source / f"{split}_candidates.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["selection_status"] == "requires_generation":
                    candidates[row["canonical_id"]] = row

    inputs = {
        split: source / f"{split}_generation_selection.json" for split in ("train", "valid")
    }
    selected: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, Any] = {}
    for split, count in (("train", args.train_count), ("valid", args.valid_count)):
        rows = load_rows(inputs[split])
        selected[split], stats[split] = select(rows, candidates, count)
        write_json(output / f"{split}_selection.json", {
            "version": VERSION, "seed": SEED, "split": split, "rows": selected[split],
        })

    combined = selected["valid"] + selected["train"]
    write_json(output / "valid_first_train_valid_selection.json", {
        "version": VERSION,
        "seed": SEED,
        "ordering": "all canonical VALID first, followed by TRAIN",
        "rows": combined,
    })
    fields = ("canonical_id", "question_normalized_sha256", "duplicate_cluster_id", "source_group_id")
    overlap = {
        field: sorted({row[field] for row in selected["train"]} & {row[field] for row in selected["valid"]})
        for field in fields
    }
    if any(overlap.values()):
        raise RuntimeError(f"TRAIN/VALID overlap: {overlap}")
    manifest = {
        "version": VERSION,
        "status": "complete_no_api",
        "seed": SEED,
        "selection_rule": "Exact proportional source-component quotas and exact 50/35/15 axis-count quotas; greedily minimize six-axis marginal deficits with stable seed+canonical_id tie breaking.",
        "corrected_57_policy": "already excluded by parent requires_generation selection; reuse later without regeneration",
        "ordering": "VALID first to preserve canonical VALID coverage if the shared budget is exhausted",
        "counts": {split: len(rows) for split, rows in selected.items()},
        "stratification": stats,
        "train_valid_overlap_counts": {field: len(values) for field, values in overlap.items()},
        "source_files": {path.name: {"sha256": sha256_file(path)} for path in inputs.values()},
        "outputs": {
            path.name: {"sha256": sha256_file(path)}
            for path in sorted(output.glob("*.json")) if path.name != "manifest.json"
        },
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
