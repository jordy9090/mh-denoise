#!/usr/bin/env python3
"""Build deterministic TRAIN/VALID production selections from existing QC only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from build_development_corruption_shard import BUILD_SEED
from fullpaper_acl_pipeline import DEFAULT_OUTPUT_DIR, read_jsonl, sha256_file, stable_random_key, write_json, write_jsonl
from run_gemma_corruption_pilot_v2 import balanced_specs
from run_paired_generator_diagnostic import deterministic_generation_seed


VERSION = "fullpaper-production-selection-v1"
SELECTION_SEED = BUILD_SEED
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "production_selection_v1_20260909"
DEFAULT_CORRECTED = DEFAULT_OUTPUT_DIR / "development_training_shard_120_corrected_v2"


def choose_one_per_question(
    canonical: list[dict[str, Any]],
    qc_by_id: dict[str, dict[str, Any]],
    held_ids: set[str],
    reusable_ids: set[str],
    split: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in canonical:
        qc = qc_by_id.get(row["canonical_id"])
        if (
            row.get("split") != split
            or row["canonical_id"] in held_ids
            or not qc
            or qc.get("qc_ok") is not True
            or qc.get("eligible") is not True
            or qc.get("baseline_degraded_axes") != []
        ):
            continue
        question_group = row.get("question_normalized_sha256")
        if not question_group:
            raise RuntimeError(f"Missing normalized question ID: {row['canonical_id']}")
        groups.setdefault(question_group, []).append(row)

    selected = []
    for question_group, rows in groups.items():
        rows.sort(
            key=lambda row: (
                0 if row["canonical_id"] in reusable_ids else 1,
                stable_random_key(SELECTION_SEED, row["canonical_id"]),
            )
        )
        chosen = rows[0]
        selected.append(
            {
                "canonical_id": chosen["canonical_id"],
                "split": split,
                "source": chosen["source"],
                "source_component": chosen["source_component"],
                "question_normalized_sha256": question_group,
                "duplicate_cluster_id": chosen["duplicate_cluster_id"],
                "source_group_id": chosen["source_group_id"],
                "selection_status": (
                    "reuse_existing_corrected_pair"
                    if chosen["canonical_id"] in reusable_ids
                    else "requires_generation"
                ),
                "candidate_response_count_for_question": len(rows),
            }
        )
    selected.sort(key=lambda row: stable_random_key(SELECTION_SEED + 1, row["canonical_id"]))
    return selected


def attach_balanced_axes(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs, stats = balanced_specs(len(rows), SELECTION_SEED)
    ordered_specs = sorted(
        enumerate(specs),
        key=lambda item: stable_random_key(SELECTION_SEED + 2, f"{item[0]}:{'+'.join(item[1])}"),
    )
    result = []
    for row, (_, combo) in zip(rows, ordered_specs, strict=True):
        axes = sorted(
            combo,
            key=lambda axis: stable_random_key(SELECTION_SEED, f"{row['canonical_id']}:{axis}"),
        )
        result.append(
            {
                "canonical_id": row["canonical_id"],
                "split": row["split"],
                "question_normalized_sha256": row["question_normalized_sha256"],
                "duplicate_cluster_id": row["duplicate_cluster_id"],
                "source_group_id": row["source_group_id"],
                "intended_axes": axes,
                "axis_count": len(axes),
                "generation_seeds": [
                    {
                        "stage_index": stage,
                        "axis": axis,
                        "attempt_1": deterministic_generation_seed(
                            BUILD_SEED, row["canonical_id"], stage, 1
                        ),
                        "attempt_2": deterministic_generation_seed(
                            BUILD_SEED, row["canonical_id"], stage, 2
                        ),
                    }
                    for stage, axis in enumerate(axes, 1)
                ],
            }
        )
    return result, stats


def split_source_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "unique_questions": len(rows),
        "by_source": dict(sorted(Counter(row["source"] for row in rows).items())),
        "by_source_component": dict(
            sorted(Counter(f"{row['source']}:{row['source_component']}" for row in rows).items())
        ),
        "reuse_existing_corrected_pair": sum(
            row["selection_status"] == "reuse_existing_corrected_pair" for row in rows
        ),
        "requires_generation": sum(row["selection_status"] == "requires_generation" for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", default=str(DEFAULT_OUTPUT_DIR / "canonical_clean_qa.jsonl"))
    parser.add_argument("--clean-qc", default=str(DEFAULT_OUTPUT_DIR / "clean_target_qc/corpus_qc.jsonl"))
    parser.add_argument("--holds", default=str(DEFAULT_CORRECTED / "held_out.jsonl"))
    parser.add_argument("--corrected-sft", default=str(DEFAULT_CORRECTED / "train_sft.jsonl"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--dev120-manifest",
        default=str(DEFAULT_OUTPUT_DIR / "development_training_shard_120/build_manifest.json"),
    )
    parser.add_argument("--input-usd-per-million", type=float, default=2.0)
    parser.add_argument("--output-usd-per-million", type=float, default=8.0)
    parser.add_argument("--recommended-budget-margin", type=float, default=1.20)
    args = parser.parse_args()

    paths = {
        "canonical": Path(args.canonical).resolve(),
        "clean_qc": Path(args.clean_qc).resolve(),
        "holds": Path(args.holds).resolve(),
        "corrected_sft": Path(args.corrected_sft).resolve(),
        "dev120_manifest": Path(args.dev120_manifest).resolve(),
    }
    canonical = list(read_jsonl(paths["canonical"]))
    qc_rows = list(read_jsonl(paths["clean_qc"]))
    qc_by_id = {row["canonical_id"]: row for row in qc_rows}
    held_ids = {row["canonical_id"] for row in read_jsonl(paths["holds"])}
    reusable_ids = {row["metadata"]["canonical_id"] for row in read_jsonl(paths["corrected_sft"])}
    if held_ids & reusable_ids:
        raise RuntimeError("Held IDs overlap the reusable corrected pair export")

    selected_by_split = {
        split: choose_one_per_question(canonical, qc_by_id, held_ids, reusable_ids, split)
        for split in ("train", "valid")
    }
    for field in ("canonical_id", "question_normalized_sha256", "duplicate_cluster_id", "source_group_id"):
        left = {row[field] for row in selected_by_split["train"]}
        right = {row[field] for row in selected_by_split["valid"]}
        overlap = left & right
        if overlap:
            raise RuntimeError(f"Canonical TRAIN/VALID overlap for {field}: {sorted(overlap)[:5]}")

    generation_rows = [
        row
        for split in ("train", "valid")
        for row in selected_by_split[split]
        if row["selection_status"] == "requires_generation"
    ]
    generation_rows.sort(key=lambda row: stable_random_key(SELECTION_SEED + 3, row["canonical_id"]))
    assigned, assignment_stats = attach_balanced_axes(generation_rows)
    assigned_by_split = {
        split: [row for row in assigned if row["split"] == split] for split in ("train", "valid")
    }

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        write_jsonl(output / f"{split}_candidates.jsonl", selected_by_split[split])
        write_json(
            output / f"{split}_generation_selection.json",
            {
                "version": VERSION,
                "seed": SELECTION_SEED,
                "split": split,
                "rows": assigned_by_split[split],
            },
        )
    write_json(
        output / "train_valid_generation_selection.json",
        {"version": VERSION, "seed": SELECTION_SEED, "splits": ["train", "valid"], "rows": assigned},
    )

    dev120 = json.loads(paths["dev120_manifest"].read_text(encoding="utf-8"))
    usage = dev120["usage"]
    denominator = 120
    historical = {
        "input_rows": denominator,
        "pre_audit_accepted_pairs": 70,
        "corrected_final_pairs": len(reusable_ids),
        "gpt_requests": usage["judge_calls"],
        "gpt_input_tokens": usage["judge_prompt_tokens"],
        "gpt_output_tokens": usage["judge_completion_tokens"],
    }
    historical["recorded_cost_usd"] = (
        historical["gpt_input_tokens"] * args.input_usd_per_million
        + historical["gpt_output_tokens"] * args.output_usd_per_million
    ) / 1_000_000
    per_input = {
        "gpt_requests": historical["gpt_requests"] / denominator,
        "gpt_input_tokens": historical["gpt_input_tokens"] / denominator,
        "gpt_output_tokens": historical["gpt_output_tokens"] / denominator,
        "recorded_cost_usd": historical["recorded_cost_usd"] / denominator,
    }

    def project(row_count: int) -> dict[str, float | int]:
        calls = row_count * per_input["gpt_requests"]
        input_tokens = row_count * per_input["gpt_input_tokens"]
        output_tokens = row_count * per_input["gpt_output_tokens"]
        return {
            "generation_inputs": row_count,
            "projected_gpt_requests": calls,
            "projected_gpt_input_tokens": input_tokens,
            "projected_gpt_output_tokens": output_tokens,
            "projected_cost_usd": (
                input_tokens * args.input_usd_per_million
                + output_tokens * args.output_usd_per_million
            ) / 1_000_000,
        }

    projections = {
        split: project(len(assigned_by_split[split])) for split in ("train", "valid")
    }
    projections["train_plus_valid"] = project(len(assigned))
    total_projection = projections["train_plus_valid"]
    cost_projection = {
        "basis": "linear projection from existing dev120 GPT-4.1 judge usage; local Qwen tokens excluded",
        "warning": "The shared budget guard changes the spending ceiling, not request count, token use, or unit price.",
        "pricing_usd_per_million_tokens": {
            "input": args.input_usd_per_million,
            "output": args.output_usd_per_million,
        },
        "historical_dev120": historical,
        "per_dev120_input": per_input,
        "projection": projections,
        "recommended_operational_caps_at_margin": {
            "margin": args.recommended_budget_margin,
            "max_api_requests": math.ceil(
                float(total_projection["projected_gpt_requests"]) * args.recommended_budget_margin
            ),
            "max_api_usd": math.ceil(
                float(total_projection["projected_cost_usd"]) * args.recommended_budget_margin
            ),
        },
    }
    write_json(output / "cost_projection.json", cost_projection)

    manifest = {
        "version": VERSION,
        "status": "complete_existing_qc_only_no_api",
        "selection_seed": SELECTION_SEED,
        "selection_rule": (
            "Filter to qc_ok=true, eligible=true, baseline_degraded_axes=[] and not held; "
            "group by canonical split and question_normalized_sha256; prefer an existing corrected "
            "pair canonical_id, otherwise choose the minimum stable hash of seed+canonical_id."
        ),
        "reusable_pair_policy": "57 corrected TRAIN pairs are retained for training and omitted from paid regeneration",
        "source_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "counts": {split: split_source_counts(rows) for split, rows in selected_by_split.items()},
        "generation_total": len(assigned),
        "assignment_stats": assignment_stats,
        "cost_projection": cost_projection,
        "split_overlap_checks": {
            "canonical_id": 0,
            "question_normalized_sha256": 0,
            "duplicate_cluster_id": 0,
            "source_group_id": 0,
        },
        "outputs": {
            path.name: {"sha256": sha256_file(path)}
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
