#!/usr/bin/env python3
"""Build a read-only-derived data inventory and a bounded Proposed smoke input.

This script never calls a model or API.  It verifies the frozen artifact hashes,
joins canonical rows to completed clean-target QC, excludes the corrected-dev120
hold list, and adapts up to eight retained dev examples to the legacy Proposed
inference contract without changing any source or candidate text.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/fullpaper_acl_pipeline"
CORRECTED = DATA / "development_training_shard_120_corrected_v2"
OUTPUT = DATA / "main_experiment_readiness_20260909"

INPUTS = {
    "canonical": DATA / "canonical_clean_qa.jsonl",
    "split_manifest": DATA / "split_manifest.json",
    "clean_qc": DATA / "clean_target_qc/corpus_qc.jsonl",
    "corrected_manifest": CORRECTED / "repair_manifest.json",
    "corrected_sft": CORRECTED / "train_sft.jsonl",
    "corrected_dpo": CORRECTED / "train_dpo.jsonl",
    "verified_spans": CORRECTED / "source_verified_spans.jsonl",
    "held": CORRECTED / "held_out.jsonl",
}

EXPECTED_SHA256 = {
    "canonical": "0394c6665eb47d59192cea418f66f7c30f7427485257cfd75c1931c63258f9ef",
    "split_manifest": "9d40ab371b71c040042a506529f58c576f423a4e199ffb3ec1042d3f23291392",
    "clean_qc": "c2ead45bee318a429e83b5e61e144d863b570b55657efd3392a8efb57a0f52b1",
    "corrected_manifest": "ae2f2c256c8fd2284f34d46aff83359c81ac5d2b4d2f55208572afeaae58406f",
    "corrected_sft": "3337394155a0760170c21c84e8d5ba4ae179a12643e0b65aca6aa5d9f3aff574",
    "corrected_dpo": "7fe74cb9a9b157b2063ffa3634a54d89c52f75142d14011294697863c911703d",
    "verified_spans": "e7e62818e589d71f5e1318109d660d4bdbe90a3565a29c8a927b36c13bb04eed",
    "held": "e8d42b75aba75126cd297a860ce76d002c93f7fc25289178a5efd5f40173b647",
}

AXES = [
    "overall_quality",
    "empathy",
    "specificity",
    "factual_consistency",
    "medical_boundary",
    "toxicity_or_harm",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def count_by(rows: list[dict], field: str) -> dict[str, int]:
    return dict(sorted(Counter(row[field] for row in rows).items()))


def unique_questions_by(rows: list[dict], field: str) -> dict[str, int]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped[row[field]].add(row["question_normalized_sha256"])
    return {key: len(value) for key, value in sorted(grouped.items())}


def main() -> None:
    actual_hashes = {name: sha256(path) for name, path in INPUTS.items()}
    mismatches = {
        name: {"expected": EXPECTED_SHA256[name], "actual": actual}
        for name, actual in actual_hashes.items()
        if actual != EXPECTED_SHA256[name]
    }
    if mismatches:
        raise RuntimeError(f"Frozen input hash mismatch: {json.dumps(mismatches, indent=2)}")

    canonical_rows = read_jsonl(INPUTS["canonical"])
    canonical = {row["canonical_id"]: row for row in canonical_rows}
    if len(canonical) != len(canonical_rows):
        raise RuntimeError("Duplicate canonical_id in canonical dataset")

    qc_rows = read_jsonl(INPUTS["clean_qc"])
    qc = {row["canonical_id"]: row for row in qc_rows}
    if len(qc) != len(qc_rows):
        raise RuntimeError("Duplicate canonical_id in clean-target QC checkpoint")

    missing = sorted(set(qc) - set(canonical))
    if missing:
        raise RuntimeError(f"QC rows missing from canonical data: {missing[:5]}")

    eligible_train = [
        canonical[canonical_id]
        for canonical_id, result in qc.items()
        if result.get("qc_ok") is True
        and result.get("baseline_degraded_axes") == []
        and canonical[canonical_id].get("split") == "train"
    ]
    held_rows = read_jsonl(INPUTS["held"])
    held_ids = {row["canonical_id"] for row in held_rows}
    eligible_after_holds = [row for row in eligible_train if row["canonical_id"] not in held_ids]

    corrected_sft = read_jsonl(INPUTS["corrected_sft"])
    corrected_dpo = read_jsonl(INPUTS["corrected_dpo"])
    if len(corrected_sft) != 57 or len(corrected_dpo) != 57:
        raise RuntimeError("Corrected dev120 export no longer contains exactly 57 SFT/DPO rows")
    corrected_ids = {row["metadata"]["canonical_id"] for row in corrected_sft}
    if corrected_ids & held_ids:
        raise RuntimeError("Held IDs leaked into corrected SFT export")
    if corrected_ids != {row["metadata"]["canonical_id"] for row in corrected_dpo}:
        raise RuntimeError("Corrected SFT/DPO IDs differ")
    if any(canonical[row_id]["split"] != "train" for row_id in corrected_ids):
        raise RuntimeError("VALID/TEST row found in corrected training export")
    if any(
        row_id not in qc
        or qc[row_id].get("qc_ok") is not True
        or qc[row_id].get("baseline_degraded_axes") != []
        for row_id in corrected_ids
    ):
        raise RuntimeError("Corrected training export contains a non-clean-QC target")

    candidate_risk_spans: dict[str, list[dict]] = defaultdict(list)
    realized_by_id = {
        row["metadata"]["canonical_id"]: set(row["metadata"].get("realized_axes", []))
        for row in corrected_sft
    }
    for span in read_jsonl(INPUTS["verified_spans"]):
        row_id = span["canonical_id"]
        if span["side"] == "candidate" and span["axis"] in realized_by_id.get(row_id, set()):
            candidate_risk_spans[row_id].append(span)
    proposed_span_ids = set(candidate_risk_spans)

    # Deterministic coverage-first choice: one row per intended axis, then fill
    # by canonical_id.  Every selected row has exact candidate-side risk spans.
    usable_dev = [
        row for row in corrected_sft
        if row["metadata"]["canonical_id"] in proposed_span_ids
    ]
    usable_dev.sort(key=lambda row: row["metadata"]["canonical_id"])
    selected: list[dict] = []
    selected_ids: set[str] = set()
    for axis in AXES:
        match = next(
            (
                row for row in usable_dev
                if axis in row["metadata"].get("intended_axes", [])
                and row["metadata"]["canonical_id"] not in selected_ids
            ),
            None,
        )
        if match is not None:
            selected.append(match)
            selected_ids.add(match["metadata"]["canonical_id"])
    for row in usable_dev:
        if len(selected) >= 8:
            break
        row_id = row["metadata"]["canonical_id"]
        if row_id not in selected_ids:
            selected.append(row)
            selected_ids.add(row_id)

    smoke_rows = []
    for row in selected[:8]:
        row_id = row["metadata"]["canonical_id"]
        source = canonical[row_id]
        adapted = {
            "id": row_id,
            "canonical_id": row_id,
            "question": row["input"]["question"],
            "unsafe_response": row["input"]["corrupted_response"],
            "safe_response": row["target"],
            "source": source["source"],
            "source_component": source["source_component"],
            "split": source["split"],
            "intended_axes": row["metadata"].get("intended_axes", []),
            "realized_axes": row["metadata"].get("realized_axes", []),
            "source_verified_candidate_risk_spans": candidate_risk_spans[row_id],
            "derived_from": str(INPUTS["corrected_sft"]),
        }
        # Exact text preservation is the point of this adapter.
        assert adapted["question"] == row["input"]["question"]
        assert adapted["unsafe_response"] == row["input"]["corrupted_response"]
        assert adapted["safe_response"] == row["target"]
        smoke_rows.append(adapted)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    smoke_path = OUTPUT / "proposed_smoke_input_8.jsonl"
    write_jsonl(smoke_path, smoke_rows)

    split_manifest = json.loads(INPUTS["split_manifest"].read_text(encoding="utf-8"))
    manifest = {
        "status": "derived_inventory_and_smoke_input_ready",
        "policy": "qc_ok == true AND baseline_degraded_axes == [] AND split == train",
        "frozen_input_sha256": actual_hashes,
        "split_assertions": split_manifest["assertions"],
        "clean_qc_checkpoint": {
            "rows": len(qc_rows),
            "unique_canonical_ids": len(qc),
            "train_rows_with_valid_qc_result": sum(
                result.get("qc_ok") is True and canonical[row_id]["split"] == "train"
                for row_id, result in qc.items()
            ),
        },
        "eligible_train_before_corrected_holds": {
            "response_rows": len(eligible_train),
            "normalized_unique_questions": len({row["question_normalized_sha256"] for row in eligible_train}),
            "duplicate_clusters": len({row["duplicate_cluster_id"] for row in eligible_train}),
            "source_response_rows": count_by(eligible_train, "source"),
            "source_component_response_rows": count_by(eligible_train, "source_component"),
        },
        "held_exclusion": {
            "known_held_ids": len(held_ids),
            "intersecting_eligible_train_ids": len(held_ids & {row["canonical_id"] for row in eligible_train}),
        },
        "usable_train_after_corrected_holds": {
            "response_rows": len(eligible_after_holds),
            "normalized_unique_questions_one_answer_each": len(
                {row["question_normalized_sha256"] for row in eligible_after_holds}
            ),
            "duplicate_clusters": len({row["duplicate_cluster_id"] for row in eligible_after_holds}),
            "source_response_rows": count_by(eligible_after_holds, "source"),
            "source_component_response_rows": count_by(eligible_after_holds, "source_component"),
            "source_unique_questions": unique_questions_by(eligible_after_holds, "source"),
            "source_component_unique_questions": unique_questions_by(eligible_after_holds, "source_component"),
            "valid_or_test_rows": sum(row["split"] != "train" for row in eligible_after_holds),
        },
        "corrected_dev120": {
            "sft_rows": len(corrected_sft),
            "dpo_rows": len(corrected_dpo),
            "clean_qc_eligible_train_rows": sum(
                row_id in qc
                and qc[row_id].get("qc_ok") is True
                and qc[row_id].get("baseline_degraded_axes") == []
                and canonical[row_id]["split"] == "train"
                for row_id in corrected_ids
            ),
            "rows_with_exact_candidate_span_on_realized_axis": len(proposed_span_ids),
            "rows_without_exact_candidate_span_on_realized_axis": sorted(corrected_ids - proposed_span_ids),
            "candidate_span_count_on_realized_axes": sum(map(len, candidate_risk_spans.values())),
            "rows_with_any_source_verified_candidate_span": len(
                {
                    span["canonical_id"]
                    for span in read_jsonl(INPUTS["verified_spans"])
                    if span["side"] == "candidate"
                }
            ),
        },
        "proposed_smoke_input": {
            "path": str(smoke_path),
            "rows": len(smoke_rows),
            "canonical_ids": [row["canonical_id"] for row in smoke_rows],
            "intended_axis_coverage": sorted(
                {axis for row in smoke_rows for axis in row["intended_axes"]}
            ),
            "sha256": sha256(smoke_path),
        },
    }
    manifest_path = OUTPUT / "data_readiness_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
