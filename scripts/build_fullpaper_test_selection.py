#!/usr/bin/env python3
"""Build the canonical full-paper TEST corruption selection without generating data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_fullpaper_production_selection import (
    SELECTION_SEED,
    attach_balanced_axes,
    choose_one_per_question,
)
from fullpaper_acl_pipeline import (
    DEFAULT_OUTPUT_DIR,
    normalize_text,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)


VERSION = "fullpaper-test-selection-v2-20261008"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "frozen_revision2_test_selection_v1_20261008"
DEFAULT_FROZEN = DEFAULT_OUTPUT_DIR / "production_accepted652_reuse_export_reviewed_v2_20261004"
DEFAULT_PILOT_SELECTIONS = (
    DEFAULT_OUTPUT_DIR / "gemma_pilot_selection_manifest.json",
    DEFAULT_OUTPUT_DIR / "gemma_pilot_100_v2_selection.json",
)


def load_pilot_test_ids(paths: list[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    ids: set[str] = set()
    provenance: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(f"Pilot selection does not contain rows: {path}")
        test_ids = {
            str(row["canonical_id"])
            for row in rows
            if row.get("split") == "test"
        }
        ids.update(test_ids)
        provenance.append(
            {"path": str(path.resolve()), "sha256": sha256_file(path), "test_ids": sorted(test_ids)}
        )
    return ids, provenance


def expand_pilot_test_question_exclusions(
    canonical: list[dict[str, Any]], pilot_test_ids: set[str]
) -> tuple[set[str], set[str]]:
    """Resolve pilot TEST IDs to normalized-question groups and every TEST ID in them."""

    canonical_by_id = {str(row["canonical_id"]): row for row in canonical}
    missing_ids = sorted(pilot_test_ids - set(canonical_by_id))
    if missing_ids:
        raise ValueError(f"Pilot TEST IDs are missing from canonical data: {missing_ids}")

    non_test_ids = sorted(
        canonical_id
        for canonical_id in pilot_test_ids
        if canonical_by_id[canonical_id].get("split") != "test"
    )
    if non_test_ids:
        raise ValueError(f"Pilot TEST IDs do not resolve to canonical TEST rows: {non_test_ids}")

    pilot_question_groups: set[str] = set()
    for canonical_id in pilot_test_ids:
        question_group = canonical_by_id[canonical_id].get("question_normalized_sha256")
        if not question_group:
            raise ValueError(f"Pilot TEST row has no normalized question hash: {canonical_id}")
        pilot_question_groups.add(str(question_group))

    excluded_test_ids = {
        str(row["canonical_id"])
        for row in canonical
        if row.get("split") == "test"
        and str(row.get("question_normalized_sha256")) in pilot_question_groups
    }
    return pilot_question_groups, excluded_test_ids


def unique_values(rows: list[dict[str, Any]], field: str) -> set[str]:
    return {str(row[field]) for row in rows if row.get(field) is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", default=str(DEFAULT_OUTPUT_DIR / "canonical_clean_qa.jsonl"))
    parser.add_argument("--clean-qc", default=str(DEFAULT_OUTPUT_DIR / "clean_target_qc/corpus_qc.jsonl"))
    parser.add_argument(
        "--holds",
        default=str(DEFAULT_OUTPUT_DIR / "development_training_shard_120_corrected_v2/held_out.jsonl"),
    )
    parser.add_argument("--frozen-dir", default=str(DEFAULT_FROZEN))
    parser.add_argument("--pilot-selection", action="append", default=[])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    canonical_path = Path(args.canonical).resolve()
    clean_qc_path = Path(args.clean_qc).resolve()
    holds_path = Path(args.holds).resolve()
    frozen_dir = Path(args.frozen_dir).resolve()
    pilot_paths = [Path(path).resolve() for path in args.pilot_selection] or [
        path.resolve() for path in DEFAULT_PILOT_SELECTIONS
    ]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    canonical = list(read_jsonl(canonical_path))
    qc_rows = list(read_jsonl(clean_qc_path))
    qc_by_id = {str(row["canonical_id"]): row for row in qc_rows}
    held_ids = {str(row["canonical_id"]) for row in read_jsonl(holds_path)}
    pilot_test_ids, pilot_provenance = load_pilot_test_ids(pilot_paths)
    pilot_test_question_groups, pilot_question_group_test_ids = (
        expand_pilot_test_question_exclusions(canonical, pilot_test_ids)
    )

    raw_eligible = [
        row
        for row in canonical
        if row.get("split") == "test"
        and (qc := qc_by_id.get(str(row["canonical_id"]))) is not None
        and qc.get("qc_ok") is True
        and qc.get("eligible") is True
        and qc.get("baseline_degraded_axes") == []
        and row["canonical_id"] not in held_ids
    ]
    excluded_pilot_eligible_ids = sorted(unique_values(raw_eligible, "canonical_id") & pilot_test_ids)
    excluded_pilot_question_group_eligible_ids = sorted(
        unique_values(raw_eligible, "canonical_id") & pilot_question_group_test_ids
    )
    selected_candidates = choose_one_per_question(
        canonical,
        qc_by_id,
        held_ids | pilot_question_group_test_ids,
        reusable_ids=set(),
        split="test",
    )
    assigned, assignment_stats = attach_balanced_axes(selected_candidates)
    for row in assigned:
        row["generation_objective"] = "response_pair"

    frozen_rows = [
        row
        for name in ("sft_train.jsonl", "sft_valid.jsonl")
        for row in read_jsonl(frozen_dir / name)
    ]
    canonical_by_id = {str(row["canonical_id"]): row for row in canonical}
    selected_canonical = [canonical_by_id[str(row["canonical_id"])] for row in selected_candidates]
    overlap = {
        field: len(unique_values(selected_candidates, field) & unique_values(frozen_rows, field))
        for field in (
            "canonical_id",
            "question_normalized_sha256",
            "duplicate_cluster_id",
            "source_group_id",
        )
    }
    overlap.update(
        {
            "question_exact_sha256": len(
                {sha256_text(str(row["question"])) for row in selected_canonical}
                & {sha256_text(str(row["question"])) for row in frozen_rows}
            ),
            "response_exact_sha256": len(
                {sha256_text(str(row["clean_response"])) for row in selected_canonical}
                & {sha256_text(str(row["safe_response"])) for row in frozen_rows}
            ),
            "response_normalized_sha256": len(
                {sha256_text(normalize_text(row["clean_response"])) for row in selected_canonical}
                & {sha256_text(normalize_text(row["safe_response"])) for row in frozen_rows}
            ),
        }
    )
    if any(overlap.values()):
        raise RuntimeError(f"TEST overlaps frozen TRAIN/VALID: {overlap}")
    if unique_values(selected_candidates, "canonical_id") & pilot_test_ids:
        raise RuntimeError("Pilot TEST ID survived exclusion")
    surviving_pilot_exposed_question_groups = sorted(
        unique_values(selected_candidates, "question_normalized_sha256")
        & pilot_test_question_groups
    )
    if surviving_pilot_exposed_question_groups:
        raise RuntimeError(
            "Pilot-exposed normalized TEST question survived exclusion: "
            f"{surviving_pilot_exposed_question_groups}"
        )
    if len(unique_values(selected_candidates, "question_normalized_sha256")) != len(selected_candidates):
        raise RuntimeError("TEST selection contains repeated normalized questions")
    ordered_ids = [str(row["canonical_id"]) for row in assigned]

    candidates_path = output / "test_candidates.jsonl"
    selection_path = output / "test_generation_selection.json"
    excluded_path = output / "excluded_development_pilot_test_ids.txt"
    expanded_excluded_path = output / "excluded_development_pilot_question_group_test_ids.txt"
    manifest_path = output / "manifest.json"
    write_jsonl(candidates_path, selected_candidates)
    write_json(
        selection_path,
        {"version": VERSION, "seed": SELECTION_SEED, "split": "test", "rows": assigned},
    )
    excluded_path.write_text("\n".join(sorted(pilot_test_ids)) + "\n", encoding="utf-8")
    expanded_excluded_path.write_text(
        "\n".join(sorted(pilot_question_group_test_ids)) + "\n", encoding="utf-8"
    )

    manifest = {
        "version": VERSION,
        "status": "complete_existing_qc_only_no_generation",
        "selection_seed": SELECTION_SEED,
        "selection_rule": (
            "Filter canonical TEST to qc_ok=true, eligible=true, baseline_degraded_axes=[] and not held; "
            "exclude every canonical TEST row whose normalized question matches any TEST ID used by "
            "the two recorded corruption-development pilots; group the remainder by "
            "question_normalized_sha256; choose the minimum stable hash of seed+canonical_id; attach "
            "the frozen balanced six-axis assignment and deterministic stage seeds."
        ),
        "source_sha256": {
            "canonical": sha256_file(canonical_path),
            "clean_qc": sha256_file(clean_qc_path),
            "holds": sha256_file(holds_path),
            "frozen_manifest": sha256_file(frozen_dir / "frozen_manifest.json"),
        },
        "pilot_exclusion_sources": pilot_provenance,
        "counts": {
            "clean_qc_eligible_candidate_rows_before_pilot_exclusion": len(raw_eligible),
            "clean_qc_eligible_unique_questions_before_pilot_exclusion": len(
                unique_values(raw_eligible, "question_normalized_sha256")
            ),
            "historical_pilot_test_ids_union": len(pilot_test_ids),
            "eligible_rows_removed_by_pilot_id": len(excluded_pilot_eligible_ids),
            "eligible_pilot_ids_removed": excluded_pilot_eligible_ids,
            "pilot_normalized_question_groups": len(pilot_test_question_groups),
            "canonical_test_ids_in_pilot_question_groups": len(pilot_question_group_test_ids),
            "eligible_rows_removed_by_pilot_question_group": len(
                excluded_pilot_question_group_eligible_ids
            ),
            "eligible_pilot_question_group_ids_removed": (
                excluded_pilot_question_group_eligible_ids
            ),
            "selected_generation_rows": len(assigned),
            "selected_unique_questions": len(unique_values(selected_candidates, "question_normalized_sha256")),
            "pilot_exposed_question_groups_retained_via_nonpilot_duplicate_id": len(
                surviving_pilot_exposed_question_groups
            ),
            "by_source": dict(sorted(Counter(row["source"] for row in selected_candidates).items())),
            "axis_count": dict(sorted(Counter(row["axis_count"] for row in assigned).items())),
        },
        "selection_id_sha256": {
            "ordered_newline": sha256_text("\n".join(ordered_ids) + "\n"),
            "sorted_newline": sha256_text("\n".join(sorted(ordered_ids)) + "\n"),
        },
        "pilot_exposed_question_groups_retained_via_nonpilot_duplicate_id": (
            surviving_pilot_exposed_question_groups
        ),
        "frozen_train_valid_overlap": overlap,
        "assignment_stats": assignment_stats,
        "outputs": {
            candidates_path.name: {"path": str(candidates_path), "sha256": sha256_file(candidates_path)},
            selection_path.name: {"path": str(selection_path), "sha256": sha256_file(selection_path)},
            excluded_path.name: {"path": str(excluded_path), "sha256": sha256_file(excluded_path)},
            expanded_excluded_path.name: {
                "path": str(expanded_excluded_path),
                "sha256": sha256_file(expanded_excluded_path),
            },
        },
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
