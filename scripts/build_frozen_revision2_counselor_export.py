#!/usr/bin/env python3
"""Export 15 TEST questions with three full-text responses under blind A/B/C labels.

The counselor-facing JSONL contains no canonical IDs, model/method fields, gold
responses, or internal scores.  A separate internal JSONL preserves the exact
method mapping and source file/physical-line provenance.  No model or API is
invoked by this utility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_frozen_revision2_test_partial_export import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl_with_locations,
    sha256_file,
    sha256_text,
)


VERSION = "frozen-revision2-counselor-blind-export-v1-20261008"
DEFAULT_PARTIAL_DIR = Path(
    "data/fullpaper_acl_pipeline/frozen_revision2_test_partial15_v1_20261008"
)
DEFAULT_RUN_DIR = Path(
    "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910"
)
DEFAULT_GENERATION_DIR = DEFAULT_RUN_DIR / "gemma/generation_1536_test_partial15_v1_20261008"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "counselor_test_partial15_v1_20261008"
METHOD_SPECS = {
    "SFT": {"response_field": "sft_response"},
    "DPO": {"response_field": "sft_response"},
    "Mask OFF": {"response_field": "final_response"},
}


def load_indexed_source(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    indexed: dict[str, dict[str, Any]] = {}
    lines: dict[str, int] = {}
    for row, line_number in read_jsonl_with_locations(path):
        canonical_id = str(row.get("canonical_id") or row.get("id") or "")
        if not canonical_id:
            raise ValueError(f"Output row has no canonical_id/id: {path}:{line_number}")
        if canonical_id in indexed:
            raise ValueError(f"Duplicate output ID {canonical_id}: {path}")
        indexed[canonical_id] = row
        lines[canonical_id] = line_number
    return indexed, lines


def balanced_blind_orders(count: int, seed: int) -> list[list[str]]:
    """Return content-independent, position-balanced method orders.

    Each consecutive three-question block uses rotations of a deterministically
    hash-permuted base order.  For 15 questions, every method appears exactly
    five times in A, B, and C.
    """

    methods = list(METHOD_SPECS)
    orders: list[list[str]] = []
    for block_start in range(0, count, len(methods)):
        block_number = block_start // len(methods)
        base = sorted(
            methods,
            key=lambda method: sha256_text(f"{seed}\0{block_number}\0{method}"),
        )
        for offset in range(min(len(methods), count - block_start)):
            orders.append(base[offset:] + base[:offset])
    return orders


def _existing_export_is_valid(
    manifest: dict[str, Any], source_hashes: dict[str, str], partial_manifest_hash: str
) -> bool:
    if manifest.get("status") != "complete_counselor_blinded_15x3_export":
        return False
    if manifest.get("inputs", {}).get("partial_manifest", {}).get("sha256") != partial_manifest_hash:
        return False
    recorded = {
        name: record.get("sha256")
        for name, record in manifest.get("inputs", {}).get("model_outputs", {}).items()
    }
    if recorded != source_hashes:
        return False
    for record in manifest.get("outputs", {}).values():
        path = Path(str(record.get("path") or ""))
        if not path.exists() or sha256_file(path) != record.get("sha256"):
            return False
    return True


def build_counselor_export(
    partial_dir: Path,
    sft_path: Path,
    dpo_path: Path,
    mask_off_path: Path,
    output_dir: Path,
    *,
    blind_seed: int = 20260910,
) -> dict[str, Any]:
    partial_dir = partial_dir.resolve()
    source_paths = {
        "SFT": sft_path.resolve(),
        "DPO": dpo_path.resolve(),
        "Mask OFF": mask_off_path.resolve(),
    }
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    partial_manifest_path = partial_dir / "manifest.json"
    partial_manifest = json.loads(partial_manifest_path.read_text(encoding="utf-8"))
    if partial_manifest.get("status") != "complete_frozen_partial_test_export":
        raise RuntimeError("Partial TEST export is not complete/frozen")
    partial_selection_path = partial_dir / "partial_selection.json"
    partial_selection = json.loads(partial_selection_path.read_text(encoding="utf-8"))
    selection_rows = partial_selection.get("rows")
    if not isinstance(selection_rows, list) or len(selection_rows) != 15:
        raise RuntimeError("Counselor export requires exactly 15 frozen questions")
    selected_ids = [str(row.get("canonical_id") or "") for row in selection_rows]
    if "" in selected_ids or len(set(selected_ids)) != 15:
        raise RuntimeError("Partial selection has invalid or duplicate IDs")
    if [row.get("partial_selection_rank") for row in selection_rows] != list(range(1, 16)):
        raise RuntimeError("Partial selection ranks are not the frozen 1..15 order")

    pair_path = partial_dir / "test_pairs.jsonl"
    pair_index, _ = load_indexed_source(pair_path)
    if set(pair_index) != set(selected_ids):
        raise RuntimeError("Partial TEST pair membership differs from partial selection")

    source_hashes = {method: sha256_file(path) for method, path in source_paths.items()}
    partial_manifest_hash = sha256_file(partial_manifest_path)
    final_manifest_path = output_dir / "manifest.json"
    if final_manifest_path.exists():
        existing = json.loads(final_manifest_path.read_text(encoding="utf-8"))
        if _existing_export_is_valid(existing, source_hashes, partial_manifest_hash):
            return existing
        raise RuntimeError("Existing counselor export differs from the requested immutable inputs")

    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    source_lines: dict[str, dict[str, int]] = {}
    for method, path in source_paths.items():
        indexed[method], source_lines[method] = load_indexed_source(path)
        if set(indexed[method]) != set(selected_ids):
            missing = set(selected_ids) - set(indexed[method])
            extra = set(indexed[method]) - set(selected_ids)
            raise RuntimeError(
                f"{method} output membership mismatch: missing={sorted(missing)} extra={sorted(extra)}"
            )

    blind_orders = balanced_blind_orders(15, blind_seed)
    blind_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    label_counts: Counter[tuple[str, str]] = Counter()
    exact_match_counts: Counter[str] = Counter()
    labels = ("A", "B", "C")
    for zero_index, (canonical_id, order) in enumerate(
        zip(selected_ids, blind_orders, strict=True)
    ):
        case_id = f"TEST-{zero_index + 1:02d}"
        pair = pair_index[canonical_id]
        question = pair.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Partial pair has an empty question: {canonical_id}")
        responses: list[dict[str, str]] = []
        method_texts: dict[str, str] = {}
        for label, method in zip(labels, order, strict=True):
            source_row = indexed[method][canonical_id]
            if source_row.get("question") != question:
                raise RuntimeError(f"{method} question differs from frozen partial input: {canonical_id}")
            for field in ("unsafe_response", "safe_response"):
                if field in source_row and source_row.get(field) != pair.get(field):
                    raise RuntimeError(
                        f"{method} {field} differs from frozen partial input: {canonical_id}"
                    )
            response_field = str(METHOD_SPECS[method]["response_field"])
            response = source_row.get(response_field)
            if not isinstance(response, str) or not response.strip():
                raise ValueError(
                    f"{method} output has empty {response_field}: {canonical_id}"
                )
            response_id = f"{case_id}-{label}"
            responses.append(
                {"response_id": response_id, "response_label": label, "response_text": response}
            )
            response_hash = sha256_text(response)
            mapping_rows.append(
                {
                    "case_id": case_id,
                    "response_id": response_id,
                    "response_label": label,
                    "method": method,
                    "canonical_id": canonical_id,
                    "partial_selection_rank": zero_index + 1,
                    "frozen_selection_rank": selection_rows[zero_index].get(
                        "frozen_selection_rank"
                    ),
                    "question_normalized_sha256": selection_rows[zero_index].get(
                        "question_normalized_sha256"
                    ),
                    "question_sha256": sha256_text(question),
                    "response_sha256": response_hash,
                    "source": {
                        "path": str(source_paths[method]),
                        "line": source_lines[method][canonical_id],
                        "file_sha256": source_hashes[method],
                        "response_field": response_field,
                        "source_row_id": str(
                            source_row.get("canonical_id") or source_row.get("id")
                        ),
                    },
                }
            )
            method_texts[method] = response
            label_counts[(method, label)] += 1
        if method_texts["SFT"] == method_texts["DPO"]:
            exact_match_counts["sft_equals_dpo"] += 1
        if method_texts["SFT"] == method_texts["Mask OFF"]:
            exact_match_counts["sft_equals_mask_off"] += 1
        if method_texts["DPO"] == method_texts["Mask OFF"]:
            exact_match_counts["dpo_equals_mask_off"] += 1
        blind_rows.append(
            {
                "case_id": case_id,
                "question_number": zero_index + 1,
                "question": question,
                "responses": responses,
            }
        )

    blind_path = output_dir / "counselor_blinded_15x3.jsonl"
    mapping_path = output_dir / "internal_model_mapping.jsonl"
    atomic_write_jsonl(blind_path, blind_rows)
    atomic_write_jsonl(mapping_path, mapping_rows)

    mapping_by_response_id = {row["response_id"]: row for row in mapping_rows}
    blind_schema_keys = {"case_id", "question_number", "question", "responses"}
    response_schema_keys = {"response_id", "response_label", "response_text"}
    checks = {
        "partial_manifest_complete": partial_manifest.get("status")
        == "complete_frozen_partial_test_export",
        "exact_15_questions": len(blind_rows) == 15,
        "exact_45_responses": len(mapping_rows) == 45
        and sum(len(row["responses"]) for row in blind_rows) == 45,
        "source_id_sets_match_partial": all(
            set(indexed[method]) == set(selected_ids) for method in METHOD_SPECS
        ),
        "questions_match_partial_exactly": all(
            row["question"] == pair_index[selected_ids[index]]["question"]
            for index, row in enumerate(blind_rows)
        ),
        "all_responses_nonempty": all(
            response["response_text"].strip()
            for row in blind_rows
            for response in row["responses"]
        ),
        "response_text_hashes_match_mapping": all(
            sha256_text(response["response_text"])
            == mapping_by_response_id[response["response_id"]]["response_sha256"]
            for row in blind_rows
            for response in row["responses"]
        ),
        # This checks identity-bearing schema, not ordinary words that may
        # coincidentally occur in an unmodified question/response (e.g. "mask").
        "blind_file_contains_no_model_names": all(
            set(row) == blind_schema_keys
            and all(set(response) == response_schema_keys for response in row["responses"])
            for row in blind_rows
        ),
        "balanced_method_positions": all(
            label_counts[(method, label)] == 5
            for method in METHOD_SPECS
            for label in labels
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)

    input_records: dict[str, Any] = {}
    for method, path in source_paths.items():
        record: dict[str, Any] = {
            "path": str(path),
            "sha256": source_hashes[method],
            "rows": len(indexed[method]),
            "response_field": METHOD_SPECS[method]["response_field"],
        }
        adjacent_manifest = path.with_suffix(".manifest.json")
        if adjacent_manifest.exists():
            record["generation_manifest"] = {
                "path": str(adjacent_manifest),
                "sha256": sha256_file(adjacent_manifest),
            }
        input_records[method] = record
    manifest = {
        "version": VERSION,
        "status": "complete_counselor_blinded_15x3_export",
        "purpose": "counselor_review_before_automatic_judging",
        "counts": {"questions": 15, "responses": 45, "responses_per_question": 3},
        "blinding": {
            "seed": blind_seed,
            "method": "content-independent hash-permuted three-row rotating blocks",
            "labels": list(labels),
            "method_position_counts_internal": {
                method: {label: label_counts[(method, label)] for label in labels}
                for method in METHOD_SPECS
            },
        },
        "exact_response_match_counts_internal": dict(sorted(exact_match_counts.items())),
        "checks": checks,
        "inputs": {
            "partial_manifest": {
                "path": str(partial_manifest_path),
                "sha256": partial_manifest_hash,
            },
            "partial_selection": {
                "path": str(partial_selection_path),
                "sha256": sha256_file(partial_selection_path),
            },
            "test_pairs": {"path": str(pair_path), "sha256": sha256_file(pair_path)},
            "model_outputs": input_records,
        },
        "outputs": {
            blind_path.name: {"path": str(blind_path), "sha256": sha256_file(blind_path)},
            mapping_path.name: {
                "path": str(mapping_path),
                "sha256": sha256_file(mapping_path),
            },
        },
    }
    atomic_write_json(final_manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial-dir", default=str(DEFAULT_PARTIAL_DIR))
    parser.add_argument("--sft", default=str(DEFAULT_GENERATION_DIR / "sft_test_outputs.jsonl"))
    parser.add_argument("--dpo", default=str(DEFAULT_GENERATION_DIR / "dpo_test_outputs.jsonl"))
    parser.add_argument(
        "--mask-off",
        default=str(DEFAULT_GENERATION_DIR / "proposed_mask_off_test_outputs.jsonl"),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--blind-seed", type=int, default=20260910)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_counselor_export(
        Path(args.partial_dir),
        Path(args.sft),
        Path(args.dpo),
        Path(args.mask_off),
        Path(args.output_dir),
        blind_seed=args.blind_seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
