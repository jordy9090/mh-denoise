#!/usr/bin/env python3
"""Utilities for the H100 frozen Revision 2 TEST surface-relaxed run."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


METHOD_SPECS = {
    "SFT": {"response_field": "sft_response", "output_key": "sft"},
    "DPO": {"response_field": "sft_response", "output_key": "dpo"},
    "Proposed": {"response_field": "final_response", "output_key": "proposed"},
}
LABELS = ("A", "B", "C")
EXPORT_VERSION = "h100-test-surface-relaxed-expert-export-v1-20261008"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(row)
    return rows


def read_jsonl_with_lines(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    rows: dict[str, dict[str, Any]] = {}
    lines: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            canonical_id = row_id(row)
            if canonical_id in rows:
                raise ValueError(f"Duplicate canonical ID {canonical_id}: {path}")
            rows[canonical_id] = row
            lines[canonical_id] = line_number
    return rows, lines


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".tmp.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def row_id(row: dict[str, Any]) -> str:
    canonical_id = str(row.get("canonical_id") or row.get("id") or "")
    if not canonical_id:
        raise ValueError("Row is missing canonical_id/id")
    return canonical_id


def ordered_ids_from_json(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Ordered JSON has no rows list: {path}")
    ids = [str(row.get("canonical_id") or row.get("id") or "") for row in rows]
    if "" in ids or len(ids) != len(set(ids)):
        raise ValueError(f"Ordered rows contain empty or duplicate canonical IDs: {path}")
    return ids


def expand_existing(paths: list[str], globs: list[str]) -> list[Path]:
    expanded = [Path(path).resolve() for path in paths]
    for pattern in globs:
        expanded.extend(Path(path).resolve() for path in glob.glob(pattern))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in expanded:
        key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        unique.append(path)
    return unique


def index_existing(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    rows: dict[str, dict[str, Any]] = {}
    sources: dict[str, Path] = {}
    for path in paths:
        for row in read_jsonl(path):
            canonical_id = row_id(row)
            if canonical_id in rows:
                if sha256_text(json.dumps(rows[canonical_id], sort_keys=True, ensure_ascii=False)) != sha256_text(
                    json.dumps(row, sort_keys=True, ensure_ascii=False)
                ):
                    raise RuntimeError(f"Conflicting existing rows for {canonical_id}: {sources[canonical_id]} and {path}")
                continue
            rows[canonical_id] = row
            sources[canonical_id] = path
    return rows, sources


def prepare_missing(args: argparse.Namespace) -> None:
    ordered_ids = ordered_ids_from_json(Path(args.ordered_json).resolve())
    input_rows = {row_id(row): row for row in read_jsonl(Path(args.input_jsonl).resolve())}
    existing_paths = expand_existing(args.existing, args.existing_glob)
    existing_rows, _ = index_existing(existing_paths)
    missing_ids = [canonical_id for canonical_id in ordered_ids if canonical_id not in existing_rows]
    missing_rows = [input_rows[canonical_id] for canonical_id in missing_ids]
    output_jsonl = Path(args.output_jsonl).resolve()
    atomic_write_jsonl(output_jsonl, missing_rows)
    status = {
        "status": "complete_missing_subset",
        "ordered_rows": len(ordered_ids),
        "existing_rows_for_ordered_ids": sum(canonical_id in existing_rows for canonical_id in ordered_ids),
        "missing_rows": len(missing_rows),
        "missing_ids": missing_ids,
        "input": {"path": str(Path(args.input_jsonl).resolve()), "sha256": sha256_file(Path(args.input_jsonl).resolve())},
        "existing_sources": [
            {"path": str(path), "sha256": sha256_file(path), "rows": len(read_jsonl(path))}
            for path in existing_paths
        ],
        "output": {"path": str(output_jsonl), "sha256": sha256_file(output_jsonl), "rows": len(missing_rows)},
    }
    atomic_write_json(Path(args.status_json).resolve(), status)
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))


def generation_completion(method: str, rows: list[dict[str, Any]]) -> dict[str, int]:
    if method in {"sft", "dpo"}:
        generations = [row.get("sft_generation") or {} for row in rows]
        return {
            "eos_reached": sum(bool(item.get("eos_reached")) for item in generations),
            "length_limit_reached": sum(bool(item.get("length_limit_reached")) for item in generations),
            "source_input_truncated": sum(bool(item.get("source_input_truncated")) for item in generations),
        }
    generations = [row.get("denoiser_generation") or {} for row in rows]
    return {
        "sft_length_limit_reached": sum(bool((row.get("sft_generation") or {}).get("length_limit_reached")) for row in rows),
        "denoiser_length_limit_reached": sum(bool(item.get("length_limit_reached")) for item in generations),
        "denoiser_eos_reached": sum(bool(item.get("eos_reached")) for item in generations),
        "denoiser_source_input_truncated": sum(bool(item.get("source_input_truncated")) for item in generations),
    }


def merge_outputs(args: argparse.Namespace) -> None:
    ordered_ids = ordered_ids_from_json(Path(args.ordered_json).resolve())
    input_path = Path(args.input_jsonl).resolve()
    input_rows = {row_id(row): row for row in read_jsonl(input_path)}
    source_paths = expand_existing(args.existing, args.existing_glob) + [
        Path(path).resolve() for path in args.new if Path(path).exists()
    ]
    available_rows, row_sources = index_existing(source_paths)
    missing = [canonical_id for canonical_id in ordered_ids if canonical_id not in available_rows]
    if missing:
        raise RuntimeError(f"Missing generated rows for ordered IDs: {missing}")
    merged = [available_rows[canonical_id] for canonical_id in ordered_ids]
    for row in merged:
        canonical_id = row_id(row)
        if canonical_id not in input_rows:
            raise RuntimeError(f"Merged row is not in input membership: {canonical_id}")
        if row.get("question") != input_rows[canonical_id].get("question"):
            raise RuntimeError(f"Generated question differs from frozen input: {canonical_id}")
    output_path = Path(args.output_jsonl).resolve()
    atomic_write_jsonl(output_path, merged)
    counts_by_source = Counter(str(row_sources[row_id(row)]) for row in merged)
    manifest = {
        "status": args.status,
        "method": args.method,
        "input": {"path": str(input_path), "sha256": sha256_file(input_path), "rows": len(input_rows)},
        "output": {"path": str(output_path), "sha256": sha256_file(output_path), "rows": len(merged)},
        "ordered_json": {"path": str(Path(args.ordered_json).resolve()), "sha256": sha256_file(Path(args.ordered_json).resolve())},
        "source_files": [
            {"path": str(path), "sha256": sha256_file(path), "rows": len(read_jsonl(path))}
            for path in source_paths
        ],
        "rows_reused_by_source": dict(sorted(counts_by_source.items())),
        "canonical_ids_ordered": ordered_ids,
        "generation_completion": generation_completion(args.method, merged),
    }
    atomic_write_json(Path(args.manifest_json).resolve(), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def balanced_orders(count: int, seed: int) -> list[list[str]]:
    methods = list(METHOD_SPECS)
    orders: list[list[str]] = []
    for block_start in range(0, count, len(methods)):
        block_number = block_start // len(methods)
        base = sorted(methods, key=lambda method: sha256_text(f"{seed}\0{block_number}\0{method}"))
        for offset in range(min(len(methods), count - block_start)):
            orders.append(base[offset:] + base[:offset])
    return orders


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def legacy_replay(row: dict[str, Any]) -> dict[str, Any]:
    settings = row.get("acceptance_settings") or {}
    min_word_count = int(settings.get("min_word_count", row.get("min_word_count", 20)) or 20)
    min_risk_delta = float(settings.get("min_risk_delta", 0.01))
    min_focus_risk_delta = float(settings.get("min_focus_risk_delta", 0.0))
    specificity_min_ratio = float(settings.get("specificity_min_ratio", 0.6))
    keyword_overlap_slack = float(settings.get("keyword_overlap_slack", 0.15))
    gate_strategy = str(settings.get("gate_strategy", row.get("gate_strategy", "overall")))

    sft_response = clean_text(row.get("sft_response"))
    denoiser_response = clean_text(row.get("denoiser_response"))
    if not row.get("used_denoiser"):
        return {
            "legacy_replay_used_denoiser": False,
            "legacy_replay_accepted_denoiser": False,
            "legacy_replay_reject_reasons": [],
            "legacy_replay_final_response": sft_response,
            "legacy_replay_final_response_sha256": sha256_text(sft_response),
        }

    reasons: list[str] = []
    if not denoiser_response:
        reasons.append("empty_denoiser_response")
    if int(row.get("denoiser_word_count") or 0) < min_word_count:
        reasons.append("short_denoiser_response")
    if float(row.get("denoiser_risk_score")) > float(row.get("sft_risk_score")) - min_risk_delta:
        reasons.append("risk_not_improved")
    if gate_strategy != "overall":
        if float(row.get("denoiser_focus_risk_score")) > float(row.get("sft_focus_risk_score")) - min_focus_risk_delta:
            reasons.append("focus_risk_not_improved")
    if int(row.get("denoiser_bad_safety_count") or 0) > int(row.get("sft_bad_safety_count") or 0):
        reasons.append("bad_safety_increased")
    if float(row.get("specificity_ratio") or 0.0) < specificity_min_ratio:
        reasons.append("specificity_ratio_low")
    if int(row.get("denoiser_generic_count") or 0) > int(row.get("sft_generic_count") or 0):
        reasons.append("genericity_increased")
    min_overlap = max(0.0, float(row.get("sft_keyword_overlap") or 0.0) - keyword_overlap_slack)
    if float(row.get("denoiser_keyword_overlap") or 0.0) < min_overlap:
        reasons.append("question_keyword_overlap_low")
    accepted = not reasons
    final_response = denoiser_response if accepted else sft_response
    return {
        "legacy_replay_used_denoiser": True,
        "legacy_replay_accepted_denoiser": accepted,
        "legacy_replay_reject_reasons": reasons,
        "legacy_replay_final_response": final_response,
        "legacy_replay_final_response_sha256": sha256_text(final_response),
    }


def source_record(path: Path, rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "rows": len(rows)}


def build_expert_export(args: argparse.Namespace) -> None:
    partial_dir = Path(args.partial_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    partial_selection_path = partial_dir / "partial_selection.json"
    pair_path = partial_dir / "test_pairs.jsonl"
    selected_ids = ordered_ids_from_json(partial_selection_path)
    pair_rows, _ = read_jsonl_with_lines(pair_path)
    if set(pair_rows) != set(selected_ids):
        raise RuntimeError("Partial pair membership differs from partial selection")

    output_paths = {
        "SFT": Path(args.sft).resolve(),
        "DPO": Path(args.dpo).resolve(),
        "Proposed": Path(args.proposed).resolve(),
    }
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    source_lines: dict[str, dict[str, int]] = {}
    for method, path in output_paths.items():
        indexed[method], source_lines[method] = read_jsonl_with_lines(path)
        if set(indexed[method]) != set(selected_ids):
            raise RuntimeError(f"{method} output membership mismatch")

    blind_orders = balanced_orders(len(selected_ids), args.blind_seed)
    blind_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    rating_rows: list[dict[str, Any]] = []
    exact_match_counts: Counter[str] = Counter()
    position_counts: Counter[tuple[str, str]] = Counter()

    for zero_index, (canonical_id, order) in enumerate(zip(selected_ids, blind_orders, strict=True)):
        case_id = f"TEST-{zero_index + 1:02d}"
        pair = pair_rows[canonical_id]
        question = pair.get("question")
        if not isinstance(question, str) or not question.strip():
            raise RuntimeError(f"Empty question in frozen pair: {canonical_id}")
        responses: list[dict[str, str]] = []
        method_texts: dict[str, str] = {}
        for label, method in zip(LABELS, order, strict=True):
            source_row = indexed[method][canonical_id]
            if source_row.get("question") != question:
                raise RuntimeError(f"{method} question differs from frozen pair: {canonical_id}")
            response_field = str(METHOD_SPECS[method]["response_field"])
            response = source_row.get(response_field)
            if not isinstance(response, str) or not response.strip():
                raise RuntimeError(f"{method} has empty {response_field}: {canonical_id}")
            response_id = f"{case_id}-{label}"
            responses.append({"response_id": response_id, "response_label": label, "response_text": response})
            response_hash = sha256_text(response)
            mapping_rows.append(
                {
                    "case_id": case_id,
                    "question_number": zero_index + 1,
                    "response_id": response_id,
                    "response_label": label,
                    "method": method,
                    "canonical_id": canonical_id,
                    "question_sha256": sha256_text(question),
                    "response_sha256": response_hash,
                    "source": {
                        "path": str(output_paths[method]),
                        "line": source_lines[method][canonical_id],
                        "file_sha256": sha256_file(output_paths[method]),
                        "response_field": response_field,
                    },
                }
            )
            rating_rows.append(
                {
                    "case_id": case_id,
                    "question_number": zero_index + 1,
                    "response_id": response_id,
                    "response_label": label,
                    "Overall Quality": "",
                    "Empathy": "",
                    "Specificity": "",
                    "Medical Advice": "",
                    "Notes": "",
                }
            )
            method_texts[method] = response
            position_counts[(method, label)] += 1
        for left, right in (("SFT", "DPO"), ("SFT", "Proposed"), ("DPO", "Proposed")):
            if method_texts[left] == method_texts[right]:
                exact_match_counts[f"{left}_equals_{right}"] += 1
        blind_rows.append(
            {
                "case_id": case_id,
                "question_number": zero_index + 1,
                "question": question,
                "responses": responses,
            }
        )

    blind_path = output_dir / f"expert_blinded_cases_{len(selected_ids)}x3.jsonl"
    mapping_path = output_dir / "researcher_model_mapping.jsonl"
    rating_csv_path = output_dir / f"expert_rating_form_{len(selected_ids)}x3.csv"
    rating_md_path = output_dir / f"expert_rating_form_{len(selected_ids)}x3.md"
    proposed_provenance_path = output_dir / "researcher_proposed_provenance.jsonl"
    manifest_path = output_dir / "export_manifest.json"
    zip_path = output_dir / f"h100_test_surface_relaxed_expert_packet_{len(selected_ids)}x3.zip"

    atomic_write_jsonl(blind_path, blind_rows)
    atomic_write_jsonl(mapping_path, mapping_rows)

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=output_dir, prefix=rating_csv_path.name + ".tmp.", delete=False
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "question_number",
                "response_id",
                "response_label",
                "Overall Quality",
                "Empathy",
                "Specificity",
                "Medical Advice",
                "Notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rating_rows)
        temporary_csv = Path(handle.name)
    os.replace(temporary_csv, rating_csv_path)

    md_lines = [
        "# Expert Rating Form",
        "",
        "Rate each anonymized response on: Overall Quality, Empathy, Specificity, Medical Advice.",
        "",
        "| case_id | response_id | label | Overall Quality | Empathy | Specificity | Medical Advice | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rating_rows:
        md_lines.append(
            f"| {row['case_id']} | {row['response_id']} | {row['response_label']} |  |  |  |  |  |"
        )
    atomic_write_text(rating_md_path, "\n".join(md_lines) + "\n")

    proposed_provenance: list[dict[str, Any]] = []
    for canonical_id in selected_ids:
        row = indexed["Proposed"][canonical_id]
        proposed_provenance.append(
            {
                "canonical_id": canonical_id,
                "question_sha256": sha256_text(str(row.get("question") or "")),
                "acceptance_policy": row.get("acceptance_policy"),
                "used_denoiser": row.get("used_denoiser"),
                "denoiser_call_reason": row.get("denoiser_call_reason"),
                "accepted_denoiser": row.get("accepted_denoiser"),
                "reject_reason": row.get("reject_reason"),
                "surface_heuristic_flags": row.get("surface_heuristic_flags"),
                "sft_risk_score": row.get("sft_risk_score"),
                "denoiser_risk_score": row.get("denoiser_risk_score"),
                "final_risk_score": row.get("final_risk_score"),
                "sft_focus_risk_score": row.get("sft_focus_risk_score"),
                "denoiser_focus_risk_score": row.get("denoiser_focus_risk_score"),
                "final_focus_risk_score": row.get("final_focus_risk_score"),
                "sft_response": row.get("sft_response"),
                "denoiser_response": row.get("denoiser_response"),
                "final_response": row.get("final_response"),
                "legacy_replay": legacy_replay(row),
                "source": {
                    "path": str(output_paths["Proposed"]),
                    "line": source_lines["Proposed"][canonical_id],
                    "file_sha256": sha256_file(output_paths["Proposed"]),
                },
            }
        )
    atomic_write_jsonl(proposed_provenance_path, proposed_provenance)

    checks = {
        "questions_have_three_responses": all(len(row["responses"]) == 3 for row in blind_rows),
        "blind_rows_have_no_method_or_canonical_id": all(
            set(row) == {"case_id", "question_number", "question", "responses"}
            and all(set(response) == {"response_id", "response_label", "response_text"} for response in row["responses"])
            for row in blind_rows
        ),
        "mapping_rows_count": len(mapping_rows) == len(selected_ids) * 3,
        "rating_rows_count": len(rating_rows) == len(selected_ids) * 3,
        "proposed_policy_surface_relaxed_v1": all(
            indexed["Proposed"][canonical_id].get("acceptance_policy") == "surface_relaxed_v1"
            for canonical_id in selected_ids
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)

    manifest = {
        "version": EXPORT_VERSION,
        "status": "complete_h100_test_surface_relaxed_expert_export",
        "counts": {
            "questions": len(selected_ids),
            "responses": len(selected_ids) * 3,
            "responses_per_question": 3,
        },
        "evaluation_axes": ["Overall Quality", "Empathy", "Specificity", "Medical Advice"],
        "blinding": {
            "seed": args.blind_seed,
            "labels": list(LABELS),
            "method_position_counts_internal": {
                method: {label: position_counts[(method, label)] for label in LABELS}
                for method in METHOD_SPECS
            },
        },
        "exact_response_match_counts_internal": dict(sorted(exact_match_counts.items())),
        "checks": checks,
        "inputs": {
            "partial_selection": {"path": str(partial_selection_path), "sha256": sha256_file(partial_selection_path)},
            "test_pairs": {"path": str(pair_path), "sha256": sha256_file(pair_path)},
            "model_outputs": {
                method: source_record(output_paths[method], indexed[method]) for method in METHOD_SPECS
            },
        },
        "outputs": {
            blind_path.name: {"path": str(blind_path), "sha256": sha256_file(blind_path)},
            mapping_path.name: {"path": str(mapping_path), "sha256": sha256_file(mapping_path)},
            rating_csv_path.name: {"path": str(rating_csv_path), "sha256": sha256_file(rating_csv_path)},
            rating_md_path.name: {"path": str(rating_md_path), "sha256": sha256_file(rating_md_path)},
            proposed_provenance_path.name: {
                "path": str(proposed_provenance_path),
                "sha256": sha256_file(proposed_provenance_path),
            },
        },
    }
    atomic_write_json(manifest_path, manifest)

    with tempfile.NamedTemporaryFile(dir=output_dir, prefix=zip_path.name + ".tmp.", delete=False) as handle:
        temporary_zip = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in (blind_path, rating_csv_path, rating_md_path, mapping_path, proposed_provenance_path, manifest_path):
                archive.write(path, arcname=path.name)
        os.replace(temporary_zip, zip_path)
    finally:
        if temporary_zip.exists():
            temporary_zip.unlink()
    manifest["outputs"][zip_path.name] = {"path": str(zip_path), "sha256": sha256_file(zip_path)}
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    missing = subparsers.add_parser("prepare-missing")
    missing.add_argument("--ordered-json", required=True)
    missing.add_argument("--input-jsonl", required=True)
    missing.add_argument("--output-jsonl", required=True)
    missing.add_argument("--status-json", required=True)
    missing.add_argument("--existing", action="append", default=[])
    missing.add_argument("--existing-glob", action="append", default=[])
    missing.set_defaults(func=prepare_missing)

    merge = subparsers.add_parser("merge-outputs")
    merge.add_argument("--ordered-json", required=True)
    merge.add_argument("--input-jsonl", required=True)
    merge.add_argument("--output-jsonl", required=True)
    merge.add_argument("--manifest-json", required=True)
    merge.add_argument("--method", choices=["sft", "dpo", "proposed"], required=True)
    merge.add_argument("--status", required=True)
    merge.add_argument("--existing", action="append", default=[])
    merge.add_argument("--existing-glob", action="append", default=[])
    merge.add_argument("--new", action="append", default=[])
    merge.set_defaults(func=merge_outputs)

    export = subparsers.add_parser("build-expert-export")
    export.add_argument("--partial-dir", required=True)
    export.add_argument("--sft", required=True)
    export.add_argument("--dpo", required=True)
    export.add_argument("--proposed", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--blind-seed", type=int, default=20260910)
    export.set_defaults(func=build_expert_export)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
