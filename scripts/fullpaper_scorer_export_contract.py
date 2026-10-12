#!/usr/bin/env python3
"""Shared validation for exact-span scorer exports.

The scorer sees only ``scorer_input_text(question, span)``.  Provenance such as
side, canonical ID, and offsets is audit metadata and must never make identical
model inputs look distinct for supervision-conflict checks.
"""
from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from typing import Any

from fullpaper_risk_contract import AXES, scorer_input_text


VERSION = "fullpaper-scorer-export-contract-v2-20260924"


def _input_text(row: dict[str, Any]) -> str:
    return scorer_input_text(str(row["question"]), str(row["span"]))


def _input_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_non_null_collisions(
    rows: list[dict[str, Any]], *, token_ids: list[list[int]] | None = None
) -> dict[tuple[Any, str], list[tuple[int, int]]]:
    """Return 0/1 collisions keyed by actual string or final token IDs and axis."""
    grouped: dict[tuple[Any, str], list[tuple[int, int]]] = defaultdict(list)
    for index, row in enumerate(rows):
        model_key: Any = tuple(token_ids[index]) if token_ids is not None else _input_text(row)
        for axis in AXES:
            label = row.get("labels", {}).get(axis)
            if label in (0, 1):
                grouped[(model_key, axis)].append((index, int(label)))
    return {
        key: members
        for key, members in grouped.items()
        if {label for _, label in members} == {0, 1}
    }


def mask_collisions(
    rows: list[dict[str, Any]],
    collisions: dict[tuple[Any, str], list[tuple[int, int]]],
    *,
    stage: str,
) -> list[dict[str, Any]]:
    """Mask every annotation in a conflicting input-axis group and ledger it."""
    ledger: list[dict[str, Any]] = []
    for (model_key, axis), members in collisions.items():
        member_records = []
        for index, label in members:
            row = rows[index]
            member_records.append({
                "row_index": index,
                "canonical_id": row.get("canonical_id"),
                "split": row.get("split"),
                "span_side": row.get("span_side"),
                "span_start": row.get("span_start"),
                "span_end": row.get("span_end"),
                "original_label": label,
                "original_basis": row.get("label_basis", {}).get(axis),
                "source_records": copy.deepcopy(row.get("source_records", [])),
            })
            row.setdefault("labels", {})[axis] = None
            row.setdefault("label_mask", {})[axis] = False
            row.setdefault("label_basis", {})[axis] = f"unknown_masked_{stage}_collision"
        model_input = _input_text(rows[members[0][0]])
        ledger.append({
            "contract_version": VERSION,
            "stage": stage,
            "axis": axis,
            "model_input_sha256": _input_sha256(model_input),
            "token_ids_sha256": (
                hashlib.sha256(",".join(map(str, model_key)).encode("ascii")).hexdigest()
                if isinstance(model_key, tuple) else None
            ),
            "original_labels": sorted({label for _, label in members}),
            "members": member_records,
            "resolution": "all_related_labels_masked_unknown",
        })
    return ledger


def assert_no_non_null_collisions(
    rows: list[dict[str, Any]], *, token_ids: list[list[int]] | None = None
) -> None:
    remaining = find_non_null_collisions(rows, token_ids=token_ids)
    if remaining:
        raise RuntimeError(f"Non-null scorer input-axis conflicts remain: {len(remaining)}")


def deduplicate_same_inputs(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collapse identical scorer inputs while preserving every origin record.

    Opposing labels must already have been masked.  Equal labels on equal
    model inputs contribute once, not once per clean/candidate offset.  The
    returned ledger retains all physical origins and per-axis contribution
    counts for audit.
    """
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    order: list[str] = []
    for index, row in enumerate(rows):
        key = _input_text(row)
        if key not in grouped:
            order.append(key)
        grouped[key].append((index, row))

    deduplicated: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for model_input in order:
        members = grouped[model_input]
        splits = {str(row.get("split")) for _, row in members}
        if len(splits) != 1:
            raise RuntimeError("Identical scorer model input occurs across data splits")
        merged = copy.deepcopy(members[0][1])
        provenance_members = []
        duplicate_axis_contributions: dict[str, int] = {}
        for index, row in members:
            provenance_members.append({
                "source_row_index": index,
                "canonical_id": row.get("canonical_id"),
                "split": row.get("split"),
                "span_side": row.get("span_side"),
                "span_start": row.get("span_start"),
                "span_end": row.get("span_end"),
                "span_source_sha256": row.get("span_source_sha256"),
                "source_records": copy.deepcopy(row.get("source_records", [])),
            })
        for axis in AXES:
            contributions = [
                (index, int(row.get("labels", {}).get(axis)))
                for index, row in members
                if row.get("labels", {}).get(axis) in (0, 1)
            ]
            values = {label for _, label in contributions}
            if len(values) > 1:
                raise RuntimeError(
                    f"Cannot deduplicate opposing non-null scorer labels: {axis}"
                )
            label = next(iter(values)) if values else None
            merged.setdefault("labels", {})[axis] = label
            merged.setdefault("label_mask", {})[axis] = label is not None
            if len(contributions) > 1:
                duplicate_axis_contributions[axis] = len(contributions) - 1
                merged.setdefault("label_basis", {})[axis] = (
                    "explicit_deduplicated_same_input_same_label"
                )
            elif len(contributions) == 1:
                source_index = contributions[0][0]
                merged.setdefault("label_basis", {})[axis] = rows[source_index].get(
                    "label_basis", {}
                ).get(axis)
        unique_records: list[dict[str, Any]] = []
        seen_records: set[str] = set()
        for _, row in members:
            for record in row.get("source_records", []):
                fingerprint = repr(sorted(record.items()))
                if fingerprint not in seen_records:
                    seen_records.add(fingerprint)
                    unique_records.append(copy.deepcopy(record))
        merged["source_records"] = unique_records
        merged["provenance_members"] = provenance_members
        merged["deduplicated_physical_rows"] = len(members)
        deduplicated.append(merged)
        if len(members) > 1:
            ledger.append({
                "contract_version": VERSION,
                "model_input_sha256": _input_sha256(model_input),
                "split": next(iter(splits)),
                "physical_rows": len(members),
                "removed_physical_rows": len(members) - 1,
                "duplicate_axis_contributions_removed": duplicate_axis_contributions,
                "members": provenance_members,
                "resolution": "single_model_input_with_merged_provenance",
            })
    return deduplicated, ledger


def _char_range(text: str, value: str, start: int = 0) -> tuple[int, int]:
    stripped = value.strip()
    offset = text.find(stripped, start)
    if offset < 0:
        raise AssertionError("Serialized scorer field cannot be located")
    return offset, offset + len(stripped)


def _range_preserved(offsets: list[tuple[int, int]], start: int, end: int, text: str) -> bool:
    covered = bytearray(end - start)
    for token_start, token_end in offsets:
        left, right = max(start, int(token_start)), min(end, int(token_end))
        if left < right:
            covered[left - start : right - start] = b"\x01" * (right - left)
    return all(covered[index] or text[start + index].isspace() for index in range(end - start))


def audit_tokenization_and_mask(
    rows: list[dict[str, Any]], tokenizer: Any, *, max_length: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[int]], list[dict[str, Any]]]:
    """Use the training tokenizer/serialization and mask supervision lost to truncation."""
    if max_length < 1:
        raise ValueError("scorer max_length must be positive")
    token_ids: list[list[int]] = []
    truncation_ledger: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        text = _input_text(row)
        question_range = _char_range(text, str(row["question"]))
        span_range = _char_range(text, str(row["span"]), question_range[1])
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        if "offset_mapping" not in encoded:
            raise RuntimeError("Scorer tokenizer must provide offset_mapping for evidence-preservation audit")
        ids = list(encoded["input_ids"])
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        token_ids.append(ids)
        question_preserved = _range_preserved(offsets, *question_range, text)
        span_preserved = _range_preserved(offsets, *span_range, text)
        untruncated = tokenizer(text, truncation=False)
        original_tokens = len(untruncated["input_ids"])
        measurements.append({
            "row_index": index,
            "canonical_id": row.get("canonical_id"),
            "split": row.get("split"),
            "original_tokens": original_tokens,
            "final_tokens": len(ids),
            "truncated": original_tokens > len(ids),
            "question_preserved": question_preserved,
            "span_preserved": span_preserved,
            "model_input_sha256": _input_sha256(text),
        })
        if question_preserved and span_preserved:
            continue
        masked = {}
        for axis in AXES:
            label = row.get("labels", {}).get(axis)
            if label in (0, 1):
                masked[axis] = {
                    "original_label": int(label),
                    "original_basis": row.get("label_basis", {}).get(axis),
                }
                row["labels"][axis] = None
                row["label_mask"][axis] = False
                row["label_basis"][axis] = "unknown_masked_scorer_input_truncation"
        truncation_ledger.append({
            **measurements[-1],
            "masked_axes": masked,
            "resolution": "affected_supervision_masked_unknown",
        })

    token_collisions = find_non_null_collisions(rows, token_ids=token_ids)
    token_collision_ledger = mask_collisions(
        rows, token_collisions, stage="tokenized_input"
    )
    assert_no_non_null_collisions(rows, token_ids=token_ids)
    summary = {
        "contract_version": VERSION,
        "max_length": max_length,
        "tokenizer_class": type(tokenizer).__name__,
        "truncation_side": getattr(tokenizer, "truncation_side", None),
        "rows": len(rows),
        "max_original_tokens": max((item["original_tokens"] for item in measurements), default=0),
        "max_final_tokens": max((item["final_tokens"] for item in measurements), default=0),
        "truncated_rows": sum(item["truncated"] for item in measurements),
        "question_not_preserved": sum(not item["question_preserved"] for item in measurements),
        "span_not_preserved": sum(not item["span_preserved"] for item in measurements),
        "token_collision_groups": len(token_collisions),
    }
    return summary, truncation_ledger, token_ids, token_collision_ledger


__all__ = [
    "VERSION",
    "assert_no_non_null_collisions",
    "audit_tokenization_and_mask",
    "deduplicate_same_inputs",
    "find_non_null_collisions",
    "mask_collisions",
]
