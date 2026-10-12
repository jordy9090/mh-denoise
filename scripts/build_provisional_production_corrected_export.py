#!/usr/bin/env python3
"""Build a review-gated, provisional corrected export without rewriting production.

The review ledger is a JSONL file with one row per reviewed canonical ID:

  {"canonical_id": "qa_...", "clean_disposition": "retained|held|unresolved",
   "decision_source": "review artifact name", "decision_evidence": ["..."],
   "scorer_annotations": [{"side": "candidate|clean", "axis": "...",
       "start": 0, "end": 12, "text": "exact source text", "source_sha256": "...",
       "scope": "local_defect|positive_support|holistic|omission",
       "label": 1|0|null, "annotation_source": "...",
       "annotation_evidence": "..."}]}

Only an explicit retained decision is exported.  Missing review rows are
unresolved, never silently retained.  `local_defect` requires an explicit 1;
`positive_support` requires an explicit 0 (it never implies 0 by itself);
`holistic` and `omission` always remain unknown for the local scorer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fullpaper_risk_contract import AXES, scorer_input_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRODUCTION = ROOT / "data/fullpaper_acl_pipeline/production_run_train1000_valid200_local_qwen35_27b_v3r1_20260910"
SCOPES = {"local_defect", "positive_support", "holistic", "omission", "unresolved"}
DISPOSITIONS = {"retained", "held", "unresolved"}
VERSION = "provisional-production-corrected-export-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_span_key(record: dict[str, Any]) -> tuple[str, str, int, int, str, str]:
    return (
        str(record["canonical_id"]), str(record["side"]), int(record["start"]), int(record["end"]),
        str(record["text"]), str(record["source_sha256"]),
    )


def require_review_ledger(rows: list[dict[str, Any]], accepted_ids: set[str]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        canonical_id = str(row.get("canonical_id") or "")
        disposition = row.get("clean_disposition")
        if not canonical_id or canonical_id not in accepted_ids:
            raise ValueError(f"Review ledger references a non-accepted canonical_id: {canonical_id!r}")
        if canonical_id in by_id:
            raise ValueError(f"Review ledger contains duplicate canonical_id: {canonical_id}")
        if disposition not in DISPOSITIONS:
            raise ValueError(f"Invalid clean_disposition for {canonical_id}: {disposition!r}")
        if not str(row.get("decision_source") or "").strip() or not row.get("decision_evidence"):
            raise ValueError(f"Review decision lacks source/evidence: {canonical_id}")
        annotations = row.get("scorer_annotations", [])
        if not isinstance(annotations, list):
            raise ValueError(f"scorer_annotations must be a list: {canonical_id}")
        by_id[canonical_id] = row
    return by_id


def audit_review_ledger(
    reviewed_cases: Path,
    decisions_csv: Path,
    hold_ids_file: Path,
    accepted_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Translate supplied human audit artifacts without inferring new decisions.

    `no_obvious_integrity_defect_in_this_read` is a limited integrity read,
    not approval of full clean eligibility.  It remains unresolved along with
    ambiguous-rubric and case-study records until a separate review decides.
    """
    cases = read_jsonl(reviewed_cases)
    csv_rows = list(csv.DictReader(decisions_csv.open(encoding="utf-8", newline="")))
    by_case = {str(row.get("canonical_id") or ""): row for row in cases}
    by_csv = {str(row.get("canonical_id") or ""): row for row in csv_rows}
    if len(by_case) != len(cases) or len(by_csv) != len(csv_rows) or set(by_case) != set(by_csv):
        raise ValueError("Reviewed JSONL and review CSV must have the same unique canonical IDs")
    if not set(by_case) <= set(accepted_by_id):
        raise ValueError("Review artifacts reference IDs outside production accepted")
    for canonical_id, case in by_case.items():
        accepted = accepted_by_id[canonical_id]
        csv_row = by_csv[canonical_id]
        if case.get("recommendation") != csv_row.get("recommendation"):
            raise ValueError(f"Review recommendation differs between JSONL/CSV: {canonical_id}")
        if case.get("question") != accepted["question"] or case.get("clean_response") != accepted["clean_response"]:
            raise ValueError(f"Review text does not match immutable production row: {canonical_id}")
        if case.get("split") != accepted["split"]:
            raise ValueError(f"Review split differs from immutable production row: {canonical_id}")
    hold_ids = {line.strip() for line in hold_ids_file.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}
    review_holds = {canonical_id for canonical_id, case in by_case.items() if case.get("recommendation") == "hold_clean_integrity"}
    if hold_ids != review_holds:
        raise ValueError("recommended_clean_hold_ids.txt does not exactly match review hold decisions")

    converted: list[dict[str, Any]] = []
    for canonical_id, case in by_case.items():
        recommendation = str(case["recommendation"])
        if recommendation == "hold_clean_integrity":
            disposition = "held"
        else:
            disposition = "unresolved"
        converted.append({
            "canonical_id": canonical_id,
            "clean_disposition": disposition,
            "decision_source": "reviewed_clean_cases.jsonl + clean_review_decisions.csv",
            "decision_evidence": [str(case.get("reason_ko") or ""), str(case.get("clean_excerpt") or "")],
            "scorer_annotations": [],
            "clean_review": {
                "recommendation": recommendation,
                "review_group": case.get("review_group"),
                "reviewer": case.get("reviewer"),
                "reason_ko": case.get("reason_ko"),
                "clean_excerpt": case.get("clean_excerpt"),
                "source": case.get("source"),
                "reviewed_clean_sha256": hashlib.sha256(case["clean_response"].encode()).hexdigest(),
            },
        })
    return require_review_ledger(converted, set(accepted_by_id)), {
        "reviewed_clean_cases.jsonl": sha256(reviewed_cases),
        "clean_review_decisions.csv": sha256(decisions_csv),
        "recommended_clean_hold_ids.txt": sha256(hold_ids_file),
    }


def annotation_label(annotation: dict[str, Any]) -> int | None:
    """Return the only permissible local-scoring label for a review annotation."""
    scope, label = annotation.get("scope"), annotation.get("label")
    if scope not in SCOPES:
        raise ValueError(f"Unknown evidence scope: {scope!r}")
    if scope in {"holistic", "omission", "unresolved"}:
        if label is not None:
            raise ValueError(f"{scope} cannot create a local scorer label")
        return None
    if scope == "local_defect":
        if label != 1:
            raise ValueError("local_defect requires an explicit label=1")
        return 1
    if label not in (None, 0):
        raise ValueError("positive_support may only have explicit label=0 or null")
    return label


def annotation_key(annotation: dict[str, Any], canonical_id: str) -> tuple[str, str, int, int, str, str]:
    required = ("side", "axis", "start", "end", "text", "source_sha256", "scope", "annotation_source")
    missing = [field for field in required if annotation.get(field) in (None, "")]
    if missing:
        raise ValueError(f"Scorer annotation lacks fields for {canonical_id}: {missing}")
    if annotation["side"] not in {"clean", "candidate"} or annotation["axis"] not in AXES:
        raise ValueError(f"Invalid scorer annotation side/axis for {canonical_id}")
    return (canonical_id, str(annotation["side"]), int(annotation["start"]), int(annotation["end"]),
            str(annotation["text"]), str(annotation["source_sha256"]))


def collision_key(row: dict[str, Any], axis: str) -> tuple[str, str]:
    # Axis is part of the prediction target even though the six-axis classifier
    # emits all dimensions from one shared question+sentence serialization.
    return (hashlib.sha256(scorer_input_text(row["question"], row["span"]).encode()).hexdigest(), axis)


def find_non_null_collisions(rows: list[dict[str, Any]], token_keys: list[str] | None = None) -> dict[tuple[str, str], list[tuple[int, int]]]:
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for index, row in enumerate(rows):
        for axis in AXES:
            values = row.get("label_candidates", {}).get(axis, [row["labels"][axis]])
            key = ((token_keys[index] if token_keys else collision_key(row, axis)[0]), axis)
            grouped[key].extend((index, label) for label in values if label is not None)
    return {key: values for key, values in grouped.items() if {label for _, label in values} == {0, 1}}


def apply_collision_mask(rows: list[dict[str, Any]], collisions: dict[tuple[str, str], list[tuple[int, int]]], kind: str) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    for (input_digest, axis), members in collisions.items():
        changed = []
        for row_index, old_label in members:
            row = rows[row_index]
            row["labels"][axis] = None
            row["label_mask"][axis] = False
            row["label_basis"][axis] = "unknown_conflicting_reviewer_labels"
            row.setdefault("label_candidates", {})[axis] = [None]
            changed.append({"canonical_id": row["canonical_id"], "old_label": old_label})
        ledger.append({
            "kind": kind,
            "input_sha256": input_digest,
            "axis": axis,
            "resolution": "all_conflicting_non_null_labels_masked_unknown",
            "members": changed,
        })
    return ledger


def build_scorer_rows(sft_rows: list[dict[str, Any]], reviews: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    scorer_rows: list[dict[str, Any]] = []
    for source in sft_rows:
        metadata, question = source["metadata"], source["input"]["question"]
        canonical_id = metadata["canonical_id"]
        spans = metadata.get("span_supervision", [])
        texts = {"clean": source["target"], "candidate": source["input"]["corrupted_response"]}
        grouped: dict[tuple[str, str, int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
        for span in spans:
            key = exact_span_key(span)
            side, start, end, text, source_hash = key[1:]
            if texts[side][start:end] != text or hashlib.sha256(texts[side].encode()).hexdigest() != source_hash:
                raise RuntimeError(f"Source span integrity failure: {canonical_id}/{side}")
            grouped[key].append(span)
        annotations: dict[tuple[str, str, int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
        for annotation in reviews[canonical_id].get("scorer_annotations", []):
            key = annotation_key(annotation, canonical_id)
            if key not in grouped:
                raise ValueError(f"Review annotation has no exact source_verified_span: {canonical_id}/{annotation['axis']}")
            if annotation["axis"] not in {record["axis"] for record in grouped[key]}:
                raise ValueError(f"Review annotation axis lacks matching source evidence: {canonical_id}/{annotation['axis']}")
            annotation_label(annotation)
            annotations[key].append(annotation)
        for key, source_records in grouped.items():
            _, side, start, end, text, source_hash = key
            labels = {axis: None for axis in AXES}
            label_candidates = {axis: [] for axis in AXES}
            basis = {axis: "unknown_no_reviewer_local_annotation" for axis in AXES}
            for annotation in annotations.get(key, []):
                axis, label = annotation["axis"], annotation_label(annotation)
                if label is not None:
                    label_candidates[axis].append(label)
                    if labels[axis] is not None and labels[axis] != label:
                        basis[axis] = "pending_conflicting_reviewer_labels"
                    labels[axis] = label
                    basis[axis] = f"reviewer_{annotation['scope']}_explicit_label_{label}"
            scorer_rows.append({
                "canonical_id": canonical_id,
                "question": question,
                "span": text,
                "span_side": side,
                "span_start": start,
                "span_end": end,
                "span_source_sha256": source_hash,
                "split": metadata["split"],
                "labels": labels,
                "label_mask": {axis: labels[axis] is not None for axis in AXES},
                "label_basis": basis,
                "label_candidates": label_candidates,
                "source_records": source_records,
                "review_annotations": annotations.get(key, []),
                "question_normalized_sha256": metadata["question_normalized_sha256"],
                "duplicate_cluster_id": metadata["duplicate_cluster_id"],
                "source_group_id": metadata["source_group_id"],
            })
    return scorer_rows


def legacy_ideal_safe(axis: str, score: dict[str, Any]) -> bool:
    if axis == "medical_boundary":
        return score.get("violation") is False
    if axis == "toxicity_or_harm":
        return score.get("score") == 1
    return score.get("score") == (4 if axis == "factual_consistency" else 5)


def legacy_auto_scorer_rows(sft_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute the pre-repair labels solely for collision accounting.

    These rows are never exported as supervision.  This makes the reported
    collision count an actual calculation rather than a copied audit number.
    """
    rows: list[dict[str, Any]] = []
    for source in sft_rows:
        meta = source["metadata"]
        texts = {"clean": source["target"], "candidate": source["input"]["corrupted_response"]}
        grouped: dict[tuple[str, int, int, str], set[str]] = defaultdict(set)
        for record in meta.get("span_supervision", []):
            side, start, end, text = record["side"], int(record["start"]), int(record["end"]), record["text"]
            if texts[side][start:end] != text:
                raise RuntimeError(f"Legacy collision audit saw invalid offset: {meta['canonical_id']}")
            grouped[(side, start, end, text)].add(record["axis"])
        scores = {"clean": meta["paired_qc"]["clean_scores"], "candidate": meta["paired_qc"]["candidate_scores"]}
        realized = set(meta["realized_axes"])
        for (side, start, end, text), axes in grouped.items():
            labels = {}
            for axis in AXES:
                if side == "candidate" and axis in realized and axis in axes:
                    labels[axis] = 1
                elif axis in axes and legacy_ideal_safe(axis, scores[side][axis]):
                    labels[axis] = 0
                else:
                    labels[axis] = None
            rows.append({"canonical_id": meta["canonical_id"], "question": source["input"]["question"],
                         "span": text, "span_side": side, "span_start": start, "span_end": end,
                         "split": meta["split"], "labels": labels})
    return rows


def make_legacy_collision_ledger(rows: list[dict[str, Any]], collisions: dict[tuple[str, str], list[tuple[int, int]],], resolution: str) -> list[dict[str, Any]]:
    return [{
        "kind": "legacy_auto_span_label_collision",
        "input_sha256": input_digest,
        "axis": axis,
        "resolution": resolution,
        "members": [{"canonical_id": rows[index]["canonical_id"], "split": rows[index]["split"],
                     "side": rows[index]["span_side"], "start": rows[index]["span_start"],
                     "end": rows[index]["span_end"], "old_label": label}
                    for index, label in members],
    } for (input_digest, axis), members in collisions.items()]


def tokenizer_collision_keys(tokenizer_dir: Path, rows: list[dict[str, Any]], max_length: int) -> tuple[list[str], dict[str, Any]]:
    from transformers import AutoTokenizer  # Import/load only when explicitly requested.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    keys = []
    truncations = 0
    for row in rows:
        full = tokenizer(scorer_input_text(row["question"], row["span"]), truncation=False)["input_ids"]
        encoded = tokenizer(scorer_input_text(row["question"], row["span"]), truncation=True, max_length=max_length)["input_ids"]
        truncations += int(len(full) > len(encoded))
        keys.append(hashlib.sha256(json.dumps(encoded, separators=(",", ":")).encode()).hexdigest())
    return keys, {"status": "completed_local_files_only", "tokenizer_dir": str(tokenizer_dir),
                  "max_length": max_length, "rows": len(rows), "truncated_rows": truncations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", default=str(DEFAULT_PRODUCTION))
    parser.add_argument("--review-ledger", help="Existing normalized review JSONL; no inferred decisions are accepted")
    parser.add_argument("--reviewed-clean-cases", help="Supplied reviewed_clean_cases.jsonl")
    parser.add_argument("--clean-review-decisions", help="Supplied clean_review_decisions.csv")
    parser.add_argument("--recommended-hold-ids", help="Supplied recommended_clean_hold_ids.txt")
    parser.add_argument("--collision-audit", help="Optional supplied legacy collision audit JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-dir", help="Optional existing local scorer tokenizer; never downloaded")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    production, output = Path(args.production_dir).resolve(), Path(args.output_dir).resolve()
    normalized = bool(args.review_ledger)
    audit_paths = (args.reviewed_clean_cases, args.clean_review_decisions, args.recommended_hold_ids)
    if normalized == all(audit_paths) or (not normalized and not all(audit_paths)):
        raise ValueError("Provide exactly --review-ledger or all three supplied clean-review artifacts")
    ledger_path = Path(args.review_ledger).resolve() if args.review_ledger else None
    if ledger_path is not None and not ledger_path.is_file():
        raise FileNotFoundError(f"Required existing review ledger is absent: {ledger_path}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite corrected export: {output}")

    required = ("accepted.jsonl", "train_sft.jsonl", "train_dpo.jsonl", "valid_sft.jsonl", "valid_dpo.jsonl", "production_manifest.json")
    source_hashes = {name: sha256(production / name) for name in required}
    accepted = read_jsonl(production / "accepted.jsonl")
    accepted_by_id = {row["canonical_id"]: row for row in accepted}
    if len(accepted_by_id) != len(accepted):
        raise RuntimeError("Production accepted IDs are not unique")
    if ledger_path is not None:
        reviews = require_review_ledger(read_jsonl(ledger_path), set(accepted_by_id))
        review_hashes = {"normalized_review_ledger.jsonl": sha256(ledger_path)}
    else:
        reviews, review_hashes = audit_review_ledger(
            Path(args.reviewed_clean_cases).resolve(), Path(args.clean_review_decisions).resolve(),
            Path(args.recommended_hold_ids).resolve(), accepted_by_id,
        )
    decisions = []
    for canonical_id in sorted(accepted_by_id):
        row = reviews.get(canonical_id)
        if row is None:
            decisions.append({"canonical_id": canonical_id, "clean_disposition": "unresolved", "decision_source": "missing_review_record", "decision_evidence": []})
        else:
            decisions.append({k: row[k] for k in ("canonical_id", "clean_disposition", "decision_source", "decision_evidence", "clean_review") if k in row})
    retained_ids = {row["canonical_id"] for row in decisions if row["clean_disposition"] == "retained"}
    held = [row for row in decisions if row["clean_disposition"] == "held"]
    unresolved = [row for row in decisions if row["clean_disposition"] == "unresolved"]

    source_sft = {split: read_jsonl(production / f"{split}_sft.jsonl") for split in ("train", "valid")}
    source_dpo = {split: read_jsonl(production / f"{split}_dpo.jsonl") for split in ("train", "valid")}
    corrected_sft, corrected_dpo, router_rows = {}, {}, {}
    for split in ("train", "valid"):
        sft = [row for row in source_sft[split] if row["metadata"]["canonical_id"] in retained_ids]
        dpo = [row for row in source_dpo[split] if row["metadata"]["canonical_id"] in retained_ids]
        if [row["metadata"]["canonical_id"] for row in sft] != [row["metadata"]["canonical_id"] for row in dpo]:
            raise RuntimeError(f"SFT/DPO membership or order mismatch: {split}")
        for s_row, d_row in zip(sft, dpo, strict=True):
            if s_row["input"] != d_row["input"] or s_row["target"] != d_row["chosen"] or s_row["input"]["corrupted_response"] != d_row["rejected"]:
                raise RuntimeError(f"SFT/DPO pair contract failure: {s_row['metadata']['canonical_id']}")
        corrected_sft[split] = [
            {**row, "metadata": {**row["metadata"], "clean_review": reviews[row["metadata"]["canonical_id"]].get("clean_review", {})}}
            for row in sft
        ]
        corrected_dpo[split] = [
            {**row, "metadata": {**row["metadata"], "clean_review": reviews[row["metadata"]["canonical_id"]].get("clean_review", {})}}
            for row in dpo
        ]
        router_rows[split] = [{
            "canonical_id": row["metadata"]["canonical_id"], "question": row["input"]["question"],
            "candidate_response": row["input"]["corrupted_response"],
            "labels": {axis: int(axis in row["metadata"]["realized_axes"]) for axis in AXES},
            "split": split, "question_normalized_sha256": row["metadata"]["question_normalized_sha256"],
            "duplicate_cluster_id": row["metadata"]["duplicate_cluster_id"], "source_group_id": row["metadata"]["source_group_id"],
        } for row in corrected_sft[split]]

    scorer_rows = build_scorer_rows(corrected_sft["train"] + corrected_sft["valid"], reviews)
    reviewed_source_sft = [
        row for row in source_sft["train"] + source_sft["valid"]
        if row["metadata"]["canonical_id"] in reviews
    ]
    all_review_scorer_rows = build_scorer_rows(reviewed_source_sft, reviews)
    pre_filter = find_non_null_collisions(all_review_scorer_rows) if all_review_scorer_rows else {}
    pre_mask = find_non_null_collisions(scorer_rows)
    collision_ledger = apply_collision_mask(scorer_rows, pre_mask, "serialized_input_axis_conflict")
    tokenizer_audit: dict[str, Any] = {"status": "not_run_no_explicit_cached_tokenizer"}
    if args.tokenizer_dir:
        token_keys, tokenizer_audit = tokenizer_collision_keys(Path(args.tokenizer_dir).resolve(), scorer_rows, args.max_length)
        token_collisions = find_non_null_collisions(scorer_rows, token_keys)
        collision_ledger.extend(apply_collision_mask(scorer_rows, token_collisions, "tokenized_input_axis_conflict"))
    remaining = find_non_null_collisions(scorer_rows)
    if remaining:
        raise RuntimeError(f"Unresolved non-null scorer collisions remain: {len(remaining)}")

    legacy_all = legacy_auto_scorer_rows(source_sft["train"] + source_sft["valid"])
    legacy_retained = legacy_auto_scorer_rows(corrected_sft["train"] + corrected_sft["valid"])
    legacy_all_collisions, legacy_retained_collisions = find_non_null_collisions(legacy_all), find_non_null_collisions(legacy_retained)
    legacy_collision_records = make_legacy_collision_ledger(
        legacy_all, legacy_all_collisions,
        "removed_by_clean_exclusion" if not legacy_retained_collisions else "masked_no_reviewer_local_scope",
    )
    supplied_collision_audit = None
    if args.collision_audit:
        supplied_path = Path(args.collision_audit).resolve()
        supplied_collision_audit = {"path": str(supplied_path), "sha256": sha256(supplied_path), "groups": len(json.loads(supplied_path.read_text(encoding="utf-8")))}
        if supplied_collision_audit["groups"] != len(legacy_all_collisions):
            raise RuntimeError("Supplied collision audit group count differs from recomputed legacy collision count")

    output.mkdir(parents=True)
    outputs: dict[str, list[dict[str, Any]]] = {
        "retained_canonical_ids.jsonl": [row for row in decisions if row["clean_disposition"] == "retained"],
        "held_canonical_ids.jsonl": held,
        "unresolved_canonical_ids.jsonl": unresolved,
        "review_ledger.jsonl": decisions,
        "train_sft.jsonl": corrected_sft["train"], "train_dpo.jsonl": corrected_dpo["train"],
        "valid_sft.jsonl": corrected_sft["valid"], "valid_dpo.jsonl": corrected_dpo["valid"],
        "router_train.jsonl": router_rows["train"], "router_valid.jsonl": router_rows["valid"],
        "scorer_train_verified_spans.jsonl": [row for row in scorer_rows if row["split"] == "train"],
        "scorer_valid_verified_spans.jsonl": [row for row in scorer_rows if row["split"] == "valid"],
        "scorer_collision_ledger.jsonl": collision_ledger,
        "legacy_scorer_collision_ledger.jsonl": legacy_collision_records,
    }
    for name, rows in outputs.items():
        write_jsonl(output / name, rows)
    label_counts = {split: {kind: dict(Counter(axis for row in scorer_rows if row["split"] == split for axis, value in row["labels"].items() if value == target)) for kind, target in (("positive", 1), ("negative", 0), ("unknown", None))} for split in ("train", "valid")}
    output_hashes = {name: sha256(output / name) for name in outputs}
    if source_hashes != {name: sha256(production / name) for name in required}:
        raise RuntimeError("Production artifact changed while corrected export was built")
    manifest = {
        "version": VERSION, "status": "provisional_corrected_export_not_training_ready",
        "review_policy": "only explicit retained review decisions are exported; held and unresolved are excluded",
        "production_dir": str(production), "production_artifact_sha256": source_hashes,
        "review_artifact_sha256": review_hashes,
        "counts": {"production_accepted": len(accepted), "retained": len(retained_ids), "held": len(held), "unresolved": len(unresolved)},
        "scorer": {"scope_policy": "only explicit local_defect=1 or positive_support=0 annotations; all other cells unknown",
                   "legacy_collisions_recomputed_before_clean_filter": len(legacy_all_collisions),
                   "legacy_collisions_removed_by_clean_exclusion": len(set(legacy_all_collisions) - set(legacy_retained_collisions)),
                   "legacy_collisions_remaining_after_clean_filter": len(legacy_retained_collisions),
                   "review_scope_collisions_before_clean_filter": len(pre_filter), "review_scope_collisions_removed_by_clean_exclusion": len(set(pre_filter) - set(pre_mask)),
                   "collisions_masked": len(collision_ledger), "remaining_non_null_collisions": 0,
                   "masked_cells": sum(len(item["members"]) for item in collision_ledger), "label_counts": label_counts,
                   "tokenizer_audit": tokenizer_audit, "supplied_collision_audit": supplied_collision_audit},
        "outputs": {name: {"rows": len(rows), "sha256": output_hashes[name]} for name, rows in outputs.items()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
