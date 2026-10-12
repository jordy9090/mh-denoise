#!/usr/bin/env python3
"""Build a separate reuse-adjudicated export for the immutable 652 accepted pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_provisional_production_corrected_export import (
    find_non_null_collisions as find_legacy_non_null_collisions,
    legacy_auto_scorer_rows,
)
from fullpaper_acl_pipeline import read_jsonl, sha256_file
from fullpaper_risk_contract import AXES
from fullpaper_scorer_export_contract import (
    assert_no_non_null_collisions,
    audit_tokenization_and_mask,
    deduplicate_same_inputs,
    find_non_null_collisions,
    mask_collisions,
)
from rejudge_production_reuse_subset import adjudicate_grade
from source_integrity_contract import surface_flags


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRODUCTION = ROOT / "data/fullpaper_acl_pipeline/production_run_train1000_valid200_local_qwen35_27b_v3r1_20260910"
DEFAULT_PRIOR = Path(str(DEFAULT_PRODUCTION) + "_provisional_corrected_v2_20260926/review_ledger.jsonl")
DEFAULT_CODEX = ROOT / "data/fullpaper_acl_pipeline/codex_clean_candidate_review_batch20_v2_20260926/cumulative_ai_review_ledger.jsonl"
VERSION = "production-reuse-adjudicated-export-v2-20261001"
REVIEWED_VERSION = "production-reuse-reviewed-export-v2-20261004"
CATEGORIES = (
    "reusable", "exclude_original_answer", "regenerate_candidate", "unresolved",
)


def rows(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_scoped_review_identity(
    review: dict[str, Any], accepted_by_id: dict[str, dict[str, Any]],
) -> None:
    canonical_id = review["canonical_id"]
    if canonical_id not in accepted_by_id:
        raise RuntimeError(f"Scoped review ID is absent from production: {canonical_id}")
    side = review["side"]
    if side not in {"clean", "candidate"}:
        raise RuntimeError(f"Invalid scoped review side: {canonical_id}: {side}")
    source = accepted_by_id[canonical_id]
    text = source["clean_response" if side == "clean" else "corrupted_response"]
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != review["source_response_sha256"]:
        raise RuntimeError(f"Scoped review response hash mismatch: {canonical_id}: {side}")
    span = review.get("span", "")
    if span and span not in text:
        raise RuntimeError(f"Scoped review exact span mismatch: {canonical_id}: {side}")


def apply_pair_review_overrides(
    decisions: list[dict[str, Any]], review_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    exclusions = {
        row["canonical_id"]: row for row in review_rows
        if row["action"] == "exclude_pair"
    }
    output = []
    for decision in decisions:
        review = exclusions.get(decision["canonical_id"])
        if review is None:
            output.append(decision)
            continue
        output.append({
            **decision,
            "reuse_disposition": "exclude_original_answer",
            "clean_assessment": "issue",
            "pair_assessment": "not_applicable",
            "decision_source": review["review_source"],
            "decision_evidence": [review["reason"]],
            "superseded_decision": {
                "reuse_disposition": decision["reuse_disposition"],
                "clean_assessment": decision["clean_assessment"],
                "pair_assessment": decision["pair_assessment"],
                "decision_source": decision["decision_source"],
                "decision_evidence": decision["decision_evidence"],
            },
        })
    return output


def apply_annotation_review_overrides(
    annotation_ledger: list[dict[str, Any]], review_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    retained = list(annotation_ledger)
    masked: list[dict[str, Any]] = []
    applications: list[dict[str, Any]] = []
    for review in review_rows:
        if review["action"] != "mask_scorer_label":
            applications.append({**review, "application_status": "recorded_no_label_change"})
            continue
        matches = [
            item for item in retained
            if item["canonical_id"] == review["canonical_id"]
            and item["side"] == review["side"]
            and item["axis"] == review["axis"]
            and item["text"] == review["span"]
            and item["source_sha256"] == review["source_response_sha256"]
            and item["label"] == review["previous_label"]
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Scoped label review expected one annotation, found {len(matches)}: "
                f"{review['canonical_id']} / {review['axis']}"
            )
        target = matches[0]
        retained.remove(target)
        masked.append({
            **target,
            "resolution": "unknown_scoped_case_review",
            "review_source": review["review_source"],
            "review_reason": review["reason"],
            "previous_label": target["label"],
            "new_label": None,
            "label_mask": False,
        })
        applications.append({**review, "application_status": "masked_to_unknown"})
    return retained, masked, applications


def classify_one(
    row: dict[str, Any], *, prior_hold: bool, codex: dict[str, Any] | None,
    warning: bool, local: dict[str, Any] | None,
) -> dict[str, Any]:
    # A review of the previous candidate cannot adjudicate a newly generated
    # candidate.  The replacement's complete current paired QC is the valid
    # content-bound judgment; the older Codex record remains in provenance.
    if local is not None and local.get("replacement_generation_run"):
        codex = None
    clean_surface = surface_flags(row["question"], row["clean_response"], "clean")
    candidate_surface = surface_flags(row["question"], row["corrupted_response"], "candidate")
    base = {
        "canonical_id": row["canonical_id"], "split": row["split"],
        "source": row["source"], "intended_axes": row["intended_axes"],
        "legacy_realized_axes": row["realized_axes"],
        "clean_surface_findings": clean_surface,
        "candidate_surface_findings": candidate_surface,
    }
    if prior_hold:
        return {**base, "reuse_disposition": "exclude_original_answer",
                "clean_assessment": "issue", "pair_assessment": "not_applicable",
                "decision_source": "existing_explicit_clean_hold",
                "decision_evidence": ["Existing evidence-backed clean hold is preserved."]}
    if codex is not None:
        review = codex["ai_review"]
        clean = review["clean_integrity"]
        if clean["status"] == "hold":
            return {**base, "reuse_disposition": "exclude_original_answer",
                    "clean_assessment": "issue", "pair_assessment": "not_applicable",
                    "decision_source": review["review_source"],
                    "decision_evidence": clean.get("issue_quotes", []) + [clean["reason_ko"]]}
        proposal = review["proposal"]
        if proposal == "use":
            disposition, pair = "reusable", "usable"
        elif proposal == "hold":
            disposition, pair = "regenerate_candidate", "generation_failed_review"
        else:
            disposition, pair = "unresolved", "ambiguous_review"
        return {**base, "reuse_disposition": disposition,
                "clean_assessment": "suitable", "pair_assessment": pair,
                "decision_source": review["review_source"],
                "decision_evidence": [clean["reason_ko"], review["intended_defect"]["reason_ko"], review["summary_ko"]["candidate"]]}
    if clean_surface:
        return {**base, "reuse_disposition": "exclude_original_answer",
                "clean_assessment": "issue", "pair_assessment": "not_applicable",
                "decision_source": "current_deterministic_source_integrity_contract",
                "decision_evidence": [item["text"] for item in clean_surface]}
    if candidate_surface:
        return {**base, "reuse_disposition": "regenerate_candidate",
                "clean_assessment": "suitable", "pair_assessment": "candidate_integrity_failure",
                "decision_source": "current_deterministic_source_integrity_contract",
                "decision_evidence": [item["text"] for item in candidate_surface]}
    if local is not None:
        local_decision_source = (
            "current_v6_local_paired_qc_after_candidate_regeneration"
            if local.get("replacement_generation_run")
            else "current_v6_local_paired_qc"
        )
        if local.get("status") != "complete":
            return {**base, "reuse_disposition": "unresolved",
                    "clean_assessment": "unresolved", "pair_assessment": "unresolved",
                    "decision_source": "current_v6_local_paired_qc_technical_failure",
                    "decision_evidence": [str(local.get("failure_reason") or "technical failure")]}
        if local["clean_status"] == "original_answer_issue":
            return {**base, "reuse_disposition": "exclude_original_answer",
                    "clean_assessment": "issue", "pair_assessment": local["pair_status"],
                    "decision_source": local_decision_source,
                    "decision_evidence": [item.get("reason", "") for item in local.get("clean_signals", [])]}
        if local["clean_status"] == "unresolved":
            return {**base, "reuse_disposition": "unresolved",
                    "clean_assessment": "unresolved", "pair_assessment": local.get("pair_status", "unresolved"),
                    "decision_source": local_decision_source + "_internal_clean_conflict",
                    "decision_evidence": [item.get("reason", "") for item in local.get("clean_signals", [])]}
        mapping = {
            "reusable": ("reusable", "usable"),
            "regenerate_candidate": ("regenerate_candidate", "generation_failed_current_qc"),
            "unresolved": ("unresolved", "unresolved_current_qc"),
        }
        disposition, pair = mapping[local["pair_status"]]
        return {**base, "reuse_disposition": disposition,
                "clean_assessment": "suitable", "pair_assessment": pair,
                "decision_source": local_decision_source,
                "decision_evidence": [str(local.get("pair_reason") or "current pair QC passed")]}
    if warning:
        return {**base, "reuse_disposition": "unresolved",
                "clean_assessment": "unresolved", "pair_assessment": "unresolved",
                "decision_source": "missing_required_warning_rejudge",
                "decision_evidence": ["Warning row has no terminal current-QC result."]}
    return {**base, "reuse_disposition": "unresolved",
            "clean_assessment": "not_currently_revalidated",
            "pair_assessment": "legacy_reuse_candidate",
            "decision_source": "legacy_accepted_pair_plus_no_clean_warning",
            "decision_evidence": [
                "Legacy accepted status and absence of the historical clean-warning flag make this a reuse candidate, "
                "but no current complete-context v6 re-review is available."
            ]}


def exact_occurrences(text: str, span: str) -> list[int]:
    starts, cursor = [], 0
    while span:
        found = text.find(span, cursor)
        if found < 0:
            break
        starts.append(found)
        cursor = found + 1
    return starts


def codex_annotations(review: dict[str, Any], source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted, excluded = [], []
    texts = {"clean": source["target"], "candidate": source["input"]["corrupted_response"]}
    old_spans = source["metadata"].get("span_supervision", [])
    for item in review["ai_review"].get("marked_candidate_span_reviews", []):
        label = item.get("proposed_binary_label")
        if label not in (0, 1):
            excluded.append({"canonical_id": review["canonical_id"], **item,
                             "resolution": "unknown_no_binary_local_review"})
            continue
        side, axis, span = item["side"], item["axis"], item["text"]
        matches = [record for record in old_spans if record["side"] == side and record["axis"] == axis and record["text"] == span]
        if matches:
            start, end = int(matches[0]["start"]), int(matches[0]["end"])
        else:
            starts = exact_occurrences(texts[side], span)
            if len(starts) != 1:
                excluded.append({"canonical_id": review["canonical_id"], **item,
                                 "resolution": "unknown_non_unique_or_missing_exact_span"})
                continue
            start, end = starts[0], starts[0] + len(span)
        if texts[side][start:end] != span:
            raise RuntimeError("Codex annotation exact-span integrity failure")
        accepted.append({
            "canonical_id": review["canonical_id"], "side": side, "axis": axis,
            "start": start, "end": end, "text": span,
            "source_sha256": hashlib.sha256(texts[side].encode()).hexdigest(),
            "scope": "local_defect" if label == 1 else "positive_support",
            "label": label, "annotation_source": review["ai_review"]["review_source"],
            "annotation_evidence": item["reason_ko"],
        })
    return accepted, excluded


def local_annotations(local: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in local.get("grade", {}).get("local_supervision", []):
        if item.get("scope") not in {"local_defect", "local_support"} or item.get("label") not in (0, 1):
            continue
        result.append({
            "canonical_id": local["canonical_id"], "side": item["side"],
            "axis": item["axis"], "start": item["start"], "end": item["end"],
            "text": item["text"], "source_sha256": item["source_sha256"],
            "scope": "local_defect" if item["label"] == 1 else "positive_support",
            "label": item["label"], "annotation_source": item["annotation_source"],
            "annotation_evidence": item["reason"],
        })
    return result


def router_labels_for_pair(
    source: dict[str, Any], *, local: dict[str, Any] | None,
    codex: dict[str, Any] | None,
) -> tuple[dict[str, int | None], dict[str, bool], dict[str, Any]]:
    """Use the latest observed axes and never invent a zero for an unjudged axis."""
    legacy = list(source["metadata"].get("realized_axes", []))
    if local is not None and local.get("status") == "complete":
        observed = set(local.get("current_realized_axes", []))
        labels = {axis: int(axis in observed) for axis in AXES}
        return labels, {axis: True for axis in AXES}, {
            "source": "current_local_paired_qc", "version": local.get("grade", {}).get(
                "local_supervision", [{}]
            )[0].get("annotation_source") if local.get("grade", {}).get("local_supervision") else "local-qwen35-27b-paired-qc-v6-20260926",
            "legacy_realized_axes": legacy, "latest_realized_axes": sorted(observed),
        }
    if codex is not None:
        review = codex["ai_review"]
        positive = set(review.get("intended_defect", {}).get("axes_present", []))
        positive.update(review.get("collateral_damage", {}).get("axes", []))
        negative = set(review.get("intended_defect", {}).get("axes_not_supported", []))
        labels = {axis: 1 if axis in positive else 0 if axis in negative else None for axis in AXES}
        return labels, {axis: labels[axis] is not None for axis in AXES}, {
            "source": review.get("review_source"), "version": codex.get("schema_version"),
            "legacy_realized_axes": legacy, "latest_realized_axes": sorted(positive),
            "explicit_negative_axes": sorted(negative),
        }
    return {axis: None for axis in AXES}, {axis: False for axis in AXES}, {
        "source": "no_current_axis_judgment", "version": None,
        "legacy_realized_axes": legacy, "latest_realized_axes": [],
    }


def raw_scorer_rows(
    source_rows: list[dict[str, Any]], annotations_by_id: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for source in source_rows:
        meta, question = source["metadata"], source["input"]["question"]
        canonical_id = meta["canonical_id"]
        texts = {"clean": source["target"], "candidate": source["input"]["corrupted_response"]}
        grouped: dict[tuple[str, int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in meta.get("span_supervision", []):
            side, start, end, text = record["side"], int(record["start"]), int(record["end"]), record["text"]
            digest = hashlib.sha256(texts[side].encode()).hexdigest()
            if texts[side][start:end] != text or record["source_sha256"] != digest:
                raise RuntimeError(f"Legacy source span integrity failure: {canonical_id}")
            grouped[(side, start, end, text, digest)].append(record)
        for annotation in annotations_by_id.get(canonical_id, []):
            key = (annotation["side"], int(annotation["start"]), int(annotation["end"]), annotation["text"], annotation["source_sha256"])
            side, start, end, text, digest = key
            if texts[side][start:end] != text or hashlib.sha256(texts[side].encode()).hexdigest() != digest:
                raise RuntimeError(f"Reviewed source span integrity failure: {canonical_id}")
            grouped.setdefault(key, [])
        for (side, start, end, text, digest), source_records in grouped.items():
            annotations = [item for item in annotations_by_id.get(canonical_id, [])
                           if (item["side"], item["start"], item["end"], item["text"], item["source_sha256"])
                           == (side, start, end, text, digest)]
            candidates = {axis: [item["label"] for item in annotations if item["axis"] == axis] for axis in AXES}
            labels = {}
            basis = {}
            for axis in AXES:
                values = set(candidates[axis])
                labels[axis] = next(iter(values)) if len(values) == 1 else None
                basis[axis] = (
                    f"reviewed_explicit_label_{labels[axis]}" if len(values) == 1
                    else "pending_conflicting_reviewed_labels" if len(values) > 1
                    else "unknown_no_local_review"
                )
            output.append({
                "canonical_id": canonical_id, "question": question, "span": text,
                "span_side": side, "span_start": start, "span_end": end,
                "span_source_sha256": digest, "split": meta["split"],
                "labels": labels, "label_mask": {axis: labels[axis] is not None for axis in AXES},
                "label_basis": basis, "label_candidates": candidates,
                "source_records": source_records, "review_annotations": annotations,
                "question_normalized_sha256": meta["question_normalized_sha256"],
                "duplicate_cluster_id": meta["duplicate_cluster_id"],
                "source_group_id": meta["source_group_id"],
            })
    return output


def deduplicate_scorer_rows(values: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    deduped, ledger = deduplicate_same_inputs(values)
    for row in deduped:
        row.pop("label_candidates", None)
        row["source_provenance"] = [{
            "canonical_id": item.get("canonical_id"),
            "side": item.get("span_side"),
            "start": item.get("span_start"),
            "end": item.get("span_end"),
            "source_sha256": item.get("span_source_sha256"),
        } for item in row.get("provenance_members", [])]
    return deduped, ledger


def label_counts(values: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    result = {}
    for split in ("train", "valid"):
        split_rows = [row for row in values if row["split"] == split]
        result[split] = {
            axis: {
                "0": sum(row["labels"][axis] == 0 for row in split_rows),
                "1": sum(row["labels"][axis] == 1 for row in split_rows),
                "unknown": sum(row["labels"][axis] is None for row in split_rows),
            } for axis in AXES
        }
    return result


def representative_rows(decisions: list[dict[str, Any]], accepted_by_id: dict[str, dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    output = []
    for category in CATEGORIES:
        choices = [row for row in decisions if row["reuse_disposition"] == category]
        picked, seen = [], set()
        for decision in choices:
            source = accepted_by_id[decision["canonical_id"]]
            key = (source["source"], source["split"])
            if key in seen and len(picked) < min(limit, len(choices)):
                continue
            seen.add(key)
            picked.append(decision)
            if len(picked) == limit:
                break
        for decision in choices:
            if len(picked) == limit:
                break
            if decision not in picked:
                picked.append(decision)
        for decision in picked:
            source = accepted_by_id[decision["canonical_id"]]
            output.append({**decision, "question": source["question"],
                           "clean_response": source["clean_response"],
                           "corrupted_response": source["corrupted_response"]})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", default=str(DEFAULT_PRODUCTION))
    parser.add_argument("--prior-review-ledger", default=str(DEFAULT_PRIOR))
    parser.add_argument("--codex-review-ledger", default=str(DEFAULT_CODEX))
    parser.add_argument("--local-rejudge-checkpoint", required=True)
    parser.add_argument(
        "--replacement-production-dir",
        help="Optional candidate-regeneration production run; only its accepted replacements are overlaid.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--scoped-review-ledger",
        help="Optional hash-bound pair/label decisions applied after the preserved QC adjudication.",
    )
    parser.add_argument(
        "--first-experiment-approved", action="store_true",
        help="Record explicit authorization for this frozen export to feed the first experiment.",
    )
    args = parser.parse_args()
    production, output = Path(args.production_dir).resolve(), Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite export: {output}")
    output.mkdir(parents=True)

    required = ("accepted.jsonl", "train_sft.jsonl", "train_dpo.jsonl", "valid_sft.jsonl", "valid_dpo.jsonl", "production_manifest.json")
    source_hashes = {name: sha256_file(production / name) for name in required}
    immutable_accepted = rows(production / "accepted.jsonl")
    immutable_accepted_by_id = {row["canonical_id"]: row for row in immutable_accepted}
    if len(immutable_accepted_by_id) != 652:
        raise RuntimeError("Expected exactly 652 unique production accepted rows")
    prior_path, codex_path = Path(args.prior_review_ledger).resolve(), Path(args.codex_review_ledger).resolve()
    prior = {row["canonical_id"]: row for row in rows(prior_path)}
    codex = {row["canonical_id"]: row for row in rows(codex_path)}
    local_path = Path(args.local_rejudge_checkpoint).resolve()
    local_rows = rows(local_path)
    local = {row["canonical_id"]: row for row in local_rows}
    if len(local) != len(local_rows):
        raise RuntimeError("Local rejudge checkpoint contains duplicate IDs")
    replacement_dir = (
        Path(args.replacement_production_dir).resolve()
        if args.replacement_production_dir else None
    )
    replacement_by_id: dict[str, dict[str, Any]] = {}
    replacement_hashes: dict[str, str] = {}
    if replacement_dir is not None:
        replacement_required = (
            "accepted.jsonl", "train_sft.jsonl", "train_dpo.jsonl",
            "valid_sft.jsonl", "valid_dpo.jsonl", "production_manifest.json",
        )
        replacement_hashes = {
            name: sha256_file(replacement_dir / name) for name in replacement_required
        }
        replacement_rows = rows(replacement_dir / "accepted.jsonl")
        replacement_by_id = {row["canonical_id"]: row for row in replacement_rows}
        if len(replacement_by_id) != len(replacement_rows):
            raise RuntimeError("Replacement production accepted rows contain duplicate IDs")
        unknown = set(replacement_by_id) - set(immutable_accepted_by_id)
        if unknown:
            raise RuntimeError(f"Replacement run contains IDs outside production: {sorted(unknown)[:5]}")
        for canonical_id, replacement in replacement_by_id.items():
            original = immutable_accepted_by_id[canonical_id]
            for key in (
                "question", "clean_response", "split", "source",
                "question_normalized_sha256", "duplicate_cluster_id", "source_group_id",
            ):
                if replacement.get(key) != original.get(key):
                    raise RuntimeError(f"Replacement changed immutable {key}: {canonical_id}")
            if replacement.get("intended_axes") != original.get("intended_axes"):
                raise RuntimeError(f"Replacement changed intended axes: {canonical_id}")
            replacement_decision = adjudicate_grade(replacement, replacement["final_grade"])
            if (
                replacement_decision["clean_status"] != "suitable"
                or replacement_decision["pair_status"] != "reusable"
            ):
                raise RuntimeError(
                    f"Replacement accepted row fails current adjudication: {canonical_id}: "
                    f"{replacement_decision}"
                )
            local[canonical_id] = {
                "canonical_id": canonical_id,
                "split": replacement["split"],
                "source": replacement["source"],
                "status": "complete",
                "intended_axes": replacement["intended_axes"],
                "legacy_realized_axes": original["realized_axes"],
                "current_realized_axes": replacement["realized_axes"],
                **replacement_decision,
                "grade": replacement["final_grade"],
                "replacement_generation_run": str(replacement_dir),
                "replacement_candidate_sha256": hashlib.sha256(
                    replacement["corrupted_response"].encode()
                ).hexdigest(),
            }
    accepted = [replacement_by_id.get(row["canonical_id"], row) for row in immutable_accepted]
    accepted_by_id = {row["canonical_id"]: row for row in accepted}
    scoped_review_path = Path(args.scoped_review_ledger).resolve() if args.scoped_review_ledger else None
    scoped_review_rows = rows(scoped_review_path) if scoped_review_path else []
    for review in scoped_review_rows:
        validate_scoped_review_identity(review, accepted_by_id)
    warning_ids = {row["canonical_id"] for row in rows(production / "qwen27_clean_target_suspicious.jsonl")}
    held_ids = {canonical_id for canonical_id, row in prior.items() if row["clean_disposition"] == "held"}

    decisions = [classify_one(
        row, prior_hold=row["canonical_id"] in held_ids,
        codex=codex.get(row["canonical_id"]), warning=row["canonical_id"] in warning_ids,
        local=local.get(row["canonical_id"]),
    ) for row in accepted]
    decisions = apply_pair_review_overrides(decisions, scoped_review_rows)
    if {row["canonical_id"] for row in decisions} != set(accepted_by_id):
        raise RuntimeError("Adjudication coverage mismatch")
    reusable_ids = {row["canonical_id"] for row in decisions if row["reuse_disposition"] == "reusable"}

    source_sft = {split: rows(production / f"{split}_sft.jsonl") for split in ("train", "valid")}
    source_dpo = {split: rows(production / f"{split}_dpo.jsonl") for split in ("train", "valid")}
    if replacement_dir is not None:
        for split in ("train", "valid"):
            replacement_sft = {
                row["metadata"]["canonical_id"]: row
                for row in rows(replacement_dir / f"{split}_sft.jsonl")
            }
            replacement_dpo = {
                row["metadata"]["canonical_id"]: row
                for row in rows(replacement_dir / f"{split}_dpo.jsonl")
            }
            expected = {
                canonical_id for canonical_id, row in replacement_by_id.items()
                if row["split"] == split
            }
            if set(replacement_sft) != expected or set(replacement_dpo) != expected:
                raise RuntimeError(f"Replacement SFT/DPO membership mismatch: {split}")
            source_sft[split] = [
                replacement_sft.get(row["metadata"]["canonical_id"], row)
                for row in source_sft[split]
            ]
            source_dpo[split] = [
                replacement_dpo.get(row["metadata"]["canonical_id"], row)
                for row in source_dpo[split]
            ]
    source_by_id = {row["metadata"]["canonical_id"]: row for split in ("train", "valid") for row in source_sft[split]}
    corrected_sft, corrected_dpo, router = {}, {}, {}
    router_axis_change_ledger: list[dict[str, Any]] = []
    for split in ("train", "valid"):
        sft = [row for row in source_sft[split] if row["metadata"]["canonical_id"] in reusable_ids]
        dpo = [row for row in source_dpo[split] if row["metadata"]["canonical_id"] in reusable_ids]
        if [row["metadata"]["canonical_id"] for row in sft] != [row["metadata"]["canonical_id"] for row in dpo]:
            raise RuntimeError(f"SFT/DPO order mismatch: {split}")
        decision_by_id = {row["canonical_id"]: row for row in decisions}
        corrected_sft[split] = [{**row, "metadata": {**row["metadata"], "reuse_adjudication": decision_by_id[row["metadata"]["canonical_id"]]}} for row in sft]
        corrected_dpo[split] = [{**row, "metadata": {**row["metadata"], "reuse_adjudication": decision_by_id[row["metadata"]["canonical_id"]]}} for row in dpo]
        router[split] = []
        for row in corrected_sft[split]:
            canonical_id = row["metadata"]["canonical_id"]
            labels, label_mask, judgment = router_labels_for_pair(
                row, local=local.get(canonical_id), codex=codex.get(canonical_id),
            )
            axis_metadata = {
                "legacy_realized_axes": list(row["metadata"].get("realized_axes", [])),
                "latest_router_labels": labels,
                "latest_router_label_mask": label_mask,
                "latest_axis_judgment": judgment,
                "realized_axes_field_status": "legacy_preserved_not_authoritative_for_router",
            }
            row["metadata"].update(axis_metadata)
            matching_dpo = next(
                item for item in corrected_dpo[split]
                if item["metadata"]["canonical_id"] == canonical_id
            )
            matching_dpo["metadata"].update(axis_metadata)
            router[split].append({
                "canonical_id": canonical_id,
                "question": row["input"]["question"],
                "candidate_response": row["input"]["corrupted_response"],
                "labels": labels,
                "label_mask": label_mask,
                "axis_judgment": judgment,
                "split": split,
                "question_normalized_sha256": row["metadata"]["question_normalized_sha256"],
                "duplicate_cluster_id": row["metadata"]["duplicate_cluster_id"],
                "source_group_id": row["metadata"]["source_group_id"],
            })
            legacy = set(judgment["legacy_realized_axes"])
            latest = set(judgment["latest_realized_axes"])
            if legacy != latest or not all(label_mask.values()):
                router_axis_change_ledger.append({
                    "canonical_id": canonical_id,
                    "split": split,
                    "legacy_realized_axes": sorted(legacy),
                    "latest_realized_axes": sorted(latest),
                    "changed_axes": sorted(legacy ^ latest),
                    "unknown_axes": sorted(axis for axis in AXES if not label_mask[axis]),
                    "judgment_source": judgment["source"],
                    "judgment_version": judgment["version"],
                })

    exported_ids = {
        split: {row["metadata"]["canonical_id"] for row in corrected_sft[split]}
        for split in ("train", "valid")
    }
    if exported_ids["train"] & exported_ids["valid"]:
        raise RuntimeError("Reuse export moved or duplicated an ID across splits")
    if exported_ids["train"] | exported_ids["valid"] != reusable_ids:
        raise RuntimeError("Reuse export membership differs from adjudication")
    original_sft_by_id = {
        row["metadata"]["canonical_id"]: row
        for split in ("train", "valid") for row in source_sft[split]
    }
    for split in ("train", "valid"):
        for row in corrected_sft[split]:
            canonical_id = row["metadata"]["canonical_id"]
            original = original_sft_by_id[canonical_id]
            if row["metadata"]["split"] != split:
                raise RuntimeError(f"Split metadata changed: {canonical_id}")
            if (row["input"] != original["input"] or row["target"] != original["target"]):
                raise RuntimeError(f"Question/response text changed: {canonical_id}")

    annotations_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    annotation_ledger, excluded_annotations = [], []
    for canonical_id in reusable_ids:
        accepted_annotations: list[dict[str, Any]] = []
        if canonical_id in local and local[canonical_id].get("replacement_generation_run"):
            accepted_annotations = local_annotations(local[canonical_id])
        elif canonical_id in codex:
            accepted_annotations, excluded = codex_annotations(codex[canonical_id], source_by_id[canonical_id])
            excluded_annotations.extend(excluded)
        elif canonical_id in local:
            accepted_annotations = local_annotations(local[canonical_id])
        annotations_by_id[canonical_id].extend(accepted_annotations)
        annotation_ledger.extend(accepted_annotations)

    annotation_ledger, review_masked_annotations, review_application_ledger = (
        apply_annotation_review_overrides(annotation_ledger, scoped_review_rows)
    )
    annotations_by_id = defaultdict(list)
    for annotation in annotation_ledger:
        annotations_by_id[annotation["canonical_id"]].append(annotation)
    excluded_annotations.extend(review_masked_annotations)

    reusable_sft = corrected_sft["train"] + corrected_sft["valid"]
    scorer_raw = raw_scorer_rows(reusable_sft, annotations_by_id)
    # Keep candidate-level annotations until this point so opposing reviewed
    # labels attached to one physical span are ledgered, not silently reduced
    # to an unknown cell by the row builder.
    string_collisions = find_legacy_non_null_collisions(scorer_raw)
    scorer_collision_ledger = mask_collisions(
        scorer_raw, string_collisions, stage="serialized_input"
    )
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    tokenizer_audit, truncation_ledger, token_ids, token_collision_ledger = (
        audit_tokenization_and_mask(
            scorer_raw, tokenizer, max_length=args.max_length
        )
    )
    tokenizer_audit.update({
        "status": "completed_local_files_only", "tokenizer_dir": str(tokenizer_dir),
    })
    scorer_collision_ledger.extend(token_collision_ledger)
    assert_no_non_null_collisions(scorer_raw)
    assert_no_non_null_collisions(scorer_raw, token_ids=token_ids)
    scorer_rows, dedup_ledger = deduplicate_scorer_rows(scorer_raw)
    assert_no_non_null_collisions(scorer_rows)
    trainable = [row for row in scorer_rows if any(value is not None for value in row["labels"].values())]

    legacy_all = legacy_auto_scorer_rows(source_sft["train"] + source_sft["valid"])
    clean_survivors = {row["canonical_id"] for row in decisions if row["reuse_disposition"] != "exclude_original_answer"}
    legacy_clean = [row for row in legacy_all if row["canonical_id"] in clean_survivors]
    legacy_reusable = [row for row in legacy_all if row["canonical_id"] in reusable_ids]
    legacy_all_c = find_legacy_non_null_collisions(legacy_all)
    legacy_clean_c = find_legacy_non_null_collisions(legacy_clean)
    legacy_reuse_c = find_legacy_non_null_collisions(legacy_reusable)
    legacy_collision_ledger = []
    for (input_digest, axis), members in legacy_all_c.items():
        key = (input_digest, axis)
        if key not in legacy_clean_c:
            resolution = "removed_by_original_answer_exclusion"
        elif key not in legacy_reuse_c:
            resolution = "removed_by_other_nonreuse_disposition"
        else:
            resolution = "unsupported_legacy_response_labels_not_carried_to_local_scorer"
        legacy_collision_ledger.append({
            "input_sha256": input_digest, "axis": axis,
            "original_labels": sorted({label for _, label in members}),
            "resolution": resolution,
            "members": [{
                "canonical_id": legacy_all[index]["canonical_id"],
                "split": legacy_all[index]["split"],
                "side": legacy_all[index]["span_side"],
                "span": legacy_all[index]["span"],
                "old_label": label,
            } for index, label in members],
        })

    proposed = {
        split: [{
            "id": row["metadata"]["canonical_id"],
            "canonical_id": row["metadata"]["canonical_id"],
            "question": row["input"]["question"],
            "unsafe_response": row["input"]["corrupted_response"],
            "safe_response": row["target"],
            "split": split,
            "source": row["metadata"]["source"],
            "source_component": row["metadata"]["source_component"],
            "intended_axes": row["metadata"]["intended_axes"],
            "legacy_realized_axes": row["metadata"].get("legacy_realized_axes", []),
            "latest_router_labels": row["metadata"]["latest_router_labels"],
            "latest_router_label_mask": row["metadata"]["latest_router_label_mask"],
            "verified_spans": row["metadata"].get("span_supervision", []),
            "question_normalized_sha256": row["metadata"]["question_normalized_sha256"],
            "duplicate_cluster_id": row["metadata"]["duplicate_cluster_id"],
            "source_group_id": row["metadata"]["source_group_id"],
            "metadata": row["metadata"],
        } for row in corrected_sft[split]]
        for split in ("train", "valid")
    }
    outputs: dict[str, list[dict[str, Any]]] = {
        "adjudication_ledger.jsonl": decisions,
        "reusable.jsonl": [row for row in decisions if row["reuse_disposition"] == "reusable"],
        "excluded_original_answer.jsonl": [row for row in decisions if row["reuse_disposition"] == "exclude_original_answer"],
        "regenerate_candidate.jsonl": [row for row in decisions if row["reuse_disposition"] == "regenerate_candidate"],
        "unresolved.jsonl": [row for row in decisions if row["reuse_disposition"] == "unresolved"],
        "train_sft.jsonl": corrected_sft["train"], "train_dpo.jsonl": corrected_dpo["train"],
        "valid_sft.jsonl": corrected_sft["valid"], "valid_dpo.jsonl": corrected_dpo["valid"],
        "sft_train.jsonl": proposed["train"], "sft_valid.jsonl": proposed["valid"],
        "proposed_train.jsonl": proposed["train"], "proposed_valid.jsonl": proposed["valid"],
        "dpo_train_raw.jsonl": corrected_dpo["train"], "dpo_valid_raw.jsonl": corrected_dpo["valid"],
        "router_train.jsonl": router["train"], "router_valid.jsonl": router["valid"],
        "router_axis_change_ledger.jsonl": router_axis_change_ledger,
        "scorer_train_all.jsonl": [row for row in scorer_rows if row["split"] == "train"],
        "scorer_valid_all.jsonl": [row for row in scorer_rows if row["split"] == "valid"],
        "scorer_train_trainable.jsonl": [row for row in trainable if row["split"] == "train"],
        "scorer_valid_trainable.jsonl": [row for row in trainable if row["split"] == "valid"],
        "scorer_train_verified_spans.jsonl": [row for row in trainable if row["split"] == "train"],
        "scorer_valid_verified_spans.jsonl": [row for row in trainable if row["split"] == "valid"],
        "scorer_annotation_ledger.jsonl": annotation_ledger,
        "scorer_excluded_annotations.jsonl": excluded_annotations,
        "scorer_collision_ledger.jsonl": scorer_collision_ledger,
        "scorer_truncation_ledger.jsonl": truncation_ledger,
        "scorer_deduplication_ledger.jsonl": dedup_ledger,
        "legacy_collision_accounting.jsonl": legacy_collision_ledger,
        "representative_examples.jsonl": representative_rows(decisions, accepted_by_id),
        "scoped_review_ledger.jsonl": scoped_review_rows,
        "scoped_review_application_ledger.jsonl": review_application_ledger,
    }
    for name, value in outputs.items():
        write_jsonl(output / name, value)
    if source_hashes != {name: sha256_file(production / name) for name in required}:
        raise RuntimeError("Immutable production inputs changed during export")
    category_counts = dict(Counter(row["reuse_disposition"] for row in decisions))
    category_by_split = {split: dict(Counter(row["reuse_disposition"] for row in decisions if row["split"] == split)) for split in ("train", "valid")}
    local_usage = {
        key: sum(float(row.get("usage", {}).get(key, 0)) for row in local_rows)
        for key in (
            "judge_calls", "judge_prompt_tokens", "judge_completion_tokens",
            "judge_seconds", "wall_seconds_allocated",
        )
    }
    manifest = {
        "version": REVIEWED_VERSION if scoped_review_rows else VERSION,
        "status": (
            "frozen_for_first_experiment_not_clinically_adjudicated"
            if args.first_experiment_approved
            else "provisional_reuse_export_complete_not_training_approved"
        ),
        "production_dir": str(production), "production_artifact_sha256": source_hashes,
        "review_artifact_sha256": {
            "prior_review_ledger": sha256_file(prior_path), "codex_review_ledger": sha256_file(codex_path),
            "local_rejudge_checkpoint": sha256_file(local_path),
            "replacement_production": {
                "path": str(replacement_dir), "artifacts": replacement_hashes,
            } if replacement_dir is not None else None,
            "scoped_review_ledger": sha256_file(scoped_review_path) if scoped_review_path else None,
        },
        "policy": {
            "pair_and_scorer_adjudicated_separately": True,
            "unknown_local_labels_do_not_remove_valid_pairs": True,
            "missing_manual_or_training_approval_is_not_a_content_exclusion": True,
            "response_score_or_omission_not_promoted_to_local_label": True,
            "original_text_and_split_preserved": True,
        },
        "integrity": {
            "adjudication_coverage_unique_ids": len(decisions),
            "exported_unique_ids": len(exported_ids["train"] | exported_ids["valid"]),
            "train_valid_id_overlap": len(exported_ids["train"] & exported_ids["valid"]),
            "question_response_text_preserved": True,
            "immutable_question_clean_and_split_preserved_across_replacements": True,
            "accepted_candidate_replacements": len(replacement_by_id),
            "split_preserved": True,
            "exact_span_offset_and_source_hash_validated": True,
        },
        "counts": {"production_accepted": len(accepted), **category_counts, "by_split": category_by_split},
        "local_rejudge": {
            "rows": len(local_rows),
            "result_counts": dict(Counter(row.get("status") for row in local_rows)),
            "clean_counts": dict(Counter(row.get("clean_status") for row in local_rows)),
            "pair_counts": dict(Counter(row.get("pair_status") for row in local_rows)),
            "usage": local_usage,
        },
        "scorer": {
            "all_rows": len(scorer_rows), "trainable_rows": len(trainable),
            "label_counts": label_counts(scorer_rows),
            "string_collision_groups_masked": len(string_collisions),
            "token_collision_groups_masked": len(token_collision_ledger),
            "truncation_rows_masked": len(truncation_ledger),
            "final_collision_groups": len(find_non_null_collisions(scorer_rows)),
            "deduplicated_groups": len(dedup_ledger), "tokenizer_audit": tokenizer_audit,
        },
        "router": {
            "rows": sum(len(router[split]) for split in ("train", "valid")),
            "label_counts": label_counts(router["train"] + router["valid"]),
            "axis_change_or_unknown_rows": len(router_axis_change_ledger),
            "latest_observed_labels_used": True,
            "unjudged_axes_masked_not_defaulted_to_zero": True,
        },
        "legacy_collision_accounting": {
            "original_groups": len(legacy_all_c),
            "removed_by_original_answer_exclusion": len(set(legacy_all_c) - set(legacy_clean_c)),
            "removed_by_other_nonreuse_dispositions": len(set(legacy_clean_c) - set(legacy_reuse_c)),
            "remaining_in_reusable_if_legacy_labels_were_used": len(legacy_reuse_c),
            "resolved_by_not_carrying_unsupported_legacy_labels": len(legacy_reuse_c),
            "final_scoped_collision_groups": len(find_non_null_collisions(scorer_rows)),
        },
        "readiness": {
            "content_reuse_export_complete": True,
            "training_ready": bool(args.first_experiment_approved),
            "scope": "first_backbone_experiment",
            "clinical_expert_adjudication": False,
            "reason": (
                "Explicit user authorization for the first experiment; six-case assistant review is not clinical-expert adjudication."
                if args.first_experiment_approved
                else "Reuse adjudication is complete, but this task does not grant main-training approval."
            ),
            "usable_scorer_labels_remain": any(any(value is not None for value in row["labels"].values()) for row in scorer_rows),
        },
        "scoped_review": {
            "rows": len(scoped_review_rows),
            "excluded_pairs": sum(row["action"] == "exclude_pair" for row in scoped_review_rows),
            "masked_labels": sum(row["action"] == "mask_scorer_label" for row in scoped_review_rows),
            "note_only_rows": sum(row["action"] == "retain_with_note" for row in scoped_review_rows),
            "review_source": sorted({row["review_source"] for row in scoped_review_rows}),
        },
        "outputs": {name: {"rows": len(value), "sha256": sha256_file(output / name)} for name, value in outputs.items()},
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
