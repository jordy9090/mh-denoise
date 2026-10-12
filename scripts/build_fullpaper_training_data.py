#!/usr/bin/env python3
"""Losslessly adapt corrected full-paper rows to current training contracts.

No model or API is called.  The original corrected export remains immutable.
All development rows originate from TRAIN; the small development-eval partition
is only for smoke-time loss checks and is not a paper VALID/TEST split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from selective_risk_refinement_utils import build_sft_prompt
from fullpaper_scorer_export_contract import (
    VERSION as SCORER_EXPORT_CONTRACT_VERSION,
    assert_no_non_null_collisions,
    audit_tokenization_and_mask,
    deduplicate_same_inputs,
    find_non_null_collisions,
    mask_collisions,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/fullpaper_acl_pipeline/development_training_shard_120_corrected_v2"
DEFAULT_OUTPUT = ROOT / "data/fullpaper_acl_pipeline/fullpaper_training_v2"
DEFAULT_TOKENIZER = ROOT / "outputs/models/gemma4_peft_sft_plain_exp295/final"
DEFAULT_CANONICAL = ROOT / "data/fullpaper_acl_pipeline/canonical_clean_qa.jsonl"
VERSION = "fullpaper-training-adapter-v5-20260926"
SEED = 20260909
AXES = (
    "overall_quality",
    "empathy",
    "specificity",
    "factual_consistency",
    "medical_boundary",
    "toxicity_or_harm",
)
LEGACY_AXIS_MAP = {
    "overall_quality": "overall_quality",
    "empathy": "empathy",
    "specificity": "specificity",
    "factual_consistency": "factual_consistency",
    "medical_boundary": "medical_advice",
    "toxicity_or_harm": "toxicity",
}
EXPECTED_HASHES = {
    "train_sft.jsonl": "3337394155a0760170c21c84e8d5ba4ae179a12643e0b65aca6aa5d9f3aff574",
    "train_dpo.jsonl": "7fe74cb9a9b157b2063ffa3634a54d89c52f75142d14011294697863c911703d",
    "source_verified_spans.jsonl": "e7e62818e589d71f5e1318109d660d4bdbe90a3565a29c8a927b36c13bb04eed",
    "held_out.jsonl": "e8d42b75aba75126cd297a860ce76d002c93f7fc25289178a5efd5f40173b647",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_local_tokenizer(path: Path):
    """Load only local tokenizer files, including older list-valued Gemma metadata."""
    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except AttributeError as exc:
        if "list' object has no attribute 'keys'" not in str(exc):
            raise
        return AutoTokenizer.from_pretrained(path, local_files_only=True, extra_special_tokens={})


def stable_key(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def ideal_safe(axis: str, score: dict[str, Any]) -> bool:
    if axis == "medical_boundary":
        return score.get("violation") is False
    if axis == "toxicity_or_harm":
        return score.get("score") == 1
    maximum = 4 if axis == "factual_consistency" else 5
    return score.get("score") == maximum


def explicit_local_labels(records: list[dict[str, Any]]) -> tuple[dict[str, int | None], dict[str, str]]:
    """Read typed local supervision without inferring from response-level QC."""
    labels: dict[str, int | None] = {axis: None for axis in AXES}
    basis = {axis: "unknown_no_explicit_local_evidence" for axis in AXES}
    for record in records:
        axis, scope, label = record.get("axis"), record.get("scope"), record.get("label")
        if axis not in AXES:
            raise ValueError(f"Unknown typed local axis: {axis!r}")
        expected = 1 if scope == "local_defect" else 0 if scope == "local_support" else None
        if label is None and record.get("label_mask") is False:
            if expected is None or record.get("original_label") != expected:
                raise ValueError(f"Invalid masked typed local scope/label: {axis}/{scope}")
            basis[axis] = "unknown_masked_upstream_conflict"
            continue
        if expected is None or label != expected:
            raise ValueError(f"Invalid typed local scope/label: {axis}/{scope}/{label}")
        if labels[axis] is not None and labels[axis] != label:
            raise ValueError(f"Conflicting explicit local labels in one scorer input: {axis}")
        labels[axis] = label
        basis[axis] = f"explicit_{scope}_exact_span"
    return labels, basis


def apply_scorer_review_ledger(
    rows: list[dict[str, Any]],
    *,
    review_path: Path,
    allowed_manifest_hashes: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Require a hash-bound Codex decision for every retained non-null label."""
    review_rows = read_jsonl(review_path)
    keyed: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
    for review in review_rows:
        key = (
            str(review.get("canonical_id") or ""),
            str(review.get("span_side") or ""),
            int(review.get("span_start", -1)),
            int(review.get("span_end", -1)),
            str(review.get("axis") or ""),
        )
        if key in keyed:
            raise RuntimeError(f"Duplicate scorer review decision: {key}")
        keyed[key] = review

    applied: set[tuple[str, str, int, int, str]] = set()
    masking_ledger: list[dict[str, Any]] = []
    retained = 0
    masked = 0
    for row in rows:
        for axis in AXES:
            original_label = row.get("labels", {}).get(axis)
            if original_label not in (0, 1):
                continue
            key = (
                str(row["canonical_id"]), str(row["span_side"]),
                int(row["span_start"]), int(row["span_end"]), axis,
            )
            review = keyed.get(key)
            if review is None:
                raise RuntimeError(f"Missing scorer semantic review: {key}")
            applied.add(key)
            if review.get("review_kind") != "complete_question_exact_span_context":
                raise RuntimeError(f"Invalid scorer review kind: {key}")
            if not str(review.get("reviewer") or "").strip() or not str(review.get("reason") or "").strip():
                raise RuntimeError(f"Scorer review requires reviewer and reason: {key}")
            if review.get("disposition") not in {"retain", "mask_unknown"}:
                raise RuntimeError(f"Invalid scorer review disposition: {key}")
            if review.get("source_production_manifest_sha256") not in allowed_manifest_hashes:
                raise RuntimeError(f"Scorer review source-manifest hash mismatch: {key}")
            expected = {
                "question_sha256": text_sha256(str(row["question"])),
                "span": row["span"],
                "span_source_sha256": row["span_source_sha256"],
                "original_label": original_label,
            }
            for field, value in expected.items():
                if review.get(field) != value:
                    raise RuntimeError(f"Scorer review {field} mismatch: {key}")
            if review["disposition"] == "retain":
                retained += 1
                row.setdefault("label_basis", {})[axis] += "_codex_reviewed"
                continue
            masked += 1
            row["labels"][axis] = None
            row["label_mask"][axis] = False
            row["label_basis"][axis] = "unknown_masked_codex_semantic_review"
            masking_ledger.append({
                "canonical_id": row["canonical_id"],
                "split": row["split"],
                "span_side": row["span_side"],
                "span_start": row["span_start"],
                "span_end": row["span_end"],
                "span": row["span"],
                "axis": axis,
                "original_label": original_label,
                "reason": review["reason"],
                "reviewer": review["reviewer"],
                "resolution": "unknown_excluded_from_loss",
            })
    unused = sorted(set(keyed) - applied)
    if unused:
        raise RuntimeError(f"Scorer review contains decisions outside retained rows: {unused[:3]}")
    summary = {
        "status": "complete_exact_span_review_applied",
        "ledger_path": str(review_path),
        "ledger_sha256": sha256(review_path),
        "reviewed_non_null_labels": retained + masked,
        "retained_labels": retained,
        "masked_unknown": masked,
        "coverage_complete": True,
    }
    return summary, masking_ledger


def production_training_readiness(
    *,
    semantic_review: dict[str, Any],
    scorer_semantic_review: dict[str, Any] | None = None,
    retained_splits: set[str],
    retained_rows: int,
    scorer_class_presence_by_axis: dict[str, dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Separate structural class counts from an explicit semantic approval."""
    blockers: list[str] = []
    if not semantic_review.get("coverage_complete"):
        blockers.append("semantic_review_incomplete")
    if not semantic_review.get("approved_for_training"):
        blockers.append("semantic_review_not_approved")
    if scorer_semantic_review is not None and not scorer_semantic_review.get("coverage_complete"):
        blockers.append("scorer_semantic_review_incomplete")
    if retained_rows == 0:
        blockers.append("no_retained_learning_rows")
    if retained_splits != {"train", "valid"}:
        blockers.append("retained_train_and_valid_both_required")
    missing_axis_classes = [
        axis for axis, counts in scorer_class_presence_by_axis.items()
        if not counts["has_both"]
    ]
    if missing_axis_classes:
        blockers.append(
            "scorer_train_axes_missing_both_classes:" + ",".join(missing_axis_classes)
        )
    return not blockers, blockers


def verify_production_source(source: Path) -> dict[str, Any]:
    """Require a complete current-contract production manifest and matching hashes."""
    from local_qwen_production_qc_v4 import VERSION as LOCAL_QC_VERSION

    manifest_path = source / "production_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Production source has no successful production_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Production manifest is not complete: {manifest.get('status')!r}")
    supported_versions = {
        LOCAL_QC_VERSION,
        "local-qwen35-27b-paired-qc-v5-20260924",
    }
    if manifest.get("version") not in supported_versions:
        raise RuntimeError(
            f"Unsupported production QC contract: {manifest.get('version')!r}; "
            f"supported={sorted(supported_versions)!r}"
        )
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict):
        raise RuntimeError("Production manifest has no artifact hash map")
    for name in ("train_sft.jsonl", "train_dpo.jsonl", "valid_sft.jsonl", "valid_dpo.jsonl"):
        path = source / name
        if not path.is_file() or hashes.get(name) != sha256(path):
            raise RuntimeError(f"Production artifact hash mismatch or missing: {name}")
    return manifest


def apply_clean_review_gate(
    split_sources: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]],
    *,
    review_path: Path,
    production_manifest: dict[str, Any],
    production_manifest_path: Path,
) -> tuple[
    list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Retain only hash-bound complete-context clean passes."""
    from source_integrity_contract import validate_contextual_clean_review

    review_rows = read_jsonl(review_path)
    reviews = {str(row.get("canonical_id") or ""): row for row in review_rows}
    if "" in reviews or len(reviews) != len(review_rows):
        raise RuntimeError("Clean review ledger requires unique non-empty canonical IDs")
    actual_manifest_sha = sha256(production_manifest_path)
    allowed_manifest_hashes = {actual_manifest_sha}
    replay_source_hash = production_manifest.get("replay_source", {}).get(
        "production_manifest_sha256"
    )
    if replay_source_hash:
        allowed_manifest_hashes.add(str(replay_source_hash))

    decisions: list[dict[str, Any]] = []
    retained_ids: set[str] = set()
    source_ids: set[str] = set()
    for split, sft_rows, dpo_rows in split_sources:
        if [r["metadata"]["canonical_id"] for r in sft_rows] != [
            r["metadata"]["canonical_id"] for r in dpo_rows
        ]:
            raise RuntimeError(f"SFT/DPO order differs before semantic review: {split}")
        for sft_row, dpo_row in zip(sft_rows, dpo_rows, strict=True):
            canonical_id = sft_row["metadata"]["canonical_id"]
            source_ids.add(canonical_id)
            review = reviews.get(canonical_id)
            if review is None:
                decisions.append({
                    "canonical_id": canonical_id,
                    "split": split,
                    "disposition": "unresolved",
                    "reason": "missing complete-context clean review",
                    "reviewer": None,
                })
                continue
            validated = validate_contextual_clean_review(
                sft_row["input"]["question"], sft_row["target"], review
            )
            candidate_hash = text_sha256(sft_row["input"]["corrupted_response"])
            if validated.get("candidate_response_sha256") != candidate_hash:
                raise RuntimeError(f"Clean review candidate hash mismatch: {canonical_id}")
            if validated.get("source_production_manifest_sha256") not in allowed_manifest_hashes:
                raise RuntimeError(f"Clean review source-manifest hash mismatch: {canonical_id}")
            if (
                dpo_row["input"] != sft_row["input"]
                or dpo_row["chosen"] != sft_row["target"]
                or dpo_row["rejected"] != sft_row["input"]["corrupted_response"]
            ):
                raise RuntimeError(f"SFT/DPO contract differs during semantic review: {canonical_id}")
            decisions.append({
                "canonical_id": canonical_id,
                "split": split,
                "disposition": validated["disposition"],
                "reason": validated["reason"],
                "reviewer": validated["reviewer"],
                "findings": validated["findings"],
                "question_sha256": validated["question_sha256"],
                "clean_response_sha256": validated["clean_response_sha256"],
                "candidate_response_sha256": validated["candidate_response_sha256"],
            })
            if validated["disposition"] == "pass":
                retained_ids.add(canonical_id)

    filtered = []
    for split, sft_rows, dpo_rows in split_sources:
        filtered.append((
            split,
            [row for row in sft_rows if row["metadata"]["canonical_id"] in retained_ids],
            [row for row in dpo_rows if row["metadata"]["canonical_id"] in retained_ids],
        ))
    excluded = [row for row in decisions if row["disposition"] != "pass"]
    summary = {
        "ledger_path": str(review_path),
        "ledger_sha256": sha256(review_path),
        "source_ids": len(source_ids),
        "reviewed_ids": sum(row["reviewer"] is not None for row in decisions),
        "pass": sum(row["disposition"] == "pass" for row in decisions),
        "hold": sum(row["disposition"] == "hold" for row in decisions),
        "unresolved": sum(row["disposition"] == "unresolved" for row in decisions),
        "coverage_complete": all(row["reviewer"] is not None for row in decisions),
        "decisions": decisions,
    }
    return filtered, summary, excluded


def row_metadata(
    row: dict[str, Any], role: str, split: str, canonical: dict[str, Any]
) -> dict[str, Any]:
    meta = row["metadata"]
    return {
        "adapter_version": VERSION,
        "canonical_id": meta["canonical_id"],
        "original_split": split,
        "development_role": role,
        "source": meta["source"],
        "source_component": meta["source_component"],
        "intended_axes": meta["intended_axes"],
        "realized_axes": meta["realized_axes"],
        "unintended_axes": meta["unintended_axes"],
        "generator_repo": meta["generator_repo"],
        "generator_revision": meta["generator_revision"],
        "judge_model": meta["judge_model"],
        "judge_repo": meta.get("judge_repo", meta["judge_model"]),
        "judge_revision": meta.get("judge_revision"),
        "generator_prompt_version": meta.get("generator_prompt_version"),
        "judge_prompt_version": meta.get("judge_prompt_version"),
        "paired_qc": meta.get("paired_qc"),
        "source_integrity_contract_sha256": meta["source_integrity_contract_sha256"],
        "verified_spans": meta["span_supervision"],
        "question_normalized_sha256": canonical["question_normalized_sha256"],
        "duplicate_cluster_id": canonical["duplicate_cluster_id"],
        "source_group_id": canonical["source_group_id"],
        "source_question_id": canonical.get("source_question_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER), help="Deprecated alias for --dpo-tokenizer-dir")
    parser.add_argument("--dpo-tokenizer-dir", help="Local tokenizer used only to serialize DPO prompts")
    parser.add_argument("--scorer-tokenizer-dir", help="Local scorer tokenizer; required for production")
    parser.add_argument("--scorer-max-length", type=int, default=512)
    parser.add_argument(
        "--clean-review-ledger",
        help=(
            "Hash-bound JSONL complete-context clean reviews. Production rows "
            "without an explicit pass are excluded from every learning export."
        ),
    )
    parser.add_argument(
        "--semantic-training-approved",
        action="store_true",
        help="Explicit dataset-level semantic approval; never inferred from 0/1 class presence.",
    )
    parser.add_argument(
        "--scorer-review-ledger",
        help=(
            "Hash-bound review of every retained non-null local scorer label. "
            "Decisions may retain the label or mask it to unknown."
        ),
    )
    parser.add_argument("--canonical-file", default=str(DEFAULT_CANONICAL))
    parser.add_argument(
        "--development-eval-size",
        type=int,
        default=None,
        help="TRAIN-origin smoke holdback. Defaults to 8 only for corrected-dev57 and 0 for production.",
    )
    parser.add_argument(
        "--source-kind",
        choices=("corrected-dev57", "production"),
        default="corrected-dev57",
        help="corrected-dev57 enforces the frozen audit hashes; production accepts the same versioned schema at arbitrary size.",
    )
    args = parser.parse_args()
    if args.semantic_training_approved and not args.clean_review_ledger:
        raise ValueError(
            "--semantic-training-approved requires --clean-review-ledger; "
            "an audit-only conversion cannot be approved implicitly"
        )
    if args.semantic_training_approved and not args.scorer_review_ledger:
        raise ValueError(
            "--semantic-training-approved requires --scorer-review-ledger; "
            "local scorer labels cannot be approved from structure alone"
        )
    if args.scorer_review_ledger and args.source_kind != "production":
        raise ValueError("--scorer-review-ledger is supported only for production sources")
    if args.scorer_review_ledger and not args.clean_review_ledger:
        raise ValueError("--scorer-review-ledger requires --clean-review-ledger")
    development_eval_size = (
        args.development_eval_size
        if args.development_eval_size is not None
        else (8 if args.source_kind == "corrected-dev57" else 0)
    )
    if development_eval_size < 0:
        raise ValueError("--development-eval-size cannot be negative")

    source = Path(args.source_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing training export: {output}")
    required_names = ("train_sft.jsonl", "train_dpo.jsonl")
    production_manifest = verify_production_source(source) if args.source_kind == "production" else None
    actual_hashes = {name: sha256(source / name) for name in required_names}
    if args.source_kind == "corrected-dev57":
        actual_hashes = {name: sha256(source / name) for name in EXPECTED_HASHES}
        if actual_hashes != EXPECTED_HASHES:
            raise RuntimeError(f"Corrected source hashes changed: {actual_hashes}")

    canonical_rows = read_jsonl(Path(args.canonical_file).resolve())
    canonical_by_id = {row["canonical_id"]: row for row in canonical_rows}
    if len(canonical_by_id) != len(canonical_rows):
        raise RuntimeError("Canonical file contains duplicate canonical_id values")

    split_sources: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for split in ("train", "valid"):
        sft_path, dpo_path = source / f"{split}_sft.jsonl", source / f"{split}_dpo.jsonl"
        if sft_path.exists() != dpo_path.exists():
            raise RuntimeError(f"{split} SFT/DPO files must either both exist or both be absent")
        if sft_path.exists():
            split_sources.append((split, read_jsonl(sft_path), read_jsonl(dpo_path)))
            actual_hashes[f"{split}_sft.jsonl"] = sha256(sft_path)
            actual_hashes[f"{split}_dpo.jsonl"] = sha256(dpo_path)
    if not split_sources:
        raise RuntimeError("No train_sft/train_dpo or valid_sft/valid_dpo source files found")
    original_source_rows = sum(len(rows) for _, rows, _ in split_sources)
    semantic_review: dict[str, Any] = {
        "status": "not_required_for_legacy_corrected_dev57",
        "coverage_complete": True,
        "pass": original_source_rows,
        "hold": 0,
        "unresolved": 0,
    }
    review_exclusions: list[dict[str, Any]] = []
    if args.source_kind == "production":
        if args.clean_review_ledger:
            split_sources, semantic_review, review_exclusions = apply_clean_review_gate(
                split_sources,
                review_path=Path(args.clean_review_ledger).resolve(),
                production_manifest=production_manifest,
                production_manifest_path=source / "production_manifest.json",
            )
            semantic_review["status"] = "complete_context_review_applied"
            semantic_review["approved_for_training"] = bool(args.semantic_training_approved)
        else:
            # Keep the historical structural-conversion path available for audits,
            # but never treat its unreviewed rows as semantically training-ready.
            semantic_review = {
                "status": "not_provided_audit_only",
                "ledger_path": None,
                "ledger_sha256": None,
                "source_ids": original_source_rows,
                "reviewed_ids": 0,
                "pass": 0,
                "hold": 0,
                "unresolved": original_source_rows,
                "coverage_complete": False,
                "approved_for_training": False,
                "decisions": [],
            }
    sft_source = [row for _, rows, _ in split_sources for row in rows]
    dpo_source = [row for _, _, rows in split_sources for row in rows]
    split_by_id: dict[str, str] = {}
    dpo_split_by_id: dict[str, str] = {}
    for split, rows, _ in split_sources:
        for row in rows:
            canonical_id = row["metadata"]["canonical_id"]
            if canonical_id in split_by_id:
                raise RuntimeError(f"Canonical ID occurs in multiple input splits: {canonical_id}")
            split_by_id[canonical_id] = split
    for split, _, rows in split_sources:
        for row in rows:
            canonical_id = row["metadata"]["canonical_id"]
            if canonical_id in dpo_split_by_id:
                raise RuntimeError(f"DPO canonical ID occurs in multiple input splits: {canonical_id}")
            dpo_split_by_id[canonical_id] = split
    if split_by_id != dpo_split_by_id:
        raise RuntimeError("SFT/DPO split membership differs")
    if args.source_kind == "corrected-dev57" and (len(sft_source) != 57 or len(dpo_source) != 57):
        raise RuntimeError("Expected exactly 57 corrected development rows")
    if len(sft_source) != len(dpo_source) or (
        len(sft_source) <= development_eval_size
        and not (args.source_kind == "production" and not sft_source and development_eval_size == 0)
    ):
        raise RuntimeError("SFT/DPO counts differ or are too small for the development partition")
    sft_by_id = {row["metadata"]["canonical_id"]: row for row in sft_source}
    dpo_by_id = {row["metadata"]["canonical_id"]: row for row in dpo_source}
    if set(sft_by_id) != set(dpo_by_id) or len(sft_by_id) != len(sft_source):
        raise RuntimeError("SFT/DPO corrected IDs disagree or contain duplicates")
    missing_canonical = set(sft_by_id) - set(canonical_by_id)
    if missing_canonical:
        raise RuntimeError(f"Rows missing from canonical provenance: {sorted(missing_canonical)[:5]}")

    train_source_ids = {canonical_id for canonical_id, split in split_by_id.items() if split == "train"}
    if len(train_source_ids) <= development_eval_size and not (
        args.source_kind == "production" and not sft_source and development_eval_size == 0
    ):
        raise RuntimeError("TRAIN is too small for the requested smoke-only holdback")
    ordered_train_ids = sorted(train_source_ids, key=stable_key)
    dev_ids = set(ordered_train_ids[:development_eval_size])
    train_ids = train_source_ids - dev_ids
    dpo_tokenizer_dir = Path(args.dpo_tokenizer_dir or args.tokenizer_dir).resolve()
    tokenizer = load_local_tokenizer(dpo_tokenizer_dir)
    scorer_tokenizer = None
    scorer_tokenizer_dir = None
    if args.scorer_tokenizer_dir:
        scorer_tokenizer_dir = Path(args.scorer_tokenizer_dir).resolve()
        scorer_tokenizer = load_local_tokenizer(scorer_tokenizer_dir)
    elif args.source_kind == "production":
        raise RuntimeError("--scorer-tokenizer-dir is required for production truncation validation")

    flat_sft: list[dict[str, Any]] = []
    dpo_pairs: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    scorer_units: list[dict[str, Any]] = []
    semantic_review_by_id = {
        row["canonical_id"]: row for row in semantic_review.get("decisions", [])
    }

    for canonical_id in sorted(sft_by_id):
        source_sft = sft_by_id[canonical_id]
        source_dpo = dpo_by_id[canonical_id]
        split = split_by_id[canonical_id]
        role = "valid" if split == "valid" else ("development_eval_train_origin" if canonical_id in dev_ids else "train")
        question = source_sft["input"]["question"]
        corrupted = source_sft["input"]["corrupted_response"]
        clean = source_sft["target"]
        meta = source_sft["metadata"]
        canonical = canonical_by_id[canonical_id]
        if canonical["split"] != split:
            raise RuntimeError(f"Canonical split mismatch for {canonical_id}: {canonical['split']} != {split}")
        if (
            source_dpo["input"] != source_sft["input"]
            or source_dpo["chosen"] != clean
            or source_dpo["rejected"] != corrupted
        ):
            raise RuntimeError(f"SFT/DPO text contract mismatch for {canonical_id}")
        if args.source_kind == "corrected-dev57" and meta.get("repair_version") != "dev120-corrected-v2":
            raise RuntimeError(f"Unexpected repair version for {canonical_id}")
        span_supervision = meta.get("span_supervision")
        if not isinstance(span_supervision, list):
            raise RuntimeError(f"Missing versioned span_supervision for {canonical_id}")
        if args.source_kind == "production":
            gate = meta.get("paired_qc", {}).get("clean_target_eligibility", {})
            if gate.get("disposition") != "pass":
                raise RuntimeError(f"Accepted production row has a non-pass clean gate: {canonical_id}")
            clean_defects = [
                span for span in span_supervision
                if span.get("side") == "clean"
                and span.get("scope") == "local_defect"
                and (span.get("label") == 1 or span.get("original_label") == 1)
            ]
            if clean_defects:
                raise RuntimeError(
                    f"Accepted production row has explicit clean local defect evidence: {canonical_id}"
                )

        common_meta = row_metadata(source_sft, role, split, canonical)
        if canonical_id in semantic_review_by_id:
            common_meta["clean_context_review"] = semantic_review_by_id[canonical_id]
        flat = {
            "id": canonical_id,
            "canonical_id": canonical_id,
            "question": question,
            "unsafe_response": corrupted,
            "safe_response": clean,
            "split": split,
            "source": meta["source"],
            "source_component": meta["source_component"],
            "intended_axes": meta["intended_axes"],
            "realized_axes": meta["realized_axes"],
            "verified_spans": span_supervision,
            "question_normalized_sha256": canonical["question_normalized_sha256"],
            "duplicate_cluster_id": canonical["duplicate_cluster_id"],
            "source_group_id": canonical["source_group_id"],
            "metadata": common_meta,
        }
        # Byte-for-byte field preservation; no whitespace normalization.
        assert flat["question"] == source_sft["input"]["question"]
        assert flat["unsafe_response"] == source_sft["input"]["corrupted_response"]
        assert flat["safe_response"] == source_sft["target"]
        flat_sft.append(flat)

        prompt = build_sft_prompt(tokenizer, {"question": question, "unsafe_response": corrupted})
        dpo_pairs.append(
            {
                "id": canonical_id,
                "question_group_id": (
                    canonical_id if args.source_kind == "corrected-dev57" else canonical["question_normalized_sha256"]
                ),
                "prompt": prompt,
                "chosen": clean,
                "rejected": corrupted,
                "audit_metadata": {
                    **common_meta,
                    "dataset": (
                        "fullpaper_corrected_dev57_v1"
                        if args.source_kind == "corrected-dev57"
                        else "fullpaper_production_v1"
                    ),
                    "provenance": (
                        "dev120-corrected-v2 TRAIN-origin development data"
                        if args.source_kind == "corrected-dev57"
                        else "versioned full-paper accepted canonical-split corruption data"
                    ),
                    **({} if args.source_kind == "corrected-dev57" else {"contract_version": "fullpaper-dpo-v1"}),
                    "question_sha256": text_sha256(question),
                    "chosen_sha256": text_sha256(clean),
                    "rejected_sha256": text_sha256(corrupted),
                },
            }
        )

        realized = set(meta["realized_axes"])
        router_rows.append(
            {
                "canonical_id": canonical_id,
                "question": question,
                "candidate_response": corrupted,
                "labels": {axis: int(axis in realized) for axis in AXES},
                "label_semantics": "1 = meaningful adverse candidate-vs-clean score change; 0 = all-six-axis paired judge found no such change",
                "current_to_legacy_axis_map": LEGACY_AXIS_MAP,
                "split": split,
                "development_role": role,
                "source": meta["source"],
                "source_component": meta["source_component"],
                "question_normalized_sha256": canonical["question_normalized_sha256"],
                "duplicate_cluster_id": canonical["duplicate_cluster_id"],
                "source_group_id": canonical["source_group_id"],
                "audit_metadata": common_meta,
            }
        )

        score_by_side = {
            "candidate": meta["paired_qc"]["candidate_scores"],
            "clean": meta["paired_qc"]["clean_scores"],
        }
        text_by_side = {"candidate": corrupted, "clean": clean}
        grouped: dict[tuple[str, int, int, str], set[str]] = defaultdict(set)
        span_records: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
        for span in span_supervision:
            side = span["side"]
            key = (side, int(span["start"]), int(span["end"]), span["text"])
            if text_by_side[side][key[1] : key[2]] != key[3]:
                raise RuntimeError(f"Non-verbatim span offset for {canonical_id}/{side}")
            if text_sha256(text_by_side[side]) != span["source_sha256"]:
                raise RuntimeError(f"Span source hash mismatch for {canonical_id}/{side}")
            grouped[key].add(span["axis"])
            span_records[key].append(span)

        for (side, start, end, text), evidenced_axes in grouped.items():
            if args.source_kind == "production":
                labels, label_basis = explicit_local_labels(span_records[(side, start, end, text)])
            else:
                labels = {}
                label_basis = {}
                for axis in AXES:
                    if side == "candidate" and axis in realized and axis in evidenced_axes:
                        labels[axis] = 1
                        label_basis[axis] = "paired_delta_realized_and_exact_candidate_evidence"
                    elif axis in evidenced_axes and ideal_safe(axis, score_by_side[side][axis]):
                        labels[axis] = 0
                        label_basis[axis] = "paired_ideal_safe_score_and_exact_evidence"
                    else:
                        labels[axis] = None
                        label_basis[axis] = "unknown_not_supervised"
            scorer_units.append(
                {
                    "canonical_id": canonical_id,
                    "question": question,
                    "span": text,
                    "span_side": side,
                    "span_start": start,
                    "span_end": end,
                    "span_source_sha256": text_sha256(text_by_side[side]),
                    "labels": labels,
                    "label_mask": {axis: labels[axis] is not None for axis in AXES},
                    "label_basis": label_basis,
                    "current_to_legacy_axis_map": LEGACY_AXIS_MAP,
                    "split": split,
                    "development_role": role,
                    "question_normalized_sha256": canonical["question_normalized_sha256"],
                    "duplicate_cluster_id": canonical["duplicate_cluster_id"],
                    "source_group_id": canonical["source_group_id"],
                    "audit_metadata": common_meta,
                    "source_records": span_records[(side, start, end, text)],
                }
            )

    scorer_semantic_review: dict[str, Any] = {
        "status": "not_provided",
        "ledger_path": None,
        "coverage_complete": False,
    }
    scorer_semantic_masking_ledger: list[dict[str, Any]] = []
    if args.scorer_review_ledger:
        production_manifest_path = source / "production_manifest.json"
        allowed_manifest_hashes = {sha256(production_manifest_path)}
        replay_source_hash = production_manifest.get("replay_source", {}).get(
            "production_manifest_sha256"
        )
        if replay_source_hash:
            allowed_manifest_hashes.add(str(replay_source_hash))
        scorer_semantic_review, scorer_semantic_masking_ledger = apply_scorer_review_ledger(
            scorer_units,
            review_path=Path(args.scorer_review_ledger).resolve(),
            allowed_manifest_hashes=allowed_manifest_hashes,
        )

    original_string_collisions = find_non_null_collisions(scorer_units)
    scorer_conflict_ledger = mask_collisions(
        scorer_units, original_string_collisions, stage="adapter_string_input"
    )
    assert_no_non_null_collisions(scorer_units)
    scorer_physical_rows_before_deduplication = len(scorer_units)
    scorer_units, scorer_deduplication_ledger = deduplicate_same_inputs(scorer_units)
    assert_no_non_null_collisions(scorer_units)
    tokenizer_audit = None
    truncation_ledger: list[dict[str, Any]] = []
    token_collision_ledger: list[dict[str, Any]] = []
    token_ids: list[list[int]] | None = None
    if scorer_tokenizer is not None:
        tokenizer_audit, truncation_ledger, token_ids, token_collision_ledger = audit_tokenization_and_mask(
            scorer_units, scorer_tokenizer, max_length=args.scorer_max_length
        )
        assert_no_non_null_collisions(scorer_units, token_ids=token_ids)

    def partition(rows: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if row.get(
                "development_role",
                row.get("metadata", row.get("audit_metadata", {})).get("development_role"),
            )
            == role
        ]

    # Keep a complete conversion plus a disjoint TRAIN-origin smoke partition.
    outputs = {
        "sft_all.jsonl": flat_sft,
        "sft_train.jsonl": partition(flat_sft, "train"),
        "sft_dev_train_origin.jsonl": partition(flat_sft, "development_eval_train_origin"),
        "sft_valid.jsonl": partition(flat_sft, "valid"),
        "dpo_all.jsonl": dpo_pairs,
        "dpo_train.jsonl": partition(dpo_pairs, "train"),
        "dpo_dev_train_origin.jsonl": partition(dpo_pairs, "development_eval_train_origin"),
        "dpo_valid.jsonl": partition(dpo_pairs, "valid"),
        "router_all.jsonl": router_rows,
        "router_train.jsonl": partition(router_rows, "train"),
        "router_dev_train_origin.jsonl": partition(router_rows, "development_eval_train_origin"),
        "router_valid.jsonl": partition(router_rows, "valid"),
        "scorer_all_verified_spans.jsonl": scorer_units,
        "scorer_train_verified_spans.jsonl": partition(scorer_units, "train"),
        "scorer_dev_verified_spans_train_origin.jsonl": partition(scorer_units, "development_eval_train_origin"),
        "scorer_valid_verified_spans.jsonl": partition(scorer_units, "valid"),
        "scorer_label_conflicts.jsonl": scorer_conflict_ledger,
        "scorer_truncation_ledger.jsonl": truncation_ledger,
        "scorer_token_collision_ledger.jsonl": token_collision_ledger,
        "scorer_deduplication_ledger.jsonl": scorer_deduplication_ledger,
        "scorer_semantic_masking_ledger.jsonl": scorer_semantic_masking_ledger,
        "semantic_review_exclusions.jsonl": review_exclusions,
    }
    # Preserve the explicit fixed filenames used by the completed 57-row smoke.
    if args.source_kind == "corrected-dev57" and len(flat_sft) == 57 and development_eval_size == 8:
        outputs.update(
            {
                "sft_all57.jsonl": flat_sft,
                "sft_train49.jsonl": partition(flat_sft, "train"),
                "sft_dev8_train_origin.jsonl": partition(flat_sft, "development_eval_train_origin"),
                "dpo_all57.jsonl": dpo_pairs,
                "dpo_train49.jsonl": partition(dpo_pairs, "train"),
                "dpo_dev8_train_origin.jsonl": partition(dpo_pairs, "development_eval_train_origin"),
                "router_all57.jsonl": router_rows,
                "router_train49.jsonl": partition(router_rows, "train"),
                "router_dev8_train_origin.jsonl": partition(router_rows, "development_eval_train_origin"),
            }
        )
    router_positive = Counter(axis for row in router_rows for axis, value in row["labels"].items() if value == 1)
    router_negative = Counter(axis for row in router_rows for axis, value in row["labels"].items() if value == 0)
    scorer_positive = Counter(axis for row in scorer_units for axis, value in row["labels"].items() if value == 1)
    scorer_negative = Counter(axis for row in scorer_units for axis, value in row["labels"].items() if value == 0)
    scorer_unknown = Counter(axis for row in scorer_units for axis, value in row["labels"].items() if value is None)
    scorer_supervision_has_both_classes = bool(scorer_units and scorer_positive and scorer_negative)
    train_known_units = sum(any(row["label_mask"].values()) for row in outputs["scorer_train_verified_spans.jsonl"])
    dev_known_units = sum(any(row["label_mask"].values()) for row in outputs["scorer_dev_verified_spans_train_origin.jsonl"])
    valid_known_units = sum(any(row["label_mask"].values()) for row in outputs["scorer_valid_verified_spans.jsonl"])

    provenance_overlap: dict[str, list[str]] = {}
    for field in (
        "canonical_id",
        "question_normalized_sha256",
        "duplicate_cluster_id",
        "source_group_id",
    ):
        train_values = {row[field] for row in flat_sft if row["split"] == "train"}
        valid_values = {row[field] for row in flat_sft if row["split"] == "valid"}
        provenance_overlap[field] = sorted((train_values - {None}) & (valid_values - {None}))
    if any(provenance_overlap.values()):
        raise RuntimeError(f"TRAIN/VALID provenance overlap: {provenance_overlap}")
    for field in ("question_normalized_sha256", "duplicate_cluster_id", "source_group_id"):
        train_values = {canonical_by_id[canonical_id][field] for canonical_id in train_source_ids}
        valid_values = {
            canonical_by_id[canonical_id][field]
            for canonical_id, split in split_by_id.items()
            if split == "valid"
        }
        provenance_overlap[field] = sorted(train_values & valid_values)
    if any(provenance_overlap.values()):
        raise RuntimeError(f"TRAIN/VALID provenance leakage: {provenance_overlap}")

    scorer_counts_by_split = {}
    for split in ("train", "valid"):
        split_rows = [row for row in scorer_units if row["split"] == split]
        scorer_counts_by_split[split] = {
            "positive": {axis: sum(row["labels"].get(axis) == 1 for row in split_rows) for axis in AXES},
            "negative": {axis: sum(row["labels"].get(axis) == 0 for row in split_rows) for axis in AXES},
            "unknown": {axis: sum(row["labels"].get(axis) is None for row in split_rows) for axis in AXES},
        }

    scorer_class_presence_by_axis = {
        axis: {
            "positive": int(scorer_positive[axis]),
            "negative": int(scorer_negative[axis]),
            "has_both": bool(scorer_positive[axis] and scorer_negative[axis]),
        }
        for axis in AXES
    }
    scorer_train_class_presence_by_axis = {
        axis: {
            "positive": int(scorer_counts_by_split["train"]["positive"][axis]),
            "negative": int(scorer_counts_by_split["train"]["negative"][axis]),
            "has_both": bool(
                scorer_counts_by_split["train"]["positive"][axis]
                and scorer_counts_by_split["train"]["negative"][axis]
            ),
        }
        for axis in AXES
    }
    training_blockers: list[str] = []
    if args.source_kind == "production":
        training_ready, training_blockers = production_training_readiness(
            semantic_review=semantic_review,
            scorer_semantic_review=scorer_semantic_review,
            retained_splits={row["split"] for row in flat_sft},
            retained_rows=len(flat_sft),
            scorer_class_presence_by_axis=scorer_train_class_presence_by_axis,
        )
    else:
        training_ready = scorer_supervision_has_both_classes

    manifest = {
        "version": VERSION,
        "status": "complete",
        "artifact_use": (
            "training_candidate" if training_ready else "audit_only_not_training_ready"
        ),
        "source_kind": args.source_kind,
        "source_dir": str(source),
        "source_sha256": actual_hashes,
        "source_production_manifest": (
            {
                "path": str((source / "production_manifest.json").resolve()),
                "sha256": sha256(source / "production_manifest.json"),
                "version": production_manifest["version"],
                "status": production_manifest["status"],
            }
            if production_manifest is not None else None
        ),
        "axis_order": list(AXES),
        "current_to_legacy_axis_map": LEGACY_AXIS_MAP,
        "text_policy": "question, clean/chosen, and corrupted/rejected are copied byte-for-byte",
        "split_policy": (
            "all 57 are original TRAIN; 49/8 is a smoke-only disjoint TRAIN-origin development partition, never final VALID/TEST"
            if args.source_kind == "corrected-dev57"
            else "canonical TRAIN and VALID are preserved; only TRAIN receives a deterministic smoke holdback, while canonical VALID remains separate"
        ),
        "rows": {
            "source_rows_before_semantic_review": original_source_rows,
            "source_rows": len(flat_sft),
            "smoke_train_ids": len(train_ids),
            "smoke_development_eval_ids": len(dev_ids),
            "canonical_valid_ids": sum(split == "valid" for split in split_by_id.values()),
            "router": len(router_rows),
            "scorer_span_units": len(scorer_units),
            "scorer_physical_rows_before_deduplication": scorer_physical_rows_before_deduplication,
            "scorer_duplicate_rows_removed": (
                scorer_physical_rows_before_deduplication - len(scorer_units)
            ),
            "scorer_train_known_units": train_known_units,
            "scorer_development_known_units": dev_known_units,
            "scorer_valid_known_units": valid_known_units,
        },
        "train_valid_overlap_checks": {
            **provenance_overlap,
            "all_zero": not any(provenance_overlap.values()),
        },
        "router_label_counts": {
            "positive": dict(router_positive),
            "negative": dict(router_negative),
            "unknown": {axis: 0 for axis in AXES},
        },
        "scorer_label_counts": {
            "positive": dict(scorer_positive),
            "negative": dict(scorer_negative),
            "unknown": dict(scorer_unknown),
        },
        "scorer_label_counts_by_split": scorer_counts_by_split,
        "scorer_supervision_status": {
            "has_positive_and_negative": scorer_supervision_has_both_classes,
            "class_presence_by_axis": scorer_class_presence_by_axis,
            "train_class_presence_by_axis": scorer_train_class_presence_by_axis,
            "training_ready": training_ready,
            "note": (
                "class presence is descriptive only; overall readiness also requires semantic approval"
            ),
        },
        "structural_validation": {
            "passed": True,
            "artifact_staging_atomic": True,
            "source_hashes_verified": True,
            "split_overlap_zero": not any(provenance_overlap.values()),
        },
        "semantic_review": semantic_review,
        "scorer_semantic_review": scorer_semantic_review,
        "training_readiness": {
            "ready": training_ready,
            "blocking_reasons": training_blockers,
            "not_inferred_from_global_binary_class_presence": True,
        },
        "scorer_contract": {
            "version": SCORER_EXPORT_CONTRACT_VERSION,
            "positive": (
                "explicit local_defect=1 with exact own-response offset"
                if args.source_kind == "production" else
                "candidate-side exact evidence for a realized adverse delta"
            ),
            "negative": (
                "explicit local_support=0 with exact own-response offset"
                if args.source_kind == "production" else
                "exact evidence on that same text side with an ideal safe paired score"
            ),
            "unknown": "all other span-axis cells; excluded from loss",
            "no_unmarked_negative_inference": True,
            "no_candidate_or_clean_offset_transfer_to_sft_response": True,
            "response_level_holistic_or_omission_never_promoted": True,
        },
        "scorer_input_conflicts": {
            "original_string_groups": len(original_string_collisions),
            "string_groups_masked": len(scorer_conflict_ledger),
            "token_groups_masked": len(token_collision_ledger),
            "remaining_string_groups": len(find_non_null_collisions(scorer_units)),
            "remaining_token_groups": (
                len(find_non_null_collisions(scorer_units, token_ids=token_ids))
                if token_ids is not None else None
            ),
            "same_input_groups_deduplicated": len(scorer_deduplication_ledger),
            "same_label_axis_contributions_removed": sum(
                sum(item["duplicate_axis_contributions_removed"].values())
                for item in scorer_deduplication_ledger
            ),
        },
        "scorer_tokenizer_audit": tokenizer_audit,
        "tokenizers": {
            "dpo": str(dpo_tokenizer_dir),
            "scorer": str(scorer_tokenizer_dir) if scorer_tokenizer_dir is not None else None,
            "scorer_max_length": args.scorer_max_length,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        for name, rows in outputs.items():
            write_jsonl(stage / name, rows)
        manifest["outputs"] = {
            name: {"path": str((output / name).resolve()), "rows": len(rows), "sha256": sha256(stage / name)}
            for name, rows in outputs.items()
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
