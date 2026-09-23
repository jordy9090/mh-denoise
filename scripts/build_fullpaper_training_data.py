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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from selective_risk_refinement_utils import build_sft_prompt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/fullpaper_acl_pipeline/development_training_shard_120_corrected_v2"
DEFAULT_OUTPUT = ROOT / "data/fullpaper_acl_pipeline/fullpaper_training_v2"
DEFAULT_TOKENIZER = ROOT / "outputs/models/gemma4_peft_sft_plain_exp295/final"
DEFAULT_CANONICAL = ROOT / "data/fullpaper_acl_pipeline/canonical_clean_qa.jsonl"
VERSION = "fullpaper-training-adapter-v2"
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
        if expected is None or label != expected:
            raise ValueError(f"Invalid typed local scope/label: {axis}/{scope}/{label}")
        if labels[axis] is not None and labels[axis] != label:
            raise ValueError(f"Conflicting explicit local labels in one scorer input: {axis}")
        labels[axis] = label
        basis[axis] = f"explicit_{scope}_exact_span"
    return labels, basis


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
    parser.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER))
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
    development_eval_size = (
        args.development_eval_size
        if args.development_eval_size is not None
        else (8 if args.source_kind == "corrected-dev57" else 0)
    )
    if development_eval_size < 0:
        raise ValueError("--development-eval-size cannot be negative")

    source = Path(args.source_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    required_names = ("train_sft.jsonl", "train_dpo.jsonl")
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
    if len(sft_source) != len(dpo_source) or len(sft_source) <= development_eval_size:
        raise RuntimeError("SFT/DPO counts differ or are too small for the development partition")
    sft_by_id = {row["metadata"]["canonical_id"]: row for row in sft_source}
    dpo_by_id = {row["metadata"]["canonical_id"]: row for row in dpo_source}
    if set(sft_by_id) != set(dpo_by_id) or len(sft_by_id) != len(sft_source):
        raise RuntimeError("SFT/DPO corrected IDs disagree or contain duplicates")
    missing_canonical = set(sft_by_id) - set(canonical_by_id)
    if missing_canonical:
        raise RuntimeError(f"Rows missing from canonical provenance: {sorted(missing_canonical)[:5]}")

    train_source_ids = {canonical_id for canonical_id, split in split_by_id.items() if split == "train"}
    if len(train_source_ids) <= development_eval_size:
        raise RuntimeError("TRAIN is too small for the requested smoke-only holdback")
    ordered_train_ids = sorted(train_source_ids, key=stable_key)
    dev_ids = set(ordered_train_ids[:development_eval_size])
    train_ids = train_source_ids - dev_ids
    tokenizer = load_local_tokenizer(Path(args.tokenizer_dir).resolve())

    flat_sft: list[dict[str, Any]] = []
    dpo_pairs: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    scorer_units: list[dict[str, Any]] = []

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

        common_meta = row_metadata(source_sft, role, split, canonical)
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
    for name, rows in outputs.items():
        write_jsonl(output / name, rows)

    router_positive = Counter(axis for row in router_rows for axis, value in row["labels"].items() if value == 1)
    router_negative = Counter(axis for row in router_rows for axis, value in row["labels"].items() if value == 0)
    scorer_positive = Counter(axis for row in scorer_units for axis, value in row["labels"].items() if value == 1)
    scorer_negative = Counter(axis for row in scorer_units for axis, value in row["labels"].items() if value == 0)
    scorer_unknown = Counter(axis for row in scorer_units for axis, value in row["labels"].items() if value is None)
    if args.source_kind == "production" and flat_sft and (
        not scorer_units or not scorer_positive or not scorer_negative
    ):
        raise RuntimeError(
            "Typed production scorer export requires at least one explicit local_defect=1 "
            "and one explicit local_support=0; response-level evidence cannot substitute"
        )
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

    manifest = {
        "version": VERSION,
        "status": "complete",
        "source_kind": args.source_kind,
        "source_dir": str(source),
        "source_sha256": actual_hashes,
        "axis_order": list(AXES),
        "current_to_legacy_axis_map": LEGACY_AXIS_MAP,
        "text_policy": "question, clean/chosen, and corrupted/rejected are copied byte-for-byte",
        "split_policy": (
            "all 57 are original TRAIN; 49/8 is a smoke-only disjoint TRAIN-origin development partition, never final VALID/TEST"
            if args.source_kind == "corrected-dev57"
            else "canonical TRAIN and VALID are preserved; only TRAIN receives a deterministic smoke holdback, while canonical VALID remains separate"
        ),
        "rows": {
            "source_rows": len(flat_sft),
            "smoke_train_ids": len(train_ids),
            "smoke_development_eval_ids": len(dev_ids),
            "canonical_valid_ids": sum(split == "valid" for split in split_by_id.values()),
            "router": len(router_rows),
            "scorer_span_units": len(scorer_units),
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
        "scorer_contract": {
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
        "outputs": {
            name: {"path": str((output / name).resolve()), "rows": len(rows), "sha256": sha256(output / name)}
            for name, rows in outputs.items()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
