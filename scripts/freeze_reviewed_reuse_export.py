#!/usr/bin/env python3
"""Validate and hash-bind a reviewed production-reuse export for one experiment."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from fullpaper_risk_contract import AXES


VERSION = "reviewed-reuse-first-experiment-freeze-v1-20261004"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def duplicate_summary(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = collections.Counter(normalized(row[field]) for row in rows)
    return {
        "groups": sum(value > 1 for value in counts.values()),
        "extra_rows": sum(value - 1 for value in counts.values()),
    }


def ensure_alias(source: Path, target: Path) -> None:
    if target.exists():
        if sha256(source) != sha256(target):
            raise RuntimeError(f"Existing DPO alias differs: {target}")
        return
    shutil.copyfile(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    args = parser.parse_args()
    root = Path(args.export_dir).resolve()
    frozen = root / "frozen_manifest.json"
    if frozen.exists():
        raise FileExistsError(f"Refusing to replace existing freeze: {frozen}")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != "production-reuse-reviewed-export-v2-20261004":
        raise RuntimeError("The export is not the reviewed Revision 2 data version")
    if manifest.get("readiness", {}).get("training_ready") is not True:
        raise RuntimeError("The reviewed export lacks first-experiment authorization")

    ensure_alias(root / "dpo_train_raw.jsonl", root / "dpo_train.jsonl")
    ensure_alias(root / "dpo_valid_raw.jsonl", root / "dpo_valid.jsonl")
    names = (
        "sft_train.jsonl", "sft_valid.jsonl", "dpo_train.jsonl", "dpo_valid.jsonl",
        "router_train.jsonl", "router_valid.jsonl",
        "scorer_train_verified_spans.jsonl", "scorer_valid_verified_spans.jsonl",
    )
    data = {name: read_jsonl(root / name) for name in names}

    split_reports = {}
    for split in ("train", "valid"):
        sft = data[f"sft_{split}.jsonl"]
        dpo = data[f"dpo_{split}.jsonl"]
        router = data[f"router_{split}.jsonl"]
        scorer = data[f"scorer_{split}_verified_spans.jsonl"]
        sft_by_id = {row["canonical_id"]: row for row in sft}
        dpo_by_id = {row["metadata"]["canonical_id"]: row for row in dpo}
        router_by_id = {row["canonical_id"]: row for row in router}
        if len(sft_by_id) != len(sft) or len(dpo_by_id) != len(dpo) or len(router_by_id) != len(router):
            raise RuntimeError(f"Duplicate pair ID in {split} SFT/DPO/Router")
        if set(sft_by_id) != set(dpo_by_id) or set(sft_by_id) != set(router_by_id):
            raise RuntimeError(f"SFT/DPO/Router pair ID mismatch in {split}")
        for canonical_id, row in sft_by_id.items():
            dpo_row, router_row = dpo_by_id[canonical_id], router_by_id[canonical_id]
            if not (
                dpo_row["input"]["question"] == row["question"]
                and dpo_row["input"]["corrupted_response"] == row["unsafe_response"]
                and dpo_row["chosen"] == row["safe_response"]
                and dpo_row["rejected"] == row["unsafe_response"]
                and router_row["question"] == row["question"]
                and router_row["candidate_response"] == row["unsafe_response"]
            ):
                raise RuntimeError(f"Cross-task q/y/u mismatch: {canonical_id}")
        if not {row["canonical_id"] for row in scorer} <= set(sft_by_id):
            raise RuntimeError(f"Scorer contains a pair outside frozen {split} pairs")
        split_reports[split] = {
            "pairs": len(sft),
            "scorer_rows": len(scorer),
            "unique_questions": len({row["question_normalized_sha256"] for row in sft}),
            "unique_duplicate_clusters": len({row["duplicate_cluster_id"] for row in sft}),
            "unique_source_groups": len({row["source_group_id"] for row in sft}),
            "normalized_duplicates": {
                field: duplicate_summary(sft, field)
                for field in ("question", "safe_response", "unsafe_response")
            },
        }

    train = data["sft_train.jsonl"]
    valid = data["sft_valid.jsonl"]
    overlap = {
        field: len({row[field] for row in train} & {row[field] for row in valid})
        for field in ("canonical_id", "question_normalized_sha256", "duplicate_cluster_id", "source_group_id")
    }
    if any(overlap.values()):
        raise RuntimeError(f"Frozen TRAIN/VALID provenance overlap: {overlap}")

    label_counts = {}
    conflicts = []
    active: dict[tuple[str, str, str], set[int]] = collections.defaultdict(set)
    for split in ("train", "valid"):
        label_counts[split] = {}
        rows = data[f"scorer_{split}_verified_spans.jsonl"]
        for axis in AXES:
            values = [row["labels"].get(axis) for row in rows]
            label_counts[split][axis] = {
                "0": sum(value == 0 for value in values),
                "1": sum(value == 1 for value in values),
                "unknown": sum(value is None for value in values),
            }
        for row in rows:
            for axis in AXES:
                value = row["labels"].get(axis)
                if value is not None:
                    active[(row["question"], row["span"], axis)].add(int(value))
    for (question, span, axis), values in active.items():
        if len(values) > 1:
            conflicts.append({
                "input_sha256": hashlib.sha256((question + "\n" + span).encode()).hexdigest(),
                "axis": axis, "labels": sorted(values),
            })
    if conflicts:
        raise RuntimeError(f"Final active scorer conflicts remain: {conflicts[:3]}")

    files = {
        name: {"rows": len(data[name]), "sha256": sha256(root / name)}
        for name in names
    }
    payload = {
        "version": VERSION,
        "status": "frozen_complete",
        "scope": "first_backbone_experiment",
        "clinical_expert_adjudication": False,
        "source_export": str(root),
        "source_manifest_sha256": sha256(manifest_path),
        "files": files,
        "splits": split_reports,
        "train_valid_overlap": overlap,
        "scorer_label_counts": label_counts,
        "final_active_conflicts": 0,
        "checks": {
            "sft_dpo_router_same_pair_ids": True,
            "sft_dpo_router_exact_q_y_u_match": True,
            "scorer_pair_ids_subset_of_frozen_pairs": True,
            "train_valid_overlap_zero": True,
        },
    }
    frozen.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
