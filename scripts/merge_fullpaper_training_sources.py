#!/usr/bin/env python3
"""Merge accepted production pairs with frozen corrected TRAIN pairs without rewriting text."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


VERSION = "fullpaper-training-source-merge-v1"


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


def ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["metadata"]["canonical_id"]) for row in rows]


def question_key(row: dict[str, Any]) -> str:
    value = row["metadata"].get("question_normalized_sha256")
    if not value:
        raise RuntimeError(f"Missing normalized question ID for {row['metadata'].get('canonical_id')}")
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", required=True)
    parser.add_argument("--corrected-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    production = Path(args.production_dir).resolve()
    corrected = Path(args.corrected_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    source_paths = {
        "production_train_sft": production / "train_sft.jsonl",
        "production_train_dpo": production / "train_dpo.jsonl",
        "production_valid_sft": production / "valid_sft.jsonl",
        "production_valid_dpo": production / "valid_dpo.jsonl",
        "corrected_train_sft": corrected / "train_sft.jsonl",
        "corrected_train_dpo": corrected / "train_dpo.jsonl",
    }
    rows = {name: read_jsonl(path) for name, path in source_paths.items()}
    for prefix in ("production_train", "production_valid", "corrected_train"):
        sft_ids, dpo_ids = ids(rows[f"{prefix}_sft"]), ids(rows[f"{prefix}_dpo"])
        if sft_ids != dpo_ids or len(sft_ids) != len(set(sft_ids)):
            raise RuntimeError(f"SFT/DPO order, membership, or uniqueness failure: {prefix}")

    production_train_ids = set(ids(rows["production_train_sft"]))
    corrected_ids = set(ids(rows["corrected_train_sft"]))
    valid_ids = set(ids(rows["production_valid_sft"]))
    if production_train_ids & corrected_ids:
        raise RuntimeError("Production TRAIN duplicates a frozen corrected TRAIN pair")
    if valid_ids & (production_train_ids | corrected_ids):
        raise RuntimeError("Canonical VALID overlaps merged TRAIN")

    # The frozen corrected rows have priority.  If a separately generated row
    # ever shares their normalized question, omit that generated duplicate from
    # both contracts instead of silently creating two targets for one question.
    corrected_question_ids = {question_key(row) for row in rows["corrected_train_sft"]}
    kept_indexes = [
        index
        for index, row in enumerate(rows["production_train_sft"])
        if question_key(row) not in corrected_question_ids
    ]
    dropped_question_duplicates = [
        ids(rows["production_train_sft"])[index]
        for index in range(len(rows["production_train_sft"]))
        if index not in set(kept_indexes)
    ]
    production_train_sft = [rows["production_train_sft"][index] for index in kept_indexes]
    production_train_dpo = [rows["production_train_dpo"][index] for index in kept_indexes]
    final_train_question_ids = [question_key(row) for row in rows["corrected_train_sft"] + production_train_sft]
    if len(final_train_question_ids) != len(set(final_train_question_ids)):
        raise RuntimeError("Merged TRAIN still contains multiple answers for one normalized question")
    valid_question_ids = {question_key(row) for row in rows["production_valid_sft"]}
    if valid_question_ids & set(final_train_question_ids):
        raise RuntimeError("Canonical VALID normalized question overlaps merged TRAIN")

    train_sft = rows["corrected_train_sft"] + production_train_sft
    train_dpo = rows["corrected_train_dpo"] + production_train_dpo
    outputs = {
        "train_sft.jsonl": train_sft,
        "train_dpo.jsonl": train_dpo,
        "valid_sft.jsonl": rows["production_valid_sft"],
        "valid_dpo.jsonl": rows["production_valid_dpo"],
    }
    for name, data in outputs.items():
        write_jsonl(output / name, data)

    manifest = {
        "version": VERSION,
        "status": "complete",
        "text_policy": "source rows copied exactly; no response text rewritten",
        "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
        "counts": {
            "corrected_train_reused": len(corrected_ids),
            "production_train_accepted": len(production_train_ids),
            "production_train_retained_after_question_dedup": len(production_train_sft),
            "production_train_question_duplicates_dropped": len(dropped_question_duplicates),
            "merged_train": len(train_sft),
            "canonical_valid_accepted": len(valid_ids),
        },
        "duplicate_id_checks": {
            "corrected_vs_production_train": 0,
            "train_vs_valid": 0,
            "final_train_normalized_question_duplicates": 0,
            "train_valid_normalized_question_overlap": 0,
        },
        "dropped_production_question_duplicate_ids": dropped_question_duplicates,
        "outputs": {name: {"rows": len(data), "sha256": sha256(output / name)} for name, data in outputs.items()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
