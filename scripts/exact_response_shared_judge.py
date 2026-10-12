#!/usr/bin/env python3
"""Pre-deduplicate judge inputs and expand exact-response-shared judgments."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


JUDGE_FIELDS = (
    "judge_model",
    "judge_raw",
    "judge_ok",
    "judge_scores",
    "judge_rationale",
    "judge_error",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def response_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("question") or ""), str(row.get("response") or "")


def response_hash(key: tuple[str, str]) -> str:
    return hashlib.sha256((key[0] + "\0" + key[1]).encode("utf-8")).hexdigest()


def prepare(args: argparse.Namespace) -> None:
    inputs = [Path(path).resolve() for path in args.input]
    rows = [row for path in inputs for row in read_jsonl(path)]
    example_ids = [str(row.get("example_id") or "") for row in rows]
    if "" in example_ids or len(example_ids) != len(set(example_ids)):
        raise RuntimeError("Prepared judge rows require unique non-empty example_id values")
    if any(not key[0] or not key[1] for key in map(response_key, rows)):
        raise RuntimeError("Prepared judge rows require non-empty question and response")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[response_key(row)].append(row)

    representatives: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    representative_by_key: dict[tuple[str, str], str] = {}
    for row in rows:
        key = response_key(row)
        if key not in representative_by_key:
            representative_by_key[key] = str(row["example_id"])
            members = grouped[key]
            representatives.append(
                {
                    **row,
                    "exact_response_hash": response_hash(key),
                    "exact_response_member_example_ids": [item["example_id"] for item in members],
                    "exact_response_member_systems": [item.get("system") for item in members],
                }
            )
        member_rows.append(
            {
                **row,
                "exact_response_hash": response_hash(key),
                "exact_response_representative_example_id": representative_by_key[key],
            }
        )

    representatives_path = Path(args.representatives).resolve()
    members_path = Path(args.members).resolve()
    write_jsonl(representatives_path, representatives)
    write_jsonl(members_path, member_rows)
    manifest = {
        "status": "complete_exact_response_pre_deduplication_no_api_calls",
        "rule": "exact prepared (question, response); first input/file order is representative",
        "input_priority": [str(path) for path in inputs],
        "input_rows": len(rows),
        "unique_question_response": len(representatives),
        "duplicate_rows_sharing_scores": len(rows) - len(representatives),
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in inputs],
        "outputs": {
            "representatives": {
                "path": str(representatives_path),
                "sha256": sha256(representatives_path),
            },
            "members": {"path": str(members_path), "sha256": sha256(members_path)},
        },
    }
    Path(args.manifest).resolve().write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def expand(args: argparse.Namespace) -> None:
    members_path = Path(args.members).resolve()
    judged_path = Path(args.judged_representatives).resolve()
    members = read_jsonl(members_path)
    judged_rows = read_jsonl(judged_path)
    judged = {str(row.get("example_id") or ""): row for row in judged_rows}
    if "" in judged or len(judged) != len(judged_rows):
        raise RuntimeError("Judged representatives require unique non-empty example_id values")

    expanded = []
    for row in members:
        representative_id = str(row["exact_response_representative_example_id"])
        representative = judged.get(representative_id)
        if representative is None:
            raise RuntimeError(f"Missing judged representative: {representative_id}")
        if response_key(row) != response_key(representative):
            raise RuntimeError(f"Representative text mismatch: {representative_id}")
        expanded.append(
            {
                **row,
                **{field: representative.get(field) for field in JUDGE_FIELDS},
                "exact_response_shared": row["example_id"] != representative_id,
                "exact_response_shared_from_example_id": representative_id,
            }
        )

    output = Path(args.output).resolve()
    write_jsonl(output, expanded)
    manifest = {
        "status": "complete_exact_response_shared_expansion",
        "member_rows": len(members),
        "judged_representatives": len(judged_rows),
        "expanded_rows": len(expanded),
        "shared_rows": sum(row["exact_response_shared"] for row in expanded),
        "all_judgments_ok": all(row.get("judge_ok") is True for row in expanded),
        "inputs": {
            "members": {"path": str(members_path), "sha256": sha256(members_path)},
            "judged_representatives": {"path": str(judged_path), "sha256": sha256(judged_path)},
        },
        "output": {"path": str(output), "sha256": sha256(output)},
    }
    Path(args.manifest).resolve().write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--input", action="append", required=True)
    prepare_parser.add_argument("--representatives", required=True)
    prepare_parser.add_argument("--members", required=True)
    prepare_parser.add_argument("--manifest", required=True)
    prepare_parser.set_defaults(function=prepare)

    expand_parser = subparsers.add_parser("expand")
    expand_parser.add_argument("--members", required=True)
    expand_parser.add_argument("--judged-representatives", required=True)
    expand_parser.add_argument("--output", required=True)
    expand_parser.add_argument("--manifest", required=True)
    expand_parser.set_defaults(function=expand)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
