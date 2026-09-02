from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from dpo_utils import (
    build_completion_message,
    build_prompt_messages,
    clean_text,
    extract_refinement_fields,
    file_sha256,
    normalize_text,
    question_key,
    read_jsonl,
    row_key,
    validate_pair_record,
    validate_prompt_contract,
    write_json,
    write_jsonl,
)


def _base_record(
    source_row: Mapping[str, Any],
    *,
    index: int,
    rejected: str,
    pair_id: str,
    pair_regime: str,
    rejected_source: str,
    hard_metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    question, initial_draft, chosen = extract_refinement_fields(source_row)
    prompt = build_prompt_messages(question, initial_draft)
    validate_prompt_contract(prompt, question, initial_draft)

    record: Dict[str, Any] = {
        "pair_id": pair_id,
        "prompt": prompt,
        "chosen": build_completion_message(chosen),
        "rejected": build_completion_message(rejected),
        "metadata": {
            "source_row_id": row_key(source_row, index),
            "question_id": question_key(source_row),
            "question": question,
            "initial_draft": initial_draft,
            "pair_regime": pair_regime,
            "rejected_source": rejected_source,
        },
    }
    if hard_metadata:
        record["metadata"]["hard_negative"] = dict(hard_metadata)
    validate_pair_record(record)
    return record


def build_minimal_records(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        _, rejected, _ = extract_refinement_fields(row)
        rid = row_key(row, index)
        records.append(
            _base_record(
                row,
                index=index,
                rejected=rejected,
                pair_id=f"{rid}__minimal",
                pair_regime="minimal",
                rejected_source="input_unsafe_response",
            )
        )
    return records


def _index_hard_candidates(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    indexed: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        source_row_id = clean_text(row.get("source_row_id"))
        if not source_row_id:
            raise ValueError("Every hard-candidate row must contain source_row_id")
        if source_row_id in indexed:
            raise ValueError(f"Duplicate hard-candidate source_row_id: {source_row_id}")
        indexed[source_row_id] = row
    return indexed


def build_hard_records(
    rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    k: int,
) -> List[Dict[str, Any]]:
    if k <= 1:
        raise ValueError("hard-pair candidate budget k must be at least 2")
    candidates_by_id = _index_hard_candidates(candidate_rows)
    records: List[Dict[str, Any]] = []

    for index, row in enumerate(rows):
        rid = row_key(row, index)
        candidate_record = candidates_by_id.get(rid)
        if candidate_record is None:
            raise ValueError(f"Missing hard candidates for source row {rid}")
        observed_k = int(candidate_record.get("candidate_budget_k", -1))
        if observed_k != k:
            raise ValueError(f"Expected candidate budget K={k} for {rid}; found {observed_k}")
        rejected_record = candidate_record.get("hard_rejected")
        if not isinstance(rejected_record, Mapping):
            raise ValueError(f"Missing hard_rejected mapping for {rid}")

        rejected = clean_text(rejected_record.get("text"))
        if not rejected:
            raise ValueError(f"Selected hard negative for {rid} is empty")
        _, _, chosen = extract_refinement_fields(row)
        if normalize_text(rejected) == normalize_text(chosen):
            raise ValueError(f"Selected hard negative for {rid} matches the chosen response")

        metadata = {
            "candidate_budget_k": k,
            "hard_score": rejected_record.get("hard_score"),
            "policy_logp_per_token": rejected_record.get("policy_logp_per_token"),
            "router_risk": rejected_record.get("router_risk"),
            "chosen_similarity": rejected_record.get("chosen_similarity"),
            "origin": rejected_record.get("origin"),
            "weak_dimension": rejected_record.get("weak_dimension"),
            "matches_input_unsafe": candidate_record.get("hard_rejected_matches_input"),
            "selection_version": candidate_record.get("selection_version"),
        }
        records.append(
            _base_record(
                row,
                index=index,
                rejected=rejected,
                pair_id=f"{rid}__hard_k{k}",
                pair_regime=f"hard_k{k}",
                rejected_source=clean_text(rejected_record.get("origin")) or "automated_hard_pool",
                hard_metadata=metadata,
            )
        )
    return records


def _check_expected(rows: Sequence[Mapping[str, Any]], expected_rows: int, expected_questions: int) -> None:
    if expected_rows >= 0 and len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} source rows; found {len(rows)}")
    n_questions = len({question_key(row) for row in rows})
    if expected_questions >= 0 and n_questions != expected_questions:
        raise ValueError(f"Expected {expected_questions} unique questions; found {n_questions}")


def _check_disjoint(rows: Sequence[Mapping[str, Any]], other_paths: Iterable[str]) -> Dict[str, int]:
    current = {question_key(row) for row in rows}
    counts: Dict[str, int] = {}
    for path in other_paths:
        other_rows = read_jsonl(path)
        overlap = current & {question_key(row) for row in other_rows}
        counts[path] = len(overlap)
        if overlap:
            examples = sorted(overlap)[:10]
            raise ValueError(f"Question leakage with {path}: count={len(overlap)}, examples={examples}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exp295 DPO preference pairs without counselor annotations.")
    parser.add_argument("--input", required=True, help="Original exp295 train or validation JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("minimal", "hard_k4"), required=True)
    parser.add_argument("--hard_candidates", default=None)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--expected_rows", type=int, default=-1)
    parser.add_argument("--expected_questions", type=int, default=-1)
    parser.add_argument("--disjoint_with", action="append", default=[])
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    _check_expected(rows, args.expected_rows, args.expected_questions)
    disjoint_counts = _check_disjoint(rows, args.disjoint_with)

    if args.mode == "minimal":
        records = build_minimal_records(rows)
        candidate_path = None
    else:
        if not args.hard_candidates:
            raise ValueError("--hard_candidates is required for hard_k4 mode")
        candidate_rows = read_jsonl(args.hard_candidates)
        records = build_hard_records(rows, candidate_rows, k=args.k)
        candidate_path = args.hard_candidates

    for record in records:
        validate_pair_record(record)
    write_jsonl(records, args.output)

    pair_counts = Counter(record["metadata"]["source_row_id"] for record in records)
    manifest = {
        "mode": args.mode,
        "k": 1 if args.mode == "minimal" else args.k,
        "source_path": args.input,
        "source_sha256": file_sha256(args.input),
        "hard_candidate_path": candidate_path,
        "hard_candidate_sha256": file_sha256(candidate_path) if candidate_path else None,
        "output_path": args.output,
        "output_sha256": file_sha256(args.output),
        "source_rows": len(rows),
        "source_questions": len({question_key(row) for row in rows}),
        "preference_pairs": len(records),
        "pairs_per_source_min": min(pair_counts.values()) if pair_counts else 0,
        "pairs_per_source_max": max(pair_counts.values()) if pair_counts else 0,
        "question_overlap_checks": disjoint_counts,
        "prompt_contract": "fixed instruction + question + initial draft only",
        "chosen_source": "safe_response",
        "rejected_source": "unsafe_response" if args.mode == "minimal" else "automated hard-negative pool",
        "counselor_annotation_used": False,
    }
    manifest_path = str(Path(args.output).with_suffix(Path(args.output).suffix + ".manifest.json"))
    write_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
