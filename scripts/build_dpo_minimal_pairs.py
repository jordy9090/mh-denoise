#!/usr/bin/env python3
"""Build and audit the exp295 legacy/core DPO-minimal preference pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from selective_risk_refinement_utils import build_sft_prompt, clean_text


DIMENSIONS = (
    "overall_quality",
    "empathy",
    "specificity",
    "medical_advice",
    "factual_consistency",
    "toxicity",
)
AUDIT_ONLY_FIELDS = ("target_dimension", "violation_vector", "brief_reason")
EXPECTED_ROWS = {"train": 1242, "valid": 174, "test": 354}
EXPECTED_QUESTIONS = {"train": 207, "valid": 29, "test": 59}
EXPECTED_PROVENANCE = {
    "counselbench_eval_strict": 99,
    "counselchat_judged": 196,
}
DATASET_NAME = "exp295_legacy_core"
DATASET_PROVENANCE = "CounselBench-Eval 99 + CounselChat 196"


class DataContractError(ValueError):
    """Raised when an exp295 artifact does not satisfy the fixed contract."""


def read_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataContractError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise DataContractError(f"Expected an object in {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_question(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def question_group_id(row: Mapping[str, Any]) -> str:
    row_id = str(row.get("id", "")).strip()
    dimension = str(row.get("target_dimension", "")).strip()
    suffix = f"_{dimension}"
    if not row_id or dimension not in DIMENSIONS or not row_id.endswith(suffix):
        raise DataContractError(
            f"Cannot derive question group from id={row_id!r}, target_dimension={dimension!r}"
        )
    return row_id[: -len(suffix)]


def validate_split(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    if split not in EXPECTED_ROWS:
        raise DataContractError(f"Unsupported split: {split}")
    if len(rows) != EXPECTED_ROWS[split]:
        raise DataContractError(
            f"{split}: expected {EXPECTED_ROWS[split]} rows, found {len(rows)}"
        )

    required = {
        "id",
        "question",
        "safe_response",
        "unsafe_response",
        "target_dimension",
        "violation_vector",
        "brief_reason",
    }
    row_ids = set()
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise DataContractError(f"{split}[{index}] missing fields: {sorted(missing)}")
        for field in ("id", "question", "safe_response", "unsafe_response", "brief_reason"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise DataContractError(f"{split}[{index}] has empty/non-string {field}")
        if row["id"] in row_ids:
            raise DataContractError(f"{split}: duplicate row id {row['id']!r}")
        row_ids.add(row["id"])

        dimension = row["target_dimension"]
        vector = row["violation_vector"]
        if dimension not in DIMENSIONS:
            raise DataContractError(f"{split}[{index}] unknown dimension {dimension!r}")
        if not isinstance(vector, dict) or set(vector) != set(DIMENSIONS):
            raise DataContractError(f"{split}[{index}] invalid violation_vector keys")
        if any(vector[name] not in (0, 1) for name in DIMENSIONS):
            raise DataContractError(f"{split}[{index}] violation_vector is not binary")
        if sum(vector.values()) != 1 or vector[dimension] != 1:
            raise DataContractError(f"{split}[{index}] violation_vector is not target one-hot")
        if clean_text(row["safe_response"]).casefold() == clean_text(row["unsafe_response"]).casefold():
            raise DataContractError(f"{split}[{index}] chosen and rejected are identical")
        grouped[question_group_id(row)].append(row)

    if len(grouped) != EXPECTED_QUESTIONS[split]:
        raise DataContractError(
            f"{split}: expected {EXPECTED_QUESTIONS[split]} question groups, found {len(grouped)}"
        )

    dimension_counts = Counter()
    normalized_questions = set()
    for group_id, group_rows in grouped.items():
        dimensions = Counter(row["target_dimension"] for row in group_rows)
        if dimensions != Counter({name: 1 for name in DIMENSIONS}):
            raise DataContractError(f"{split}/{group_id}: expected exactly one row per dimension")
        questions = {normalize_question(row["question"]) for row in group_rows}
        safe_responses = {clean_text(row["safe_response"]) for row in group_rows}
        if len(questions) != 1 or len(safe_responses) != 1:
            raise DataContractError(f"{split}/{group_id}: inconsistent question or safe response")
        normalized_questions.update(questions)
        dimension_counts.update(dimensions)

    if len(normalized_questions) != EXPECTED_QUESTIONS[split]:
        raise DataContractError(f"{split}: normalized question collision detected")

    return {
        "rows": len(rows),
        "questions": len(grouped),
        "dimension_counts": dict(sorted(dimension_counts.items())),
        "question_group_ids": sorted(grouped),
        "normalized_questions": sorted(normalized_questions),
    }


def validate_no_split_leakage(split_audits: Mapping[str, Mapping[str, Any]]) -> None:
    split_names = ("train", "valid", "test")
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            left = split_audits[left_name]
            right = split_audits[right_name]
            group_overlap = set(left["question_group_ids"]) & set(right["question_group_ids"])
            question_overlap = set(left["normalized_questions"]) & set(right["normalized_questions"])
            if group_overlap or question_overlap:
                raise DataContractError(
                    f"Leakage between {left_name} and {right_name}: "
                    f"group_ids={len(group_overlap)}, normalized_questions={len(question_overlap)}"
                )


def validate_provenance(
    provenance_rows: Sequence[Mapping[str, Any]],
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    if len(provenance_rows) != 295:
        raise DataContractError(f"Expected 295 provenance rows, found {len(provenance_rows)}")
    source_counts = Counter(str(row.get("safe_target_mix_source", "")) for row in provenance_rows)
    if source_counts != Counter(EXPECTED_PROVENANCE):
        raise DataContractError(
            f"Unexpected provenance distribution: {dict(source_counts)}; expected {EXPECTED_PROVENANCE}"
        )

    by_question: Dict[str, Mapping[str, Any]] = {}
    for row in provenance_rows:
        key = normalize_question(row.get("question"))
        if not key or key in by_question:
            raise DataContractError("Missing or duplicate normalized question in provenance file")
        by_question[key] = row

    split_distributions: Dict[str, Dict[str, int]] = {}
    for split, rows in split_rows.items():
        per_question: Dict[str, str] = {}
        for row in rows:
            key = normalize_question(row["question"])
            source_row = by_question.get(key)
            if source_row is None:
                raise DataContractError(f"{split}: question is absent from provenance file")
            if clean_text(source_row.get("safe_response")) != clean_text(row["safe_response"]):
                raise DataContractError(f"{split}: safe response disagrees with provenance for {row['id']}")
            per_question[key] = str(source_row["safe_target_mix_source"])
        split_distributions[split] = dict(sorted(Counter(per_question.values()).items()))

    return {
        "label": DATASET_PROVENANCE,
        "question_counts": dict(sorted(source_counts.items())),
        "split_question_counts": split_distributions,
    }


def build_pair(
    row: Mapping[str, Any],
    prompt_builder: Callable[[Mapping[str, Any]], str],
) -> Dict[str, Any]:
    # Deliberately pass only q+d into the existing SFT prompt builder. This makes
    # it impossible for audit-only fields to affect the model-visible prompt.
    prompt_row = {
        "question": row["question"],
        "unsafe_response": row["unsafe_response"],
    }
    pair = {
        "id": row["id"],
        "question_group_id": question_group_id(row),
        "prompt": prompt_builder(prompt_row),
        "chosen": clean_text(row["safe_response"]),
        "rejected": clean_text(row["unsafe_response"]),
        "audit_metadata": {
            "dataset": DATASET_NAME,
            "provenance": DATASET_PROVENANCE,
            "target_dimension": row["target_dimension"],
            "violation_vector": row["violation_vector"],
            "brief_reason": row["brief_reason"],
            "source": row.get("source"),
            "generator": row.get("generator"),
            "version": row.get("version"),
        },
    }
    if any(str(pair["audit_metadata"][name]) in pair["prompt"] for name in AUDIT_ONLY_FIELDS):
        # This check is only meaningful for exact serialized values. The stronger
        # guarantee is the q+d-only prompt_row above.
        raise DataContractError(f"Audit-only value leaked into prompt for {row['id']}")
    return pair


def build_split_pairs(
    rows: Sequence[Mapping[str, Any]],
    split: str,
    prompt_builder: Callable[[Mapping[str, Any]], str],
) -> List[Dict[str, Any]]:
    if split == "test":
        raise DataContractError("Preference-pair construction is forbidden for exp295 legacy test")
    validate_split(rows, split)
    return [build_pair(row, prompt_builder) for row in rows]


def audit_tokenization(
    pair_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_prompt_length: int = 768,
    max_completion_length: int = 256,
) -> Dict[str, int]:
    eos = tokenizer.eos_token or ""
    prompt_lengths: List[int] = []
    chosen_lengths: List[int] = []
    rejected_lengths: List[int] = []
    chosen_prefix_mismatches = 0
    rejected_prefix_mismatches = 0
    for row in pair_rows:
        prompt_ids = tokenizer(text=row["prompt"])["input_ids"]
        chosen_full = tokenizer(text=row["prompt"] + row["chosen"] + eos)["input_ids"]
        rejected_full = tokenizer(text=row["prompt"] + row["rejected"] + eos)["input_ids"]
        chosen_mismatch = chosen_full[: len(prompt_ids)] != prompt_ids
        rejected_mismatch = rejected_full[: len(prompt_ids)] != prompt_ids
        chosen_prefix_mismatches += int(chosen_mismatch)
        rejected_prefix_mismatches += int(rejected_mismatch)
        prompt_lengths.append(len(prompt_ids))
        chosen_lengths.append(len(chosen_full) - len(prompt_ids))
        rejected_lengths.append(len(rejected_full) - len(prompt_ids))

    if chosen_prefix_mismatches or rejected_prefix_mismatches:
        raise DataContractError(
            "Tokenizer prompt/completion boundary mismatch: "
            f"chosen={chosen_prefix_mismatches}, rejected={rejected_prefix_mismatches}"
        )
    return {
        "rows": len(pair_rows),
        "prompt_max_tokens": max(prompt_lengths, default=0),
        "prompt_over_limit": sum(length > max_prompt_length for length in prompt_lengths),
        "chosen_completion_max_tokens": max(chosen_lengths, default=0),
        "chosen_completion_over_limit": sum(length > max_completion_length for length in chosen_lengths),
        "rejected_completion_max_tokens": max(rejected_lengths, default=0),
        "rejected_completion_over_limit": sum(length > max_completion_length for length in rejected_lengths),
        "chosen_prefix_mismatches": chosen_prefix_mismatches,
        "rejected_prefix_mismatches": rejected_prefix_mismatches,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--valid_file", required=True)
    parser.add_argument("--test_file", required=True, help="Validated for leakage only; no pairs are emitted")
    parser.add_argument("--provenance_file", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = {
        "train": Path(args.train_file),
        "valid": Path(args.valid_file),
        "test": Path(args.test_file),
    }
    split_rows = {name: read_jsonl(path) for name, path in input_paths.items()}
    split_audits = {name: validate_split(rows, name) for name, rows in split_rows.items()}
    validate_no_split_leakage(split_audits)
    provenance = validate_provenance(read_jsonl(args.provenance_file), split_rows)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)

    def prompt_builder(row: Mapping[str, Any]) -> str:
        return build_sft_prompt(tokenizer, dict(row))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "train": output_dir / "train.jsonl",
        "valid": output_dir / "valid.jsonl",
    }
    built_pairs = {}
    tokenization_audits = {}
    for split in ("train", "valid"):
        built_pairs[split] = build_split_pairs(split_rows[split], split, prompt_builder)
        tokenization_audits[split] = audit_tokenization(built_pairs[split], tokenizer)
        write_jsonl(built_pairs[split], output_paths[split])

    manifest = {
        "dataset": DATASET_NAME,
        "display_name": "exp295 legacy/core dataset",
        "provenance": provenance,
        "preference_contract": {
            "prompt": "existing sft_plain(question, unsafe_response)",
            "chosen": "safe_response",
            "rejected": "unsafe_response",
            "audit_only_fields": list(AUDIT_ONLY_FIELDS),
            "test_policy": "validated for leakage only; no preference pairs; no model selection",
            "test_result_label": "exp295 legacy test",
        },
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "rows": split_audits[name]["rows"],
                "questions": split_audits[name]["questions"],
            }
            for name, path in input_paths.items()
        },
        "provenance_file": {
            "path": str(Path(args.provenance_file).resolve()),
            "sha256": sha256_file(args.provenance_file),
        },
        "tokenizer_dir": str(Path(args.tokenizer_dir).resolve()),
        "tokenization_audit": tokenization_audits,
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "rows": EXPECTED_ROWS[name],
            }
            for name, path in output_paths.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
