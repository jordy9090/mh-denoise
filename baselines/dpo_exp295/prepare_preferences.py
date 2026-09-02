#!/usr/bin/env python3
"""Build leakage-safe DPO preference files for the exp295 refinement split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SYSTEM_PROMPT = (
    "You are a careful mental-health response refiner. Produce one final response "
    "that is safe, empathetic, specific, and professionally bounded. Preserve useful "
    "context, avoid diagnosis and direct medical instructions, and return only the "
    "response that should be shown to the user."
)

EXPECTED_COUNTS = {"train": 1242, "valid": 174, "test": 354}


class DataValidationError(ValueError):
    """Raised when the exp295 preference data violates a required invariant."""


@dataclass(frozen=True)
class SplitSpec:
    name: str
    path: Path
    expected_rows: int


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataValidationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise DataValidationError(f"{path}:{line_number}: each row must be an object")
            rows.append(row)
    return rows


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).casefold()


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w']+", normalize_text(text), flags=re.UNICODE))


def first_text(row: Mapping[str, Any], names: Sequence[str], *, field_label: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise DataValidationError(f"missing {field_label}; checked fields={list(names)}")


def get_question(row: Mapping[str, Any]) -> str:
    return first_text(row, ("question", "query", "user_question"), field_label="question")


def get_chosen(row: Mapping[str, Any]) -> str:
    return first_text(
        row,
        ("safe_response", "target_response", "response"),
        field_label="safe/chosen response",
    )


def get_rejected(row: Mapping[str, Any]) -> str:
    return first_text(
        row,
        ("unsafe_response", "corrupted_response", "bad_response"),
        field_label="unsafe/rejected response",
    )


def get_dimension(row: Mapping[str, Any]) -> str:
    for name in ("target_dimension", "dimension", "violated_dimension"):
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def get_source_id(row: Mapping[str, Any], split: str, row_index: int) -> str:
    for name in ("id", "example_id", "pair_id"):
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    payload = f"{split}\0{row_index}\0{get_question(row)}\0{get_rejected(row)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def get_question_key(row: Mapping[str, Any]) -> str:
    explicit = row.get("question_id")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()

    source_id = str(row.get("id", "")).strip()
    dimension = get_dimension(row)
    suffix = f"_{dimension}"
    if source_id and source_id.endswith(suffix):
        return source_id[: -len(suffix)]

    question = normalize_text(get_question(row))
    return "q_" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:20]


def build_prompt_messages(question: str, initial_draft: str) -> list[dict[str, str]]:
    """Return the fixed x=(q,d) prompt. No flaw label is exposed to the policy."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Mental-health question:\n{question.strip()}\n\n"
                f"Potentially unsafe initial draft:\n{initial_draft.strip()}\n\n"
                "Return only the final response."
            ),
        },
    ]


def text_similarity(candidate: str, chosen: str) -> float:
    """Deterministic near-safe hardness score in [0, 1]."""
    c_norm = normalize_text(candidate)
    y_norm = normalize_text(chosen)
    sequence = SequenceMatcher(None, c_norm, y_norm).ratio()
    c_tokens, y_tokens = tokens(candidate), tokens(chosen)
    union = c_tokens | y_tokens
    jaccard = len(c_tokens & y_tokens) / max(1, len(union))
    c_len, y_len = max(1, len(c_tokens)), max(1, len(y_tokens))
    length_ratio = min(c_len, y_len) / max(c_len, y_len)
    return 0.60 * sequence + 0.30 * jaccard + 0.10 * length_ratio


def validate_split_rows(rows: list[dict[str, Any]], split: str, expected_rows: int) -> dict[str, Any]:
    if expected_rows >= 0 and len(rows) != expected_rows:
        raise DataValidationError(
            f"{split}: expected {expected_rows} rows, found {len(rows)}. "
            "This baseline is locked to the exp295 split."
        )

    ids: set[str] = set()
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    dimensions: Counter[str] = Counter()
    exact_pair_duplicates = 0
    seen_pairs: set[tuple[str, str, str]] = set()

    for index, row in enumerate(rows):
        question = get_question(row)
        chosen = get_chosen(row)
        rejected = get_rejected(row)
        if normalize_text(chosen) == normalize_text(rejected):
            raise DataValidationError(f"{split} row {index}: chosen and rejected are identical")

        source_id = get_source_id(row, split, index)
        if source_id in ids:
            raise DataValidationError(f"{split}: duplicate source id {source_id!r}")
        ids.add(source_id)

        qkey = get_question_key(row)
        groups[qkey].append(row)
        dimensions[get_dimension(row)] += 1

        pair_key = (normalize_text(question), normalize_text(chosen), normalize_text(rejected))
        if pair_key in seen_pairs:
            exact_pair_duplicates += 1
        seen_pairs.add(pair_key)

    inconsistent_safe_groups: list[str] = []
    for qkey, group in groups.items():
        chosen_values = {normalize_text(get_chosen(row)) for row in group}
        if len(chosen_values) != 1:
            inconsistent_safe_groups.append(qkey)
    if inconsistent_safe_groups:
        preview = inconsistent_safe_groups[:5]
        raise DataValidationError(
            f"{split}: {len(inconsistent_safe_groups)} question groups contain multiple safe targets; "
            f"examples={preview}"
        )

    group_sizes = [len(group) for group in groups.values()]
    return {
        "rows": len(rows),
        "question_groups": len(groups),
        "group_size_min": min(group_sizes) if group_sizes else 0,
        "group_size_max": max(group_sizes) if group_sizes else 0,
        "group_size_mean": sum(group_sizes) / max(1, len(group_sizes)),
        "dimensions": dict(sorted(dimensions.items())),
        "exact_pair_duplicates": exact_pair_duplicates,
    }


def _preference_row(
    *,
    source_row: Mapping[str, Any],
    split: str,
    source_index: int,
    rejected_text: str,
    rejected_dimension: str,
    pair_mode: str,
    pair_index: int,
    hardness_score: float,
) -> dict[str, Any]:
    question = get_question(source_row)
    initial_draft = get_rejected(source_row)
    chosen = get_chosen(source_row)
    source_id = get_source_id(source_row, split, source_index)
    qkey = get_question_key(source_row)
    return {
        "id": f"{source_id}__{pair_mode}__n{pair_index}",
        "source_id": source_id,
        "question_key": qkey,
        "split": split,
        "pair_mode": pair_mode,
        "pair_index": pair_index,
        "question": question,
        "initial_draft": initial_draft,
        "source_target_dimension": get_dimension(source_row),
        "rejected_target_dimension": rejected_dimension,
        "hardness_score": round(float(hardness_score), 8),
        "prompt": build_prompt_messages(question, initial_draft),
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected_text}],
    }


def build_minimal_pairs(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        rejected = get_rejected(row)
        output.append(
            _preference_row(
                source_row=row,
                split=split,
                source_index=index,
                rejected_text=rejected,
                rejected_dimension=get_dimension(row),
                pair_mode="minimal",
                pair_index=0,
                hardness_score=text_similarity(rejected, get_chosen(row)),
            )
        )
    return output


def _rank_group_candidates(
    source_row: Mapping[str, Any], group: list[Mapping[str, Any]], k: int
) -> list[tuple[str, str, float]]:
    chosen = get_chosen(source_row)
    original = get_rejected(source_row)
    original_norm = normalize_text(original)

    dedup: dict[str, tuple[str, str, float]] = {}
    for candidate_row in group:
        candidate = get_rejected(candidate_row)
        candidate_norm = normalize_text(candidate)
        if candidate_norm == normalize_text(chosen):
            continue
        score = text_similarity(candidate, chosen)
        entry = (candidate, get_dimension(candidate_row), score)
        previous = dedup.get(candidate_norm)
        if previous is None or score > previous[2]:
            dedup[candidate_norm] = entry

    if original_norm not in dedup:
        dedup[original_norm] = (original, get_dimension(source_row), text_similarity(original, chosen))
    if len(dedup) < k:
        raise DataValidationError(
            f"question group {get_question_key(source_row)!r} has {len(dedup)} unique flawed responses; K={k}"
        )

    selected: list[tuple[str, str, float]] = [dedup.pop(original_norm)]
    used_dimensions = {selected[0][1]}
    ranked = sorted(dedup.values(), key=lambda item: (-item[2], item[1], normalize_text(item[0])))

    # Fill unseen dimensions first, then use the remaining closest candidates.
    for candidate in ranked:
        if len(selected) >= k:
            break
        if candidate[1] not in used_dimensions:
            selected.append(candidate)
            used_dimensions.add(candidate[1])
    if len(selected) < k:
        selected_norm = {normalize_text(item[0]) for item in selected}
        for candidate in ranked:
            if len(selected) >= k:
                break
            if normalize_text(candidate[0]) not in selected_norm:
                selected.append(candidate)
                selected_norm.add(normalize_text(candidate[0]))

    return selected[:k]


def build_hard_k_pairs(
    rows: list[dict[str, Any]], split: str, *, k: int, seed: int
) -> list[dict[str, Any]]:
    if k < 2:
        raise DataValidationError("hard-pair K must be at least 2")
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[get_question_key(row)].append(row)

    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        candidates = _rank_group_candidates(row, groups[get_question_key(row)], k)
        # Stable seed-dependent rotation affects equally scored candidates without crossing question groups.
        rng = random.Random(f"{seed}:{split}:{get_source_id(row, split, index)}")
        tail = candidates[1:]
        grouped_by_score: defaultdict[float, list[tuple[str, str, float]]] = defaultdict(list)
        for item in tail:
            grouped_by_score[round(item[2], 12)].append(item)
        stable_tail: list[tuple[str, str, float]] = []
        for score in sorted(grouped_by_score, reverse=True):
            bucket = grouped_by_score[score]
            rng.shuffle(bucket)
            stable_tail.extend(bucket)
        candidates = [candidates[0], *stable_tail[: k - 1]]

        for pair_index, (rejected, rejected_dimension, score) in enumerate(candidates):
            output.append(
                _preference_row(
                    source_row=row,
                    split=split,
                    source_index=index,
                    rejected_text=rejected,
                    rejected_dimension=rejected_dimension,
                    pair_mode=f"hard_k{k}",
                    pair_index=pair_index,
                    hardness_score=score,
                )
            )
    return output


def validate_preference_rows(
    rows: list[dict[str, Any]], *, split: str, source_rows: int, pair_mode: str, k: int
) -> dict[str, Any]:
    expected = source_rows if pair_mode == "minimal" else source_rows * k
    if len(rows) != expected:
        raise DataValidationError(f"{split}/{pair_mode}: expected {expected} rows, found {len(rows)}")

    by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source_id"]].append(row)
        expected_prompt = build_prompt_messages(row["question"], row["initial_draft"])
        if row["prompt"] != expected_prompt:
            raise DataValidationError(
                f"{split}/{pair_mode}: prompt contract changed for source_id={row['source_id']}"
            )
        prompt_dump = json.dumps(row["prompt"], ensure_ascii=False).casefold()
        if "target_dimension" in prompt_dump or "violation dimension:" in prompt_dump:
            raise DataValidationError(
                f"{split}/{pair_mode}: flaw metadata leaked into prompt for source_id={row['source_id']}"
            )
        if normalize_text(row["chosen"][0]["content"]) == normalize_text(row["rejected"][0]["content"]):
            raise DataValidationError(f"{split}/{pair_mode}: identical chosen/rejected completion")

    if len(by_source) != source_rows:
        raise DataValidationError(
            f"{split}/{pair_mode}: expected {source_rows} source ids, found {len(by_source)}"
        )

    expected_per_source = 1 if pair_mode == "minimal" else k
    for source_id, source_pairs in by_source.items():
        if len(source_pairs) != expected_per_source:
            raise DataValidationError(
                f"{split}/{pair_mode}: source_id={source_id} has {len(source_pairs)} pairs; "
                f"expected {expected_per_source}"
            )
        rejected_set = {normalize_text(item["rejected"][0]["content"]) for item in source_pairs}
        if len(rejected_set) != expected_per_source:
            raise DataValidationError(f"{split}/{pair_mode}: duplicate negatives for source_id={source_id}")
        initial_draft = normalize_text(source_pairs[0]["initial_draft"])
        if initial_draft not in rejected_set:
            raise DataValidationError(
                f"{split}/{pair_mode}: original input draft is missing from negatives for source_id={source_id}"
            )

    scores = [float(row["hardness_score"]) for row in rows]
    return {
        "rows": len(rows),
        "source_rows": source_rows,
        "pairs_per_source": expected_per_source,
        "hardness_mean": sum(scores) / max(1, len(scores)),
        "hardness_min": min(scores) if scores else math.nan,
        "hardness_max": max(scores) if scores else math.nan,
    }


def prepare_all(
    *,
    specs: Sequence[SplitSpec],
    output_dir: Path,
    modes: Sequence[str],
    k: int,
    seed: int,
) -> dict[str, Any]:
    raw: dict[str, list[dict[str, Any]]] = {}
    manifest: dict[str, Any] = {
        "schema_version": "dpo_exp295_v1",
        "prompt_contract": "x = question + potentially unsafe initial draft; no target_dimension",
        "seed": seed,
        "hard_k": k,
        "input": {},
        "output": {},
    }

    for spec in specs:
        rows = read_jsonl(spec.path)
        raw[spec.name] = rows
        manifest["input"][spec.name] = {
            "path": str(spec.path),
            "sha256": sha256_file(spec.path),
            **validate_split_rows(rows, spec.name, spec.expected_rows),
        }

    question_keys = {split: {get_question_key(row) for row in rows} for split, rows in raw.items()}
    leakage: dict[str, list[str]] = {}
    split_names = sorted(question_keys)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlap = sorted(question_keys[left] & question_keys[right])
            if overlap:
                leakage[f"{left}__{right}"] = overlap[:20]
    if leakage:
        raise DataValidationError(f"question-group leakage across splits: {leakage}")
    manifest["cross_split_question_overlap"] = 0

    for mode in modes:
        if mode not in {"minimal", "hard_k4"}:
            raise DataValidationError(f"unsupported mode={mode!r}")
        mode_key = "minimal" if mode == "minimal" else f"hard_k{k}"
        manifest["output"][mode_key] = {}
        for split, rows in raw.items():
            if mode == "minimal":
                prepared = build_minimal_pairs(rows, split)
            else:
                prepared = build_hard_k_pairs(rows, split, k=k, seed=seed)
            stats = validate_preference_rows(
                prepared,
                split=split,
                source_rows=len(rows),
                pair_mode="minimal" if mode == "minimal" else mode_key,
                k=k,
            )
            output_path = output_dir / mode_key / f"{split}.jsonl"
            write_jsonl(prepared, output_path)
            manifest["output"][mode_key][split] = {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                **stats,
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_input", type=Path, required=True)
    parser.add_argument("--valid_input", type=Path, required=True)
    parser.add_argument("--test_input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("data/dpo_exp295"))
    parser.add_argument(
        "--mode",
        choices=("minimal", "hard_k4", "both"),
        default="both",
        help="Build exact one-negative pairs, K=4 pairs, or both.",
    )
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_train", type=int, default=EXPECTED_COUNTS["train"])
    parser.add_argument("--expected_valid", type=int, default=EXPECTED_COUNTS["valid"])
    parser.add_argument("--expected_test", type=int, default=EXPECTED_COUNTS["test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = ("minimal", "hard_k4") if args.mode == "both" else (args.mode,)
    specs = (
        SplitSpec("train", args.train_input, args.expected_train),
        SplitSpec("valid", args.valid_input, args.expected_valid),
        SplitSpec("test", args.test_input, args.expected_test),
    )
    manifest = prepare_all(
        specs=specs,
        output_dir=args.output_dir,
        modes=modes,
        k=args.k,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
