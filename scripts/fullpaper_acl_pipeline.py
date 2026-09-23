#!/usr/bin/env python3
"""Build the leakage-safe full-paper clean-QA split and corruption plan.

This pipeline intentionally stops before text generation.  The clean split is
materialized and fingerprinted first; corruption assignments are then derived
from that frozen file in a separate command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import itertools
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE1_ROOT = REPO_ROOT / "data/fullpaper_phase1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/fullpaper_acl_pipeline"
DEFAULT_SEED = 20260904
SPLIT_RATIOS = {"train": 0.80, "valid": 0.10, "test": 0.10}
AXES = (
    "empathy",
    "specificity",
    "factual_consistency",
    "medical_boundary",
    "toxicity_or_harm",
)
AXIS_COUNT_RATIOS = {1: 0.50, 2: 0.35, 3: 0.15}
SOURCE_REVISIONS = {
    "CounselBench-100": "8d56a96ea1de3f3f190f77f4ca9bc3503d731af7",
    "CounselChat": "17501f72697cf8018aaf496162e9dc1408a64e67",
    "MentalChat16K": "5f60cd380cfc58f0f12d44892bed41ee3670a70a",
}
SOURCE_REPOS = {
    "CounselChat": "nbertagnolli/counsel-chat",
    "MentalChat16K": "ShenLab/MentalChat16K",
}

SPACE_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]+>")
NON_WORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)
QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
    }
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_display(value: Any) -> str:
    if value is None:
        return ""
    value = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return SPACE_RE.sub(" ", value).strip()


def normalize_text(value: Any) -> str:
    """Conservative duplicate normalization used before any split."""

    value = html.unescape(str(value or ""))
    value = TAG_RE.sub(" ", value)
    value = unicodedata.normalize("NFKC", value).translate(QUOTE_TRANSLATION)
    value = NON_WORD_RE.sub(" ", value.casefold())
    return SPACE_RE.sub(" ", value).strip()


def combined_question(title: Any, body: Any) -> str:
    title_text = clean_display(title)
    body_text = clean_display(body)
    if title_text and body_text and normalize_text(title_text) != normalize_text(body_text):
        return f"{title_text}\n\n{body_text}"
    return title_text or body_text


def stable_random_key(seed: int, value: str) -> str:
    return sha256_text(f"{seed}\0{value}")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class OptionalSourceAdapter(Protocol):
    """Optional datasets must be explicit adapters and never default inputs."""

    name: str

    def load(self, path: Path) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class Psych8kAdapter:
    name: str = "Psych8k"

    def load(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError("Psych8k is optional; supply an authorized local JSON file explicitly")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Psych8k adapter expects a JSON list")
        return payload


@dataclass(frozen=True)
class PsyQAAdapter:
    name: str = "PsyQA"

    def load(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError("PsyQA is optional; supply an authorized local JSON file explicitly")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("PsyQA adapter expects a JSON list")
        return payload


OPTIONAL_ADAPTERS: dict[str, OptionalSourceAdapter] = {
    "psych8k": Psych8kAdapter(),
    "psyqa": PsyQAAdapter(),
}


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def load_exclusion_manifest(path: Path) -> dict[str, Any]:
    benchmark_rows = list(read_jsonl(path))
    linked_question_ids: set[str] = set()
    linked_cluster_ids: set[str] = set()
    benchmark_questions: set[str] = set()
    expected_linked_response_rows = 0

    for row in benchmark_rows:
        benchmark_questions.add(normalize_text(row["combined_question"]))
        direct = row.get("direct_counselchat_match") or {}
        if direct.get("question_id") is not None:
            linked_question_ids.add(str(direct["question_id"]))
        expected_linked_response_rows += int(direct.get("response_rows") or 0)
        linked_cluster_ids.update(str(x) for x in direct.get("matched_cluster_ids", []))
        for field in ("exact_question_matches", "normalized_only_question_matches"):
            matches = (row.get(field) or {}).get("CounselChat", [])
            linked_cluster_ids.update(str(x) for x in matches)

    for cluster_id in linked_cluster_ids:
        match = re.fullmatch(r"counselchat:questionID:(.+)", cluster_id)
        if match:
            linked_question_ids.add(match.group(1))

    if len(benchmark_rows) != 100:
        raise AssertionError(f"Expected CounselBench-100 manifest to contain 100 rows; got {len(benchmark_rows)}")
    if len(linked_question_ids) != 100:
        raise AssertionError(f"Expected 100 linked CounselChat question groups; got {len(linked_question_ids)}")

    return {
        "benchmark_rows": benchmark_rows,
        "benchmark_normalized_questions": benchmark_questions,
        "linked_question_ids": linked_question_ids,
        "linked_cluster_ids": linked_cluster_ids,
        "expected_linked_response_rows": expected_linked_response_rows,
    }


def canonical_id(row: dict[str, Any]) -> str:
    identity = "\0".join(
        [
            "fullpaper-clean-qa-v1",
            row["source_repo"],
            row["source_revision"],
            row["source_component"],
            str(row["source_row_index"]),
            row["source_group_id"],
            sha256_text(row["clean_response"]),
        ]
    )
    return "qa_" + sha256_text(identity)[:24]


def make_row(
    *,
    source: str,
    component: str,
    source_file: Path,
    row_index: int,
    source_question_id: str,
    source_group_id: str,
    source_group_basis: str,
    question: str,
    response: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    exact_question = clean_display(question)
    exact_response = clean_display(response)
    normalized_question = normalize_text(question)
    normalized_response = normalize_text(response)
    row = {
        "schema_version": "canonical-clean-qa-v1",
        "canonical_id": "",
        "source": source,
        "source_component": component,
        "source_repo": SOURCE_REPOS[source],
        "source_revision": SOURCE_REVISIONS[source],
        "source_file": str(source_file.resolve()),
        "source_row_index": row_index,
        "source_question_id": source_question_id,
        "source_group_id": source_group_id,
        "source_group_basis": source_group_basis,
        "question": question.strip(),
        "clean_response": response.strip(),
        "question_exact_sha256": sha256_text(exact_question),
        "question_normalized": normalized_question,
        "question_normalized_sha256": sha256_text(normalized_question),
        "response_exact_sha256": sha256_text(exact_response),
        "response_normalized_sha256": sha256_text(normalized_response),
        "duplicate_cluster_id": "",
        "split": "",
        "provenance": provenance,
    }
    row["canonical_id"] = canonical_id(row)
    return row


def load_immediate_sources(
    *,
    phase1_root: Path,
    exclusion_manifest: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exclusion = load_exclusion_manifest(exclusion_manifest)
    rows: list[dict[str, Any]] = []
    skipped = Counter()

    mental_sources = [
        ("interview", phase1_root / "raw_sources/mentalchat16k/Interview_Data_6K.csv"),
        ("synthetic", phase1_root / "raw_sources/mentalchat16k/Synthetic_Data_10K.csv"),
    ]
    for component, source_file in mental_sources:
        with source_file.open(encoding="utf-8-sig", newline="") as handle:
            for row_index, raw in enumerate(csv.DictReader(handle)):
                question = str(raw.get("input") or "").strip()
                response = str(raw.get("output") or "").strip()
                if not normalize_text(question):
                    skipped[f"MentalChat16K:{component}:empty_question"] += 1
                    continue
                if not normalize_text(response):
                    skipped[f"MentalChat16K:{component}:empty_response"] += 1
                    continue
                fallback_id = f"mentalchat:{component}:row:{row_index}"
                rows.append(
                    make_row(
                        source="MentalChat16K",
                        component=component,
                        source_file=source_file,
                        row_index=row_index,
                        source_question_id=fallback_id,
                        source_group_id=fallback_id,
                        source_group_basis="released schema has no transcript/session ID; immutable source-row fallback",
                        question=question,
                        response=response,
                        provenance={
                            "instruction": raw.get("instruction"),
                            "released_session_id_available": False,
                        },
                    )
                )

    counsel_file = phase1_root / "raw_sources/counsel_chat/20220401_counsel_chat.csv"
    counsel_raw: list[dict[str, str]] = []
    with counsel_file.open(encoding="utf-8-sig", newline="") as handle:
        counsel_raw = list(csv.DictReader(handle))

    # Exclusion is decided at question-group level before any response filtering.
    group_questions: dict[str, str] = {}
    for raw in counsel_raw:
        question_id = str(raw.get("questionID") or "").strip()
        group_questions.setdefault(
            question_id,
            combined_question(raw.get("questionTitle"), raw.get("questionText")),
        )
    normalized_benchmark = exclusion["benchmark_normalized_questions"]
    excluded_question_ids = set(exclusion["linked_question_ids"])
    excluded_question_ids.update(
        question_id
        for question_id, question in group_questions.items()
        if normalize_text(question) in normalized_benchmark
    )

    excluded_response_rows = 0
    for row_index, raw in enumerate(counsel_raw):
        question_id = str(raw.get("questionID") or "").strip()
        if question_id in excluded_question_ids:
            excluded_response_rows += 1
            continue
        question = combined_question(raw.get("questionTitle"), raw.get("questionText"))
        response = str(raw.get("answerText") or "").strip()
        if not normalize_text(question):
            skipped["CounselChat:empty_question"] += 1
            continue
        if not normalize_text(response):
            skipped["CounselChat:empty_response"] += 1
            continue
        rows.append(
            make_row(
                source="CounselChat",
                component="20220401",
                source_file=counsel_file,
                row_index=row_index,
                source_question_id=question_id,
                source_group_id=f"counselchat:questionID:{question_id}",
                source_group_basis="questionID; all linked therapist responses form one group",
                question=question,
                response=response,
                provenance={
                    "question_id": question_id,
                    "question_link": raw.get("questionLink"),
                    "question_title": raw.get("questionTitle"),
                    "question_text": raw.get("questionText"),
                    "topic": raw.get("topic"),
                    "therapist_info": raw.get("therapistInfo"),
                    "therapist_url": raw.get("therapistURL"),
                    "upvotes": raw.get("upvotes"),
                    "views": raw.get("views"),
                },
            )
        )

    if excluded_response_rows < exclusion["expected_linked_response_rows"]:
        raise AssertionError(
            "CounselBench exclusion removed fewer than the directly linked "
            f"CounselChat responses: {excluded_response_rows} < "
            f"{exclusion['expected_linked_response_rows']}"
        )

    intake = {
        "source_input_rows": {
            "MentalChat16K-interview": 6310,
            "MentalChat16K-synthetic": 9774,
            "CounselChat": len(counsel_raw),
        },
        "skipped_rows": dict(sorted(skipped.items())),
        "counselbench_questions": len(exclusion["benchmark_rows"]),
        "excluded_counselchat_question_groups": len(excluded_question_ids),
        "excluded_counselchat_response_rows": excluded_response_rows,
        "directly_linked_counselchat_response_rows": exclusion["expected_linked_response_rows"],
        "additional_normalized_link_response_rows": (
            excluded_response_rows - exclusion["expected_linked_response_rows"]
        ),
        "linked_counselchat_question_ids": sorted(excluded_question_ids, key=lambda x: (len(x), x)),
        "benchmark_normalized_questions": normalized_benchmark,
    }
    return rows, intake


def assign_duplicate_clusters(rows: list[dict[str, Any]]) -> None:
    uf = UnionFind(len(rows))
    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        keys = [
            ("source_group", row["source_group_id"]),
            ("question_exact", row["question_exact_sha256"]),
            ("question_normalized", row["question_normalized_sha256"]),
            ("response_exact", row["response_exact_sha256"]),
            ("response_normalized", row["response_normalized_sha256"]),
        ]
        for key in keys:
            previous = seen.setdefault(key, index)
            uf.union(index, previous)

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        members[uf.find(index)].append(index)
    for indices in members.values():
        member_ids = sorted(rows[index]["canonical_id"] for index in indices)
        cluster_id = "dup_" + sha256_text("\n".join(member_ids))[:24]
        for index in indices:
            rows[index]["duplicate_cluster_id"] = cluster_id


def largest_remainder_counts(total: int, ratios: dict[Any, float]) -> dict[Any, int]:
    raw = {key: total * ratio for key, ratio in ratios.items()}
    counts = {key: math.floor(value) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(ratios, key=lambda key: (-(raw[key] - counts[key]), str(key)))
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def assign_splits(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    cluster_indices: dict[str, list[int]] = defaultdict(list)
    sources = sorted({row["source"] for row in rows})
    for index, row in enumerate(rows):
        cluster_indices[row["duplicate_cluster_id"]].append(index)

    targets_total = largest_remainder_counts(len(rows), SPLIT_RATIOS)
    source_totals = Counter(row["source"] for row in rows)
    targets_source = {
        source: largest_remainder_counts(source_totals[source], SPLIT_RATIOS)
        for source in sources
    }
    current_total = Counter()
    current_source: dict[str, Counter[str]] = {source: Counter() for source in sources}
    split_names = tuple(SPLIT_RATIOS)

    def global_loss(
        candidate: str,
        unit_size: int,
        unit_sources: Counter[str],
    ) -> int:
        # Source-specific targets are sufficient to recover the global target
        # because they sum to the full dataset.  A second global term would
        # swamp the smaller CounselChat source while large clusters are placed.
        loss = 0
        for split in split_names:
            for source in sources:
                source_value = current_source[source][split]
                if split == candidate:
                    source_value += unit_sources[source]
                loss += (source_value - targets_source[source][split]) ** 2
        return loss

    ordered_clusters = sorted(
        cluster_indices,
        key=lambda cluster_id: (
            -len(cluster_indices[cluster_id]),
            stable_random_key(seed, cluster_id),
        ),
    )
    for cluster_id in ordered_clusters:
        indices = cluster_indices[cluster_id]
        unit_sources = Counter(rows[index]["source"] for index in indices)
        tie_order = sorted(split_names, key=lambda split: stable_random_key(seed, f"{cluster_id}:{split}"))
        chosen = min(
            tie_order,
            key=lambda split: global_loss(split, len(indices), unit_sources),
        )
        for index in indices:
            rows[index]["split"] = chosen
        current_total[chosen] += len(indices)
        for source, count in unit_sources.items():
            current_source[source][chosen] += count

    return {
        "targets": targets_total,
        "actual": dict(current_total),
        "source_targets": {source: dict(value) for source, value in targets_source.items()},
        "source_actual": {source: dict(value) for source, value in current_source.items()},
        "duplicate_clusters": len(cluster_indices),
    }


def split_invariants(
    rows: list[dict[str, Any]],
    benchmark_normalized_questions: set[str],
    excluded_question_ids: set[str],
) -> dict[str, Any]:
    errors: list[str] = []

    contaminated = [
        row["canonical_id"]
        for row in rows
        if row["question_normalized"] in benchmark_normalized_questions
        or (
            row["source"] == "CounselChat"
            and row["source_question_id"] in excluded_question_ids
        )
    ]
    if contaminated:
        errors.append(f"CounselBench contamination: {contaminated[:10]}")

    def overlaps(field: str) -> dict[str, list[str]]:
        values: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            values[str(row[field])].add(row["split"])
        return {key: sorted(value) for key, value in values.items() if len(value) > 1}

    question_overlap = overlaps("question_normalized_sha256")
    response_overlap = overlaps("response_normalized_sha256")
    cluster_overlap = overlaps("duplicate_cluster_id")
    group_overlap = overlaps("source_group_id")
    if question_overlap:
        errors.append(f"Question overlap across splits: {list(question_overlap.items())[:5]}")
    if response_overlap:
        errors.append(f"Response overlap across splits: {list(response_overlap.items())[:5]}")
    if cluster_overlap:
        errors.append(f"Duplicate-cluster overlap across splits: {list(cluster_overlap.items())[:5]}")
    if group_overlap:
        errors.append(f"Source-group overlap across splits: {list(group_overlap.items())[:5]}")
    if errors:
        raise AssertionError("\n".join(errors))
    return {
        "zero_counselbench100_contamination": True,
        "zero_normalized_question_overlap_across_splits": True,
        "zero_normalized_response_overlap_across_splits": True,
        "zero_duplicate_cluster_overlap_across_splits": True,
        "zero_source_group_overlap_across_splits": True,
    }


def build_clean(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    exclusion_path = Path(args.exclusion_manifest).resolve()
    rows, intake = load_immediate_sources(
        phase1_root=Path(args.phase1_root).resolve(),
        exclusion_manifest=exclusion_path,
    )
    assign_duplicate_clusters(rows)
    split_stats = assign_splits(rows, args.seed)
    invariants = split_invariants(
        rows,
        set(intake.pop("benchmark_normalized_questions")),
        set(intake["linked_counselchat_question_ids"]),
    )

    rows.sort(key=lambda row: (row["source"], row["source_component"], row["source_row_index"], row["canonical_id"]))
    canonical_path = output_dir / "canonical_clean_qa.jsonl"
    write_jsonl(canonical_path, rows)

    split_counts = Counter(row["split"] for row in rows)
    source_counts = Counter(row["source"] for row in rows)
    component_counts = Counter(f"{row['source']}:{row['source_component']}" for row in rows)
    manifest = {
        "manifest_version": "fullpaper-clean-split-v1",
        "seed": args.seed,
        "split_ratios": SPLIT_RATIOS,
        "canonical_file": str(canonical_path),
        "canonical_sha256": sha256_file(canonical_path),
        "row_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "source_component_counts": dict(sorted(component_counts.items())),
        "unique_question_hashes": len({row["question_normalized_sha256"] for row in rows}),
        "unique_response_hashes": len({row["response_normalized_sha256"] for row in rows}),
        "unique_duplicate_clusters": len({row["duplicate_cluster_id"] for row in rows}),
        "unique_source_groups": len({row["source_group_id"] for row in rows}),
        "split_assignment": split_stats,
        "counselbench_exclusion": {
            "manifest": str(exclusion_path),
            "manifest_sha256": sha256_file(exclusion_path),
            "benchmark_revision": SOURCE_REVISIONS["CounselBench-100"],
            "benchmark_question_clusters": intake["counselbench_questions"],
            "excluded_counselchat_question_groups": intake["excluded_counselchat_question_groups"],
            "excluded_counselchat_response_rows": intake["excluded_counselchat_response_rows"],
            "directly_linked_counselchat_response_rows": intake[
                "directly_linked_counselchat_response_rows"
            ],
            "additional_normalized_link_response_rows": intake[
                "additional_normalized_link_response_rows"
            ],
            "linked_counselchat_question_ids": intake["linked_counselchat_question_ids"],
        },
        "intake": {key: value for key, value in intake.items() if key != "linked_counselchat_question_ids"},
        "assertions": invariants,
        "optional_adapters": {
            "Psych8k": "not enabled; explicit authorized local path required",
            "PsyQA": "not enabled; explicit authorized local path required",
        },
        "known_grouping_limitation": (
            "MentalChat16K publishes no transcript/session identifier. Its immutable source-row ID is "
            "therefore the source-group fallback; duplicate question/response linkage is still clustered globally."
        ),
    }
    manifest_path = output_dir / "split_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({"canonical": str(canonical_path), "manifest": str(manifest_path), "rows": len(rows), "splits": split_counts}, default=dict))
    return manifest


def balanced_axis_specs(total: int, seed: int) -> tuple[list[tuple[str, ...]], dict[str, Any]]:
    count_quotas = largest_remainder_counts(total, AXIS_COUNT_RATIOS)
    specs: list[tuple[str, ...]] = []
    combination_counts: Counter[tuple[str, ...]] = Counter()
    for axis_count in sorted(count_quotas):
        combinations = list(itertools.combinations(AXES, axis_count))
        combinations.sort(key=lambda combo: stable_random_key(seed + axis_count, "|".join(combo)))
        quota = count_quotas[axis_count]
        base, remainder = divmod(quota, len(combinations))
        for index, combo in enumerate(combinations):
            count = base + (1 if index < remainder else 0)
            specs.extend([combo] * count)
            combination_counts[combo] += count

    rng = random.Random(seed)
    rng.shuffle(specs)
    marginal = Counter(axis for spec in specs for axis in spec)
    pairwise = Counter(pair for spec in specs for pair in itertools.combinations(spec, 2))
    for pair in itertools.combinations(AXES, 2):
        pairwise[pair] += 0
    return specs, {
        "axis_count_counts": {str(key): value for key, value in sorted(count_quotas.items())},
        "marginal_counts": {axis: marginal[axis] for axis in AXES},
        "marginal_range": max(marginal.values()) - min(marginal.values()),
        "pairwise_counts": {"+".join(pair): pairwise[pair] for pair in itertools.combinations(AXES, 2)},
        "pairwise_range": max(pairwise.values()) - min(pairwise.values()),
        "combination_counts": {"+".join(combo): count for combo, count in sorted(combination_counts.items())},
    }


def assign_corruptions(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    canonical_path = Path(args.canonical).resolve()
    split_manifest_path = Path(args.split_manifest).resolve()
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    canonical_hash = sha256_file(canonical_path)
    if split_manifest["canonical_sha256"] != canonical_hash:
        raise AssertionError("Canonical clean-QA file differs from the frozen split manifest")
    rows = list(read_jsonl(canonical_path))
    if len(rows) != split_manifest["row_count"]:
        raise AssertionError("Canonical row count differs from the frozen split manifest")

    specs, balance = balanced_axis_specs(len(rows), args.seed)
    ordered_rows = sorted(rows, key=lambda row: stable_random_key(args.seed, row["canonical_id"]))
    assignments = []
    for row, axes in zip(ordered_rows, specs, strict=True):
        assignments.append(
            {
                "assignment_version": "counselbench-aligned-atomic-v1",
                "canonical_id": row["canonical_id"],
                "split": row["split"],
                "source": row["source"],
                "intended_axes": list(axes),
                "axis_count": len(axes),
                "assignment_seed": args.seed,
                "clean_split_sha256": canonical_hash,
            }
        )
    assignments.sort(key=lambda row: row["canonical_id"])
    output_path = output_dir / "corruption_assignment.jsonl"
    write_jsonl(output_path, assignments)
    manifest = {
        "manifest_version": "corruption-assignment-v1",
        "assignment_file": str(output_path),
        "assignment_sha256": sha256_file(output_path),
        "canonical_file": str(canonical_path),
        "canonical_sha256": canonical_hash,
        "seed": args.seed,
        "row_count": len(assignments),
        "one_specification_per_canonical_row": (
            len(assignments) == len(rows) == len({row["canonical_id"] for row in assignments})
        ),
        "primary_axes": list(AXES),
        "overall_dimension_policy": "evaluation-only; not a corruption operator",
        **balance,
    }
    if not manifest["one_specification_per_canonical_row"]:
        raise AssertionError("Corruption assignment is not one-to-one with canonical rows")
    write_json(output_dir / "corruption_assignment.manifest.json", manifest)
    print(json.dumps({"assignment": str(output_path), "rows": len(assignments), **balance}))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean = subparsers.add_parser("build-clean", help="build and freeze the canonical clean split")
    clean.add_argument("--phase1-root", default=str(PHASE1_ROOT))
    clean.add_argument(
        "--exclusion-manifest",
        default=str(PHASE1_ROOT / "manifests/counselbench100_exclusion_manifest.jsonl"),
    )
    clean.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    clean.add_argument("--seed", type=int, default=DEFAULT_SEED)
    clean.set_defaults(function=build_clean)

    assignment = subparsers.add_parser("assign-corruptions", help="assign corruption specs to a frozen clean split")
    assignment.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    assignment.add_argument("--canonical", default=str(DEFAULT_OUTPUT_DIR / "canonical_clean_qa.jsonl"))
    assignment.add_argument("--split-manifest", default=str(DEFAULT_OUTPUT_DIR / "split_manifest.json"))
    assignment.add_argument("--seed", type=int, default=DEFAULT_SEED)
    assignment.set_defaults(function=assign_corruptions)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
