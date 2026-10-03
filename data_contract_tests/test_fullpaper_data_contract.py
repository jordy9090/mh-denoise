from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from corruption_contract import (  # noqa: E402
    AXES,
    CorruptionRequest,
    QCResult,
    REQUIRED_OUTPUT_FIELDS,
    apply_calibrated_thresholds,
    build_corruption_prompt,
    validate_output_record,
)
from fullpaper_acl_pipeline import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    OPTIONAL_ADAPTERS,
    balanced_axis_specs,
    normalize_text,
    read_jsonl,
    sha256_file,
)
from run_gemma_corruption_pilot import select_stratified_pilot  # noqa: E402


@pytest.fixture(scope="module")
def artifacts() -> tuple[list[dict], dict, list[dict], dict]:
    canonical_path = DEFAULT_OUTPUT_DIR / "canonical_clean_qa.jsonl"
    split_manifest_path = DEFAULT_OUTPUT_DIR / "split_manifest.json"
    assignment_path = DEFAULT_OUTPUT_DIR / "corruption_assignment.jsonl"
    assignment_manifest_path = DEFAULT_OUTPUT_DIR / "corruption_assignment.manifest.json"
    for path in (canonical_path, split_manifest_path, assignment_path, assignment_manifest_path):
        assert path.is_file(), f"Missing required pipeline artifact: {path}"
    canonical = list(read_jsonl(canonical_path))
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    assignments = list(read_jsonl(assignment_path))
    assignment_manifest = json.loads(assignment_manifest_path.read_text(encoding="utf-8"))
    return canonical, split_manifest, assignments, assignment_manifest


def values_crossing_splits(rows: list[dict], field: str) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        values[str(row[field])].add(row["split"])
    return {key: splits for key, splits in values.items() if len(splits) > 1}


def test_clean_manifest_fingerprint_and_counts(artifacts) -> None:
    canonical, manifest, _, _ = artifacts
    canonical_path = DEFAULT_OUTPUT_DIR / "canonical_clean_qa.jsonl"
    assert len(canonical) == manifest["row_count"] == 18224
    assert sha256_file(canonical_path) == manifest["canonical_sha256"]
    assert Counter(row["split"] for row in canonical) == Counter(manifest["split_counts"])
    assert Counter(row["source"] for row in canonical) == Counter(manifest["source_counts"])


def test_complete_counselbench_exclusion(artifacts) -> None:
    canonical, manifest, _, _ = artifacts
    exclusion = manifest["counselbench_exclusion"]
    excluded_ids = set(exclusion["linked_counselchat_question_ids"])
    assert exclusion["benchmark_question_clusters"] == 100
    assert exclusion["directly_linked_counselchat_response_rows"] == 588
    assert exclusion["excluded_counselchat_question_groups"] == 101
    assert exclusion["excluded_counselchat_response_rows"] == 589
    assert not [
        row
        for row in canonical
        if row["source"] == "CounselChat" and row["source_question_id"] in excluded_ids
    ]

    benchmark_manifest = Path(exclusion["manifest"])
    if not benchmark_manifest.is_file():
        benchmark_manifest = REPO_ROOT / "data/fullpaper_phase1/manifests" / benchmark_manifest.name
    benchmark_rows = list(read_jsonl(benchmark_manifest))
    benchmark_questions = {normalize_text(row["combined_question"]) for row in benchmark_rows}
    assert not benchmark_questions.intersection(row["question_normalized"] for row in canonical)


@pytest.mark.parametrize(
    "field",
    [
        "question_normalized_sha256",
        "response_normalized_sha256",
        "duplicate_cluster_id",
        "source_group_id",
    ],
)
def test_no_leakage_across_splits(artifacts, field: str) -> None:
    canonical, _, _, _ = artifacts
    assert values_crossing_splits(canonical, field) == {}


def test_every_row_has_exactly_one_balanced_assignment(artifacts) -> None:
    canonical, split_manifest, assignments, assignment_manifest = artifacts
    canonical_ids = {row["canonical_id"] for row in canonical}
    assignment_ids = [row["canonical_id"] for row in assignments]
    assert len(assignment_ids) == len(set(assignment_ids)) == len(canonical_ids)
    assert set(assignment_ids) == canonical_ids
    assert all(row["axis_count"] == len(row["intended_axes"]) for row in assignments)
    assert all(1 <= row["axis_count"] <= 3 for row in assignments)
    assert all(set(row["intended_axes"]) <= set(AXES) for row in assignments)
    assert all(row["clean_split_sha256"] == split_manifest["canonical_sha256"] for row in assignments)
    assert assignment_manifest["axis_count_counts"] == {"1": 9112, "2": 6378, "3": 2734}
    assert assignment_manifest["marginal_range"] <= 3
    assert assignment_manifest["pairwise_range"] <= 3
    assert assignment_manifest["overall_dimension_policy"].startswith("evaluation-only")


def test_assignment_schedule_is_deterministic() -> None:
    first, first_stats = balanced_axis_specs(18224, 20260904)
    second, second_stats = balanced_axis_specs(18224, 20260904)
    assert first == second
    assert first_stats == second_stats


def test_prompt_uses_behaviors_without_dataset_axis_labels() -> None:
    request = CorruptionRequest(
        canonical_id="qa_test",
        split="train",
        question="I feel anxious before work. What can I do?",
        clean_response="It makes sense to feel worried. Try a short breathing exercise before work.",
        intended_axes=("empathy", "medical_boundary", "toxicity_or_harm"),
        generation_seed=7,
        generation_attempt=1,
    )
    prompt = build_corruption_prompt(request)
    assert all(axis not in prompt.casefold() for axis in AXES)
    assert request.question in prompt
    assert request.clean_response in prompt
    assert "approximately the same response length" in prompt


def test_qc_does_not_assume_intended_equals_realized() -> None:
    request = CorruptionRequest(
        canonical_id="qa_test",
        split="valid",
        question="question",
        clean_response="answer",
        intended_axes=("specificity",),
        generation_seed=7,
        generation_attempt=1,
    )
    result = QCResult(
        realized_axes=("empathy",),
        scores={
            "axis_degradation": {axis: 0.0 for axis in AXES},
            "topic_relevance": 5.0,
            "fluency": 5.0,
            "unintended_catastrophic_degradation": 0.0,
            "medical_boundary_realized": False,
            "toxicity_or_harm_realized": False,
            "notes": "",
        },
        qc_pass=None,
        failure_reason=None,
        raw_output="{}",
    )
    uncalibrated = apply_calibrated_thresholds(request, result, None)
    assert uncalibrated.realized_axes != request.intended_axes
    assert uncalibrated.qc_pass is None


def test_generator_output_contract() -> None:
    record = {field: None for field in REQUIRED_OUTPUT_FIELDS}
    record.update(
        {
            "canonical_id": "qa_test",
            "split": "test",
            "question": "q",
            "clean_response": "clean",
            "intended_axes": ["empathy"],
            "axis_count": 1,
            "generator_repo": "repo",
            "generator_revision": "a" * 40,
            "generation_seed": 1,
            "generation_attempt": 1,
            "corrupted_response": "corrupt",
            "realized_axes": [],
            "qc_scores": {},
            "qc_pass": None,
            "qc_failure_reason": None,
        }
    )
    validate_output_record(record)


def test_stratified_pilot_selection(artifacts) -> None:
    canonical, _, assignments, _ = artifacts
    first = select_stratified_pilot(canonical, assignments, size=50, seed=20260904)
    second = select_stratified_pilot(canonical, assignments, size=50, seed=20260904)
    assert [row["canonical_id"] for row in first] == [row["canonical_id"] for row in second]
    assert Counter(row["source"] for row in first) == {"MentalChat16K": 44, "CounselChat": 6}
    assert Counter(row["axis_count"] for row in first) == {1: 25, 2: 18, 3: 7}
    marginal = Counter(axis for row in first for axis in row["intended_axes"])
    assert max(marginal.values()) - min(marginal.values()) <= 1


def test_optional_adapters_do_not_block_defaults(tmp_path: Path) -> None:
    assert set(OPTIONAL_ADAPTERS) == {"psych8k", "psyqa"}
    with pytest.raises(FileNotFoundError):
        OPTIONAL_ADAPTERS["psych8k"].load(tmp_path / "not-authorized.json")
    with pytest.raises(FileNotFoundError):
        OPTIONAL_ADAPTERS["psyqa"].load(tmp_path / "not-authorized.json")


def test_completed_pilot_contract_if_present() -> None:
    pilot_path = DEFAULT_OUTPUT_DIR / "gemma_corruption_pilot_50.jsonl"
    if not pilot_path.is_file():
        pytest.skip("Pilot has not been run")
    rows = list(read_jsonl(pilot_path))
    assert len(rows) == 50
    for row in rows:
        validate_output_record(row)
        assert row["generator_repo"] == "google/gemma-4-E4B-it"
        assert row["generator_revision"] == "ee0ef6023621cff504d758262d4e04895a5af4a2"
        assert row["corrupted_response"].strip()
        assert normalize_text(row["corrupted_response"]) != normalize_text(row["clean_response"])
        assert set(row["qc_scores"]["axis_degradation"]) == set(AXES)
        assert row["qc_pass"] is None
