import json
from pathlib import Path

import pytest

from build_fullpaper_test_selection import (
    expand_pilot_test_question_exclusions,
    load_pilot_test_ids,
)
from fullpaper_acl_pipeline import read_jsonl
from run_fullpaper_corruption_production import LOCAL_JUDGE_VERSION, export, load_selection


def canonical_row(canonical_id: str, question_hash: str, split: str = "test") -> dict:
    return {
        "canonical_id": canonical_id,
        "split": split,
        "question": f"question {canonical_id}",
        "clean_response": f"clean {canonical_id}",
        "question_normalized_sha256": question_hash,
        "duplicate_cluster_id": f"dup-{canonical_id}",
        "source_group_id": f"group-{canonical_id}",
        "source": "fixture",
        "source_component": "fixture",
    }


def qc_row(canonical_id: str) -> dict:
    return {
        "canonical_id": canonical_id,
        "qc_ok": True,
        "eligible": True,
        "baseline_degraded_axes": [],
    }


def write_selection(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")


def test_production_loader_accepts_canonical_test_split(tmp_path: Path):
    row = canonical_row("qa_test", "qhash")
    selection = tmp_path / "selection.json"
    write_selection(selection, [{"canonical_id": "qa_test", "split": "test", "intended_axes": ["empathy"]}])

    loaded = load_selection(
        selection,
        1,
        {"qa_test": row},
        {"qa_test": qc_row("qa_test")},
        set(),
        {"test"},
    )

    assert len(loaded) == 1
    assert loaded[0]["split"] == "test"


def test_production_loader_rejects_duplicate_test_question(tmp_path: Path):
    rows = [canonical_row("qa_a", "same"), canonical_row("qa_b", "same")]
    selection = tmp_path / "selection.json"
    write_selection(
        selection,
        [
            {"canonical_id": "qa_a", "split": "test", "intended_axes": ["empathy"]},
            {"canonical_id": "qa_b", "split": "test", "intended_axes": ["specificity"]},
        ],
    )

    with pytest.raises(ValueError, match="More than one response selected"):
        load_selection(
            selection,
            2,
            {row["canonical_id"]: row for row in rows},
            {row["canonical_id"]: qc_row(row["canonical_id"]) for row in rows},
            set(),
            {"test"},
        )


def test_pilot_test_ids_only_include_test_rows(tmp_path: Path):
    path = tmp_path / "pilot.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {"canonical_id": "qa_train", "split": "train"},
                    {"canonical_id": "qa_test", "split": "test"},
                ]
            }
        ),
        encoding="utf-8",
    )

    ids, provenance = load_pilot_test_ids([path])

    assert ids == {"qa_test"}
    assert provenance[0]["test_ids"] == ["qa_test"]


def test_pilot_question_exclusion_expands_to_separate_ids_with_same_question():
    canonical = [
        canonical_row("qa_pilot", "shared-question"),
        canonical_row("qa_separate", "shared-question"),
        canonical_row("qa_other", "other-question"),
        canonical_row("qa_train", "shared-question", split="train"),
    ]

    groups, excluded_ids = expand_pilot_test_question_exclusions(canonical, {"qa_pilot"})

    assert groups == {"shared-question"}
    assert excluded_ids == {"qa_pilot", "qa_separate"}


def test_pilot_question_exclusion_requires_canonical_pilot_id():
    with pytest.raises(ValueError, match="missing from canonical"):
        expand_pilot_test_question_exclusions(
            [canonical_row("qa_other", "other-question")], {"qa_missing"}
        )


def test_production_export_preserves_test_split(tmp_path: Path):
    result = {
        "status": "accepted",
        "canonical_id": "qa_test",
        "split": "test",
        "source": "fixture",
        "source_component": "fixture",
        "question": "What should I do?",
        "clean_response": "A safe answer.",
        "corrupted_response": "A degraded answer.",
        "baseline_degraded_axes": [],
        "intended_axes": ["empathy"],
        "realized_axes": ["empathy"],
        "unintended_axes": [],
        "axis_count": 1,
        "judge_model": "fixture-judge",
        "judge_repo": "fixture-judge",
        "judge_revision": "fixture-revision",
        "generator_repo": "fixture-generator",
        "generator_revision": "fixture-revision",
        "generation_objective": "response_pair",
        "question_normalized_sha256": "qhash",
        "duplicate_cluster_id": "dup",
        "source_group_id": "group",
        "generator_prompt_version": "fixture-generator-prompt",
        "judge_prompt_version": LOCAL_JUDGE_VERSION,
        "final_grade": {
            "clean_target_eligibility": {"disposition": "pass"},
            "local_supervision": [],
            "response_level_evidence": [],
        },
    }
    manifest = {"version": LOCAL_JUDGE_VERSION}

    export(tmp_path, [result], manifest)

    test_sft = list(read_jsonl(tmp_path / "test_sft.jsonl"))
    test_dpo = list(read_jsonl(tmp_path / "test_dpo.jsonl"))
    assert len(test_sft) == len(test_dpo) == 1
    assert test_sft[0]["metadata"]["split"] == "test"
    assert test_sft[0]["input"]["corrupted_response"] == "A degraded answer."
    assert test_dpo[0]["chosen"] == "A safe answer."
    assert list(read_jsonl(tmp_path / "train_sft.jsonl")) == []
    assert list(read_jsonl(tmp_path / "valid_sft.jsonl")) == []
    assert manifest["accepted_by_split"] == {"test": 1}
