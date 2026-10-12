import json
import sys
from pathlib import Path

from build_frozen_revision2_test_export import main


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_builds_flat_evaluation_only_test_export(tmp_path: Path, monkeypatch):
    production = tmp_path / "production"
    selection = tmp_path / "selection"
    frozen = tmp_path / "frozen"
    output = tmp_path / "output"
    accepted = {
        "status": "accepted",
        "canonical_id": "qa_test",
        "split": "test",
        "source": "fixture",
        "source_component": "fixture",
        "question": "A question",
        "clean_response": "A safe response",
        "corrupted_response": "An unsafe response",
        "question_normalized_sha256": "qhash",
        "duplicate_cluster_id": "dup-test",
        "source_group_id": "group-test",
        "intended_axes": ["empathy"],
        "realized_axes": ["empathy"],
        "unintended_axes": [],
        "axis_count": 1,
        "generation_objective": "response_pair",
        "generation_seeds": [{"attempt_1": 1}],
        "stage_history": [{"axis": "empathy"}],
        "usage": {"generator_calls": 1, "judge_calls": 2},
        "automatic_clean_preflight": {"eligible": True},
        "final_grade": {"clean_target_eligibility": {"disposition": "pass"}},
        "generator_repo": "generator",
        "generator_revision": "generator-revision",
        "generator_prompt_version": "generator-prompt",
        "judge_model": "judge",
        "judge_repo": "judge",
        "judge_revision": "judge-revision",
        "judge_prompt_version": "judge-prompt",
    }
    write_json(
        production / "production_manifest.json",
        {
            "status": "complete",
            "allowed_splits": ["test"],
            "run_fingerprint": "run-fingerprint",
            "structural_validation": {"passed": True},
        },
    )
    for name, rows in {
        "accepted.jsonl": [accepted],
        "rejected.jsonl": [],
        "qc_conflicts.jsonl": [],
        "qc_holds.jsonl": [],
        "source_verified_spans.jsonl": [],
    }.items():
        write_jsonl(production / name, rows)
    write_json(
        selection / "test_generation_selection.json",
        {"rows": [{"canonical_id": "qa_test", "split": "test"}]},
    )
    (selection / "excluded_development_pilot_test_ids.txt").write_text("qa_pilot\n", encoding="utf-8")
    write_jsonl(frozen / "sft_train.jsonl", [])
    write_jsonl(frozen / "sft_valid.jsonl", [])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_frozen_revision2_test_export.py",
            "--production-dir",
            str(production),
            "--selection-dir",
            str(selection),
            "--frozen-train-valid-dir",
            str(frozen),
            "--output-dir",
            str(output),
        ],
    )
    main()

    row = json.loads((output / "sft_test.jsonl").read_text())
    assert row["canonical_id"] == "qa_test"
    assert row["split"] == "test"
    assert row["question"] == "A question"
    assert row["unsafe_response"] == "An unsafe response"
    assert row["safe_response"] == "A safe response"
    assert row["generation_provenance"]["stage_history"] == [{"axis": "empathy"}]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete_evaluation_only_test_export"
    assert manifest["counts"]["accepted_pairs"] == 1
