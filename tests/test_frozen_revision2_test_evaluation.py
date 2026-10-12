import csv
import json
import sys
from pathlib import Path

from summarize_frozen_revision2_test_evaluation import main


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def scores(overall: int) -> dict:
    return {
        "overall_quality": overall,
        "empathy": 4,
        "specificity": 3,
        "medical_advice": 5,
        "factual_consistency": 5,
        "toxicity": 1,
    }


def test_summarizes_fixed_three_method_test_membership(tmp_path: Path, monkeypatch):
    common = {
        "canonical_id": "qa_test",
        "id": "qa_test",
        "split": "test",
        "question": "Question",
        "unsafe_response": "Unsafe",
        "safe_response": "Safe",
    }
    sft = tmp_path / "sft.jsonl"
    dpo = tmp_path / "dpo.jsonl"
    proposed = tmp_path / "proposed.jsonl"
    judged = tmp_path / "judged.jsonl"
    write_jsonl(sft, [{**common, "sft_response": "SFT response"}])
    write_jsonl(dpo, [{**common, "sft_response": "SFT response"}])
    write_jsonl(
        proposed,
        [
            {
                **common,
                "sft_response": "SFT response",
                "denoiser_response": "Improved response",
                "final_response": "Improved response",
                "used_denoiser": True,
                "accepted_denoiser": True,
                "reject_reason": "",
            }
        ],
    )
    write_jsonl(
        judged,
        [
            {
                "example_id": f"qa_test__{system}",
                "base_example_id": "qa_test",
                "system": system,
                "judge_ok": True,
                "judge_scores": scores(5 if system == "mask_OFF" else 4),
                "judge_rationale": "fixture",
                "exact_response_shared_from_example_id": f"qa_test__{system}",
            }
            for system in ("SFT", "DPO", "mask_OFF")
        ],
    )
    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_frozen_revision2_test_evaluation.py",
            "--sft",
            str(sft),
            "--dpo",
            str(dpo),
            "--proposed",
            str(proposed),
            "--judged",
            str(judged),
            "--output-dir",
            str(output),
        ],
    )
    main()

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["rows_per_method"] == 1
    assert manifest["proposed"]["denoiser_called"] == 1
    assert manifest["proposed"]["final_response_changed_vs_sft"] == 1
    with (output / "method_scores.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["method"] for row in rows] == ["SFT", "DPO", "mask_OFF"]
