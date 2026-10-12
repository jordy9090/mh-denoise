import argparse
import json
from pathlib import Path

from exact_response_shared_judge import expand, prepare
from run_refinement_llm_judge_budgeted import budget_projection


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_exact_question_response_is_judged_once_and_expanded(tmp_path: Path):
    sft = tmp_path / "sft.jsonl"
    dpo = tmp_path / "dpo.jsonl"
    representatives = tmp_path / "representatives.jsonl"
    members = tmp_path / "members.jsonl"
    write_jsonl(
        sft,
        [{"example_id": "q1__SFT", "question": "q", "response": "same", "system": "SFT"}],
    )
    write_jsonl(
        dpo,
        [
            {"example_id": "q1__DPO", "question": "q", "response": "same", "system": "DPO"},
            {"example_id": "q2__DPO", "question": "q2", "response": "other", "system": "DPO"},
        ],
    )
    prepare(
        argparse.Namespace(
            input=[str(sft), str(dpo)],
            representatives=str(representatives),
            members=str(members),
            manifest=str(tmp_path / "prepare_manifest.json"),
        )
    )

    reps = read_jsonl(representatives)
    assert [row["example_id"] for row in reps] == ["q1__SFT", "q2__DPO"]
    assert reps[0]["exact_response_member_example_ids"] == ["q1__SFT", "q1__DPO"]

    judged = tmp_path / "judged.jsonl"
    write_jsonl(
        judged,
        [
            {**row, "judge_ok": True, "judge_scores": {"overall_quality": index + 4}}
            for index, row in enumerate(reps)
        ],
    )
    output = tmp_path / "expanded.jsonl"
    expand(
        argparse.Namespace(
            members=str(members),
            judged_representatives=str(judged),
            output=str(output),
            manifest=str(tmp_path / "expand_manifest.json"),
        )
    )

    expanded = read_jsonl(output)
    assert expanded[0]["judge_scores"] == expanded[1]["judge_scores"]
    assert expanded[0]["exact_response_shared"] is False
    assert expanded[1]["exact_response_shared"] is True


def test_budget_projection_caps_every_retry():
    rows = [
        {
            "question": "question",
            "response": "response",
            "system": "SFT",
            "unsafe_response": "unsafe",
            "reference_safe_response": "safe",
        }
    ]
    projection = budget_projection(
        rows,
        model="gpt-4.1-mini",
        rubric_style="json",
        max_tokens=500,
        max_retries=3,
        input_price=0.40,
        output_price=1.60,
    )

    assert projection["unique_rows"] == 1
    assert projection["absolute_request_cap"] == 3
    assert projection["absolute_three_attempt_ceiling_usd"] == 3 * projection["one_attempt_ceiling_usd"]
