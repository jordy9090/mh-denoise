import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from build_frozen_revision2_counselor_export import build_counselor_export
from build_frozen_revision2_test_partial_export import build_partial_export


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection_row(index: int, question_hash: str | None = None) -> dict:
    return {
        "canonical_id": f"qa_{index}",
        "split": "test",
        "question_normalized_sha256": question_hash or f"question-hash-{index}",
        "duplicate_cluster_id": f"duplicate-{index}",
        "source_group_id": f"source-group-{index}",
        "intended_axes": ["empathy"],
        "axis_count": 1,
        "generation_objective": "response_pair",
    }


def terminal_row(spec: dict, status: str, fingerprint: str = "run-fingerprint") -> dict:
    row = {
        **spec,
        "status": status,
        "run_fingerprint": fingerprint,
        "source": "fixture",
        "source_component": "fixture",
    }
    if status == "accepted":
        row.update(
            {
                "question": f"Question for {spec['canonical_id']}?",
                "clean_response": f"Safe response for {spec['canonical_id']}.",
                "corrupted_response": f"Unsafe response for {spec['canonical_id']}.",
                "realized_axes": ["empathy"],
                "unintended_axes": [],
                "generator_repo": "generator",
                "generator_revision": "generator-revision",
                "judge_model": "judge",
                "judge_repo": "judge",
                "judge_revision": "judge-revision",
            }
        )
    else:
        row["failure_reason"] = f"fixture-{status}"
    return row


def make_contract(production: Path, selection_file: Path, worker_count: int = 1) -> None:
    write_json(
        production / "run_contract.json",
        {
            "run_fingerprint": "run-fingerprint",
            "selection_file_sha256": file_sha256(selection_file),
            "allowed_splits": ["test"],
            "worker_count": worker_count,
        },
    )


def test_partial_export_waits_then_freezes_first_accepted_in_original_order(tmp_path: Path):
    production = tmp_path / "production"
    selection_dir = tmp_path / "selection"
    output = tmp_path / "partial"
    specs = [selection_row(1, "pilot-question-hash")] + [
        selection_row(index) for index in range(2, 7)
    ]
    selection_file = selection_dir / "test_generation_selection.json"
    write_json(selection_file, {"seed": 20260910, "split": "test", "rows": specs})
    write_json(
        selection_dir / "manifest.json",
        {
            "pilot_exposed_question_groups_retained_via_nonpilot_duplicate_id": [
                "pilot-question-hash"
            ],
            "pilot_exclusion_sources": [],
        },
    )
    make_contract(production, selection_file)
    initial = [
        terminal_row(specs[0], "accepted"),
        terminal_row(specs[1], "rejected"),
        terminal_row(specs[2], "accepted"),
        terminal_row(specs[3], "qc_hold"),
    ]
    checkpoint = production / "results_checkpoint.part0.jsonl"
    write_jsonl(checkpoint, initial)

    progress = build_partial_export(
        production,
        selection_file,
        output,
        target_count=2,
        canonical_file=None,
        pilot_selection_files=[],
    )

    assert progress["status"] == "waiting_for_accepted_prefix"
    assert progress["processed_rows_observed"] == 4
    assert progress["eligible_accepted_in_completed_prefix"] == 1
    assert not (output / "sft_test.jsonl").exists()
    snapshot = [json.loads(line) for line in (output / "selection_status_snapshot.jsonl").read_text().splitlines()]
    assert snapshot[0]["decision_state"] == "accepted_but_pilot_question_excluded"
    assert snapshot[1]["decision_state"] == "rejected"
    assert snapshot[-1]["decision_state"] == "pending"

    write_jsonl(checkpoint, initial + [terminal_row(specs[4], "accepted")])
    manifest = build_partial_export(
        production,
        selection_file,
        output,
        target_count=2,
        canonical_file=None,
        pilot_selection_files=[],
    )

    assert manifest["status"] == "complete_frozen_partial_test_export"
    assert manifest["selected_ids_ordered"] == ["qa_3", "qa_5"]
    pairs = [json.loads(line) for line in (output / "test_pairs.jsonl").read_text().splitlines()]
    sft = [json.loads(line) for line in (output / "sft_test.jsonl").read_text().splitlines()]
    dpo = [json.loads(line) for line in (output / "dpo_test.jsonl").read_text().splitlines()]
    assert [row["canonical_id"] for row in pairs] == ["qa_3", "qa_5"]
    assert all(row["question"] and row["unsafe_response"] and row["safe_response"] for row in sft)
    assert [row["target"] for row in sft] == [row["safe_response"] for row in sft]
    assert [row["chosen"] for row in dpo] == [row["safe_response"] for row in dpo]
    assert [row["rejected"] for row in dpo] == [row["unsafe_response"] for row in dpo]

    frozen_sft_hash = file_sha256(output / "sft_test.jsonl")
    write_jsonl(
        checkpoint,
        initial + [terminal_row(specs[4], "accepted"), terminal_row(specs[5], "accepted")],
    )
    repeated = build_partial_export(
        production,
        selection_file,
        output,
        target_count=2,
        canonical_file=None,
        pilot_selection_files=[],
    )
    assert repeated == manifest
    assert file_sha256(output / "sft_test.jsonl") == frozen_sft_hash


def test_partial_export_does_not_jump_over_pending_worker_block(tmp_path: Path):
    production = tmp_path / "production"
    selection_file = tmp_path / "selection" / "test_generation_selection.json"
    output = tmp_path / "partial"
    specs = [selection_row(index) for index in range(1, 5)]
    write_json(selection_file, {"rows": specs, "split": "test"})
    write_json(selection_file.parent / "manifest.json", {"pilot_exclusion_sources": []})
    make_contract(production, selection_file, worker_count=2)
    # Worker 1 owns qa_3..qa_4 and may finish while qa_1 is still pending.
    write_jsonl(
        production / "results_checkpoint.part1.jsonl",
        [terminal_row(specs[2], "accepted")],
    )

    progress = build_partial_export(
        production,
        selection_file,
        output,
        target_count=1,
        canonical_file=None,
        pilot_selection_files=[],
    )

    assert progress["status"] == "waiting_for_accepted_prefix"
    assert progress["processed_rows_observed"] == 1
    assert progress["contiguous_completed_prefix_rows"] == 0
    assert progress["eligible_accepted_in_completed_prefix"] == 0


def make_partial_fixture(root: Path) -> tuple[Path, list[dict]]:
    partial = root / "partial"
    rows = [
        {
            "partial_selection_rank": index,
            "frozen_selection_rank": index + 10,
            "canonical_id": f"qa_{index:02d}",
            "question_normalized_sha256": f"normalized-{index:02d}",
        }
        for index in range(1, 16)
    ]
    write_json(partial / "partial_selection.json", {"status": "frozen", "rows": rows})
    write_json(partial / "manifest.json", {"status": "complete_frozen_partial_test_export"})
    pairs = [
        {
            "canonical_id": row["canonical_id"],
            "id": row["canonical_id"],
            "question": f"Exact question {index}\nwith second line",
            "unsafe_response": f"unsafe {index}",
            "safe_response": f"safe {index}",
        }
        for index, row in enumerate(rows, start=1)
    ]
    write_jsonl(partial / "test_pairs.jsonl", pairs)
    return partial, pairs


def model_rows(pairs: list[dict], method: str) -> list[dict]:
    rows = []
    for index, pair in enumerate(reversed(pairs), start=1):
        row = dict(pair)
        if method == "Mask OFF":
            row["final_response"] = f"  Mask response {index}\nkept verbatim  "
        else:
            row["sft_response"] = f"  {method} response {index}\nkept verbatim  "
        rows.append(row)
    return rows


def test_counselor_export_blinds_balances_and_preserves_exact_source_text(tmp_path: Path):
    partial, pairs = make_partial_fixture(tmp_path)
    sft_path = tmp_path / "sft.jsonl"
    dpo_path = tmp_path / "dpo.jsonl"
    mask_path = tmp_path / "mask.jsonl"
    output = tmp_path / "counselor"
    write_jsonl(sft_path, model_rows(pairs, "SFT"))
    write_jsonl(dpo_path, model_rows(pairs, "DPO"))
    write_jsonl(mask_path, model_rows(pairs, "Mask OFF"))

    manifest = build_counselor_export(
        partial, sft_path, dpo_path, mask_path, output, blind_seed=12345
    )

    assert manifest["status"] == "complete_counselor_blinded_15x3_export"
    assert all(manifest["checks"].values())
    blind = [json.loads(line) for line in (output / "counselor_blinded_15x3.jsonl").read_text().splitlines()]
    mapping = [json.loads(line) for line in (output / "internal_model_mapping.jsonl").read_text().splitlines()]
    assert len(blind) == 15
    assert len(mapping) == 45
    assert all(set(row) == {"case_id", "question_number", "question", "responses"} for row in blind)
    assert all("canonical_id" not in row for row in blind)
    assert all("method" not in response for row in blind for response in row["responses"])
    assert all(response["response_text"].startswith("  ") for row in blind for response in row["responses"])
    assert all(response["response_text"].endswith("  ") for row in blind for response in row["responses"])
    position_counts = Counter((row["method"], row["response_label"]) for row in mapping)
    assert set(position_counts.values()) == {5}
    assert all(row["source"]["source_row_id"] == row["canonical_id"] for row in mapping)
    assert all(row["source"]["line"] >= 1 for row in mapping)

    repeated = build_counselor_export(
        partial, sft_path, dpo_path, mask_path, output, blind_seed=12345
    )
    assert repeated == manifest


def test_counselor_export_rejects_incomplete_method_membership(tmp_path: Path):
    partial, pairs = make_partial_fixture(tmp_path)
    sft_path = tmp_path / "sft.jsonl"
    dpo_path = tmp_path / "dpo.jsonl"
    mask_path = tmp_path / "mask.jsonl"
    write_jsonl(sft_path, model_rows(pairs, "SFT"))
    write_jsonl(dpo_path, model_rows(pairs, "DPO")[:-1])
    write_jsonl(mask_path, model_rows(pairs, "Mask OFF"))

    with pytest.raises(RuntimeError, match="DPO output membership mismatch"):
        build_counselor_export(
            partial, sft_path, dpo_path, mask_path, tmp_path / "out"
        )
