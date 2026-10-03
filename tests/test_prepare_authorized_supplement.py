from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare_authorized_supplement import (
    DATASETS,
    extract_candidates,
    git_blob_sha1,
    prepare_supplement,
    validate_official_cardinality,
)


INSTRUCTION = (
    "If you are a counsellor, please answer the questions based on the "
    "description of the patient."
)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def canonical_row(
    *, canonical_id: str, question: str, answer: str, split: str, cluster: str
) -> dict:
    from scripts.fullpaper_acl_pipeline import clean_display, normalize_text, sha256_text

    return {
        "canonical_id": canonical_id,
        "split": split,
        "question_exact_sha256": sha256_text(clean_display(question)),
        "question_normalized_sha256": sha256_text(normalize_text(question)),
        "response_exact_sha256": sha256_text(clean_display(answer)),
        "response_normalized_sha256": sha256_text(normalize_text(answer)),
        "source_group_id": f"fixture:{canonical_id}",
        "duplicate_cluster_id": cluster,
    }


def write_canonical(path: Path, rows: list[dict]) -> bytes:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()
    path.write_bytes(payload)
    return payload


def psych8k_row(question: str, answer: str, **metadata) -> dict:
    return {
        "instruction": INSTRUCTION,
        "input": question,
        "output": answer,
        **metadata,
    }


def authorization_attestation(dataset_kind: str, *, valid_through: str = "2099-12-31") -> dict:
    spec = DATASETS[dataset_kind]
    return {
        "dataset_kind": dataset_kind,
        "repo": spec["repo"],
        "revision": spec["revision"],
        "status": "authorized",
        "authorized_identity": {"name": "Fixture Researcher", "account": "fixture@example.test"},
        "evidence": {
            "type": "test_fixture",
            "reference": "fixture-approval-1",
            "sha256": "0" * 64,
        },
        "allowed_uses": ["data_preparation"],
        "valid_from": "2026-01-01",
        "valid_through": valid_through,
    }


def run_fixture(
    tmp_path: Path,
    *,
    source_rows: list[dict],
    canonical_rows: list[dict],
    accepted_rows: list[dict] | None = None,
    dataset_kind: str = "psych8k",
    output_name: str = "prepared",
):
    source = tmp_path / "opaque_authorized_payload.json"
    canonical = tmp_path / "canonical.jsonl"
    authorization = tmp_path / "approval.json"
    accepted = tmp_path / "accepted.jsonl"
    output = tmp_path / output_name
    write_json(source, source_rows)
    canonical_bytes = write_canonical(canonical, canonical_rows)
    write_json(authorization, authorization_attestation(dataset_kind))
    if accepted_rows is not None:
        write_canonical(accepted, accepted_rows)
    manifest = prepare_supplement(
        dataset_kind=dataset_kind,
        input_json=source,
        canonical_path=canonical,
        accepted_652_path=accepted if accepted_rows is not None else None,
        authorization_record=authorization,
        authorization_confirmed=True,
        output_dir=output,
        enforce_cardinality=False,
        enforce_frozen_canonical=False,
        enforce_frozen_accepted=False,
    )
    clean = [
        json.loads(line)
        for line in (output / "clean_candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    plan = [
        json.loads(line)
        for line in (output / "generation_plan.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    return manifest, clean, plan, output, canonical, canonical_bytes


def test_psych8k_preparation_is_deterministic_deduplicated_and_split_safe(tmp_path):
    shared_answer = "This response already exists in canonical."
    canonical_rows = [
        canonical_row(
            canonical_id="qa-existing",
            question="I already exist.",
            answer=shared_answer,
            split="valid",
            cluster="dup-existing",
        )
    ]
    source_rows = [
        psych8k_row("I already exist!", "A second target that must be preserved."),
        psych8k_row("How can I sleep better?", "Try a regular wind-down routine."),
        psych8k_row(" HOW can I sleep better ", "A duplicate-question answer."),
        psych8k_row("how can I sleep better", " Try a regular wind-down routine. "),
        psych8k_row("Can you reflect this concern?", shared_answer),
        psych8k_row("Shared session first question", "First unique answer.", session_id="s-1"),
        psych8k_row("Shared session second question", "Second unique answer.", session_id="s-1"),
    ]

    manifest, clean, plan, output, canonical, original_canonical = run_fixture(
        tmp_path, source_rows=source_rows, canonical_rows=canonical_rows
    )

    assert {path.name for path in output.iterdir()} == {
        "clean_candidates.jsonl",
        "generation_plan.jsonl",
        "manifest.json",
    }
    assert canonical.read_bytes() == original_canonical
    assert manifest["status"] == "complete_cpu_only_no_model_no_api_no_training"
    assert manifest["dataset_identity"]["explicit_dataset_kind"] == "psych8k"
    assert manifest["dataset_identity"]["filename_used_for_identity"] is False
    assert manifest["counts"] == {
        "input_top_level_records": 7,
        "extracted_original_answer_pairs": 7,
        "excluded_total": 1,
        "excluded_by_reason": {
            "within_source_normalized_question_response_pair_duplicate": 1,
        },
        "clean_distinct_question_answer_candidates": 6,
        "clean_unique_normalized_questions": 5,
        "clean_by_split": manifest["counts"]["clean_by_split"],
        "generation_plan_candidates": 5,
        "covered_by_existing_accepted652_question": 0,
        "generation_plan_by_split": manifest["counts"]["generation_plan_by_split"],
        "generation_status": manifest["counts"]["generation_status"],
    }
    assert len(clean) == 6
    assert len({row["question_normalized_sha256"] for row in plan}) == len(plan) == 5
    assert sum("sleep better" in row["question_normalized"] for row in clean) == 2
    assert sum("sleep better" in row["question_normalized"] for row in plan) == 1
    assert all(row["schema_version"] == "canonical-clean-qa-v1" for row in clean)
    assert all(
        {"question_normalized", "response_normalized", "source_file", "provenance"}
        <= set(row)
        for row in clean
    )
    from scripts.fullpaper_acl_pipeline import canonical_id

    assert all(canonical_id(row) == row["canonical_id"] for row in clean)

    anchored = next(row for row in plan if row["clean_response"] == shared_answer)
    assert anchored["split"] == "valid"
    assert anchored["split_basis"] == "anchored_to_existing_canonical_linkage"
    assert anchored["linked_existing_duplicate_cluster_ids"] == ["dup-existing"]
    assert anchored["duplicate_cluster_id"] == "dup-existing"

    shared_session = [row for row in clean if row["source_group_id"].endswith(":s-1")]
    assert len(shared_session) == 2
    assert len({row["split"] for row in shared_session}) == 1
    assert len({row["duplicate_cluster_id"] for row in shared_session}) == 1
    assert all(row["generation"]["status"] in {"planned_not_run", "reserved_test_no_generation"} for row in plan)
    assert manifest["generation_settings"]["paid_api_enabled"] is False
    assert manifest["generation_settings"]["model_execution_performed"] is False
    assert manifest["generation_settings"]["training_performed"] is False
    assert manifest["generation_settings"]["generator"] == {
        "repo": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "temperature": 0.75,
        "top_p": 0.9,
        "do_sample": True,
        "max_input_tokens": 4096,
        "max_new_tokens_policy": "min(max(current_response_tokens + 256, 768), 1536)",
        "truncation_retry_ceiling": 2048,
        "max_semantic_attempts_per_stage": 2,
        "max_infrastructure_attempts": 4,
        "deterministic_seed": 20260910,
        "enable_thinking": False,
        "execution_status": "planned_only_not_run",
    }
    assert manifest["inputs"]["accepted_652_merge_anchor"]["rows"] == 652
    assert manifest["invariants"]["existing_accepted_652_bytes_untouched"] is True
    assert manifest["invariants"]["supplement_canonical_ids_unique"] is True
    assert manifest["invariants"]["supplement_ids_disjoint_from_frozen_canonical"] is True
    assert manifest["invariants"]["supplement_ids_disjoint_from_accepted652"] is True

    second_manifest, second_clean, second_plan, second_output, _, _ = run_fixture(
        tmp_path,
        source_rows=source_rows,
        canonical_rows=canonical_rows,
        output_name="prepared_again",
    )
    assert second_clean == clean
    assert second_plan == plan
    assert (second_output / "generation_plan.jsonl").read_bytes() == (
        output / "generation_plan.jsonl"
    ).read_bytes()
    assert (second_output / "clean_candidates.jsonl").read_bytes() == (
        output / "clean_candidates.jsonl"
    ).read_bytes()
    assert second_manifest["outputs"] == manifest["outputs"]


def test_explicit_kind_must_match_schema_and_psyqa_keeps_one_answer_per_question(tmp_path):
    sharegpt, sharegpt_identity = extract_candidates(
        [
            {
                "id": "conversation-1",
                "conversations": [
                    {
                        "from": "human",
                        "value": INSTRUCTION + " Description: I feel overwhelmed.",
                    },
                    {"from": "gpt", "value": "It makes sense to feel stretched."},
                ],
            }
        ],
        "psych8k",
        enforce_cardinality=False,
    )
    assert sharegpt[0]["question"] == "I feel overwhelmed."
    assert sharegpt_identity["detected_schema"] == "psych8k_sharegpt_human_assistant"
    assert sharegpt_identity["sharegpt_embedded_instruction_removed"] == 1
    with pytest.raises(ValueError, match="content signature mismatch"):
        extract_candidates(
            [
                {
                    "conversations": [
                        {"from": "human", "value": "Generic unrelated ShareGPT row"},
                        {"from": "gpt", "value": "Generic answer"},
                    ]
                }
            ],
            "psych8k",
            enforce_cardinality=False,
        )

    psyqa_rows = [
        {
            "questionID": 10,
            "question": "A title",
            "description": "A longer description",
            "answers": [
                {"answerID": 1, "answer_text": "First original answer"},
                {"answerID": 2, "answer_text": "Second original answer"},
            ],
        },
        {
            "questionID": 11,
            "question": "Another question",
            "description": "",
            "answers": [{"answerID": 3, "answer_text": "Third answer"}],
        },
    ]
    with pytest.raises(ValueError, match="Psych8k schema mismatch"):
        extract_candidates(psyqa_rows, "psych8k", enforce_cardinality=False)

    extracted, identity = extract_candidates(
        psyqa_rows, "psyqa-sample", enforce_cardinality=False
    )
    assert identity == {
        "detected_schema": "psyqa_question_description_answers",
        "schema_variant_counts": {"psyqa_question_description_answers": 2},
        "top_level_records": 2,
        "answer_records": 3,
    }
    assert len(extracted) == 3
    assert extracted[0]["question"] == "A title\n\nA longer description"

    manifest, clean, plan, _, _, _ = run_fixture(
        tmp_path,
        source_rows=psyqa_rows,
        canonical_rows=[],
        dataset_kind="psyqa-sample",
    )
    assert len(clean) == 3
    assert len(plan) == 2
    assert manifest["counts"]["extracted_original_answer_pairs"] == 3
    assert manifest["counts"]["excluded_by_reason"] == {}
    assert manifest["counts"]["clean_distinct_question_answer_candidates"] == 3
    assert manifest["counts"]["generation_plan_candidates"] == 2
    assert all(row["source_group_id"].startswith("psyqa:questionID:") for row in plan)


def test_accepted652_question_is_preserved_clean_but_not_regenerated(tmp_path):
    existing = canonical_row(
        canonical_id="qa-accepted",
        question="Question already reused",
        answer="Existing accepted answer",
        split="train",
        cluster="dup-accepted",
    )
    manifest, clean, plan, _, _, _ = run_fixture(
        tmp_path,
        source_rows=[
            psych8k_row("Question already reused", "A distinct supplemental original answer")
        ],
        canonical_rows=[existing],
        accepted_rows=[existing],
    )
    assert len(clean) == 1
    assert plan == []
    assert manifest["counts"]["covered_by_existing_accepted652_question"] == 1
    assert manifest["generation_plan_exclusions"][0]["reason"] == (
        "covered_by_existing_accepted652_question"
    )
    assert clean[0]["duplicate_cluster_id"] == "dup-accepted"


def test_official_cardinality_and_registered_artifact_identity(tmp_path):
    validate_official_cardinality(
        "psych8k", top_level_records=8_187, answer_records=8_187
    )
    validate_official_cardinality(
        "psyqa-sample", top_level_records=100, answer_records=268
    )
    assert DATASETS["psych8k"]["expected_bytes"] == 6_575_030
    assert DATASETS["psych8k"]["expected_git_blob_sha1"] == (
        "b0a9f254223cb3accb871872feb91156b9d9d719"
    )
    blob = tmp_path / "blob"
    blob.write_bytes(b"fixture")
    assert git_blob_sha1(blob) == "001f1993905d81b471eeaa840432cf35aedaea61"
    with pytest.raises(ValueError, match="expected 100"):
        validate_official_cardinality(
            "psyqa-sample", top_level_records=101, answer_records=268
        )
    with pytest.raises(ValueError, match="expected 8187 answer records"):
        validate_official_cardinality(
            "psych8k", top_level_records=8_187, answer_records=8_188
        )


def test_authorization_output_and_cross_split_anchor_are_fail_closed(tmp_path):
    source = tmp_path / "source.json"
    canonical = tmp_path / "canonical.jsonl"
    approval = tmp_path / "approval.json"
    output = tmp_path / "new-output"
    write_json(source, [psych8k_row("Question", "Answer")])
    write_canonical(canonical, [])
    write_json(approval, authorization_attestation("psych8k"))
    with pytest.raises(PermissionError, match="authorization-confirmed"):
        prepare_supplement(
            dataset_kind="psych8k",
            input_json=source,
            canonical_path=canonical,
            accepted_652_path=None,
            authorization_record=approval,
            authorization_confirmed=False,
            output_dir=output,
            enforce_cardinality=False,
            enforce_frozen_canonical=False,
            enforce_frozen_accepted=False,
        )
    assert not output.exists()

    expired = tmp_path / "expired-approval.json"
    expired_output = tmp_path / "expired-output"
    write_json(
        expired,
        authorization_attestation("psych8k", valid_through="2026-05-18"),
    )
    with pytest.raises(PermissionError, match="expired"):
        prepare_supplement(
            dataset_kind="psych8k",
            input_json=source,
            canonical_path=canonical,
            accepted_652_path=None,
            authorization_record=expired,
            authorization_confirmed=True,
            output_dir=expired_output,
            enforce_cardinality=False,
            enforce_frozen_canonical=False,
            enforce_frozen_accepted=False,
        )
    assert not expired_output.exists()

    wrong_kind = tmp_path / "wrong-kind-approval.json"
    wrong_kind_output = tmp_path / "wrong-kind-output"
    write_json(wrong_kind, authorization_attestation("psyqa-sample"))
    with pytest.raises(PermissionError, match="dataset_kind mismatch"):
        prepare_supplement(
            dataset_kind="psych8k",
            input_json=source,
            canonical_path=canonical,
            accepted_652_path=None,
            authorization_record=wrong_kind,
            authorization_confirmed=True,
            output_dir=wrong_kind_output,
            enforce_cardinality=False,
            enforce_frozen_canonical=False,
            enforce_frozen_accepted=False,
        )
    assert not wrong_kind_output.exists()

    with pytest.raises(RuntimeError, match="register exact approved artifact hash"):
        prepare_supplement(
            dataset_kind="psyqa-full",
            input_json=source,
            canonical_path=canonical,
            accepted_652_path=None,
            authorization_record=approval,
            authorization_confirmed=True,
            output_dir=tmp_path / "disabled-full-output",
            enforce_cardinality=False,
            enforce_frozen_canonical=False,
            enforce_frozen_accepted=False,
        )
    output.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_supplement(
            dataset_kind="psych8k",
            input_json=source,
            canonical_path=canonical,
            accepted_652_path=None,
            authorization_record=approval,
            authorization_confirmed=True,
            output_dir=output,
            enforce_cardinality=False,
            enforce_frozen_canonical=False,
            enforce_frozen_accepted=False,
        )

    canonical_rows = [
        canonical_row(
            canonical_id="qa-train",
            question="Old train question",
            answer="Train anchor answer",
            split="train",
            cluster="dup-train",
        ),
        canonical_row(
            canonical_id="qa-valid",
            question="Old valid question",
            answer="Valid anchor answer",
            split="valid",
            cluster="dup-valid",
        ),
    ]
    conflicting_source = [
        psych8k_row("New first", "Train anchor answer", session_id="same-session"),
        psych8k_row("New second", "Valid anchor answer", session_id="same-session"),
    ]
    source2 = tmp_path / "source2.json"
    canonical2 = tmp_path / "canonical2.jsonl"
    output2 = tmp_path / "conflict-output"
    write_json(source2, conflicting_source)
    write_canonical(canonical2, canonical_rows)
    with pytest.raises(RuntimeError, match="links multiple existing splits"):
        prepare_supplement(
            dataset_kind="psych8k",
            input_json=source2,
            canonical_path=canonical2,
            accepted_652_path=None,
            authorization_record=approval,
            authorization_confirmed=True,
            output_dir=output2,
            enforce_cardinality=False,
            enforce_frozen_canonical=False,
            enforce_frozen_accepted=False,
        )
    assert not output2.exists()

    same_split_rows = [
        canonical_row(
            canonical_id="qa-train-a",
            question="Old A",
            answer="Anchor A",
            split="train",
            cluster="dup-a",
        ),
        canonical_row(
            canonical_id="qa-train-b",
            question="Old B",
            answer="Anchor B",
            split="train",
            cluster="dup-b",
        ),
    ]
    bridge_source = [
        psych8k_row("Bridge first", "Anchor A", session_id="bridge"),
        psych8k_row("Bridge second", "Anchor B", session_id="bridge"),
    ]
    source3 = tmp_path / "source3.json"
    canonical3 = tmp_path / "canonical3.jsonl"
    output3 = tmp_path / "bridge-output"
    write_json(source3, bridge_source)
    write_canonical(canonical3, same_split_rows)
    with pytest.raises(RuntimeError, match="bridges multiple immutable existing duplicate clusters"):
        prepare_supplement(
            dataset_kind="psych8k",
            input_json=source3,
            canonical_path=canonical3,
            accepted_652_path=None,
            authorization_record=approval,
            authorization_confirmed=True,
            output_dir=output3,
            enforce_cardinality=False,
            enforce_frozen_canonical=False,
            enforce_frozen_accepted=False,
        )
    assert not output3.exists()


def test_schema_and_cardinality_failures_create_no_output(tmp_path):
    canonical = tmp_path / "canonical.jsonl"
    approval = tmp_path / "approval.json"
    write_canonical(canonical, [])
    write_json(approval, authorization_attestation("psych8k"))

    wrong_schema = tmp_path / "wrong-schema.json"
    wrong_output = tmp_path / "wrong-schema-output"
    write_json(
        wrong_schema,
        [
            {
                "questionID": 1,
                "question": "PsyQA-shaped",
                "description": "row",
                "answers": [{"answer_text": "answer"}],
            }
        ],
    )
    with pytest.raises(ValueError, match="Psych8k schema mismatch"):
        prepare_supplement(
            dataset_kind="psych8k",
            input_json=wrong_schema,
            canonical_path=canonical,
            accepted_652_path=None,
            authorization_record=approval,
            authorization_confirmed=True,
            output_dir=wrong_output,
            enforce_cardinality=False,
            enforce_frozen_canonical=False,
            enforce_frozen_accepted=False,
        )
    assert not wrong_output.exists()

    partial = tmp_path / "partial-psych8k.json"
    partial_output = tmp_path / "partial-output"
    write_json(partial, [psych8k_row("Question", "Answer")])
    with pytest.raises(ValueError, match="pinned artifact identity mismatch"):
        prepare_supplement(
            dataset_kind="psych8k",
            input_json=partial,
            canonical_path=canonical,
            accepted_652_path=None,
            authorization_record=approval,
            authorization_confirmed=True,
            output_dir=partial_output,
            enforce_cardinality=True,
            enforce_frozen_canonical=False,
            enforce_frozen_accepted=False,
        )
    assert not partial_output.exists()
