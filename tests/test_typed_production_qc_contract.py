import hashlib
import inspect
import json
from types import SimpleNamespace

import pytest

from scripts.build_fullpaper_training_data import (
    apply_scorer_review_ledger,
    explicit_local_labels,
    production_training_readiness,
)
from scripts.fullpaper_scorer_export_contract import (
    audit_tokenization_and_mask,
    deduplicate_same_inputs,
    find_non_null_collisions,
    mask_collisions,
)
from scripts.local_qwen_production_qc_v4 import (
    AXES,
    content_disposition,
    validate_and_map,
)
from scripts.run_fullpaper_corruption_production import (
    paired_grade_local,
    process_one_local,
    source_verified_spans,
    text_edit_trace,
)
from scripts.corruption_contract_v2 import StageRequest, build_stage_prompt
from scripts.freeze_fullpaper_main_data import require_training_ready
from scripts import local_qwen_production_qc_v4, run_fullpaper_corruption_production, source_integrity_contract


def side(axis, *, scope="none", source="none", span="", reason="No local claim."):
    result = {
        "evidence_scope": scope,
        "evidence_source": source,
        "evidence_span": span,
        "reason": reason,
    }
    result["violation" if axis == "medical_boundary" else "score"] = (
        False if axis == "medical_boundary" else (1 if axis == "toxicity_or_harm" else 4)
    )
    return result


def payload():
    checks = {
        "topic_preserved": {"value": True, "evidence_span": "", "reason": "Topic retained."},
        "fluent": {"value": True, "evidence_span": "", "reason": "Fluent."},
        "unrelated_topic_injection": {"value": False, "evidence_span": "", "reason": "None."},
        "cartoonish_or_exaggerated_shaming": {"value": False, "evidence_span": "", "reason": "None."},
        "prohibited_dangerous_content": {"value": False, "evidence_sufficient": True, "evidence_span": "", "reason": "None."},
        "editing_task_refusal_or_meta": {"value": False, "evidence_span": "", "reason": "None."},
    }
    return {
        "scores": {
            axis: {"A": side(axis), "B": side(axis), "comparison_reason": "Compared complete responses."}
            for axis in AXES
        },
        "response_eligibility": {
            label: {
                "disposition": "pass", "axis": None, "evidence_scope": "none",
                "evidence_source": "none", "evidence_span": "", "reason": "No clear material defect.",
            }
            for label in ("A", "B")
        },
        "content_checks": {"A": dict(checks), "B": dict(checks)},
        "text_reason_contradiction": {"detected": False, "reason": "No contradiction."},
        "overall_reason": "Typed QC fixture.",
    }


def test_typed_exact_local_defect_and_support_create_both_classes():
    question = "I feel anxious about tomorrow."
    clean = "It may help to take one small step and ask someone you trust for support."
    candidate = "You will definitely fail tomorrow."
    data = payload()
    data["scores"]["empathy"]["A"] = side(
        "empathy", scope="local_support", source="own_response",
        span="ask someone you trust for support", reason="Supportive and appropriate to the question.",
    )
    data["scores"]["factual_consistency"]["B"] = side(
        "factual_consistency", scope="local_defect", source="own_response",
        span="You will definitely fail tomorrow", reason="Unsupported certainty about the user's future.",
    )
    grade, invalid = validate_and_map(data, clean, candidate, "A", "B", question=question)
    assert invalid == []
    assert {(item["axis"], item["side"], item["label"]) for item in grade["local_supervision"]} == {
        ("empathy", "clean", 0), ("factual_consistency", "candidate", 1)
    }
    row = {
        "canonical_id": "fixture", "question": question, "clean_response": clean,
        "corrupted_response": candidate, "final_grade": grade,
    }
    verified, excluded = source_verified_spans(row)
    assert excluded == []
    assert {item["label"] for item in verified} == {0, 1}


def test_holistic_and_omission_never_become_local_positive():
    question = "I need a concrete next step."
    clean = "Call your sister tomorrow and ask for help."
    candidate = "Try to cope somehow."
    data = payload()
    data["scores"]["overall_quality"]["B"] = side(
        "overall_quality", scope="holistic", source="whole_response", span="",
        reason="The response is globally too thin.",
    )
    data["scores"]["specificity"]["B"] = side(
        "specificity", scope="omission", source="other_response", span="Call your sister tomorrow",
        reason="The concrete step was omitted.",
    )
    grade, _ = validate_and_map(data, clean, candidate, "A", "B", question=question)
    assert grade["local_supervision"] == []
    assert {item["scope"] for item in grade["response_level_evidence"]} >= {"holistic", "omission"}


def test_distributed_omission_accepts_whole_response_empty_without_local_label():
    data = payload()
    data["scores"]["specificity"]["B"] = side(
        "specificity", scope="omission", source="whole_response", span="",
        reason="The response omits the requested topic throughout.",
    )
    grade, invalid = validate_and_map(
        data, "A complete response.", "A thin response.", "A", "B",
        question="Please explain both topics.",
    )
    assert invalid == []
    assert grade["local_supervision"] == []
    omission = next(
        item for item in grade["response_level_evidence"]
        if item["axis"] == "specificity" and item["side"] == "candidate"
    )
    assert omission["scope"] == "omission"
    assert omission["evidence_source"] == "whole_response"


def test_non_verbatim_local_is_retried_then_can_only_be_quarantined_unknown():
    data = payload()
    data["scores"]["empathy"]["B"] = side(
        "empathy", scope="local_defect", source="own_response",
        span="You should... stop worrying", reason="Ellipsis is not verbatim evidence.",
    )
    with pytest.raises(ValueError, match="local evidence is not exact verbatim"):
        validate_and_map(
            data, "I hear you.", "You should immediately stop worrying.", "A", "B",
            question="I am worried.",
        )
    grade, invalid = validate_and_map(
        data, "I hear you.", "You should immediately stop worrying.", "A", "B",
        question="I am worried.", quarantine_invalid_evidence=True,
    )
    assert len(invalid) == 1
    assert invalid[0]["resolution"] == "non_verbatim_removed_no_local_label"
    assert grade["local_supervision"] == []


def test_clean_hold_blocks_acceptance_and_exact_export():
    question = "I have a small facial cut."
    clean = "Your mother's sleep disturbances explain this."
    candidate = "Keep the area clean."
    data = payload()
    data["response_eligibility"]["A"] = {
        "disposition": "hold", "axis": "factual_consistency",
        "evidence_scope": "local_defect", "evidence_source": "own_response",
        "evidence_span": "Your mother's sleep disturbances",
        "reason": "This asserts unsupported family history.",
    }
    grade, _ = validate_and_map(data, clean, candidate, "A", "B", question=question)
    assert content_disposition(question, candidate, grade, clean=clean) == (
        "hold", "clean_review_required"
    )


def test_clean_local_defect_with_pass_is_held_and_process_one_never_accepts(monkeypatch):
    question = "I have a small facial cut."
    clean = "Your mother has a diagnosed sleep disorder."
    candidate = "I hear how concerning this feels. Keep the area clean."
    data = payload()
    data["scores"]["factual_consistency"]["A"] = side(
        "factual_consistency", scope="local_defect", source="own_response",
        span="Your mother has a diagnosed sleep disorder",
        reason="Unsupported family medical history.",
    )
    data["scores"]["empathy"]["B"]["score"] = 2
    grade, _ = validate_and_map(data, clean, candidate, "A", "B", question=question)
    assert content_disposition(question, candidate, grade, clean=clean) == (
        "hold", "clean_review_required"
    )

    generation = SimpleNamespace(
        text=candidate, raw_output=candidate, truncation_reason=None,
        current_response_tokens=10, max_new_tokens=100, generated_tokens=12, eos_reached=True,
    )
    backend = SimpleNamespace(generate=lambda requests, allowance: [generation])
    judge = SimpleNamespace(record_generation=lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_fullpaper_corruption_production,
        "paired_grade_local",
        lambda *args, **kwargs: (grade, []),
    )
    item = {
        "clean": {
            "canonical_id": "fixture-clean-conflict", "split": "train",
            "question": question, "clean_response": clean,
        },
        "intended_axes": ["empathy"], "generation_seeds": [],
    }
    assert process_one_local(item, backend, judge)["status"] == "qc_hold"


def test_automatic_clean_preflight_holds_before_any_generator_call():
    question = "I need help organizing my day."
    clean = "Your mother has already agreed to manage your medication."
    axes = {
        axis: {
            "material_degradation": axis == "factual_consistency",
            "evidence_source": "response" if axis == "factual_consistency" else "none",
            "evidence_span": (
                "Your mother has already agreed to manage your medication"
                if axis == "factual_consistency" else ""
            ),
            "reason": "Unsupported family and treatment history." if axis == "factual_consistency" else "No defect.",
        }
        for axis in AXES
    }

    class CleanJudge:
        def set_context(self, **context):
            self.context = context

        def call(self, prompt):
            return json.dumps({"axes": axes}), {"fixture": True}

        @staticmethod
        def _parse_eligibility(data, row):
            return local_qwen_production_qc_v4.parse_eligibility(data, row)

    class NoGeneration:
        @staticmethod
        def generate(*args, **kwargs):
            raise AssertionError("generator must not run for a held clean target")

    result = process_one_local(
        {
            "clean": {
                "canonical_id": "automatic-clean-hold",
                "split": "train",
                "question": question,
                "clean_response": clean,
            },
            "intended_axes": ["empathy"],
            "generation_seeds": [],
            "automatic_clean_preflight": True,
        },
        NoGeneration(),
        CleanJudge(),
    )
    assert result["status"] == "qc_hold"
    assert result["failure_reason"] == "automatic_clean_preflight_hold"
    assert result["automatic_clean_preflight"]["eligible"] is False


@pytest.mark.parametrize("axis", ["empathy", "specificity"])
def test_local_scorer_generation_requires_one_intrinsically_judgeable_sentence(axis):
    request = StageRequest(
        "fixture", "train", "I feel ignored and need a concrete next step.",
        "Your feelings matter. Call a trusted friend tonight.",
        "Your feelings matter. Call a trusted friend tonight.",
        (axis,), (), axis, 1, 17, 1, None, "local_scorer",
    )
    prompt = build_stage_prompt(request)
    assert "question plus that single edited sentence" in prompt
    assert "provenance only and will not be treated as a label" in prompt
    assert "exactly one localized sentence edit" in prompt
    if axis == "empathy":
        assert "minimizes, dismisses, or invalidates" in prompt
        assert "Do not create the target defect by deleting validation" in prompt
    else:
        assert "visibly mismatched to the user's stated goal" in prompt
        assert "Do not create the target defect by deleting details" in prompt


def test_edit_trace_records_locations_without_claiming_ground_truth():
    before = "I hear you. Call a friend tonight."
    after = "I hear you. Ignore the problem tonight."
    trace = text_edit_trace(before, after)
    assert trace["changed_hunk_count"] >= 1
    assert trace["label_status"] == "provenance_only_not_ground_truth"
    assert any(hunk["after_text"] for hunk in trace["changed_hunks"])


def test_blind_prompt_and_ab_mapping_do_not_expose_clean_identity():
    assert "clean-target label" not in local_qwen_production_qc_v4.PAIRED_PROMPT
    assert "{clean_label}" not in local_qwen_production_qc_v4.PAIRED_PROMPT
    question = "I feel overwhelmed."
    clean = "It makes sense to feel overwhelmed."
    candidate = "Stop complaining."
    data = payload()
    data["scores"]["empathy"]["A"] = side(
        "empathy", scope="local_support", source="own_response",
        span="It makes sense to feel overwhelmed", reason="Validating statement.",
    )
    data["content_checks"]["B"]["fluent"] = {
        "value": False, "evidence_span": "Stop complaining", "reason": "Abrupt fragment."
    }
    grade, _ = validate_and_map(data, clean, candidate, "A", "B", question=question)
    assert grade["local_supervision"][0]["side"] == "clean"
    assert grade["content_checks"] == data["content_checks"]["B"]
    assert grade["response_content_checks"]["clean"] == data["content_checks"]["A"]

    swapped = payload()
    swapped["scores"]["empathy"]["B"] = data["scores"]["empathy"]["A"]
    swapped["content_checks"]["A"] = data["content_checks"]["B"]
    swapped_grade, _ = validate_and_map(swapped, candidate, clean, "B", "A", question=question)
    assert swapped_grade["local_supervision"][0]["side"] == "clean"
    assert swapped_grade["content_checks"] == swapped["content_checks"]["A"]


def test_same_actual_scorer_input_axis_conflict_is_masked_across_sides():
    rows = [
        {
            "canonical_id": "a", "question": "Q", "span": "Same empathic sentence.",
            "span_side": "clean", "labels": {axis: (0 if axis == "empathy" else None) for axis in AXES},
            "label_mask": {axis: axis == "empathy" for axis in AXES}, "label_basis": {},
        },
        {
            "canonical_id": "b", "question": "Q", "span": "Same empathic sentence.",
            "span_side": "candidate", "labels": {axis: (1 if axis == "empathy" else None) for axis in AXES},
            "label_mask": {axis: axis == "empathy" for axis in AXES}, "label_basis": {},
        },
    ]
    collisions = find_non_null_collisions(rows)
    ledger = mask_collisions(rows, collisions, stage="fixture")
    assert len(collisions) == len(ledger) == 1
    assert all(row["labels"]["empathy"] is None for row in rows)
    assert not find_non_null_collisions(rows)


def test_same_input_same_label_is_one_training_row_with_all_provenance():
    rows = [
        {
            "canonical_id": "same", "split": "train", "question": "Q", "span": "S",
            "span_side": side_name, "span_start": start, "span_end": start + 1,
            "span_source_sha256": f"hash-{side_name}",
            "labels": {axis: (0 if axis == "specificity" else None) for axis in AXES},
            "label_mask": {axis: axis == "specificity" for axis in AXES},
            "label_basis": {axis: "fixture" for axis in AXES},
            "source_records": [{"side": side_name, "start": start}],
        }
        for side_name, start in (("clean", 10), ("candidate", 20))
    ]
    merged, ledger = deduplicate_same_inputs(rows)
    assert len(merged) == len(ledger) == 1
    assert merged[0]["labels"]["specificity"] == 0
    assert len(merged[0]["provenance_members"]) == 2
    assert ledger[0]["duplicate_axis_contributions_removed"] == {"specificity": 1}


class CharTokenizer:
    truncation_side = "right"

    def __call__(self, text, truncation, max_length=None, return_offsets_mapping=False):
        width = min(len(text), max_length) if truncation else len(text)
        result = {"input_ids": [1] * width, "attention_mask": [1] * width}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(width)]
        return result


def test_actual_scorer_tokenization_masks_truncated_evidence():
    rows = [{
        "canonical_id": "long", "split": "train", "question": "q" * 40,
        "span": "critical evidence", "span_side": "candidate",
        "labels": {axis: (1 if axis == "factual_consistency" else None) for axis in AXES},
        "label_mask": {axis: axis == "factual_consistency" for axis in AXES},
        "label_basis": {axis: "fixture" for axis in AXES},
    }]
    summary, ledger, _, _ = audit_tokenization_and_mask(rows, CharTokenizer(), max_length=24)
    assert summary["span_not_preserved"] == 1
    assert ledger[0]["masked_axes"]["factual_consistency"]["original_label"] == 1
    assert rows[0]["labels"]["factual_consistency"] is None


def test_production_adapter_requires_explicit_scope_label_pair():
    records = [{"axis": "empathy", "scope": "local_support", "label": 0}]
    labels, basis = explicit_local_labels(records)
    assert labels["empathy"] == 0
    assert basis["empathy"] == "explicit_local_support_exact_span"
    with pytest.raises(ValueError, match="Invalid typed local"):
        explicit_local_labels([{"axis": "empathy", "scope": "holistic", "label": 1}])


def test_hash_bound_scorer_review_masks_response_level_rationale(tmp_path):
    question = "I am hurt."
    span = "It is hoped things improve."
    source_hash = hashlib.sha256(b"candidate response").hexdigest()
    labels = {axis: (1 if axis == "empathy" else None) for axis in AXES}
    row = {
        "canonical_id": "review-fixture", "split": "train", "question": question,
        "span": span, "span_side": "candidate", "span_start": 0,
        "span_end": len(span), "span_source_sha256": source_hash,
        "labels": labels, "label_mask": {axis: labels[axis] is not None for axis in AXES},
        "label_basis": {axis: "explicit_local_defect_exact_span" for axis in AXES},
    }
    manifest_hash = "a" * 64
    review = {
        "canonical_id": "review-fixture", "span_side": "candidate",
        "span_start": 0, "span_end": len(span), "axis": "empathy",
        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "span": span, "span_source_sha256": source_hash, "original_label": 1,
        "disposition": "mask_unknown", "reviewer": "Codex",
        "review_kind": "complete_question_exact_span_context",
        "source_production_manifest_sha256": manifest_hash,
        "reason": "The cited sentence is neutral; the rationale depends on an omission elsewhere.",
    }
    path = tmp_path / "scorer_review.jsonl"
    path.write_text(json.dumps(review) + "\n", encoding="utf-8")
    summary, ledger = apply_scorer_review_ledger(
        [row], review_path=path, allowed_manifest_hashes={manifest_hash}
    )
    assert row["labels"]["empathy"] is None
    assert row["label_mask"]["empathy"] is False
    assert summary["masked_unknown"] == 1
    assert ledger[0]["resolution"] == "unknown_excluded_from_loss"


def test_binary_classes_do_not_imply_training_ready_without_semantic_approval():
    class_presence = {
        axis: {"positive": 1, "negative": 1, "has_both": True}
        for axis in AXES
    }
    ready, blockers = production_training_readiness(
        semantic_review={"coverage_complete": True, "approved_for_training": False},
        retained_splits={"train", "valid"}, retained_rows=12,
        scorer_class_presence_by_axis=class_presence,
    )
    assert ready is False
    assert blockers == ["semantic_review_not_approved"]

    ready, blockers = production_training_readiness(
        semantic_review={"coverage_complete": True, "approved_for_training": True},
        scorer_semantic_review={"coverage_complete": False},
        retained_splits={"train", "valid"}, retained_rows=12,
        scorer_class_presence_by_axis=class_presence,
    )
    assert ready is False
    assert blockers == ["scorer_semantic_review_incomplete"]


def test_main_data_freeze_rejects_audit_only_and_legacy_class_only_manifests():
    with pytest.raises(RuntimeError, match="audit-only"):
        require_training_ready({
            "training_readiness": {
                "ready": False,
                "blocking_reasons": ["semantic_review_not_approved"],
            }
        })
    with pytest.raises(RuntimeError, match="lacks explicit semantic"):
        require_training_ready({
            "scorer_supervision_status": {
                "has_positive_and_negative": True,
                "training_ready": True,
            }
        })
    require_training_ready({"training_readiness": {"ready": True, "blocking_reasons": []}})


def test_contract_has_no_known_id_or_failure_phrase_exceptions():
    source = inspect.getsource(local_qwen_production_qc_v4) + inspect.getsource(source_integrity_contract)
    for forbidden in (
        "qa_b8711c6b200b077b7e8db6ac",
        "qa_ec895ef531a913aaee613844",
        "qa_d25175051aa75a2305ad15be",
        "your mother's sleep disturbances",
        "CBT surveys we've been working on",
    ):
        assert forbidden not in source


def test_contextual_review_is_hash_bound_and_overrides_automated_phrase_signal():
    question = "We already scheduled another appointment. What should we cover then?"
    clean = "Our next session will focus on practical ways to ask your family for help."
    data = payload()
    grade, _ = validate_and_map(data, clean, "Try a schedule.", "A", "B", question=question)
    assert content_disposition(question, "Try a schedule.", grade, clean=clean) == (
        "hold", "clean_review_required"
    )
    assert grade["automated_clean_review_signals"]
    review = {
        "disposition": "pass",
        "reviewer": "Codex fixture",
        "review_kind": "complete_question_clean_context",
        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "clean_response_sha256": hashlib.sha256(clean.encode()).hexdigest(),
        "reason": "The user explicitly states that another appointment is already scheduled.",
        "findings": [],
    }
    assert content_disposition(
        question, "Try a schedule.", grade, clean=clean,
        contextual_clean_review=review,
    ) == ("pass", None)
    assert grade["contextual_clean_review"]["disposition"] == "pass"
    bad = dict(review, question_sha256="0" * 64)
    with pytest.raises(ValueError, match="question hash mismatch"):
        content_disposition(
            question, "Try a schedule.", grade, clean=clean,
            contextual_clean_review=bad,
        )


class FakeJudge:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []
        self.contexts = []

    def set_context(self, **context):
        self.contexts.append(context)

    def call(self, prompt):
        self.prompts.append(prompt)
        return self.outputs.pop(0), {"fixture": True}


def test_judge_retry_prompt_contains_specific_validation_error_and_changes_hash():
    invalid = payload()
    invalid["content_checks"]["A"]["topic_preserved"]["evidence_span"] = "not verbatim"
    valid = payload()
    judge = FakeJudge([json.dumps(invalid), json.dumps(valid)])
    grade, failures = paired_grade_local(
        judge,
        {
            "canonical_id": "retry-fixture", "question": "Q", "clean_response": "Clean",
            "intended_axes": ["overall_quality"],
        },
        "Candidate", 1, 1,
    )
    assert grade is not None
    assert len(failures) == 1
    assert "non-verbatim own-response content-check evidence" in judge.prompts[1]
    assert hashlib.sha256(judge.prompts[0].encode()).hexdigest() != hashlib.sha256(judge.prompts[1].encode()).hexdigest()
    assert judge.contexts[1]["validation_feedback"].startswith("ValueError:")


def test_judge_retry_cap_remains_four_for_unparseable_json():
    judge = FakeJudge(["{"] * 4)
    grade, failures = paired_grade_local(
        judge,
        {
            "canonical_id": "retry-cap-fixture", "question": "Q", "clean_response": "Clean",
            "intended_axes": ["overall_quality"],
        },
        "Candidate", 1, 1,
    )
    assert grade is None
    assert len(judge.prompts) == len(failures) == 4
    assert all("JSONDecodeError" in prompt for prompt in judge.prompts[1:])


def test_local_offsets_and_hash_are_bound_to_the_scored_side():
    text = "A locally supported sentence."
    labels, _ = explicit_local_labels([{"axis": "empathy", "scope": "local_support", "label": 0}])
    assert labels["empathy"] == 0
    assert hashlib.sha256(text.encode()).hexdigest() != hashlib.sha256((text + "x").encode()).hexdigest()
