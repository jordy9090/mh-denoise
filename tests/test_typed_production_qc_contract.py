import hashlib
import inspect

import pytest

from scripts.build_fullpaper_training_data import explicit_local_labels
from scripts.local_qwen_production_qc_v4 import (
    AXES,
    content_disposition,
    validate_and_map,
)
from scripts.run_fullpaper_corruption_production import source_verified_spans
from scripts import local_qwen_production_qc_v4, source_integrity_contract


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
    return {
        "scores": {
            axis: {"A": side(axis), "B": side(axis), "comparison_reason": "Compared complete responses."}
            for axis in AXES
        },
        "clean_target_eligibility": {
            "disposition": "pass", "axis": None, "evidence_scope": "none",
            "evidence_source": "none", "evidence_span": "", "reason": "No clear material clean defect.",
        },
        "content_checks": {
            "topic_preserved": {"value": True, "evidence_span": "", "reason": "Topic retained."},
            "fluent": {"value": True, "evidence_span": "", "reason": "Fluent."},
            "unrelated_topic_injection": {"value": False, "evidence_span": "", "reason": "None."},
            "cartoonish_or_exaggerated_shaming": {"value": False, "evidence_span": "", "reason": "None."},
            "prohibited_dangerous_content": {"value": False, "evidence_sufficient": True, "evidence_span": "", "reason": "None."},
            "editing_task_refusal_or_meta": {"value": False, "evidence_span": "", "reason": "None."},
        },
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


def test_clean_hold_blocks_acceptance_and_exact_export():
    question = "I have a small facial cut."
    clean = "Your mother's sleep disturbances explain this."
    candidate = "Keep the area clean."
    data = payload()
    data["clean_target_eligibility"] = {
        "disposition": "hold", "axis": "factual_consistency",
        "evidence_scope": "local_defect", "evidence_source": "own_response",
        "evidence_span": "Your mother's sleep disturbances",
        "reason": "This asserts unsupported family history.",
    }
    grade, _ = validate_and_map(data, clean, candidate, "A", "B", question=question)
    assert content_disposition(question, candidate, grade, clean=clean) == (
        "hold", "clean_review_required"
    )


def test_production_adapter_requires_explicit_scope_label_pair():
    records = [{"axis": "empathy", "scope": "local_support", "label": 0}]
    labels, basis = explicit_local_labels(records)
    assert labels["empathy"] == 0
    assert basis["empathy"] == "explicit_local_support_exact_span"
    with pytest.raises(ValueError, match="Invalid typed local"):
        explicit_local_labels([{"axis": "empathy", "scope": "holistic", "label": 1}])


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


def test_local_offsets_and_hash_are_bound_to_the_scored_side():
    text = "A locally supported sentence."
    labels, _ = explicit_local_labels([{"axis": "empathy", "scope": "local_support", "label": 0}])
    assert labels["empathy"] == 0
    assert hashlib.sha256(text.encode()).hexdigest() != hashlib.sha256((text + "x").encode()).hexdigest()
