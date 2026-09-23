import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_provisional_production_corrected_export import (  # noqa: E402
    apply_collision_mask,
    build_scorer_rows,
    find_non_null_collisions,
)
from local_qwen_production_qc_v4 import content_disposition  # noqa: E402


PRODUCTION = ROOT / "data/fullpaper_acl_pipeline/production_run_train1000_valid200_local_qwen35_27b_v3r1_20260910"


def accepted(canonical_id):
    for line in (PRODUCTION / "accepted.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["canonical_id"] == canonical_id:
            return row
    raise AssertionError(f"missing fixture row {canonical_id}")


def source_sft(canonical_id):
    for split in ("train", "valid"):
        for line in (PRODUCTION / f"{split}_sft.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["metadata"]["canonical_id"] == canonical_id:
                return row
    raise AssertionError(f"missing fixture SFT row {canonical_id}")


def review(canonical_id, annotations=()):
    return {
        "canonical_id": canonical_id,
        "clean_disposition": "retained",
        "decision_source": "fixture-reviewed-decision",
        "decision_evidence": ["fixture evidence"],
        "scorer_annotations": list(annotations),
    }


def annotation_from_span(span, *, scope, label):
    return {
        "side": span["side"], "axis": span["axis"], "start": span["start"], "end": span["end"],
        "text": span["text"], "source_sha256": span["source_sha256"], "scope": scope,
        "label": label, "annotation_source": "fixture-reviewed-local-evidence",
        "annotation_evidence": "fixture evidence supports this model input",
    }


def test_clean_gate_regression_cases_are_held_by_evidence_backed_signals():
    # The fixture supplies the structured QC decision that future production
    # requires.  The gate itself contains no canonical-ID or phrase exception.
    cases = {
        "qa_b8711c6b200b077b7e8db6ac": "your mother's sleep disturbances",
        "qa_ec895ef531a913aaee613844": "2853,You are a helpful mental health counselling assistant",
        "qa_d25175051aa75a2305ad15be": "surveys we've been working on",
    }
    for canonical_id, evidence in cases.items():
        row = accepted(canonical_id)
        grade = json.loads(json.dumps(row["final_grade"]))
        assert evidence in row["clean_response"]
        grade["clean_target_eligibility"] = {
            "disposition": "hold", "axis": "factual_consistency",
            "evidence_scope": "local_defect", "evidence_source": "own_response",
            "evidence_span": evidence,
            "reason": "The clean target asserts material context not supported by the complete user message.",
        }
        disposition, reason = content_disposition(
            row["question"], row["corrupted_response"], grade, clean=row["clean_response"]
        )
        assert (disposition, reason) == ("hold", "clean_review_required")
        assert grade["clean_review_signals"]


def test_unchanged_empathic_opening_is_not_auto_promoted_to_positive_local_defect():
    canonical_id = "qa_a4ec11b76f2d214737a06d91"
    sft = source_sft(canonical_id)
    rows = build_scorer_rows([sft], {canonical_id: review(canonical_id)})
    target = [row for row in rows if "I can see how important this seminar" in row["span"]]
    assert target
    assert all(row["labels"]["empathy"] is None for row in target)


def test_reviewed_cold_local_cue_can_remain_positive():
    canonical_id = "qa_0d2fef54e38c0130eeedd010"
    sft = source_sft(canonical_id)
    span = next(
        item for item in sft["metadata"]["span_supervision"]
        if item["side"] == "candidate" and item["axis"] == "empathy"
        and "I acknowledge" in item["text"]
    )
    rows = build_scorer_rows(
        [sft], {canonical_id: review(canonical_id, [annotation_from_span(span, scope="local_defect", label=1)])}
    )
    target = next(row for row in rows if row["span_start"] == span["start"] and row["span_side"] == "candidate")
    assert target["labels"]["empathy"] == 1


def test_unchanged_empathic_opening_is_unknown_without_reviewer_scope():
    canonical_id = "qa_7b818442efd87999948d25cc"
    sft = source_sft(canonical_id)
    rows = build_scorer_rows([sft], {canonical_id: review(canonical_id)})
    target = [row for row in rows if "I'm glad to hear that you're feeling more hopeful" in row["span"]]
    assert target
    assert all(row["labels"]["empathy"] is None for row in target)


def test_conflicting_same_input_axis_is_masked_and_ledgered():
    rows = [
        {"canonical_id": "qa-fixture-a", "question": "Q", "span": "Same sentence.",
         "labels": {"overall_quality": None, "empathy": 0, "specificity": None, "factual_consistency": None, "medical_boundary": None, "toxicity_or_harm": None},
         "label_mask": {"overall_quality": False, "empathy": True, "specificity": False, "factual_consistency": False, "medical_boundary": False, "toxicity_or_harm": False}, "label_basis": {}},
        {"canonical_id": "qa-fixture-b", "question": "Q", "span": "Same sentence.",
         "labels": {"overall_quality": None, "empathy": 1, "specificity": None, "factual_consistency": None, "medical_boundary": None, "toxicity_or_harm": None},
         "label_mask": {"overall_quality": False, "empathy": True, "specificity": False, "factual_consistency": False, "medical_boundary": False, "toxicity_or_harm": False}, "label_basis": {}},
    ]
    collisions = find_non_null_collisions(rows)
    assert len(collisions) == 1
    ledger = apply_collision_mask(rows, collisions, "fixture")
    assert len(ledger) == 1
    assert all(row["labels"]["empathy"] is None and not row["label_mask"]["empathy"] for row in rows)
    assert find_non_null_collisions(rows) == {}
