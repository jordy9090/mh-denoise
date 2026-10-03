from scripts.build_production_reuse_export import (
    classify_one,
    deduplicate_scorer_rows,
    router_labels_for_pair,
)
import json

from scripts.fullpaper_risk_contract import AXES
from scripts.rejudge_production_reuse_subset import (
    adjudicate_grade,
    paired_grade_local_many,
    read_explicit_ids,
    select_rows,
)


def grade(*, clean_hold=False, realized=("empathy",), unrelated=False):
    clean_gate = {
        "disposition": "hold" if clean_hold else "pass",
        "axis": "factual_consistency" if clean_hold else None,
        "evidence_scope": "local_defect" if clean_hold else "none",
        "evidence_source": "own_response" if clean_hold else "none",
        "evidence_span": "unsupported history" if clean_hold else "",
        "reason": "unsupported" if clean_hold else "eligible",
    }
    checks = {
        "topic_preserved": {"value": True}, "fluent": {"value": True},
        "unrelated_topic_injection": {"value": unrelated},
        "cartoonish_or_exaggerated_shaming": {"value": False},
        "prohibited_dangerous_content": {"value": False, "evidence_sufficient": True, "evidence_span": ""},
        "editing_task_refusal_or_meta": {"value": False},
    }
    return {
        "clean_target_eligibility": clean_gate, "local_supervision": [],
        "content_checks": checks, "realized_axes": list(realized),
    }


def row():
    return {
        "canonical_id": "qa-x", "question": "I feel ignored.",
        "clean_response": "unsupported history" if False else "I hear you.",
        "corrupted_response": "Stop complaining.", "intended_axes": ["empathy"],
    }


def test_pair_and_clean_are_adjudicated_separately():
    decision = adjudicate_grade(row(), grade())
    assert decision["clean_status"] == "suitable"
    assert decision["pair_status"] == "reusable"
    missing = adjudicate_grade(row(), grade(realized=()))
    assert missing["clean_status"] == "suitable"
    assert missing["pair_status"] == "regenerate_candidate"


def test_clean_issue_does_not_erase_pair_assessment():
    target = row()
    target["clean_response"] = "unsupported history"
    decision = adjudicate_grade(target, grade(clean_hold=True))
    assert decision["clean_status"] == "original_answer_issue"
    assert decision["pair_status"] == "reusable"


def test_clean_pass_plus_local_defect_is_review_required_not_auto_excluded():
    payload = grade()
    payload["local_supervision"] = [{
        "side": "clean", "scope": "local_defect", "label": 1,
        "axis": "specificity", "text": "I hear you.",
        "reason": "The span is generic.",
    }]
    decision = adjudicate_grade(row(), payload)
    assert decision["clean_status"] == "unresolved"
    assert decision["pair_status"] == "reusable"
    assert decision["clean_signals"][0]["recommended_disposition"] == "review_required"


def test_current_content_reject_requests_regeneration():
    decision = adjudicate_grade(row(), grade(unrelated=True))
    assert decision["pair_status"] == "regenerate_candidate"


def test_selection_reuses_prior_reviews_and_only_rejudges_missing_warning_rows():
    accepted = [{"canonical_id": name} for name in ("a", "b", "c", "d")]
    suspicious = [{"canonical_id": name} for name in ("a", "b", "c")]
    prior = [{"canonical_id": "a", "clean_disposition": "held"}]
    codex = [{"canonical_id": "b"}]
    assert [row["canonical_id"] for row in select_rows(accepted, suspicious, prior, codex)] == ["c"]


def test_explicit_selection_overrides_warning_filter_and_validates_ids(tmp_path):
    accepted = [{"canonical_id": name} for name in ("a", "b", "c")]
    explicit = tmp_path / "ids.txt"
    explicit.write_text("c\na\n", encoding="utf-8")
    ids = read_explicit_ids(explicit)
    assert ids == {"a", "c"}
    assert [item["canonical_id"] for item in select_rows(
        accepted, [], [], [], explicit_ids=ids,
    )] == ["a", "c"]


def accepted_row():
    return {
        "canonical_id": "qa-x", "split": "train", "source": "fixture",
        "question": "I feel ignored.", "clean_response": "I hear you.",
        "corrupted_response": "Stop complaining.", "intended_axes": ["empathy"],
        "realized_axes": ["empathy"],
    }


def test_legacy_no_warning_is_unconfirmed_candidate_not_content_exclusion():
    decision = classify_one(
        accepted_row(), prior_hold=False, codex=None, warning=False, local=None,
    )
    assert decision["reuse_disposition"] == "unresolved"
    assert decision["pair_assessment"] == "legacy_reuse_candidate"
    assert decision["decision_source"] == "legacy_accepted_pair_plus_no_clean_warning"


def test_clean_and_pair_failure_categories_are_distinct():
    clean = classify_one(
        accepted_row(), prior_hold=True, codex=None, warning=True, local=None,
    )
    assert clean["reuse_disposition"] == "exclude_original_answer"
    local = {"status": "complete", "clean_status": "suitable", "pair_status": "regenerate_candidate", "pair_reason": "missing_intended_axes:empathy"}
    pair = classify_one(
        accepted_row(), prior_hold=False, codex=None, warning=True, local=local,
    )
    assert pair["reuse_disposition"] == "regenerate_candidate"


def test_export_holds_internal_clean_qc_conflict_instead_of_excluding():
    local = {
        "status": "complete", "clean_status": "unresolved",
        "pair_status": "reusable", "clean_signals": [{"reason": "QC disagreement"}],
    }
    decision = classify_one(
        accepted_row(), prior_hold=False, codex=None, warning=True, local=local,
    )
    assert decision["reuse_disposition"] == "unresolved"
    assert decision["clean_assessment"] == "unresolved"


def test_router_uses_latest_local_axes_and_masks_unjudged_codex_axes():
    source = {
        "metadata": {"realized_axes": ["empathy", "specificity"]},
    }
    local = {
        "status": "complete", "current_realized_axes": ["specificity"],
        "grade": {"local_supervision": []},
    }
    labels, mask, provenance = router_labels_for_pair(source, local=local, codex=None)
    assert labels["empathy"] == 0 and labels["specificity"] == 1
    assert all(mask.values())
    assert provenance["legacy_realized_axes"] == ["empathy", "specificity"]

    codex = {"schema_version": "fixture-v1", "ai_review": {
        "review_source": "codex_review",
        "intended_defect": {
            "axes_present": ["empathy"], "axes_not_supported": ["specificity"],
        },
        "collateral_damage": {"axes": []},
    }}
    labels, mask, _ = router_labels_for_pair(source, local=None, codex=codex)
    assert labels["empathy"] == 1 and labels["specificity"] == 0
    assert labels["factual_consistency"] is None
    assert mask["factual_consistency"] is False


def test_new_candidate_qc_supersedes_review_of_previous_candidate():
    local = {
        "status": "complete", "clean_status": "suitable", "pair_status": "reusable",
        "pair_reason": None, "replacement_generation_run": "/fixture/replacement",
    }
    stale_codex = {"ai_review": {
        "review_source": "old_candidate_review", "proposal": "hold",
        "clean_integrity": {"status": "pass", "reason_ko": "clean is suitable"},
        "intended_defect": {"reason_ko": "old candidate failed"},
        "summary_ko": {"candidate": "old candidate only"},
    }}
    decision = classify_one(
        accepted_row(), prior_hold=False, codex=stale_codex, warning=False, local=local,
    )
    assert decision["reuse_disposition"] == "reusable"
    assert decision["decision_source"] == "current_v6_local_paired_qc_after_candidate_regeneration"


def test_scorer_dedup_preserves_one_non_null_label_and_sources():
    base = {
        "split": "train", "question": "Q", "span": "S", "span_side": "candidate",
        "span_start": 0, "span_end": 1, "span_source_sha256": "x",
        "labels": {axis: None for axis in ("overall_quality", "empathy", "specificity", "factual_consistency", "medical_boundary", "toxicity_or_harm")},
        "label_basis": {axis: "unknown" for axis in ("overall_quality", "empathy", "specificity", "factual_consistency", "medical_boundary", "toxicity_or_harm")},
        "review_annotations": [], "source_records": [], "question_normalized_sha256": "q",
        "duplicate_cluster_id": "d", "source_group_id": "s",
    }
    first = {**base, "canonical_id": "a", "labels": {**base["labels"], "empathy": 1}, "label_basis": {**base["label_basis"], "empathy": "reviewed"}}
    second = {**base, "canonical_id": "b"}
    deduped, ledger = deduplicate_scorer_rows([first, second])
    assert len(deduped) == 1 and len(ledger) == 1
    assert deduped[0]["labels"]["empathy"] == 1
    assert len(deduped[0]["source_provenance"]) == 2


def _valid_paired_payload():
    def scored(axis):
        item = {
            "evidence_scope": "none", "evidence_source": "none",
            "evidence_span": "", "reason": "No local claim.",
        }
        item["violation" if axis == "medical_boundary" else "score"] = (
            False if axis == "medical_boundary" else (1 if axis == "toxicity_or_harm" else 4)
        )
        return item

    checks = {
        "topic_preserved": {"value": True, "evidence_span": "", "reason": "Yes."},
        "fluent": {"value": True, "evidence_span": "", "reason": "Yes."},
        "unrelated_topic_injection": {"value": False, "evidence_span": "", "reason": "No."},
        "cartoonish_or_exaggerated_shaming": {"value": False, "evidence_span": "", "reason": "No."},
        "prohibited_dangerous_content": {"value": False, "evidence_sufficient": True, "evidence_span": "", "reason": "No."},
        "editing_task_refusal_or_meta": {"value": False, "evidence_span": "", "reason": "No."},
    }
    return {
        "scores": {axis: {"A": scored(axis), "B": scored(axis), "comparison_reason": "Equal."} for axis in AXES},
        "response_eligibility": {side: {"disposition": "pass", "axis": None, "evidence_scope": "none", "evidence_source": "none", "evidence_span": "", "reason": "Pass."} for side in ("A", "B")},
        "content_checks": {"A": checks, "B": checks},
        "text_reason_contradiction": {"detected": False, "reason": "No."},
        "overall_reason": "Valid fixture.",
    }


def test_batched_rejudge_preserves_per_item_records_and_usage():
    raw = json.dumps(_valid_paired_payload())

    class Backend:
        def call_many(self, prompts):
            return [(raw, {"input_tokens": 10, "output_tokens": 20, "elapsed_seconds": 1.5}) for _ in prompts]

    class Judge:
        backend = Backend()

        def __init__(self):
            self.records = []
            self.context = {}

        def _record_call(self, value):
            self.records.append(value)

    judge = Judge()
    batch = [{**accepted_row(), "canonical_id": name} for name in ("qa-a", "qa-b")]
    results = paired_grade_local_many(judge, batch, 1, 1)
    assert len(results) == 2 and all(grade is not None for grade, _, _ in results)
    assert len(judge.records) == 2
    assert {item["context"]["canonical_id"] for item in judge.records} == {"qa-a", "qa-b"}
    assert all(usage["judge_calls"] == 1 for _, _, usage in results)
