import json
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from corruption_contract_v2 import (  # noqa: E402
    AXES,
    StageRequest,
    acceptance_decision,
    build_judge_prompt,
    build_stage_prompt,
    parse_judge_result,
    validate_judge_evidence,
)
from fullpaper_acl_pipeline import read_jsonl  # noqa: E402
from run_gemma_corruption_pilot_v2 import (  # noqa: E402
    PILOT_SEED,
    attach_specs,
    balanced_specs,
    select_rows,
)


def judge_payload(violated=()):
    return {
        "axes": {
            axis: {
                "violated": axis in violated,
                "evidence_span": "bad claim" if axis in violated else "",
                "reason": "observable defect" if axis in violated else "not observed",
            }
            for axis in AXES
        },
        "topic_preserved": True,
        "fluent": True,
        "unnecessary_rewriting": False,
        "catastrophic_safety_issue": False,
        "notes": "",
    }


def test_six_axis_balancing_is_exact_and_deterministic():
    specs, stats = balanced_specs(100, PILOT_SEED)
    assert specs == balanced_specs(100, PILOT_SEED)[0]
    assert stats["axis_count_counts"] == {"1": 50, "2": 35, "3": 15}
    assert sorted(stats["marginal_counts"].values()) == [27, 27, 27, 28, 28, 28]
    assert stats["pairwise_range"] <= 1
    assert set(axis for spec in specs for axis in spec) == set(AXES)


def test_selection_is_fixed_balanced_and_unique():
    rows = list(read_jsonl(ROOT / "data/fullpaper_acl_pipeline/canonical_clean_qa.jsonl"))
    selected = select_rows(rows, 100, PILOT_SEED)
    assert len({row["canonical_id"] for row in selected}) == 100
    assert Counter(row["source"] for row in selected) == {"MentalChat16K": 88, "CounselChat": 12}
    assert Counter(row["split"] for row in selected) == {"train": 80, "valid": 10, "test": 10}
    specs, _ = balanced_specs(100, PILOT_SEED)
    attached = attach_specs(selected, specs, PILOT_SEED)
    assert Counter(row["axis_count"] for row in attached) == {1: 50, 2: 35, 3: 15}


def test_stage_prompt_is_sequential_local_and_has_no_length_constraint():
    request = StageRequest(
        canonical_id="x", split="train", question="q", clean_response="clean",
        current_response="stage one", intended_axes=("empathy", "factual_consistency"),
        completed_axes=("empathy",), target_axis="factual_consistency", stage_index=2,
        generation_seed=7, generation_attempt=1,
    )
    prompt = build_stage_prompt(request)
    assert "exactly this new change" in prompt
    assert "Preserve every controlled defect already present" in prompt
    assert "unsupported or inaccurate claim" in prompt
    assert "no required sentence count or fixed length range" in prompt


def test_judge_is_blind_to_intent_and_acceptance_requires_all_axes():
    prompt = build_judge_prompt("q", "clean", "bad claim")
    assert "intended_axes" not in prompt
    result = parse_judge_result(
        judge_payload(("empathy", "specificity", "overall_quality")), json.dumps(judge_payload())
    )
    accepted, reason = acceptance_decision(result, ("empathy", "specificity"))
    assert accepted and reason is None
    assert set(result.realized_axes) == {"empathy", "specificity", "overall_quality"}
    accepted, reason = acceptance_decision(result, ("empathy", "medical_boundary"))
    assert not accepted and "medical_boundary" in reason


def test_judge_boolean_fields_are_strict():
    payload = judge_payload(("empathy",))
    payload["fluent"] = "false"
    with pytest.raises(ValueError, match="fluent"):
        parse_judge_result(payload, json.dumps(payload))


def test_omission_and_whole_response_evidence_do_not_require_candidate_span():
    payload = judge_payload(("overall_quality", "specificity"))
    payload["axes"]["overall_quality"].update(
        evidence_type="whole_response", evidence_span="",
        reason="The response is globally disorganized and much less useful.",
    )
    payload["axes"]["specificity"].update(
        evidence_type="omission", evidence_span="call your sister",
        reason="The source-specific next step was removed.",
    )
    result = parse_judge_result(payload, json.dumps(payload))
    validate_judge_evidence(
        result,
        candidate="Try to cope somehow.",
        clean_response="It may help to call your sister tomorrow.",
        normalize=lambda text: " ".join(text.lower().split()),
    )


def test_local_span_still_must_be_verbatim_candidate_text():
    payload = judge_payload(("factual_consistency",))
    payload["axes"]["factual_consistency"]["evidence_type"] = "span"
    result = parse_judge_result(payload, json.dumps(payload))
    with pytest.raises(ValueError, match="not verbatim"):
        validate_judge_evidence(
            result,
            candidate="No matching phrase.",
            clean_response="Reference.",
            normalize=lambda text: " ".join(text.lower().split()),
        )


def test_runner_refuses_non_pilot_size():
    with pytest.raises(ValueError, match="only permits"):
        select_rows([], 101, PILOT_SEED)


def test_completed_pilot_records_require_all_intended_axes():
    path = ROOT / "data/fullpaper_acl_pipeline/gemma_corruption_pilot_100_v2.jsonl"
    if not path.exists():
        pytest.skip("v2 pilot artifact has not been generated")
    records = list(read_jsonl(path))
    assert len(records) <= 100
    for record in records:
        assert set(record["intended_axes"]) <= set(record["realized_axes"])
        assert set(record["axis_judgments"]) == set(AXES)
        assert record["qc_pass"] is True
        for decision in record["axis_judgments"].values():
            if decision["violated"]:
                assert decision["evidence_span"]
