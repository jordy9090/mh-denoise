from types import SimpleNamespace

import pytest

from scripts.fullpaper_risk_contract import (
    AXES,
    RISK_THRESHOLD_STATUS,
    ROUTER_LABEL_SEMANTICS,
    require_fullpaper_axis_order,
    router_input_text,
    scorer_input_text,
)
from scripts.train_fullpaper_risk_model import audit_train_valid_provenance, make_text
from scripts import selective_risk_refinement_utils as refinement


def test_training_and_inference_text_contract_is_shared():
    router_row = {"question": " q ", "candidate_response": " response "}
    scorer_row = {"question": " q ", "span": " span "}
    assert make_text("router", router_row) == router_input_text(" q ", " response ")
    assert make_text("scorer", scorer_row) == scorer_input_text(" q ", " span ")


def test_axis_order_and_relative_zero_semantics_are_fail_closed():
    config = SimpleNamespace(id2label={i: axis for i, axis in enumerate(AXES)})
    require_fullpaper_axis_order(config)
    assert "absolute safety" in ROUTER_LABEL_SEMANTICS["warning"]
    assert "final VALID" in RISK_THRESHOLD_STATUS
    wrong = SimpleNamespace(id2label={i: axis for i, axis in enumerate(reversed(AXES))})
    with pytest.raises(ValueError, match="axis mismatch"):
        require_fullpaper_axis_order(wrong)


def test_main_training_requires_distinct_canonical_train_valid_provenance():
    train = [{
        "canonical_id": "train-1", "split": "train",
        "question_normalized_sha256": "q-train", "duplicate_cluster_id": "d-train",
        "source_group_id": "g-train",
    }]
    valid = [{
        "canonical_id": "valid-1", "split": "valid",
        "question_normalized_sha256": "q-valid", "duplicate_cluster_id": "d-valid",
        "source_group_id": "g-valid",
    }]
    result = audit_train_valid_provenance(train, valid, require_full_provenance=True)
    assert result["all_zero"] is True

    valid[0]["source_group_id"] = "g-train"
    with pytest.raises(RuntimeError, match="provenance overlap"):
        audit_train_valid_provenance(train, valid, require_full_provenance=True)


def test_legacy_smoke_allows_missing_provenance_but_still_rejects_known_overlap():
    assert audit_train_valid_provenance([{"canonical_id": "a"}], [{"canonical_id": "b"}], False)["all_zero"]
    with pytest.raises(RuntimeError, match="provenance overlap"):
        audit_train_valid_provenance([{"canonical_id": "same"}], [{"canonical_id": "same"}], False)


def test_proposed_fullpaper_contract_uses_shared_axes_and_text_then_restores_legacy():
    refinement.configure_risk_contract("fullpaper_v1")
    assert tuple(refinement.DIMS) == AXES
    assert refinement.router_text("q", "r") == router_input_text("q", "r")
    assert refinement.risk_text("q", "span") == scorer_input_text("q", "span")
    refinement.configure_risk_contract("legacy")
    assert "medical_advice" in refinement.DIMS
