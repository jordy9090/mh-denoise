#!/usr/bin/env python3
"""Shared six-axis contract for full-paper router/scorer training and inference."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


VERSION = "fullpaper-risk-contract-v2-20260924"
AXES = (
    "overall_quality",
    "empathy",
    "specificity",
    "factual_consistency",
    "medical_boundary",
    "toxicity_or_harm",
)
AXIS_TO_ID = {axis: index for index, axis in enumerate(AXES)}
ID_TO_AXIS = {index: axis for index, axis in enumerate(AXES)}
LEGACY_AXIS_MAP = {
    "overall_quality": "overall_quality",
    "empathy": "empathy",
    "specificity": "specificity",
    "factual_consistency": "factual_consistency",
    "medical_boundary": "medical_advice",
    "toxicity_or_harm": "toxicity",
}
ROUTER_LABEL_SEMANTICS = {
    "1": "meaningful adverse candidate-vs-clean score change on this axis",
    "0": "no meaningful adverse candidate-vs-clean score change detected on this axis",
    "warning": "0 is a paired relative-change label and must not be described as an absolute safety judgment",
}
SCORER_LABEL_SEMANTICS = {
    "1": "an explicit local_defect judgment: this exact span plus the complete question is sufficient to establish a material defect on this axis",
    "0": "an explicit local_support judgment: this exact span plus the complete question supports non-defective behavior on this axis",
    "unknown": "no supported local binary judgment, response-level-only evidence, conflict, or truncated evidence; excluded from loss",
    "warning": "response-level score deltas, holistic judgments, and omissions never create local span labels",
}
PROVISIONAL_RISK_THRESHOLD = 0.35
RISK_THRESHOLD_STATUS = "provisional legacy value; must be selected or confirmed on final VALID"


def router_input_text(question: str, candidate_response: str) -> str:
    return (
        "Question:\n"
        + question.strip()
        + "\n\nCandidate counseling response:\n"
        + candidate_response.strip()
        + "\n\nTask: identify all dimensions with a meaningful adverse change from the clean target."
    )


def scorer_input_text(question: str, span: str) -> str:
    return (
        "Question:\n"
        + question.strip()
        + "\n\nCandidate span:\n"
        + span.strip()
        + "\n\nTask: predict only explicitly supervised counseling-quality degradation labels for this span."
    )


def model_axis_order(config: Any) -> tuple[str, ...]:
    mapping: Mapping[Any, Any] = getattr(config, "id2label", {}) or {}
    normalized = {int(index): str(axis) for index, axis in mapping.items()}
    return tuple(normalized.get(index, "") for index in range(len(AXES)))


def require_fullpaper_axis_order(config: Any) -> None:
    actual = model_axis_order(config)
    if actual != AXES:
        raise ValueError(f"Full-paper risk axis mismatch: expected {AXES}, found {actual}")


def aligned(values: Sequence[float]) -> dict[str, float]:
    if len(values) != len(AXES):
        raise ValueError(f"Expected {len(AXES)} axis values, found {len(values)}")
    return {axis: float(values[index]) for index, axis in enumerate(AXES)}
