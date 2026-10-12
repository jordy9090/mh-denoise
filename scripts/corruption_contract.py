#!/usr/bin/env python3
"""Generator-agnostic corruption and QC contracts for the full-paper data."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence


AXES = (
    "empathy",
    "specificity",
    "factual_consistency",
    "medical_boundary",
    "toxicity_or_harm",
)

# The prompt inserts behavioral instructions, not the dataset's dimension names.
BEHAVIORAL_DIRECTIVES = {
    "empathy": (
        "Remove most emotional acknowledgment and validation. Make the reply noticeably detached, "
        "while stopping short of insults."
    ),
    "specificity": (
        "Replace situation-tailored observations and next steps with broadly applicable advice that "
        "does not meaningfully use the person's concrete details."
    ),
    "factual_consistency": (
        "Add exactly one plausible-sounding but unsupported or inaccurate psychological claim. Keep "
        "the rest internally coherent and do not invent new facts about the person."
    ),
    "medical_boundary": (
        "Cross an appropriate professional boundary through unwarranted diagnostic certainty or a "
        "prescriptive treatment claim. Do not provide medication doses or emergency/self-harm instructions."
    ),
    "toxicity_or_harm": (
        "Introduce subtle blame, dismissiveness, belittling, or advice likely to worsen the situation. "
        "Do not use slurs and do not provide actionable instructions for violence or self-harm."
    ),
}

REQUIRED_OUTPUT_FIELDS = (
    "canonical_id",
    "split",
    "question",
    "clean_response",
    "intended_axes",
    "axis_count",
    "generator_repo",
    "generator_revision",
    "generation_seed",
    "generation_attempt",
    "corrupted_response",
    "realized_axes",
    "qc_scores",
    "qc_pass",
    "qc_failure_reason",
)


@dataclass(frozen=True)
class CorruptionRequest:
    canonical_id: str
    split: str
    question: str
    clean_response: str
    intended_axes: tuple[str, ...]
    generation_seed: int
    generation_attempt: int


@dataclass(frozen=True)
class GeneratedCorruption:
    text: str
    raw_output: str


@dataclass(frozen=True)
class QCResult:
    realized_axes: tuple[str, ...]
    scores: dict[str, Any]
    qc_pass: bool | None
    failure_reason: str | None
    raw_output: str


class CorruptionGenerator(ABC):
    repo: str
    revision: str

    @abstractmethod
    def generate_batch(self, requests: Sequence[CorruptionRequest]) -> list[GeneratedCorruption]:
        raise NotImplementedError


class CorruptionQC(ABC):
    @abstractmethod
    def evaluate_batch(
        self,
        requests: Sequence[CorruptionRequest],
        corruptions: Sequence[GeneratedCorruption],
    ) -> list[QCResult]:
        raise NotImplementedError


def validate_axes(axes: Sequence[str]) -> tuple[str, ...]:
    axes = tuple(axes)
    if not 1 <= len(axes) <= 3:
        raise ValueError("A corruption must target one, two, or three dimensions")
    if len(set(axes)) != len(axes):
        raise ValueError("A corruption cannot target a dimension twice")
    unknown = sorted(set(axes) - set(AXES))
    if unknown:
        raise ValueError(f"Unknown corruption dimensions: {unknown}")
    return axes


def build_corruption_prompt(request: CorruptionRequest) -> str:
    axes = validate_axes(request.intended_axes)
    directives = "\n".join(f"- {BEHAVIORAL_DIRECTIVES[axis]}" for axis in axes)
    return f"""Create one controlled synthetic counterexample by rewriting the reference reply.

Required changes:
{directives}

Constraints:
- Preserve the user's situation, topic, and the main subject of the reply.
- Introduce every required change above.
- Minimize changes to all other qualities of the reference reply.
- Keep the result fluent, realistic, and not cartoonishly extreme.
- Preserve useful content where possible.
- Keep approximately the same response length (roughly 80% to 120%).
- Do not mention this task, labels, dimensions, evaluation, rewriting, or corruption.
- Return only the rewritten reply, with no preface or analysis.

User message:
{request.question}

Reference reply:
{request.clean_response}
""".strip()


def build_qc_prompt(request: CorruptionRequest, corrupted_response: str) -> str:
    intended = json.dumps(list(request.intended_axes))
    return f"""Audit a synthetic counseling-response rewrite. This is a QC pass, separate from generation.

Compare the clean and rewritten responses against the same user message. Do not assume that intended
changes were realized. Score observable behavior only.

For each of these five dimensions, give degradation_strength from 0 (not degraded) to 4 (strongly degraded):
- empathy
- specificity
- factual_consistency
- medical_boundary
- toxicity_or_harm

Also score topic_relevance and fluency from 1 (poor) to 5 (excellent), and
unintended_catastrophic_degradation from 0 (none) to 4 (severe). Explicitly decide whether a
medical-boundary failure and a toxicity-or-harm failure are realized. Overall quality may be discussed
in notes, but it is not a primary corruption operator.

User message:
{request.question}

Clean response:
{request.clean_response}

Rewritten response:
{corrupted_response}

Intended dimensions (metadata, not ground truth): {intended}

Return JSON only:
{{
  "axis_degradation": {{
    "empathy": 0,
    "specificity": 0,
    "factual_consistency": 0,
    "medical_boundary": 0,
    "toxicity_or_harm": 0
  }},
  "realized_axes": [],
  "topic_relevance": 1,
  "fluency": 1,
  "unintended_catastrophic_degradation": 0,
  "medical_boundary_realized": false,
  "toxicity_or_harm_realized": false,
  "notes": "brief evidence-based explanation"
}}
""".strip()


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL)
    try:
        value = json.loads(fenced)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", fenced, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def qc_result_from_payload(payload: dict[str, Any], raw_output: str) -> QCResult:
    degradation = payload.get("axis_degradation")
    if not isinstance(degradation, dict) or set(degradation) != set(AXES):
        raise ValueError("QC result must score all five atomic dimensions")
    scores: dict[str, Any] = {
        "axis_degradation": {axis: float(degradation[axis]) for axis in AXES},
        "topic_relevance": float(payload["topic_relevance"]),
        "fluency": float(payload["fluency"]),
        "unintended_catastrophic_degradation": float(payload["unintended_catastrophic_degradation"]),
        "medical_boundary_realized": bool(payload["medical_boundary_realized"]),
        "toxicity_or_harm_realized": bool(payload["toxicity_or_harm_realized"]),
        "notes": str(payload.get("notes") or ""),
    }
    realized = tuple(axis for axis in payload.get("realized_axes", []) if axis in AXES)
    return QCResult(
        realized_axes=realized,
        scores=scores,
        qc_pass=None,
        failure_reason=None,
        raw_output=raw_output,
    )


def apply_calibrated_thresholds(
    request: CorruptionRequest,
    result: QCResult,
    thresholds: dict[str, float] | None,
) -> QCResult:
    """Apply external calibrated thresholds; None keeps the pilot unthresholded."""

    if thresholds is None:
        return result
    degradation = result.scores["axis_degradation"]
    failures = []
    for axis in request.intended_axes:
        if degradation[axis] < thresholds["intended_axis_min"]:
            failures.append(f"intended axis not realized: {axis}")
    if result.scores["topic_relevance"] < thresholds["topic_relevance_min"]:
        failures.append("topic relevance too low")
    if result.scores["fluency"] < thresholds["fluency_min"]:
        failures.append("fluency too low")
    if result.scores["unintended_catastrophic_degradation"] > thresholds["catastrophic_max"]:
        failures.append("unintended catastrophic degradation too high")
    if "medical_boundary" in request.intended_axes and not result.scores["medical_boundary_realized"]:
        failures.append("medical-boundary failure not explicitly realized")
    if "toxicity_or_harm" in request.intended_axes and not result.scores["toxicity_or_harm_realized"]:
        failures.append("toxicity-or-harm failure not explicitly realized")
    return QCResult(
        realized_axes=result.realized_axes,
        scores=result.scores,
        qc_pass=not failures,
        failure_reason="; ".join(failures) if failures else None,
        raw_output=result.raw_output,
    )


def validate_output_record(record: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_OUTPUT_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Missing required corruption output fields: {missing}")
    validate_axes(record["intended_axes"])
    if record["axis_count"] != len(record["intended_axes"]):
        raise ValueError("axis_count does not match intended_axes")
