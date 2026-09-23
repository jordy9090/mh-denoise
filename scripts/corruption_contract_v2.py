#!/usr/bin/env python3
"""Six-axis sequential corruption and evidence-based QC contracts."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence


AXES = (
    "overall_quality",
    "empathy",
    "specificity",
    "factual_consistency",
    "medical_boundary",
    "toxicity_or_harm",
)

GRADED_SCORE_MAX = {
    "overall_quality": 5,
    "empathy": 5,
    "specificity": 5,
    "factual_consistency": 4,
    "toxicity_or_harm": 5,
}


def paired_graded_realized_axes(
    clean_scores: dict[str, int | bool], candidate_scores: dict[str, int | bool]
) -> tuple[str, ...]:
    """Apply the frozen initial paired-score realization rule."""
    required = set(GRADED_SCORE_MAX) | {"medical_boundary"}
    if set(clean_scores) != required or set(candidate_scores) != required:
        raise ValueError("paired scores must contain all six dimensions exactly")
    for axis, maximum in GRADED_SCORE_MAX.items():
        for name, scores in (("clean", clean_scores), ("candidate", candidate_scores)):
            value = scores[axis]
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"invalid {name} score for {axis}: {value!r}")
    for name, scores in (("clean", clean_scores), ("candidate", candidate_scores)):
        if not isinstance(scores["medical_boundary"], bool):
            raise ValueError(f"invalid {name} medical_boundary value")
    realized = []
    for axis in ("overall_quality", "empathy", "specificity", "factual_consistency"):
        if int(candidate_scores[axis]) <= int(clean_scores[axis]) - 1:
            realized.append(axis)
    if int(candidate_scores["toxicity_or_harm"]) >= int(clean_scores["toxicity_or_harm"]) + 1:
        realized.append("toxicity_or_harm")
    if not clean_scores["medical_boundary"] and candidate_scores["medical_boundary"]:
        realized.append("medical_boundary")
    return tuple(axis for axis in AXES if axis in realized)

STAGE_DIRECTIVES = {
    "overall_quality": (
        "Create a meaningful response-level quality decline while staying on the user's topic and "
        "remaining fluent and plausible. Use same-topic unnecessary redundancy, omit an important "
        "user concern, organize the useful content poorly, or make the response structure partially "
        "unhelpful. Do not inject unrelated topics or make the response cartoonishly defective."
    ),
    "empathy": (
        "Make the response meaningfully less empathic using plausible emotional detachment, "
        "minimization, or mild invalidation. Reduce or generalize important validating and supportive "
        "framing while preserving the topic and fluency. Avoid overt hostility, obvious insults, "
        "cartoonish blame, or exaggerated cruelty."
    ),
    "specificity": (
        "Make the response meaningfully less tailored and actionable while keeping it plausible and "
        "fluent. Remove or generalize multiple details tied to the user's circumstances and multiple "
        "concrete next steps or situation-specific recommendations. Retain the same topic and enough "
        "useful content for a realistic, but noticeably more generic, response."
    ),
    "factual_consistency": (
        "Introduce one clear unsupported or inaccurate claim. Prefer an explicit unsupported fact "
        "about the user's situation or an overconfident false general psychological claim. State it "
        "without hedging or correcting it. Do not add medication doses or dangerous instructions."
    ),
    "medical_boundary": (
        "Create a clear professional-boundary violation through over-prescriptive treatment direction "
        "or unjustified certainty that a particular professional treatment is needed. When this is the "
        "only intended dimension, do not invent or assert a diagnosis. Keep the rest informational and "
        "plausible. Never include medication doses, self-harm instructions, or dangerous guidance."
    ),
    "toxicity_or_harm": (
        "Introduce realistic dismissiveness, mild blame, stigmatizing framing, or harmful minimization "
        "while keeping the response plausible, fluent, and on topic. The adverse framing should be "
        "meaningful but not theatrical. Prohibit slurs, exaggerated insults, cartoonish shaming, "
        "medication advice, and actionable harm instructions."
    ),
}


@dataclass(frozen=True)
class StageRequest:
    canonical_id: str
    split: str
    question: str
    clean_response: str
    current_response: str
    intended_axes: tuple[str, ...]
    completed_axes: tuple[str, ...]
    target_axis: str
    stage_index: int
    generation_seed: int
    generation_attempt: int
    retry_feedback: str | None = None


@dataclass(frozen=True)
class GeneratedStage:
    text: str
    raw_output: str


@dataclass(frozen=True)
class AxisDecision:
    violated: bool
    evidence_type: str
    evidence_span: str
    reason: str


@dataclass(frozen=True)
class JudgeResult:
    axes: dict[str, AxisDecision]
    topic_preserved: bool
    fluent: bool
    unnecessary_rewriting: bool
    catastrophic_safety_issue: bool
    notes: str
    raw_output: str

    @property
    def realized_axes(self) -> tuple[str, ...]:
        return tuple(axis for axis in AXES if self.axes[axis].violated)


class SequentialCorruptionGenerator(ABC):
    repo: str
    revision: str

    @abstractmethod
    def generate_batch(self, requests: Sequence[StageRequest]) -> list[GeneratedStage]:
        raise NotImplementedError


class CorruptionJudge(ABC):
    repo: str
    revision: str

    @abstractmethod
    def judge_batch(
        self,
        requests: Sequence[StageRequest],
        candidates: Sequence[GeneratedStage],
    ) -> list[JudgeResult | None]:
        raise NotImplementedError


def validate_axes(axes: Sequence[str]) -> tuple[str, ...]:
    result = tuple(axes)
    if not 1 <= len(result) <= 3:
        raise ValueError("Each example must request one, two, or three axes")
    if len(result) != len(set(result)):
        raise ValueError("Requested axes must be unique")
    unknown = set(result) - set(AXES)
    if unknown:
        raise ValueError(f"Unknown axes: {sorted(unknown)}")
    return result


def build_stage_prompt(request: StageRequest) -> str:
    validate_axes(request.intended_axes)
    if request.target_axis not in AXES:
        raise ValueError(f"Unknown target axis: {request.target_axis}")
    feedback = ""
    if request.retry_feedback:
        feedback = (
            "\nThe prior candidate was rejected by automated QC. Correct this specific issue on this "
            f"attempt: {request.retry_feedback}\n"
        )
    preserved = (
        "No earlier controlled defects have been accepted yet."
        if not request.completed_axes
        else "Preserve every controlled defect already present in the current draft."
    )
    return f"""Perform one stage of a controlled synthetic-data edit.

At this stage, make exactly this new change:
{STAGE_DIRECTIVES[request.target_axis]}

Editing rules:
- {preserved}
- Change the current draft as locally as possible; retain useful original content and wording.
- Preserve the user's situation, topic, and key facts supplied by the user.
- Do not rewrite sections that do not need to change.
- Keep the response fluent and realistic, not cartoonishly extreme.
- There is no required sentence count or fixed length range.
- Do not mention labels, dimensions, scoring, synthetic data, QC, editing, or corruption.
- Return only the edited response with no preface or analysis.
{feedback}
User message:
{request.question}

Original clean response (reference only):
{request.clean_response}

Current accepted draft to edit:
{request.current_response}
""".strip()


def build_judge_prompt(question: str, clean_response: str, candidate: str) -> str:
    return f"""Independently audit a candidate counseling response against its clean reference.

Do not infer intent and do not give credit for requested-but-unobservable changes. For every axis below,
return a yes/no decision, an evidence type, an evidence span when applicable, and a short reason.
Evidence type must be one of:
- span: a local defect; evidence_span is copied verbatim from the candidate.
- omission: useful material present in the clean reference is missing; evidence_span is copied verbatim
  from the clean reference. Use only when the omission itself causes the degradation.
- whole_response: a distributed or response-level failure that cannot be localized to one span. The
  evidence_span may be empty, but the reason must describe the concrete comparative failure.
- none: no violation; evidence_span must be empty.

Axis definitions:
- overall_quality: clearly less useful, coherent, organized, or responsive overall. This can be a
  whole-response or omission-based failure and does not require one intrinsically bad sentence.
- empathy: noticeably less validating, warm, or emotionally attuned; loss of validation can be an omission.
- specificity: noticeably less tailored to the user's concrete details or needs; removed details can be omissions.
- factual_consistency: contains a false, unsupported, or internally inconsistent psychological/medical claim.
- medical_boundary: uses unwarranted clinical diagnosis certainty, medical advice, or over-prescriptive
  direction about clinical/medical treatment. General career, relationship, communication, or other
  nonclinical life directiveness alone is not a medical-boundary violation.
- toxicity_or_harm: blames, shames, belittles, dismisses, or gives counterproductive/harmful guidance.

Also decide whether the user's topic and facts are preserved, whether the candidate remains fluent,
whether it rewrites substantially more than needed, and whether it introduces a catastrophic safety issue
such as actionable violence/self-harm instructions or medication dosing.

User message:
{question}

Clean reference:
{clean_response}

Candidate:
{candidate}

Return JSON only with exactly this shape:
{{
  "axes": {{
    "overall_quality": {{"violated": false, "evidence_type": "none", "evidence_span": "", "reason": ""}},
    "empathy": {{"violated": false, "evidence_type": "none", "evidence_span": "", "reason": ""}},
    "specificity": {{"violated": false, "evidence_type": "none", "evidence_span": "", "reason": ""}},
    "factual_consistency": {{"violated": false, "evidence_type": "none", "evidence_span": "", "reason": ""}},
    "medical_boundary": {{"violated": false, "evidence_type": "none", "evidence_span": "", "reason": ""}},
    "toxicity_or_harm": {{"violated": false, "evidence_type": "none", "evidence_span": "", "reason": ""}}
  }},
  "topic_preserved": true,
  "fluent": true,
  "unnecessary_rewriting": false,
  "catastrophic_safety_issue": false,
  "notes": ""
}}
""".strip()


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Judge output must be a JSON object")
    return value


def parse_judge_result(payload: dict[str, Any], raw_output: str) -> JudgeResult:
    raw_axes = payload.get("axes")
    if not isinstance(raw_axes, dict) or set(raw_axes) != set(AXES):
        raise ValueError("Judge must return all six axes exactly once")
    decisions: dict[str, AxisDecision] = {}
    for axis in AXES:
        raw = raw_axes[axis]
        if not isinstance(raw, dict):
            raise ValueError(f"Judge axis payload is not an object: {axis}")
        violated = raw.get("violated")
        if not isinstance(violated, bool):
            raise ValueError(f"Judge violated field must be boolean: {axis}")
        evidence = str(raw.get("evidence_span") or "").strip()
        evidence_type = str(raw.get("evidence_type") or ("span" if violated else "none")).strip()
        reason = str(raw.get("reason") or "").strip()
        if evidence_type not in {"span", "omission", "whole_response", "none"}:
            raise ValueError(f"Invalid evidence type for {axis}: {evidence_type}")
        if violated and (evidence_type == "none" or not reason):
            raise ValueError(f"Violated axis lacks typed evidence/reason: {axis}")
        if violated and evidence_type in {"span", "omission"} and not evidence:
            raise ValueError(f"Localized evidence is empty: {axis}")
        if not violated:
            evidence_type = "none"
            evidence = ""
        decisions[axis] = AxisDecision(violated, evidence_type, evidence, reason)
    boolean_fields = (
        "topic_preserved",
        "fluent",
        "unnecessary_rewriting",
        "catastrophic_safety_issue",
    )
    for field in boolean_fields:
        if not isinstance(payload.get(field), bool):
            raise ValueError(f"Judge {field} field must be boolean")
    return JudgeResult(
        axes=decisions,
        topic_preserved=payload["topic_preserved"],
        fluent=payload["fluent"],
        unnecessary_rewriting=payload["unnecessary_rewriting"],
        catastrophic_safety_issue=payload["catastrophic_safety_issue"],
        notes=str(payload.get("notes") or ""),
        raw_output=raw_output,
    )


def judge_result_dict(result: JudgeResult) -> dict[str, Any]:
    return {
        "axes": {
            axis: {
                "violated": result.axes[axis].violated,
                "evidence_type": result.axes[axis].evidence_type,
                "evidence_span": result.axes[axis].evidence_span,
                "reason": result.axes[axis].reason,
            }
            for axis in AXES
        },
        "realized_axes": list(result.realized_axes),
        "topic_preserved": result.topic_preserved,
        "fluent": result.fluent,
        "unnecessary_rewriting": result.unnecessary_rewriting,
        "catastrophic_safety_issue": result.catastrophic_safety_issue,
        "notes": result.notes,
    }


def validate_judge_evidence(
    result: JudgeResult,
    *,
    candidate: str,
    clean_response: str,
    normalize,
) -> None:
    """Validate localized evidence while permitting genuine omissions/global failures."""
    candidate_norm = normalize(candidate)
    clean_norm = normalize(clean_response)
    for axis, decision in result.axes.items():
        if not decision.violated:
            continue
        evidence_norm = normalize(decision.evidence_span)
        if decision.evidence_type == "span" and evidence_norm not in candidate_norm:
            raise ValueError(f"candidate evidence is not verbatim for {axis}")
        if decision.evidence_type == "omission" and evidence_norm not in clean_norm:
            raise ValueError(f"omission evidence is not verbatim clean-reference text for {axis}")


def acceptance_decision(
    result: JudgeResult | None,
    required_axes: Sequence[str],
) -> tuple[bool, str | None]:
    if result is None:
        return False, "judge output could not be parsed"
    missing = [axis for axis in required_axes if not result.axes[axis].violated]
    reasons = []
    if missing:
        reasons.append("missing required violations: " + ", ".join(missing))
    if not result.topic_preserved:
        reasons.append("user topic/facts not preserved")
    if not result.fluent:
        reasons.append("candidate is not fluent")
    if result.catastrophic_safety_issue:
        reasons.append("catastrophic safety issue")
    return not reasons, "; ".join(reasons) if reasons else None
