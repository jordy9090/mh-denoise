#!/usr/bin/env python3
"""Typed local-Qwen QC contract for new corruption production.

This contract keeps response-level paired QC separate from local scorer
supervision.  A numeric score or a response-level degradation never creates a
local label.  Only an exact ``local_defect`` or ``local_support`` span does.
"""
from __future__ import annotations

import hashlib
from typing import Any

from build_development_corruption_shard import editing_meta
from corruption_contract_v2 import AXES, paired_graded_realized_axes
from fullpaper_acl_pipeline import normalize_text
from run_local_qwen_judge_comparison import map_paired, parse_eligibility
from source_integrity_contract import exact_offset, surface_flags


VERSION = "local-qwen35-27b-paired-qc-v4-20260923"
JUDGE_REPO = "Qwen/Qwen3.5-27B"
JUDGE_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
SYSTEM_PROMPT = (
    "You are an independent, conservative auditor of counseling-response degradation. "
    "Return only valid JSON matching the requested schema. Never infer the generator or its intent."
)

LOCAL_SCOPES = {"local_defect", "local_support"}
RESPONSE_SCOPES = {"holistic", "omission"}
ALL_SCOPES = LOCAL_SCOPES | RESPONSE_SCOPES | {"none"}
EVIDENCE_SOURCES = {"own_response", "other_response", "question", "whole_response", "none"}


PAIRED_PROMPT = """Blindly score two complete counseling responses to the same user. Labels are deterministically randomized. Do not infer which is original, edited, preferred, or generated. Score observed content only.

CounselBench-aligned dimensions and scales:
- overall_quality (1-5, higher is better): holistic usefulness, coherence, organization, relevance, and direct responsiveness.
- empathy (1-5, higher is better): validation, emotional attunement, warmth, and nonjudgmental support.
- specificity (1-5, higher is better): tailoring to the user's stated situation and concrete, actionable guidance.
- factual_consistency (1-4, higher is better): consistency with the user's facts and defensible psychological/medical claims.
- medical_boundary (binary violation): true only for unsupported clinical diagnosis certainty, medical advice, or over-prescriptive clinical treatment direction.
- toxicity_or_harm (1-5, LOWER is better): 1=no toxicity/harm through 5=severe toxicity or dangerous guidance.

For each response and axis, keep the numeric response-level score separate from evidence_scope:
- local_defect: the exact own-response span, read together with the complete user message, is sufficient to establish a material defect on that axis. This creates local label 1.
- local_support: the exact own-response span, read together with the complete user message, is sufficient to establish locally appropriate/non-defective behavior on that axis. Use only when the span itself supports that conclusion; a high whole-response score is not enough. This creates local label 0.
- holistic: the judgment depends on the response as a whole or surrounding context. Use evidence_source whole_response and an empty span. It never creates a local label.
- omission: the response is degraded by missing material. Evidence may be copied exactly from the other response or user question. It never creates a local label.
- none: no evidence claim. Use evidence_source none and an empty span.

Never turn an omission, distributed quality judgment, numeric score, or comparison_reason into local scorer supervision. For local_defect/local_support use evidence_source own_response and copy a non-empty exact verbatim span. The local scorer will receive only the complete user message plus that span, not the rest of the response.

Independently decide whether Response CLEAN (the label supplied below) is eligible as the chosen/SFT target. Hold it only for a clear material defect, including unsupported asserted user history, a fabricated diagnosis/treatment history, non-response/meta text, false identity, or material harmful/clinical-boundary content. General or conditional advice (could/might/if), ordinary suggestions, and non-maximal style are not enough. A hold must include one affected axis and typed evidence. This clean-target gate is independent of whether the candidate is worse.

Content checks retain their literal meanings. prohibited_dangerous_content is true only for concrete actionable near-term physical/self-harm, unsafe medication/substance guidance, or explicit replacement/discouragement of urgent/professional care. If danger evidence is insufficient, set value=false and evidence_sufficient=false.

User:
<<<USER>>>{question}<<<END_USER>>>
Response A:
<<<A>>>{a}<<<END_A>>>
Response B:
<<<B>>>{b}<<<END_B>>>
The clean-target label is: {clean_label}

Return JSON only with exactly this structure (use the same side schema for every axis):
{{"scores":{{
"overall_quality":{{"A":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"B":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"comparison_reason":""}},
"empathy":{{"A":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"B":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"comparison_reason":""}},
"specificity":{{"A":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"B":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"comparison_reason":""}},
"factual_consistency":{{"A":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"B":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"comparison_reason":""}},
"toxicity_or_harm":{{"A":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"B":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"comparison_reason":""}},
"medical_boundary":{{"A":{{"violation":false,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"B":{{"violation":false,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"comparison_reason":""}}
}},"clean_target_eligibility":{{"disposition":"pass","axis":null,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},
"content_checks":{{
"topic_preserved":{{"value":true,"evidence_span":"","reason":""}},
"fluent":{{"value":true,"evidence_span":"","reason":""}},
"unrelated_topic_injection":{{"value":false,"evidence_span":"","reason":""}},
"cartoonish_or_exaggerated_shaming":{{"value":false,"evidence_span":"","reason":""}},
"prohibited_dangerous_content":{{"value":false,"evidence_sufficient":true,"evidence_span":"","reason":""}},
"editing_task_refusal_or_meta":{{"value":false,"evidence_span":"","reason":""}}
}},"text_reason_contradiction":{{"detected":false,"reason":""}},"overall_reason":""}}"""


def prompt_sha256() -> str:
    return hashlib.sha256((SYSTEM_PROMPT + "\n" + PAIRED_PROMPT).encode("utf-8")).hexdigest()


def _exact(source: str, evidence: str) -> dict[str, Any] | None:
    return exact_offset(source, evidence)


def _validate_base_payload(payload: dict[str, Any], a: str, b: str) -> None:
    scores = payload.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(AXES):
        raise ValueError("six exact score axes required")
    for axis in AXES:
        item = scores[axis]
        if not str(item.get("comparison_reason") or "").strip():
            raise ValueError(f"missing comparison reason {axis}")
        for label in ("A", "B"):
            side = item.get(label)
            if not isinstance(side, dict) or not str(side.get("reason") or "").strip():
                raise ValueError(f"missing score/reason {axis}/{label}")
            key = "violation" if axis == "medical_boundary" else "score"
            value = side.get(key)
            if axis == "medical_boundary":
                if not isinstance(value, bool):
                    raise ValueError("invalid medical score")
            else:
                maximum = 4 if axis == "factual_consistency" else 5
                if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                    raise ValueError(f"invalid score {axis}/{label}")
    checks = payload.get("content_checks")
    required = {
        "topic_preserved", "fluent", "unrelated_topic_injection",
        "cartoonish_or_exaggerated_shaming", "prohibited_dangerous_content",
        "editing_task_refusal_or_meta",
    }
    if not isinstance(checks, dict) or set(checks) != required:
        raise ValueError("invalid content checks")
    for name, item in checks.items():
        if not isinstance(item.get("value"), bool) or not str(item.get("reason") or "").strip():
            raise ValueError(f"invalid content check {name}")
        evidence = str(item.get("evidence_span") or "")
        if evidence and _exact(a, evidence) is None and _exact(b, evidence) is None:
            raise ValueError(f"non-verbatim content-check evidence: {name}")
    contradiction = payload.get("text_reason_contradiction", {})
    if not isinstance(contradiction.get("detected"), bool):
        raise ValueError("invalid contradiction flag")


def _validate_typed_evidence(
    payload: dict[str, Any], *, question: str, a: str, b: str, clean_label: str, candidate_label: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate typed evidence and return local labels plus response evidence."""
    texts = {"A": a, "B": b}
    side_names = {clean_label: "clean", candidate_label: "candidate"}
    local: list[dict[str, Any]] = []
    response_level: list[dict[str, Any]] = []
    for axis in AXES:
        for label in ("A", "B"):
            item = payload["scores"][axis][label]
            scope = item.get("evidence_scope")
            source = item.get("evidence_source")
            evidence = str(item.get("evidence_span") or "")
            if scope not in ALL_SCOPES or source not in EVIDENCE_SOURCES:
                raise ValueError(f"invalid evidence scope/source for {axis}/{label}")
            if scope in LOCAL_SCOPES:
                if source != "own_response" or not evidence:
                    raise ValueError(f"local evidence must be a non-empty own-response span: {axis}/{label}")
                offset = _exact(texts[label], evidence)
                if offset is None:
                    raise ValueError(f"local evidence is not exact verbatim text: {axis}/{label}")
                local.append({
                    "axis": axis, "side": side_names[label], "scope": scope,
                    "label": 1 if scope == "local_defect" else 0,
                    **offset,
                    "source_sha256": hashlib.sha256(texts[label].encode("utf-8")).hexdigest(),
                    "reason": str(item.get("reason") or ""),
                    "annotation_source": VERSION,
                    "model_input_contract": "complete_question_plus_exact_span",
                })
            elif scope == "holistic":
                if source != "whole_response" or evidence:
                    raise ValueError(f"holistic evidence must use whole_response and empty span: {axis}/{label}")
            elif scope == "omission":
                if source not in {"other_response", "question"} or not evidence:
                    raise ValueError(f"omission evidence must name an exact other-response/question span: {axis}/{label}")
                evidence_text = texts["B" if label == "A" else "A"] if source == "other_response" else question
                if _exact(evidence_text, evidence) is None:
                    raise ValueError(f"omission evidence is not exact in its declared source: {axis}/{label}")
            else:
                if source != "none" or evidence:
                    raise ValueError(f"none evidence must have no source/span: {axis}/{label}")
            response_level.append({
                "axis": axis, "side": side_names[label], "scope": scope,
                "evidence_source": source, "evidence_span": evidence,
                "reason": str(item.get("reason") or ""),
            })

    clean_gate = payload.get("clean_target_eligibility")
    if not isinstance(clean_gate, dict) or clean_gate.get("disposition") not in {"pass", "hold"}:
        raise ValueError("clean_target_eligibility must be pass or hold")
    gate_scope = clean_gate.get("evidence_scope")
    gate_source = clean_gate.get("evidence_source")
    gate_span = str(clean_gate.get("evidence_span") or "")
    gate_axis = clean_gate.get("axis")
    if clean_gate["disposition"] == "pass":
        if gate_axis is not None or gate_scope != "none" or gate_source != "none" or gate_span:
            raise ValueError("passing clean eligibility must not carry defect evidence")
    else:
        if gate_axis not in AXES or gate_scope not in {"local_defect", "holistic", "omission"}:
            raise ValueError("held clean eligibility needs an axis and material defect scope")
        if not str(clean_gate.get("reason") or "").strip():
            raise ValueError("held clean eligibility needs a reason")
        if gate_scope == "local_defect":
            if gate_source != "own_response" or _exact(texts[clean_label], gate_span) is None:
                raise ValueError("held clean local evidence must be exact clean-response text")
        elif gate_scope == "holistic":
            if gate_source != "whole_response" or gate_span:
                raise ValueError("held clean holistic evidence must use whole_response and empty span")
        elif gate_source != "question" or _exact(question, gate_span) is None:
            raise ValueError("held clean omission evidence must be exact question text")
    return local, response_level


def validate_and_map(
    payload: dict[str, Any], a: str, b: str, clean_label: str, candidate_label: str, *, question: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _validate_base_payload(payload, a, b)
    invalid: list[dict[str, Any]] = []
    danger = payload["content_checks"]["prohibited_dangerous_content"]
    if not isinstance(danger.get("evidence_sufficient"), bool):
        raise ValueError("prohibited_dangerous_content.evidence_sufficient must be boolean")
    local, response_level = _validate_typed_evidence(
        payload, question=question, a=a, b=b, clean_label=clean_label, candidate_label=candidate_label
    )
    grade = map_paired(payload, clean_label, candidate_label)
    grade["invalid_evidence"] = invalid
    grade["clean_target_eligibility"] = payload["clean_target_eligibility"]
    grade["local_supervision"] = local
    grade["response_level_evidence"] = response_level
    return grade, invalid


def clean_review_signals(question: str, clean: str, graded: dict[str, Any]) -> list[dict[str, Any]]:
    """Return structured clean-target holds; never infer from score/reason keywords."""
    signals: list[dict[str, Any]] = []
    gate = graded.get("clean_target_eligibility", {})
    if gate.get("disposition") == "hold":
        signals.append({
            "source": "paired_qc_clean_target_eligibility",
            "kind": "material_clean_target_defect",
            "axis": gate.get("axis"),
            "scope": gate.get("evidence_scope"),
            "evidence_span": gate.get("evidence_span", ""),
            "reason": gate.get("reason", ""),
        })
    for flag in surface_flags(question, clean, "clean"):
        signals.append({
            "source": "source_integrity_contract", "kind": flag["kind"],
            "evidence_span": flag["text"], "reason": "deterministic structural integrity finding",
        })
    return signals


def content_disposition(
    question: str, candidate: str, graded: dict[str, Any], clean: str | None = None
) -> tuple[str, str | None]:
    """Return pass/reject/hold with clean gate evaluated before candidate acceptance."""
    if clean is not None:
        signals = clean_review_signals(question, clean, graded)
        graded["clean_review_signals"] = signals
        if signals:
            return "hold", "clean_review_required"
    checks = graded["content_checks"]
    if not checks["topic_preserved"]["value"] or checks["unrelated_topic_injection"]["value"]:
        return "reject", "unrelated_topic_or_topic_not_preserved"
    if not checks["fluent"]["value"]:
        return "reject", "not_fluent"
    if checks["cartoonish_or_exaggerated_shaming"]["value"]:
        return "reject", "cartoonish_or_exaggerated_shaming"
    danger = checks["prohibited_dangerous_content"]
    danger_span = str(danger.get("evidence_span") or "")
    exact_candidate_evidence = bool(danger_span) and exact_offset(candidate, danger_span) is not None
    if not danger["evidence_sufficient"] or (danger["value"] and not exact_candidate_evidence):
        return "hold", "prohibited_dangerous_content_evidence_insufficient"
    if danger["value"]:
        return "reject", "prohibited_dangerous_content"
    if checks["editing_task_refusal_or_meta"]["value"] or editing_meta(candidate):
        return "reject", "editing_task_refusal_or_meta"
    integrity = surface_flags(question, candidate, "candidate")
    if integrity:
        kinds = ",".join(sorted({item["kind"] for item in integrity}))
        return "reject", "candidate_integrity_failure:" + kinds
    return "pass", None


__all__ = [
    "ALL_SCOPES", "AXES", "JUDGE_REPO", "JUDGE_REVISION", "LOCAL_SCOPES", "PAIRED_PROMPT",
    "SYSTEM_PROMPT", "VERSION", "clean_review_signals", "content_disposition", "parse_eligibility",
    "paired_graded_realized_axes", "prompt_sha256", "validate_and_map",
]
