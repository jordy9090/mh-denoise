#!/usr/bin/env python3
"""Typed local-Qwen QC contract for new corruption production.

This contract keeps response-level paired QC separate from local scorer
supervision.  A numeric score or a response-level degradation never creates a
local label.  Only an exact ``local_defect`` or ``local_support`` span does.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any

from build_development_corruption_shard import editing_meta
from corruption_contract_v2 import AXES, paired_graded_realized_axes
from fullpaper_acl_pipeline import normalize_text
from run_local_qwen_judge_comparison import map_paired, parse_eligibility
from source_integrity_contract import (
    exact_offset,
    surface_flags,
    validate_contextual_clean_review,
)


VERSION = "local-qwen35-27b-paired-qc-v6-20260926"
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
- omission: the response is degraded by missing material. If one exact omitted item can be cited, copy it from the other response or user question. If the omission is distributed or has no single exact anchor, use evidence_source whole_response and an empty span. It never creates a local label.
- none: no evidence claim. Use evidence_source none and an empty span.

Never turn an omission, distributed quality judgment, numeric score, or comparison_reason into local scorer supervision. For local_defect/local_support use evidence_source own_response and copy a non-empty exact verbatim span. The local scorer will receive only the complete user message plus that span, not the rest of the response.

Independently decide whether each response is eligible as a counseling target. Use the identical eligibility schema for A and B; their roles are intentionally undisclosed. Compare each complete response with the complete user message. Hold a response only for a clear material defect, including a definite user/family attribute or history unsupported by the user message, a fabricated diagnosis/treatment history, a false promise or assertion of an established counselor relationship or later session, editorial/generation meta-text, false identity, or material harmful/clinical-boundary content. Distinguish those assertions from ordinary patient-facing collaboration (for example, saying that options can be explored now), conditional advice (could/might/if), and non-maximal style. A hold must include one affected axis and typed evidence.

Apply content checks independently to A and B. Evidence spans must come from the response being checked. prohibited_dangerous_content is true only for concrete actionable near-term physical/self-harm, unsafe medication/substance guidance, or explicit replacement/discouragement of urgent/professional care. If danger evidence is insufficient, set value=false and evidence_sufficient=false.

User:
<<<USER>>>{question}<<<END_USER>>>
Response A:
<<<A>>>{a}<<<END_A>>>
Response B:
<<<B>>>{b}<<<END_B>>>

Return JSON only with exactly this structure (use the same side schema for every axis):
{{"scores":{{
"overall_quality":{{"A":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"B":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"comparison_reason":""}},
"empathy":{{"A":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"B":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"comparison_reason":""}},
"specificity":{{"A":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"B":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"comparison_reason":""}},
"factual_consistency":{{"A":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"B":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"comparison_reason":""}},
"toxicity_or_harm":{{"A":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"B":{{"score":1,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"comparison_reason":""}},
"medical_boundary":{{"A":{{"violation":false,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"B":{{"violation":false,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"comparison_reason":""}}
}},"response_eligibility":{{"A":{{"disposition":"pass","axis":null,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}},"B":{{"disposition":"pass","axis":null,"evidence_scope":"none","evidence_source":"none","evidence_span":"","reason":""}}}},
"content_checks":{{"A":{{
"topic_preserved":{{"value":true,"evidence_span":"","reason":""}},
"fluent":{{"value":true,"evidence_span":"","reason":""}},
"unrelated_topic_injection":{{"value":false,"evidence_span":"","reason":""}},
"cartoonish_or_exaggerated_shaming":{{"value":false,"evidence_span":"","reason":""}},
"prohibited_dangerous_content":{{"value":false,"evidence_sufficient":true,"evidence_span":"","reason":""}},
"editing_task_refusal_or_meta":{{"value":false,"evidence_span":"","reason":""}}
}},"B":{{
"topic_preserved":{{"value":true,"evidence_span":"","reason":""}},
"fluent":{{"value":true,"evidence_span":"","reason":""}},
"unrelated_topic_injection":{{"value":false,"evidence_span":"","reason":""}},
"cartoonish_or_exaggerated_shaming":{{"value":false,"evidence_span":"","reason":""}},
"prohibited_dangerous_content":{{"value":false,"evidence_sufficient":true,"evidence_span":"","reason":""}},
"editing_task_refusal_or_meta":{{"value":false,"evidence_span":"","reason":""}}
}}}},"text_reason_contradiction":{{"detected":false,"reason":""}},"overall_reason":""}}"""


VALIDATION_RETRY_INSTRUCTION = """

Your previous JSON failed validation.
Validation error: {validation_error}

Return the complete JSON object again with that error corrected. For
local_defect/local_support, use one contiguous verbatim own-response substring:
do not use ellipses, paraphrase whitespace, or join separate passages. If no
single local span supports the judgment, use holistic/whole_response/empty or
omission/whole_response/empty as appropriate; those scopes create no local
label. Keep every required key and return JSON only.

Previous output for correction:
<<<PREVIOUS_JSON>>>{previous_output}<<<END_PREVIOUS_JSON>>>
"""


def prompt_sha256() -> str:
    return hashlib.sha256(
        (SYSTEM_PROMPT + "\n" + PAIRED_PROMPT + "\n" + VALIDATION_RETRY_INSTRUCTION).encode("utf-8")
    ).hexdigest()


def paired_prompt(
    *, question: str, a: str, b: str,
    validation_error: str | None = None,
    previous_output: str | None = None,
) -> str:
    prompt = PAIRED_PROMPT.format(question=question, a=a, b=b)
    if validation_error is not None:
        prompt += VALIDATION_RETRY_INSTRUCTION.format(
            validation_error=validation_error,
            previous_output=previous_output or "",
        )
    return prompt


def _exact(source: str, evidence: str) -> dict[str, Any] | None:
    return exact_offset(source, evidence)


def _invalid_evidence_record(
    *, axis: str, response_label: str, scope: str, span: str, reason: str
) -> dict[str, Any]:
    return {
        "axis": axis,
        "response_label": response_label,
        "reported_scope": scope,
        "invalid_span": span,
        "reason": reason,
        "resolution": "non_verbatim_removed_no_local_label",
    }


def _validate_base_payload(
    payload: dict[str, Any], a: str, b: str, invalid: list[dict[str, Any]],
    *, quarantine_invalid_evidence: bool,
) -> None:
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
    texts = {"A": a, "B": b}
    if not isinstance(checks, dict) or set(checks) != {"A", "B"}:
        raise ValueError("content checks must be independently reported for A and B")
    for label, side_checks in checks.items():
        if not isinstance(side_checks, dict) or set(side_checks) != required:
            raise ValueError(f"invalid content checks for {label}")
        for name, item in side_checks.items():
            if not isinstance(item.get("value"), bool) or not str(item.get("reason") or "").strip():
                raise ValueError(f"invalid content check {label}/{name}")
            evidence = str(item.get("evidence_span") or "")
            if evidence and _exact(texts[label], evidence) is None:
                reason = f"non-verbatim own-response content-check evidence: {label}/{name}"
                invalid.append(_invalid_evidence_record(
                    axis=f"content_check:{name}", response_label=label,
                    scope="content_check", span=evidence, reason=reason,
                ))
                if not quarantine_invalid_evidence:
                    raise ValueError(reason)
                item["evidence_span"] = ""
                item["evidence_validation"] = "non_verbatim_removed"
            if name == "prohibited_dangerous_content" and not isinstance(item.get("evidence_sufficient"), bool):
                raise ValueError(f"invalid danger evidence sufficiency: {label}")
    contradiction = payload.get("text_reason_contradiction", {})
    if not isinstance(contradiction.get("detected"), bool):
        raise ValueError("invalid contradiction flag")


def _validate_typed_evidence(
    payload: dict[str, Any], *, question: str, a: str, b: str, clean_label: str,
    candidate_label: str, invalid: list[dict[str, Any]],
    quarantine_invalid_evidence: bool,
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
                    reason = f"local evidence is not exact verbatim text: {axis}/{label}"
                    invalid.append(_invalid_evidence_record(
                        axis=axis, response_label=label, scope=scope,
                        span=evidence, reason=reason,
                    ))
                    if not quarantine_invalid_evidence:
                        raise ValueError(reason)
                else:
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
                if source == "whole_response" and not evidence:
                    pass
                elif source in {"other_response", "question"} and evidence:
                    evidence_text = texts["B" if label == "A" else "A"] if source == "other_response" else question
                    if _exact(evidence_text, evidence) is None:
                        reason = f"omission evidence is not exact in its declared source: {axis}/{label}"
                        invalid.append(_invalid_evidence_record(
                            axis=axis, response_label=label, scope=scope,
                            span=evidence, reason=reason,
                        ))
                        if not quarantine_invalid_evidence:
                            raise ValueError(reason)
                else:
                    raise ValueError(
                        f"omission evidence must use whole_response/empty or an exact "
                        f"other-response/question span: {axis}/{label}"
                    )
            else:
                if source != "none" or evidence:
                    raise ValueError(f"none evidence must have no source/span: {axis}/{label}")
            response_level.append({
                "axis": axis, "side": side_names[label], "scope": scope,
                "evidence_source": source, "evidence_span": evidence,
                "reason": str(item.get("reason") or ""),
            })

    eligibility = payload.get("response_eligibility")
    if not isinstance(eligibility, dict) or set(eligibility) != {"A", "B"}:
        raise ValueError("response_eligibility must independently report A and B")
    for label, gate in eligibility.items():
        if not isinstance(gate, dict) or gate.get("disposition") not in {"pass", "hold"}:
            raise ValueError(f"response eligibility must be pass or hold: {label}")
        gate_scope = gate.get("evidence_scope")
        gate_source = gate.get("evidence_source")
        gate_span = str(gate.get("evidence_span") or "")
        gate_axis = gate.get("axis")
        if gate["disposition"] == "pass":
            if gate_axis is not None or gate_scope != "none" or gate_source != "none" or gate_span:
                raise ValueError(f"passing eligibility must not carry defect evidence: {label}")
        else:
            if gate_axis not in AXES or gate_scope not in {"local_defect", "holistic", "omission"}:
                raise ValueError(f"held eligibility needs an axis and material defect scope: {label}")
            if not str(gate.get("reason") or "").strip():
                raise ValueError(f"held eligibility needs a reason: {label}")
            if gate_scope == "local_defect":
                if gate_source != "own_response" or not gate_span:
                    raise ValueError(f"held local evidence must be exact own-response text: {label}")
                if _exact(texts[label], gate_span) is None:
                    reason = f"held local evidence is not exact own-response text: {label}"
                    invalid.append(_invalid_evidence_record(
                        axis=str(gate_axis), response_label=label,
                        scope="eligibility_local_defect", span=gate_span,
                        reason=reason,
                    ))
                    if not quarantine_invalid_evidence:
                        raise ValueError(reason)
                    gate["evidence_validation"] = "non_verbatim_removed"
                    gate["evidence_span"] = ""
            elif gate_scope == "holistic":
                if gate_source != "whole_response" or gate_span:
                    raise ValueError(f"held holistic evidence must use whole_response and empty span: {label}")
            elif gate_source == "whole_response" and not gate_span:
                pass
            elif gate_source != "question" or _exact(question, gate_span) is None:
                raise ValueError(
                    f"held omission evidence must use whole_response/empty or exact question text: {label}"
                )
    return local, response_level


def validate_and_map(
    payload: dict[str, Any], a: str, b: str, clean_label: str, candidate_label: str,
    *, question: str, quarantine_invalid_evidence: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = copy.deepcopy(payload)
    invalid: list[dict[str, Any]] = []
    _validate_base_payload(
        payload, a, b, invalid,
        quarantine_invalid_evidence=quarantine_invalid_evidence,
    )
    local, response_level = _validate_typed_evidence(
        payload, question=question, a=a, b=b, clean_label=clean_label,
        candidate_label=candidate_label, invalid=invalid,
        quarantine_invalid_evidence=quarantine_invalid_evidence,
    )
    mapped_payload = {**payload, "content_checks": payload["content_checks"][candidate_label]}
    grade = map_paired(mapped_payload, clean_label, candidate_label)
    grade["invalid_evidence"] = invalid
    grade["clean_target_eligibility"] = payload["response_eligibility"][clean_label]
    grade["candidate_target_eligibility"] = payload["response_eligibility"][candidate_label]
    grade["response_eligibility"] = {
        "clean": payload["response_eligibility"][clean_label],
        "candidate": payload["response_eligibility"][candidate_label],
    }
    grade["response_content_checks"] = {
        "clean": payload["content_checks"][clean_label],
        "candidate": payload["content_checks"][candidate_label],
    }
    grade["local_supervision"] = local
    grade["response_level_evidence"] = response_level
    return grade, invalid


def clean_review_signals(
    question: str,
    clean: str,
    graded: dict[str, Any],
    contextual_review: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return structured clean-target holds; never infer from score/reason keywords.

    A hash-bound complete-context review, when supplied, is authoritative over
    automated signals.  The automated result is still preserved on ``graded``
    for audit, so a reviewer pass does not erase a judge disagreement.
    """
    automated: list[dict[str, Any]] = []
    gate = graded.get("clean_target_eligibility", {})
    if gate.get("disposition") == "hold":
        automated.append({
            "source": "paired_qc_clean_target_eligibility",
            "kind": "material_clean_target_defect",
            "axis": gate.get("axis"),
            "scope": gate.get("evidence_scope"),
            "evidence_span": gate.get("evidence_span", ""),
            "reason": gate.get("reason", ""),
            "recommended_disposition": "exclude_original_answer",
        })
    for annotation in graded.get("local_supervision", []):
        if (
            annotation.get("side") == "clean"
            and annotation.get("scope") == "local_defect"
            and annotation.get("label") == 1
        ):
            automated.append({
                "source": "paired_qc_clean_local_supervision",
                "kind": "clean_local_defect_eligibility_conflict",
                "axis": annotation.get("axis"),
                "scope": annotation.get("scope"),
                "evidence_span": annotation.get("text", ""),
                "reason": annotation.get("reason", ""),
                "clean_target_eligibility": gate,
                # A local axis defect and a whole-response target pass are an
                # internal QC disagreement.  Preserve the exact local label,
                # but do not silently promote it to a material target defect.
                "recommended_disposition": (
                    "exclude_original_answer"
                    if gate.get("disposition") == "hold"
                    else "review_required"
                ),
            })
    for flag in surface_flags(question, clean, "clean"):
        automated.append({
            "source": "source_integrity_contract", "kind": flag["kind"],
            "evidence_span": flag["text"], "reason": "deterministic structural integrity finding",
            "recommended_disposition": "exclude_original_answer",
        })
    graded["automated_clean_review_signals"] = automated
    if contextual_review is None:
        return automated
    review = validate_contextual_clean_review(question, clean, contextual_review)
    graded["contextual_clean_review"] = review
    if review["disposition"] == "pass":
        return []
    return [
        {
            "source": "complete_context_clean_review",
            "kind": finding["category"],
            "evidence_span": finding["text"],
            "start": finding["start"],
            "end": finding["end"],
            "reason": finding["reason"],
            "review_disposition": review["disposition"],
            "reviewer": review["reviewer"],
        }
        for finding in review["findings"]
    ]


def content_disposition(
    question: str,
    candidate: str,
    graded: dict[str, Any],
    clean: str | None = None,
    contextual_clean_review: dict[str, Any] | None = None,
) -> tuple[str, str | None]:
    """Return pass/reject/hold with clean gate evaluated before candidate acceptance."""
    if clean is not None:
        signals = clean_review_signals(
            question, clean, graded, contextual_review=contextual_clean_review
        )
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
    "paired_graded_realized_axes", "paired_prompt", "prompt_sha256", "validate_and_map",
]
