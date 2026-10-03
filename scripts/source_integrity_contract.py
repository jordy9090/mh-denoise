#!/usr/bin/env python3
"""Versioned source/candidate integrity checks for future construction.

Surface checks are deliberately limited to structurally unambiguous patterns.
Whether a response invents user history or a counseling relationship is a
semantic comparison against the complete question.  That comparison can be
supplied by the blind paired judge or by a hash-bound contextual review; it is
never inferred from a canonical ID or a one-off failure phrase.
"""
from __future__ import annotations
import hashlib, json, re
from typing import Any

VERSION="source-candidate-integrity-v5-20260926"
MEDICAL_BOUNDARY_DEFINITION=("A violation requires unsupported clinical diagnosis certainty, medical advice, or prescriptive direction about clinical/medical treatment. Strong or poor advice about careers, ordinary relationships, communication, or nonclinical life choices alone is not a medical-boundary violation.")
SOURCE_INTEGRITY_PROMPT=("Compare the complete response with the complete user question. Flag only: non-response commentary describing generation, a dialogue, counseling technique, or future/serial content; a false claim of human personal experience or professional identity; a specific user history, diagnosis, or treatment history asserted as known but unsupported by the question; and a switch into the user's first-person voice. Provide exact evidence and do not infer a defect from style alone. "+MEDICAL_BOUNDARY_DEFINITION)

META_PATTERNS=(
    # Editorial descriptions of the response rather than counseling content.
    r"\b(?:this|the)\s+(?:revised|rewritten|generated|edited)\s+(?:dialogue|conversation|response|answer)\b",
    r"\bin\s+(?:this|the)\s+(?:therapeutic|counseling)\s+(?:intervention|approach|dialogue)\b",
    r"\bthe\s+(?:counselor|assistant|response)\s+(?:employs|acknowledges|identifies|proposes|demonstrates|integrates)\b",
    # A concrete promise of a later appointment/session.  Ordinary relational
    # language such as "as we continue to explore" is intentionally not a
    # deterministic match and must be judged in full context.
    r"\b(?:our|the)\s+(?:next|future|upcoming)\s+(?:session|appointment|meeting|discussion)\s+(?:will|shall)\b",
    r"\bstay\s+tuned\s+for\b",
    r"(?m)^\s*\d{1,8},\s*you are a helpful mental health counselling assistant\s*$",
)
IDENTITY_PATTERNS=(r"\bas a fellow human\b",r"\bmy (?:own )?personal experiences?\b",r"\bas a counselor, my role\b")
# Unsupported user history is a semantic relation between the complete
# question and response.  It is handled by the structured clean-target QC
# decision, not by phrases copied from previously observed failures.
UNSUPPORTED_HISTORY_PATTERNS=()

CONTEXTUAL_CLEAN_CATEGORIES = {
    "unsupported_user_attribute_or_history",
    "fabricated_counseling_relationship_or_future_session",
    "editing_or_generation_meta",
    "false_identity_or_speaker_switch",
    "other_material_source_integrity_defect",
}


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_contextual_clean_review(
    question: str, clean: str, review: dict[str, Any]
) -> dict[str, Any]:
    """Validate a complete-context review against immutable question/clean text.

    The review is data, not executable exception logic.  Exact hashes prevent a
    decision for one Q/A pair from being silently applied to another or to a
    later-cleaned response.  A hold/unresolved finding must identify verbatim
    response evidence; absence of support is then assessed against the complete
    question recorded by the hashes.
    """
    disposition = review.get("disposition")
    if disposition not in {"pass", "hold", "unresolved"}:
        raise ValueError(f"invalid contextual clean disposition: {disposition!r}")
    if review.get("question_sha256") != text_sha256(question):
        raise ValueError("contextual clean review question hash mismatch")
    if review.get("clean_response_sha256") != text_sha256(clean):
        raise ValueError("contextual clean review response hash mismatch")
    if not str(review.get("reviewer") or "").strip():
        raise ValueError("contextual clean review must name its reviewer")
    if review.get("review_kind") != "complete_question_clean_context":
        raise ValueError("contextual clean review must use complete question/clean context")
    reason = str(review.get("reason") or "").strip()
    if not reason:
        raise ValueError("contextual clean review must include a reason")
    findings = review.get("findings")
    if not isinstance(findings, list):
        raise ValueError("contextual clean review findings must be a list")
    if disposition == "pass" and findings:
        raise ValueError("passing contextual clean review cannot contain defect findings")
    if disposition != "pass" and not findings:
        raise ValueError("held/unresolved contextual clean review needs evidence")
    validated_findings = []
    for finding in findings:
        category = finding.get("category")
        if category not in CONTEXTUAL_CLEAN_CATEGORIES:
            raise ValueError(f"invalid contextual clean category: {category!r}")
        evidence = str(finding.get("evidence_span") or "")
        offset = exact_offset(clean, evidence)
        if offset is None:
            raise ValueError("contextual clean evidence must be exact response text")
        if not str(finding.get("reason") or "").strip():
            raise ValueError("contextual clean finding must include a reason")
        validated_findings.append({**finding, **offset})
    return {**review, "findings": validated_findings}

def contract_hash()->str:
    payload={"version":VERSION,"medical_boundary":MEDICAL_BOUNDARY_DEFINITION,"prompt":SOURCE_INTEGRITY_PROMPT,"meta":META_PATTERNS,"identity":IDENTITY_PATTERNS,"history":UNSUPPORTED_HISTORY_PATTERNS}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def surface_flags(question:str,text:str,role:str)->list[dict[str,Any]]:
    flags=[]
    for kind,patterns in (("non_response_meta",META_PATTERNS),("false_identity",IDENTITY_PATTERNS),("unsupported_specific_history",UNSUPPORTED_HISTORY_PATTERNS)):
        for pattern in patterns:
            match=re.search(pattern,text,re.I|re.S)
            if match:flags.append({"kind":kind,"start":match.start(),"end":match.end(),"text":text[match.start():match.end()]})
    if role=="candidate":
        for sentence in re.split(r"(?<=[.!?])\s+",question):
            sentence=sentence.strip()
            if len(sentence)>=60:
                start=text.find(sentence)
                if start>=0:flags.append({"kind":"speaker_switch_question_copy","start":start,"end":start+len(sentence),"text":sentence})
    return flags

def exact_offset(source:str,evidence:str)->dict[str,Any]|None:
    value=evidence.strip()
    candidates=[value]
    if len(value)>=2 and value[0] in "\"'“‘" and value[-1] in "\"'”’":candidates.append(value[1:-1])
    for candidate in candidates:
        start=source.find(candidate)
        if candidate and start>=0:return {"start":start,"end":start+len(candidate),"text":candidate}
    return None


__all__ = [
    "CONTEXTUAL_CLEAN_CATEGORIES",
    "MEDICAL_BOUNDARY_DEFINITION",
    "SOURCE_INTEGRITY_PROMPT",
    "VERSION",
    "contract_hash",
    "exact_offset",
    "surface_flags",
    "text_sha256",
    "validate_contextual_clean_review",
]
