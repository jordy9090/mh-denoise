#!/usr/bin/env python3
"""Versioned narrow source/candidate integrity checks for future construction."""
from __future__ import annotations
import hashlib, json, re
from typing import Any

VERSION="source-candidate-integrity-v4"
MEDICAL_BOUNDARY_DEFINITION=("A violation requires unsupported clinical diagnosis certainty, medical advice, or prescriptive direction about clinical/medical treatment. Strong or poor advice about careers, ordinary relationships, communication, or nonclinical life choices alone is not a medical-boundary violation.")
SOURCE_INTEGRITY_PROMPT=("Compare the complete response with the complete user question. Flag only: non-response commentary describing generation, a dialogue, counseling technique, or future/serial content; a false claim of human personal experience or professional identity; a specific user history, diagnosis, or treatment history asserted as known but unsupported by the question; and a switch into the user's first-person voice. Provide exact evidence and do not infer a defect from style alone. "+MEDICAL_BOUNDARY_DEFINITION)

META_PATTERNS=(r"\bthis revised dialogue\b",r"\bin this therapeutic (?:intervention|approach)\b",r"\bthe counselor (?:employs|acknowledges|identifies|proposes)\b",r"\bas we continue to explore\b",r"\bstay tuned for\b",r"(?m)^\s*\d{1,8},\s*you are a helpful mental health counselling assistant\s*$")
IDENTITY_PATTERNS=(r"\bas a fellow human\b",r"\bmy (?:own )?personal experiences?\b",r"\bas a counselor, my role\b")
# Unsupported user history is a semantic relation between the complete
# question and response.  It is handled by the structured clean-target QC
# decision, not by phrases copied from previously observed failures.
UNSUPPORTED_HISTORY_PATTERNS=()

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
