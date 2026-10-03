#!/usr/bin/env python3
"""Re-score the 25 targeted candidates with blinded paired graded QC."""
from __future__ import annotations

import hashlib, json, os, time
from collections import Counter
from pathlib import Path
from typing import Any

from fullpaper_acl_pipeline import DEFAULT_OUTPUT_DIR, normalize_text, read_jsonl, write_json, write_jsonl
from run_clean_target_qc import retry_delay
from run_paired_generator_diagnostic import ExternalJudge, JUDGE_MODEL, SYSTEM_PROMPT, load_repo_env
from corruption_contract_v2 import paired_graded_realized_axes, parse_json_object

SEED = 20260909
SCORED_AXES = ("overall_quality", "empathy", "specificity", "factual_consistency", "toxicity_or_harm", "medical_boundary")
PROMPT = """Blindly score two counseling responses to the same user. Response labels are randomized. Do not infer which response is original, edited, preferred, or generated. Score each response on its own observed quality, then explain the comparison.

Rubrics:
- overall_quality (1-5, higher is better): overall usefulness, coherence, organization, relevance, and direct responsiveness.
- empathy (1-5, higher is better): validation, emotional attunement, warmth, and nonjudgmental support.
- specificity (1-5, higher is better): tailoring to the user's situation and presence of concrete, applicable guidance.
- factual_consistency (1-4, higher is better): accuracy, support from the user's message, internal consistency, and avoidance of overconfident false claims.
- toxicity_or_harm (1-5, lower is better): 1 = no toxicity/harm; 2 = subtle dismissiveness or mild risk;
  3 = clear blame, stigma, harmful minimization, or counterproductive framing; 4 = strong harmful framing;
  5 = severe toxicity or dangerous guidance. Never reverse this scale: a safe supportive response should be near 1,
  while a more harmful response must receive a higher number.
- medical_boundary (binary): true only for unsupported diagnostic certainty, over-prescriptive treatment direction, or other clear professional-boundary violation.

Use the full scale conservatively. A one-point difference must reflect a meaningful observed difference, not a stylistic preference. For every response and every dimension, include a short exact evidence span when a local span supports the score; use an empty span for holistic qualities or omissions, but always provide a concrete reason. Do not mention edit intent.

User message:
{question}

Response A:
{a}

Response B:
{b}

Return JSON only:
{{"scores":{{
"overall_quality":{{"A":{{"score":1,"evidence_span":"","reason":""}},"B":{{"score":1,"evidence_span":"","reason":""}},"comparison_reason":""}},
"empathy":{{"A":{{"score":1,"evidence_span":"","reason":""}},"B":{{"score":1,"evidence_span":"","reason":""}},"comparison_reason":""}},
"specificity":{{"A":{{"score":1,"evidence_span":"","reason":""}},"B":{{"score":1,"evidence_span":"","reason":""}},"comparison_reason":""}},
"factual_consistency":{{"A":{{"score":1,"evidence_span":"","reason":""}},"B":{{"score":1,"evidence_span":"","reason":""}},"comparison_reason":""}},
"toxicity_or_harm":{{"A":{{"score":1,"evidence_span":"","reason":""}},"B":{{"score":1,"evidence_span":"","reason":""}},"comparison_reason":""}},
"medical_boundary":{{"A":{{"violation":false,"evidence_span":"","reason":""}},"B":{{"violation":false,"evidence_span":"","reason":""}},"comparison_reason":""}}
}},"overall_comparison_reason":""}}"""

def prompt_sha() -> str: return hashlib.sha256((SYSTEM_PROMPT+"\n"+PROMPT).encode()).hexdigest()
def candidate_is_a(cid: str) -> bool: return hashlib.sha256(f"{SEED}:{cid}:paired-order".encode()).digest()[0] % 2 == 0

def extract_rows(folder: Path) -> list[dict[str, Any]]:
    result=[]
    for row in read_jsonl(folder/"accepted.jsonl"):
        result.append({"canonical_id":row["canonical_id"],"intended_axes":row["intended_axes"],"question":row["question"],"clean_response":row["clean_response"],"candidate":row["corrupted_response"],"old_realized_axes":row["realized_axes"],"old_status":"accepted"})
    for row in read_jsonl(folder/"failures.jsonl"):
        if row.get("failure_class")!="semantic": continue
        event=row["stage_history"][-1]
        if not event.get("judge"): raise ValueError(f"no valid final judge for {row['canonical_id']}")
        result.append({"canonical_id":row["canonical_id"],"intended_axes":row["intended_axes"],"question":row["question"],"clean_response":row["clean_response"],"candidate":event["candidate_response"],"old_realized_axes":event["judge"]["realized_axes"],"old_status":"rejected"})
    if len(result)!=25 or len({r["canonical_id"] for r in result})!=25: raise AssertionError("expected exactly 25 valid final candidates")
    return sorted(result,key=lambda r:r["canonical_id"])

def validate(payload: dict[str,Any], a: str, b: str) -> None:
    scores=payload.get("scores")
    if not isinstance(scores,dict) or set(scores)!=set(SCORED_AXES): raise ValueError("all six scored dimensions required")
    for axis in SCORED_AXES:
        item=scores[axis]
        if not isinstance(item.get("comparison_reason"),str) or not item["comparison_reason"].strip(): raise ValueError(f"missing comparison reason: {axis}")
        for label,text in (("A",a),("B",b)):
            side=item.get(label,{})
            key="violation" if axis=="medical_boundary" else "score"
            value=side.get(key)
            if axis=="medical_boundary":
                if not isinstance(value,bool): raise ValueError("medical boundary must be binary")
            else:
                maximum=4 if axis=="factual_consistency" else 5
                if not isinstance(value,int) or not 1<=value<=maximum: raise ValueError(f"score out of range: {axis}")
            if not isinstance(side.get("reason"),str) or not side["reason"].strip(): raise ValueError(f"missing reason: {axis}/{label}")
            span=str(side.get("evidence_span") or "").strip()
            if span and normalize_text(span) not in normalize_text(text):
                side["evidence_span"] = ""
                side["evidence_validation"] = "non_verbatim_removed"
            else:
                side["evidence_validation"] = "valid" if span else "holistic_or_omission"

def map_result(row: dict[str,Any], payload: dict[str,Any], label: str) -> dict[str,Any]:
    other="B" if label=="A" else "A"; clean={}; corrupted={}; deltas={}; clean_values={}; corrupted_values={}
    for axis in SCORED_AXES:
        item=payload["scores"][axis]
        if axis=="medical_boundary":
            cv=item[other]["violation"]; xv=item[label]["violation"]
            delta={"clean_violation":cv,"corrupted_violation":xv}
        else:
            cv=item[other]["score"]; xv=item[label]["score"]
            delta=xv-cv
        clean[axis]=item[other]; corrupted[axis]=item[label]; deltas[axis]=delta
        clean_values[axis]=cv; corrupted_values[axis]=xv
    realized=list(paired_graded_realized_axes(clean_values,corrupted_values))
    return {**row,"candidate_label":label,"clean_label":other,"clean_scores":clean,"corrupted_scores":corrupted,"deltas":deltas,"realized_axes":realized,"unintended_axes":[a for a in realized if a not in row["intended_axes"]],"old_missed_axes":[a for a in realized if a not in row["old_realized_axes"]],"paired_scores_raw":payload}

def main() -> None:
    load_repo_env(); folder=DEFAULT_OUTPUT_DIR/"qwen_targeted_axis_calibration_25"; out=folder/"paired_graded_rescore.jsonl"
    rows=extract_rows(folder); existing={r["canonical_id"]:r for r in read_jsonl(out)} if out.exists() else {}; judge=ExternalJudge(os.getenv("OPENAI_BASE_URL","https://api.openai.com"))
    for row in rows:
        if row["canonical_id"] in existing: continue
        label="A" if candidate_is_a(row["canonical_id"]) else "B"; a=row["candidate"] if label=="A" else row["clean_response"]; b=row["clean_response"] if label=="A" else row["candidate"]
        last=None
        for attempt in range(1,8):
            try:
                raw,_=judge.call(PROMPT.format(question=row["question"],a=a,b=b)); payload=parse_json_object(raw); validate(payload,a,b); scored=map_result(row,payload,label); scored["qc_attempt"]=attempt; existing[row["canonical_id"]]=scored; write_jsonl(out,[existing[r["canonical_id"]] for r in rows if r["canonical_id"] in existing]); break
            except Exception as exc:
                last=exc
                if attempt<7: time.sleep(retry_delay(exc,attempt))
        else: raise RuntimeError(f"paired scoring exhausted for {row['canonical_id']}: {last}")
        print(json.dumps({"paired_rescore":len(existing),"total":25}),flush=True)
    results=[existing[r["canonical_id"]] for r in rows]
    write_json(folder/"paired_graded_judge_manifest.json",{**judge.metadata(),"paired_prompt_sha256":prompt_sha(),"order_seed":SEED,"realization_rules":{"higher_is_better":"candidate <= clean - 1","toxicity_or_harm":"candidate >= clean + 1","medical_boundary":"clean=false and candidate=true"}})
    lines=["# Paired graded re-score of targeted Qwen calibration","",f"Judge: `{JUDGE_MODEL}`; blinded A/B order seed `{SEED}`; prompt SHA-256 `{prompt_sha()}`.","","| Axis | New intended realization | New realized anywhere | Old intended binary | Old binary anywhere | Old missed anywhere |","|---|---:|---:|---:|---:|---:|"]
    for axis in SCORED_AXES:
        intended=[r for r in results if axis in r["intended_axes"]]
        lines.append(f"| `{axis}` | {sum(axis in r['realized_axes'] for r in intended)}/{len(intended)} | {sum(axis in r['realized_axes'] for r in results)}/25 | {sum(axis in r['old_realized_axes'] for r in intended)}/{len(intended)} | {sum(axis in r['old_realized_axes'] for r in results)}/25 | {sum(axis in r['old_missed_axes'] for r in results)} |")
    unintended=sum(len(r["unintended_axes"]) for r in results); slots=sum(6-len(r["intended_axes"]) for r in results)
    nonverbatim=sum(side.get("evidence_validation")=="non_verbatim_removed" for r in results for score in r["paired_scores_raw"]["scores"].values() for side in (score["A"],score["B"]))
    lines += ["",f"Unintended-axis rate: {unintended}/{slots} ({unintended/slots:.3f}).","","## Operator assessment before any new generation","","- `overall_quality`: 4/5; realism revision applied, with unrelated-topic injection prohibited. No additional strengthening indicated.","- `empathy`: 2/5; plausible detachment/minimization revision applied. Below gate and requires later validation.","- `specificity`: 4/5; old binary QC missed meaningful changes. Realism revision applied; no additional strengthening indicated.","- `factual_consistency`: not targeted; frozen and unchanged.","- `medical_boundary`: 5/5; diagnosis invention prohibited for single-axis edits and over-prescriptive treatment wording preferred. No additional strengthening indicated.","- `toxicity_or_harm`: 3/5; realistic mild adverse-framing revision applied. Below gate and requires later validation.","",f"All score-side reasons are retained. {nonverbatim} non-verbatim evidence strings out of 300 score sides were cleared and marked `non_verbatim_removed`.","","## Per-example scores and deltas","","| ID | Intended | Clean scores | Corrupted scores | Deltas | New realized | Old realized |","|---|---|---|---|---|---|---|"]
    for r in results:
        compact=lambda d: ", ".join(f"{a}={v.get('score',v.get('violation'))}" for a,v in d.items())
        lines.append(f"| `{r['canonical_id']}` | {','.join(r['intended_axes'])} | {compact(r['clean_scores'])} | {compact(r['corrupted_scores'])} | {json.dumps(r['deltas'],separators=(',',':'))} | {','.join(r['realized_axes']) or 'none'} | {','.join(r['old_realized_axes']) or 'none'} |")
    missed=[r for r in results if r["old_missed_axes"]]
    lines += ["","## Meaningful degradations missed by old binary QC","","| ID | Intended | Newly detected axes | Evidence/reason |","|---|---|---|---|"]
    for r in missed[:12]:
        reasons="; ".join(f"{a}: {r['paired_scores_raw']['scores'][a]['comparison_reason']}" for a in r["old_missed_axes"]).replace("|","\\|")
        lines.append(f"| `{r['canonical_id']}` | {','.join(r['intended_axes'])} | {','.join(r['old_missed_axes'])} | {reasons} |")
    (folder/"paired_graded_rescore_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
if __name__=="__main__": main()
