#!/usr/bin/env python3
"""Build the bounded 120-input Qwen development corruption shard."""
from __future__ import annotations

import argparse, hashlib, json, os, re, time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from corruption_contract_v2 import AXES, STAGE_DIRECTIVES, StageRequest, build_stage_prompt, paired_graded_realized_axes, parse_json_object
from fullpaper_acl_pipeline import DEFAULT_OUTPUT_DIR, largest_remainder_counts, normalize_text, read_jsonl, sha256_file, stable_random_key, write_json, write_jsonl
from run_clean_target_qc import retry_delay
from run_gemma_corruption_pilot_v2 import attach_specs, balanced_specs
from run_paired_generator_diagnostic import DEFAULT_REQUIRED_FREE_VRAM_MIB, ELIGIBILITY_PROMPT, ExternalJudge, GENERATORS, JUDGE_MODEL, JUDGE_SEED, JUDGE_TEMPERATURE, LocalGenerator, deterministic_generation_seed, generation_stop_metadata, gpu_status, load_repo_env, refusal
from source_integrity_contract import VERSION as SOURCE_INTEGRITY_VERSION, contract_hash as source_integrity_contract_hash, surface_flags
from shared_api_budget import BudgetExceeded

BUILD_SEED=20260910
BUILD_SIZE=120
OUTPUT_NAME="development_training_shard_120"
BUILD_JUDGE_MAX_TOKENS=2400
PAIRED_PROMPT="""Blindly score two complete counseling responses to the same user. Labels are deterministically randomized. Do not infer which is original, edited, preferred, or generated. The complete texts are between explicit delimiters. Score observed content only and do not claim that text is absent if it appears in both responses.

Rubrics:
- overall_quality, empathy, specificity: integer 1-5, higher is better.
- factual_consistency: integer 1-4, higher is better.
- toxicity_or_harm: integer 1-5, LOWER is better: 1=no toxicity/harm, 2=subtle dismissiveness or mild risk, 3=clear blame/stigma/harmful minimization, 4=strong harmful framing, 5=severe toxicity or dangerous guidance. Never reverse this scale.
- medical_boundary: binary violation; true only for unsupported clinical diagnosis certainty, medical advice, or over-prescriptive direction about clinical/medical treatment. General career, relationship, communication, or other nonclinical life directiveness alone is not a medical-boundary violation.

A one-point difference must be meaningful rather than stylistic. For every response and dimension give a concrete reason and an exact short evidence span when local evidence exists. Empty evidence is allowed for holistic qualities or omissions. Assess all dimensions without knowing requested axes.

Content checks apply to Response A and B as counseling responses. An appropriate patient-facing safety boundary is not an editing-task refusal. Flag only editing/instruction meta-refusals. Flag unrelated-topic injection, exaggerated insults/cartoonish shaming, and actionable dangerous content including physical/self-harm instructions or medication dosing.

User:
<<<USER>>>{question}<<<END_USER>>>
Response A:
<<<A>>>{a}<<<END_A>>>
Response B:
<<<B>>>{b}<<<END_B>>>

Return JSON only with exactly this structure:
{{"scores":{{
"overall_quality":{{"A":{{"score":1,"evidence_span":"","reason":""}},"B":{{"score":1,"evidence_span":"","reason":""}},"comparison_reason":""}},
"empathy":{{"A":{{"score":1,"evidence_span":"","reason":""}},"B":{{"score":1,"evidence_span":"","reason":""}},"comparison_reason":""}},
"specificity":{{"A":{{"score":1,"evidence_span":"","reason":""}},"B":{{"score":1,"evidence_span":"","reason":""}},"comparison_reason":""}},
"factual_consistency":{{"A":{{"score":1,"evidence_span":"","reason":""}},"B":{{"score":1,"evidence_span":"","reason":""}},"comparison_reason":""}},
"toxicity_or_harm":{{"A":{{"score":1,"evidence_span":"","reason":""}},"B":{{"score":1,"evidence_span":"","reason":""}},"comparison_reason":""}},
"medical_boundary":{{"A":{{"violation":false,"evidence_span":"","reason":""}},"B":{{"violation":false,"evidence_span":"","reason":""}},"comparison_reason":""}}
}},"content_checks":{{
"topic_preserved":{{"value":true,"evidence_span":"","reason":""}},
"fluent":{{"value":true,"evidence_span":"","reason":""}},
"unrelated_topic_injection":{{"value":false,"evidence_span":"","reason":""}},
"cartoonish_or_exaggerated_shaming":{{"value":false,"evidence_span":"","reason":""}},
"prohibited_dangerous_content":{{"value":false,"evidence_span":"","reason":""}},
"editing_task_refusal_or_meta":{{"value":false,"evidence_span":"","reason":""}}
}},"text_reason_contradiction":{{"detected":false,"reason":""}},"overall_reason":""}}"""

ACCEPTANCE_RULES={"score_drop":1,"toxicity_increase":1,"medical_boundary":"clean_false_candidate_true","require_all_intended":True,"allow_additional":True,"content_checks_required":True,"max_semantic_attempts_per_stage":2,"max_infrastructure_attempts":4,"truncation_ceiling":2048}

def htext(value:str)->str:return hashlib.sha256(value.encode()).hexdigest()
def canonical_json_hash(value:Any)->str:return htext(json.dumps(value,sort_keys=True,separators=(",",":")))
def paired_prompt_hash()->str:return htext(PAIRED_PROMPT)
def generation_prompt_hash()->str:return canonical_json_hash(STAGE_DIRECTIVES)
def clean_qc_hash()->str:return htext(ELIGIBILITY_PROMPT)

def previous_calibration_clusters(root:Path,by_id:dict[str,dict[str,Any]])->set[str]:
    clusters=set()
    for path in root.rglob("*selection*.json"):
        if OUTPUT_NAME in str(path): continue
        try: payload=json.loads(path.read_text())
        except Exception: continue
        rows=payload.get("rows",[]) if isinstance(payload,dict) else []
        for item in rows:
            if isinstance(item,dict) and item.get("canonical_id") in by_id: clusters.add(by_id[item["canonical_id"]]["duplicate_cluster_id"])
    return clusters

def select(canonical:list[dict[str,Any]],qc_rows:list[dict[str,Any]],root:Path)->list[dict[str,Any]]:
    by_id={r["canonical_id"]:r for r in canonical}; old=previous_calibration_clusters(root,by_id)
    valid={r["canonical_id"]:r for r in qc_rows if r.get("qc_ok") and r.get("baseline_degraded_axes")==[]}
    pool=[]; seen=set(); seen_q=set()
    for cid in sorted(valid,key=lambda x:stable_random_key(BUILD_SEED,x)):
        row=by_id.get(cid)
        if not row or row["split"]!="train" or row["duplicate_cluster_id"] in old or row["duplicate_cluster_id"] in seen or row["question_normalized_sha256"] in seen_q: continue
        seen.add(row["duplicate_cluster_id"]);seen_q.add(row["question_normalized_sha256"]);pool.append(row)
    train=[r for r in canonical if r["split"]=="train"]
    strata=Counter((r["source"],r["source_component"]) for r in train)
    quotas=largest_remainder_counts(BUILD_SIZE,{k:v/len(train) for k,v in strata.items()})
    picked=[]
    for cell,q in sorted(quotas.items()):
        options=[r for r in pool if (r["source"],r["source_component"])==cell]
        if len(options)<q: raise RuntimeError(f"eligible QC pool short for {cell}: need {q}, have {len(options)}")
        picked.extend(options[:q])
    picked.sort(key=lambda r:stable_random_key(BUILD_SEED+1,r["canonical_id"]))
    if len(picked)!=BUILD_SIZE or len({r["duplicate_cluster_id"] for r in picked})!=BUILD_SIZE:raise AssertionError("invalid build selection")
    specs,stats=balanced_specs(BUILD_SIZE,BUILD_SEED)
    assigned=attach_specs(picked,specs,BUILD_SEED+2)
    for item in assigned:
        axes=tuple(sorted(item["intended_axes"],key=lambda a:stable_random_key(BUILD_SEED,f"{item['clean']['canonical_id']}:{a}")))
        item["intended_axes"]=list(axes);item["generation_seeds"]=[{"stage_index":i+1,"axis":axis,"attempt_1":deterministic_generation_seed(BUILD_SEED,item["clean"]["canonical_id"],i+1,1),"attempt_2":deterministic_generation_seed(BUILD_SEED,item["clean"]["canonical_id"],i+1,2)} for i,axis in enumerate(axes)]
    return assigned

def load_or_freeze_selection(out:Path,canonical:list[dict[str,Any]],qc:list[dict[str,Any]],root:Path)->list[dict[str,Any]]:
    path=out/"selection.json"
    if path.exists(): payload=json.loads(path.read_text())
    else:
        assigned=select(canonical,qc,root)
        payload={"version":"development-training-shard-selection-v1","seed":BUILD_SEED,"rows":[{"canonical_id":x["clean"]["canonical_id"],"intended_axes":x["intended_axes"],"axis_count":x["axis_count"],"generation_seeds":x["generation_seeds"]} for x in assigned]}
        write_json(path,payload)
    by_id={r["canonical_id"]:r for r in canonical}; qc_by={r["canonical_id"]:r for r in qc if r.get("qc_ok")}
    result=[]
    for slot,item in enumerate(payload["rows"]):
        row=by_id[item["canonical_id"]]; q=qc_by.get(row["canonical_id"])
        if row["split"]!="train" or not q or q.get("baseline_degraded_axes")!=[]:raise AssertionError("frozen selection contains an ineligible target")
        result.append({"slot":slot,"clean":row,"eligibility":q,**{k:item[k] for k in ("intended_axes","axis_count","generation_seeds")}})
    if len(result)!=BUILD_SIZE:raise AssertionError("selection must have 120 rows")
    return result

def candidate_a(cid:str,stage:int,attempt:int)->bool:return hashlib.sha256(f"{BUILD_SEED}:{cid}:{stage}:{attempt}:ab".encode()).digest()[0]%2==0
def editing_meta(text:str)->bool:
    n=normalize_text(text)[:500]
    return bool(re.search(r"\b(cannot|can't|unable to)\b.{0,80}\b(comply|instruction|request|edit|rewrite|corrupt|dimension|prompt)\b",n)) or "synthetic data" in n

def validate_paired(payload:dict[str,Any],a:str,b:str)->list[dict[str,Any]]:
    scores=payload.get("scores"); invalid=[]
    if not isinstance(scores,dict) or set(scores)!=set(AXES):raise ValueError("six exact score axes required")
    for axis in AXES:
        item=scores[axis]
        if not str(item.get("comparison_reason","")).strip():raise ValueError(f"missing comparison reason {axis}")
        values={}
        for label,text in (("A",a),("B",b)):
            side=item.get(label,{})
            key="violation" if axis=="medical_boundary" else "score"; value=side.get(key); values[label]=value
            if axis=="medical_boundary":
                if not isinstance(value,bool):raise ValueError("invalid medical score")
            else:
                maximum=4 if axis=="factual_consistency" else 5
                if isinstance(value,bool) or not isinstance(value,int) or not 1<=value<=maximum:raise ValueError(f"invalid score {axis}/{label}")
            if not str(side.get("reason","")).strip():raise ValueError(f"missing reason {axis}/{label}")
            span=str(side.get("evidence_span") or "").strip()
            if span and normalize_text(span) not in normalize_text(text):
                invalid.append({"axis":axis,"response_label":label,"invalid_span":span,"reason":"non_verbatim_evidence_removed"});side["evidence_span"]="";side["evidence_validation"]="non_verbatim_removed"
            else:side["evidence_validation"]="valid" if span else "holistic_or_omission"
    checks=payload.get("content_checks")
    required={"topic_preserved","fluent","unrelated_topic_injection","cartoonish_or_exaggerated_shaming","prohibited_dangerous_content","editing_task_refusal_or_meta"}
    if not isinstance(checks,dict) or set(checks)!=required:raise ValueError("invalid content checks")
    for name,item in checks.items():
        if not isinstance(item.get("value"),bool) or not str(item.get("reason","")).strip():raise ValueError(f"invalid content check {name}")
        span=str(item.get("evidence_span") or "").strip()
        if span and normalize_text(span) not in normalize_text(a) and normalize_text(span) not in normalize_text(b):invalid.append({"axis":name,"response_label":"unknown","invalid_span":span,"reason":"non_verbatim_evidence_removed"});item["evidence_span"]="";item["evidence_validation"]="non_verbatim_removed"
    contradiction=payload.get("text_reason_contradiction",{})
    if not isinstance(contradiction.get("detected"),bool):raise ValueError("invalid contradiction flag")
    return invalid

def paired_grade(judge:ExternalJudge,row:dict[str,Any],candidate:str,stage:int,attempt:int)->tuple[dict[str,Any]|None,list[dict[str,Any]],list[dict[str,Any]]]:
    label="A" if candidate_a(row["canonical_id"],stage,attempt) else "B"; other="B" if label=="A" else "A";a=candidate if label=="A" else row["clean_response"];b=row["clean_response"] if label=="A" else candidate
    infra=[]
    for ia in range(1,5):
        try:
            raw,_=judge.call(PAIRED_PROMPT.format(question=row["question"],a=a,b=b)); payload=parse_json_object(raw); invalid=validate_paired(payload,a,b);break
        except BudgetExceeded:
            raise
        except Exception as exc:
            infra.append({"kind":"paired_judge_parse_or_validation","attempt":ia,"reason":f"{type(exc).__name__}: {exc}"})
            if ia<4:time.sleep(retry_delay(exc,ia))
    else:return None,[],infra
    clean_values={};candidate_values={};clean_scores={};candidate_scores={}
    for axis in AXES:
        key="violation" if axis=="medical_boundary" else "score";clean_scores[axis]=payload["scores"][axis][other];candidate_scores[axis]=payload["scores"][axis][label];clean_values[axis]=clean_scores[axis][key];candidate_values[axis]=candidate_scores[axis][key]
    realized=list(paired_graded_realized_axes(clean_values,candidate_values))
    mapped={"candidate_label":label,"clean_label":other,"response_a_sha256":htext(a),"response_b_sha256":htext(b),"response_a_chars":len(a),"response_b_chars":len(b),"clean_scores":clean_scores,"candidate_scores":candidate_scores,"deltas":{axis:({"clean_violation":clean_values[axis],"candidate_violation":candidate_values[axis]} if axis=="medical_boundary" else candidate_values[axis]-clean_values[axis]) for axis in AXES},"realized_axes":realized,"unintended_axes":[x for x in realized if x not in row["intended_axes"]],"content_checks":payload["content_checks"],"text_reason_contradiction":payload["text_reason_contradiction"],"overall_reason":payload.get("overall_reason",""),"invalid_evidence":invalid}
    return mapped,invalid,infra

def content_failure(question:str,candidate:str,graded:dict[str,Any])->str|None:
    c=graded["content_checks"]
    if not c["topic_preserved"]["value"] or c["unrelated_topic_injection"]["value"]:return "unrelated_topic_or_topic_not_preserved"
    if not c["fluent"]["value"]:return "not_fluent"
    if c["cartoonish_or_exaggerated_shaming"]["value"]:return "cartoonish_or_exaggerated_shaming"
    if c["prohibited_dangerous_content"]["value"]:return "prohibited_dangerous_content"
    if c["editing_task_refusal_or_meta"]["value"] or editing_meta(candidate):return "editing_task_refusal_or_meta"
    integrity=surface_flags(question,candidate,"candidate")
    if integrity:return "candidate_integrity_failure:"+",".join(sorted({x["kind"] for x in integrity}))
    return None

def process_one(item:dict[str,Any],backend:LocalGenerator,judge:ExternalJudge)->dict[str,Any]:
    row=item["clean"];current=row["clean_response"];history=[];infra=[];final_grade=None
    for stage,axis in enumerate(item["intended_axes"],1):
        feedback=None;stage_ok=False
        for semantic_attempt in (1,2):
            req=StageRequest(row["canonical_id"],row.get("split","train"),row["question"],row["clean_response"],current,tuple(item["intended_axes"]),tuple(item["intended_axes"][:stage-1]),axis,stage,deterministic_generation_seed(BUILD_SEED,row["canonical_id"],stage,semantic_attempt),semantic_attempt,feedback)
            candidate=None;allowance=None
            for ia in range(1,5):
                try:candidate=backend.generate([req],allowance)[0]
                except Exception as exc:
                    infra.append({"kind":"generation_failure","stage":stage,"semantic_attempt":semantic_attempt,"infrastructure_attempt":ia,"reason":f"{type(exc).__name__}: {exc}"});candidate=None
                    if ia<4:time.sleep(min(2**(ia-1),8))
                    continue
                if candidate.truncation_reason:
                    infra.append({"kind":"generation_truncation","stage":stage,"semantic_attempt":semantic_attempt,"infrastructure_attempt":ia,"current_response_tokens":candidate.current_response_tokens,"max_new_tokens":candidate.max_new_tokens,"generated_tokens":candidate.generated_tokens,"eos_reached":candidate.eos_reached,"reason":candidate.truncation_reason})
                    if candidate.max_new_tokens>=2048:candidate=None;break
                    allowance=min(max(candidate.max_new_tokens+256,candidate.max_new_tokens*2),2048);candidate=None;continue
                break
            if candidate is None:return {"status":"technical_failure","failure_reason":"generation_infrastructure_exhausted","stage_history":history,"infrastructure_failures":infra}
            immediate=None
            if not candidate.text:immediate="empty_generation"
            elif normalize_text(candidate.text)==normalize_text(current):immediate="unchanged_response"
            elif editing_meta(candidate.text):immediate="editing_task_refusal_or_meta"
            if immediate:
                history.append({"stage_index":stage,"target_axis":axis,"semantic_attempt":semantic_attempt,"candidate_response":candidate.text,"generation":candidate.__dict__,"grade":None,"accepted":False,"reason":immediate});feedback=immediate;continue
            grade,invalid,judge_infra=paired_grade(judge,{**row,"intended_axes":item["intended_axes"]},candidate.text,stage,semantic_attempt);infra.extend({**x,"stage":stage,"semantic_attempt":semantic_attempt} for x in judge_infra)
            if grade is None:return {"status":"technical_failure","failure_reason":"paired_judge_infrastructure_exhausted","stage_history":history,"infrastructure_failures":infra}
            conflict=[]
            if grade["clean_scores"]["specificity"]["score"]==1:conflict.append("clean_specificity_score_1")
            if grade["clean_scores"]["medical_boundary"]["violation"]:conflict.append("clean_medical_boundary_violation")
            if grade["text_reason_contradiction"]["detected"]:conflict.append("judge_text_reason_contradiction")
            if conflict:
                recheck=None
                try:recheck=judge.eligibility(row)
                except BudgetExceeded:
                    raise
                except Exception as exc:infra.append({"kind":"bounded_conflict_recheck_failure","stage":stage,"semantic_attempt":semantic_attempt,"reason":f"{type(exc).__name__}: {exc}"})
                history.append({"stage_index":stage,"target_axis":axis,"semantic_attempt":semantic_attempt,"candidate_response":candidate.text,"generation":candidate.__dict__,"grade":grade,"accepted":False,"reason":"qc_conflict","conflict_flags":conflict,"bounded_clean_recheck":recheck})
                return {"status":"qc_conflict","failure_reason":";".join(conflict),"stage_history":history,"infrastructure_failures":infra,"conflict_recheck":recheck}
            reason=content_failure(row["question"],candidate.text,grade)
            missing=[x for x in item["intended_axes"][:stage] if x not in grade["realized_axes"]]
            if not reason and missing:reason="missing_intended_axes:"+",".join(missing)
            accepted=reason is None
            history.append({"stage_index":stage,"target_axis":axis,"semantic_attempt":semantic_attempt,"generation_seed":req.generation_seed,"candidate_response":candidate.text,"generation":candidate.__dict__,"grade":grade,"accepted":accepted,"reason":reason})
            if accepted:current=candidate.text;final_grade=grade;stage_ok=True;break
            feedback=reason
        if not stage_ok:return {"status":"rejected","failure_reason":feedback or "semantic_stage_failed","stage_history":history,"infrastructure_failures":infra}
    return {"status":"accepted","corrupted_response":current,"realized_axes":final_grade["realized_axes"],"unintended_axes":final_grade["unintended_axes"],"final_grade":final_grade,"stage_history":history,"infrastructure_failures":infra}

def base_record(item:dict[str,Any])->dict[str,Any]:
    r=item["clean"]
    return {"canonical_id":r["canonical_id"],"source":r["source"],"source_component":r["source_component"],"duplicate_cluster_id":r["duplicate_cluster_id"],"source_group_id":r["source_group_id"],"split":"train","question":r["question"],"clean_response":r["clean_response"],"baseline_degraded_axes":[],"intended_axes":item["intended_axes"],"axis_count":item["axis_count"],"generation_seeds":item["generation_seeds"],"generator_repo":GENERATORS["qwen"]["repo"],"generator_revision":GENERATORS["qwen"]["revision"],"judge_model":JUDGE_MODEL}

def export(out:Path,selection:list[dict[str,Any]],results:list[dict[str,Any]],manifest:dict[str,Any])->None:
    accepted=[r for r in results if r["status"]=="accepted"];rejected=[r for r in results if r["status"]!="accepted" and r["status"]!="qc_conflict"];conflicts=[r for r in results if r["status"]=="qc_conflict"]
    write_jsonl(out/"accepted.jsonl",accepted);write_jsonl(out/"rejected.jsonl",rejected);write_jsonl(out/"qc_conflicts.jsonl",conflicts)
    sft=[];dpo=[]
    for r in accepted:
        inp={"question":r["question"],"corrupted_response":r["corrupted_response"]};meta={k:r[k] for k in ("canonical_id","source","source_component","intended_axes","realized_axes","unintended_axes","axis_count","judge_model","generator_repo","generator_revision")};meta["paired_qc"]={k:r["final_grade"][k] for k in ("clean_scores","candidate_scores","deltas","content_checks","invalid_evidence")}
        sft.append({"input":inp,"target":r["clean_response"],"metadata":meta});dpo.append({"input":inp,"chosen":r["clean_response"],"rejected":r["corrupted_response"],"metadata":meta})
    write_jsonl(out/"train_sft.jsonl",sft);write_jsonl(out/"train_dpo.jsonl",dpo)
    axis=Counter(a for r in accepted for a in r["intended_axes"]);counts=Counter(r["axis_count"] for r in accepted);sources=Counter(r["source"] for r in accepted);reasons=Counter(r.get("failure_reason","") for r in results if r["status"]!="accepted")
    infra=Counter(e["kind"] for r in results for e in r.get("infrastructure_failures",[]));total_calls=sum(r["usage"]["generator_calls"]+r["usage"]["judge_calls"] for r in results);total_tokens=sum(r["usage"][k] for r in results for k in ("generator_prompt_tokens","generator_completion_tokens","judge_prompt_tokens","judge_completion_tokens"));measured=sum(r["usage"]["generator_seconds"]+r["usage"]["judge_seconds"] for r in results)
    checks={"input_denominator_120":len(results)==120,"terminal_partition":len(accepted)+len(rejected)+len(conflicts)==120,"accepted_intended_subset_realized":all(set(r["intended_axes"])<=set(r["realized_axes"]) for r in accepted),"accepted_clean_policy":all(r["baseline_degraded_axes"]==[] for r in accepted),"accepted_train_only":all(r["split"]=="train" for r in accepted),"accepted_unique_duplicate_clusters":len({r["duplicate_cluster_id"] for r in accepted})==len(accepted),"sft_dpo_count_matches_accepted":len(sft)==len(dpo)==len(accepted),"refiner_inputs_exclude_targets_and_labels":all(set(r["input"])=={"question","corrupted_response"} for r in sft+dpo),"sft_target_is_clean":all(a["target"]==r["clean_response"] for a,r in zip(sft,accepted,strict=True)),"dpo_contract":all(a["chosen"]==r["clean_response"] and a["rejected"]==r["corrupted_response"] and a["input"]==s["input"] for a,s,r in zip(dpo,sft,accepted,strict=True))}
    if not all(checks.values()):raise AssertionError(f"export data contract failed: {checks}")
    manifest.update({"status":"complete","development_training_shard":True,"processed_inputs":len(results),"accepted_count":len(accepted),"rejected_count":len(rejected),"qc_conflict_count":len(conflicts),"accepted_by_intended_axis":dict(axis),"accepted_by_axis_count":dict(counts),"accepted_by_source":dict(sources),"rejection_reasons":dict(reasons),"infrastructure_retry_events":dict(infra),"calls_per_accepted":total_calls/len(accepted) if accepted else None,"tokens_per_accepted":total_tokens/len(accepted) if accepted else None,"measured_seconds_per_accepted":measured/len(accepted) if accepted else None,"data_contract_checks":checks})
    artifacts={name:sha256_file(out/name) for name in ("selection.json","accepted.jsonl","rejected.jsonl","qc_conflicts.jsonl","train_sft.jsonl","train_dpo.jsonl")};manifest["artifact_sha256"]=artifacts;write_json(out/"build_manifest.json",manifest)
    lines=["# Development corruption training shard (120 inputs)","",f"Accepted: {len(accepted)}/120. This is a bounded development shard, not the final experiment dataset.","",f"Technical failures: {sum(r['status']=='technical_failure' for r in results)}; QC conflicts: {len(conflicts)}.","",f"Accepted by intended axis: `{dict(axis)}`",f"Accepted by axis count: `{dict(counts)}`",f"Accepted by source: `{dict(sources)}`",f"Rejected reasons: `{dict(reasons)}`",f"Infrastructure retry events: `{dict(infra)}`",f"Calls per accepted example: {total_calls/len(accepted):.3f}",f"Tokens per accepted example: {total_tokens/len(accepted):.1f}",f"Measured generation+judge seconds per accepted example: {measured/len(accepted):.3f}","","QC conflicts were quarantined after one bounded clean-target recheck; primary paired scores were not overwritten.","","Data-contract checks:","```json",json.dumps(checks,indent=2),"```","","Exact build commands:","```bash",".venv-fullpaper/bin/python -u scripts/build_development_corruption_shard.py --worker-count 3 --worker-index 0",".venv-fullpaper/bin/python -u scripts/build_development_corruption_shard.py --worker-count 3 --worker-index 1",".venv-fullpaper/bin/python -u scripts/build_development_corruption_shard.py --worker-count 3 --worker-index 2","```","","Artifact hashes:","```json",json.dumps(artifacts,indent=2),"```",""]
    (out/"build_report.md").write_text("\n".join(lines),encoding="utf-8")

def main()->None:
    load_repo_env();p=argparse.ArgumentParser();p.add_argument("--output-dir",default=str(DEFAULT_OUTPUT_DIR/OUTPUT_NAME));p.add_argument("--selected-gpu",type=int,default=0);p.add_argument("--required-free-vram-mib",type=int,default=DEFAULT_REQUIRED_FREE_VRAM_MIB);p.add_argument("--max-input-tokens",type=int,default=4096);p.add_argument("--worker-index",type=int,default=0);p.add_argument("--worker-count",type=int,default=1);p.add_argument("--finalize-only",action="store_true");a=p.parse_args()
    if not 1<=a.worker_count<=3 or not 0<=a.worker_index<a.worker_count:raise ValueError("worker-count must be 1..3 and worker-index must be in range")
    out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True);root=DEFAULT_OUTPUT_DIR;canonical_path=root/"canonical_clean_qa.jsonl";qc_path=root/"clean_target_qc"/"corpus_qc.jsonl";split_path=root/"split_manifest.json"
    split=json.loads(split_path.read_text());assert all(split["assertions"].values())
    canonical=list(read_jsonl(canonical_path));qc=list(read_jsonl(qc_path));selection=load_or_freeze_selection(out,canonical,qc,root)
    specs=Counter(x["axis_count"] for x in selection);assert specs=={1:60,2:42,3:18}
    hashes={"generation_prompts_sha256":generation_prompt_hash(),"clean_target_qc_sha256":clean_qc_hash(),"paired_qc_sha256":paired_prompt_hash(),"acceptance_rules_sha256":canonical_json_hash(ACCEPTANCE_RULES),"source_integrity_contract_version":SOURCE_INTEGRITY_VERSION,"source_integrity_contract_sha256":source_integrity_contract_hash()}
    manifest={"version":"development-corruption-shard-v1","status":"building","development_training_shard":True,"seed":BUILD_SEED,"input_denominator":120,"axis_count_allocation":dict(specs),"intended_axis_marginals":dict(Counter(x for r in selection for x in r["intended_axes"])),"selected_by_source":dict(Counter(r["clean"]["source"] for r in selection)),"selected_by_source_component":dict(Counter(f"{r['clean']['source']}:{r['clean']['source_component']}" for r in selection)),"generator":GENERATORS["qwen"],"judge_model":JUDGE_MODEL,"judge_config":{"temperature":JUDGE_TEMPERATURE,"seed":JUDGE_SEED,"max_tokens":BUILD_JUDGE_MAX_TOKENS,"response_format":"json_object"},"hashes":hashes,"canonical_sha256":sha256_file(canonical_path),"split_manifest_sha256":sha256_file(split_path),"clean_qc_checkpoint_sha256_at_freeze":sha256_file(qc_path),"split_assertions":split["assertions"],"acceptance_rules":ACCEPTANCE_RULES}
    manifest_path=out/"build_manifest.json"
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text());
        if old.get("status")=="complete":
            if not a.finalize_only:raise RuntimeError("immutable completed shard already exists")
            combined={}
            for part in sorted(out.glob("results_checkpoint.part*.jsonl")):combined.update({r["canonical_id"]:r for r in read_jsonl(part)})
            if len(combined)!=BUILD_SIZE:raise RuntimeError(f"cannot finalize incomplete shard: {len(combined)}/120")
            results=[combined[x["clean"]["canonical_id"]] for x in selection];export(out,selection,results,old);return
        if old.get("hashes")!=hashes:raise RuntimeError("frozen configuration hash mismatch")
    else:write_json(manifest_path,manifest)
    gpu=gpu_status(a.selected_gpu,a.required_free_vram_mib);print(json.dumps({"gpu":gpu,"selection":manifest["selected_by_source"],"axis_counts":dict(specs)}),flush=True)
    if not gpu["sufficient_free_vram"]:raise RuntimeError("insufficient free VRAM")
    checkpoint=out/(f"results_checkpoint.part{a.worker_index}.jsonl" if a.worker_count>1 else "results_checkpoint.jsonl");done={r["canonical_id"]:r for r in read_jsonl(checkpoint)} if checkpoint.exists() else {}
    width=(BUILD_SIZE+a.worker_count-1)//a.worker_count;worker_selection=selection[a.worker_index*width:min((a.worker_index+1)*width,BUILD_SIZE)]
    judge=ExternalJudge(os.getenv("OPENAI_BASE_URL","https://api.openai.com"),max_tokens=BUILD_JUDGE_MAX_TOKENS);backend=LocalGenerator("qwen",1,a.max_input_tokens)
    started=time.monotonic()
    try:
        for item in worker_selection:
            cid=item["clean"]["canonical_id"]
            if cid in done:continue
            bg=(backend.calls,backend.prompt_tokens,backend.completion_tokens,backend.elapsed);bj=(judge.calls,judge.prompt_tokens,judge.completion_tokens,judge.elapsed);t=time.monotonic();result=process_one(item,backend,judge)
            usage={"generator_calls":backend.calls-bg[0],"generator_prompt_tokens":backend.prompt_tokens-bg[1],"generator_completion_tokens":backend.completion_tokens-bg[2],"generator_seconds":backend.elapsed-bg[3],"judge_calls":judge.calls-bj[0],"judge_prompt_tokens":judge.prompt_tokens-bj[1],"judge_completion_tokens":judge.completion_tokens-bj[2],"judge_seconds":judge.elapsed-bj[3],"wall_seconds":time.monotonic()-t}
            done[cid]={**base_record(item),**result,"usage":usage};write_jsonl(checkpoint,[done[x["clean"]["canonical_id"]] for x in worker_selection if x["clean"]["canonical_id"] in done])
            if len(done)%20==0:
                vals=list(done.values());print(json.dumps({"worker":a.worker_index,"processed":len(vals),"accepted":sum(r["status"]=="accepted" for r in vals),"rejected":sum(r["status"]=="rejected" for r in vals),"qc_conflicts":sum(r["status"]=="qc_conflict" for r in vals),"technical_failures":sum(r["status"]=="technical_failure" for r in vals),"calls":sum(r["usage"]["generator_calls"]+r["usage"]["judge_calls"] for r in vals),"tokens":sum(r["usage"]["generator_prompt_tokens"]+r["usage"]["generator_completion_tokens"]+r["usage"]["judge_prompt_tokens"]+r["usage"]["judge_completion_tokens"] for r in vals)}),flush=True)
    finally:backend.close()
    if len(done)!=len(worker_selection):raise RuntimeError(f"incomplete worker checkpoint {len(done)}/{len(worker_selection)}")
    if a.worker_count==1:
        results=[done[x["clean"]["canonical_id"]] for x in selection]
    else:
        combined={}
        for index in range(a.worker_count):
            part=out/f"results_checkpoint.part{index}.jsonl"
            if not part.exists():print(json.dumps({"worker":a.worker_index,"status":"complete_waiting_for_other_workers"}),flush=True);return
            combined.update({r["canonical_id"]:r for r in read_jsonl(part)})
        if len(combined)!=BUILD_SIZE:print(json.dumps({"worker":a.worker_index,"status":"complete_waiting_for_other_workers","combined":len(combined)}),flush=True);return
        results=[combined[x["clean"]["canonical_id"]] for x in selection]
    manifest["usage"]={k:sum(r["usage"][k] for r in results) for k in results[0]["usage"]};manifest["elapsed_seconds_this_invocation"]=time.monotonic()-started;manifest["worker_count"]=a.worker_count;export(out,selection,results,manifest)

if __name__=="__main__":main()
