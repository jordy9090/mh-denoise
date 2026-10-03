#!/usr/bin/env python3
"""Backfill infrastructure-exhausted slots in the targeted calibration."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from types import SimpleNamespace
from fullpaper_acl_pipeline import DEFAULT_OUTPUT_DIR, read_jsonl, stable_random_key, write_json, write_jsonl
from run_paired_generator_diagnostic import ExternalJudge, load_repo_env, run_generator
from run_qwen_targeted_axis_calibration import TARGETED_SEED, build_report, prior_ids

def main() -> None:
    load_repo_env(); p=argparse.ArgumentParser(); p.add_argument("--max-input-tokens",type=int,default=4096); a=p.parse_args()
    root=DEFAULT_OUTPUT_DIR; out=root/"qwen_targeted_axis_calibration_25"
    accepted=list(read_jsonl(out/"accepted.jsonl")); failures=list(read_jsonl(out/"failures.jsonl"))
    missing=[r["intended_axes"][0] for r in failures if r.get("failure_class")=="infrastructure"]
    if not missing: return
    canonical={r["canonical_id"]:r for r in read_jsonl(root/"canonical_clean_qa.jsonl")}
    qc=list(read_jsonl(root/"clean_target_qc"/"corpus_qc.jsonl")); qc_by={r["canonical_id"]:r for r in qc if r.get("qc_ok") and r.get("baseline_degraded_axes")==[]}
    used=prior_ids(root)|{r["canonical_id"] for r in accepted+failures}
    pool=sorted((cid for cid in qc_by if cid not in used and canonical[cid]["split"]=="train"),key=lambda cid:stable_random_key(TARGETED_SEED,cid))
    selected=[]
    for slot,(axis,cid) in enumerate(zip(missing,pool,strict=False)):
        selected.append({"slot":slot,"clean":canonical[cid],"intended_axes":(axis,),"axis_count":1,"eligibility":qc_by[cid]})
    if len(selected)!=len(missing): raise RuntimeError("insufficient fresh clean checkpoints for backfill")
    write_json(out/"infrastructure_backfill_selection.json",{"seed":TARGETED_SEED,"rows":[{"canonical_id":x["clean"]["canonical_id"],"intended_axis":x["intended_axes"][0]} for x in selected]})
    judge=ExternalJudge(os.getenv("OPENAI_BASE_URL","https://api.openai.com")); args=SimpleNamespace(batch_size=1,max_input_tokens=a.max_input_tokens)
    new_ok,new_fail,metrics=run_generator("qwen",selected,judge,args,generation_seed_base=TARGETED_SEED)
    write_jsonl(out/"infrastructure_backfill_accepted.jsonl",new_ok); write_jsonl(out/"infrastructure_backfill_failures.jsonl",new_fail); write_json(out/"infrastructure_backfill_metrics.json",metrics)
    combined_ok=accepted+new_ok
    combined_fail=[r for r in failures if r.get("failure_class")!="infrastructure"]+new_fail
    write_jsonl(out/"accepted.jsonl",combined_ok); write_jsonl(out/"failures.jsonl",combined_fail)
    combined_metrics=json.loads((out/"metrics.json").read_text())
    for key,value in metrics.items(): combined_metrics[key]=combined_metrics.get(key,0)+value
    write_json(out/"metrics.json",combined_metrics); build_report(combined_ok,combined_fail,combined_metrics,out/"report.md")
    print(json.dumps({"backfill_accepted":len(new_ok),"backfill_failed":len(new_fail)}))
if __name__=="__main__": main()
