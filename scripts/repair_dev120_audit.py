#!/usr/bin/env python3
"""Create a separate corrected dev120 export from the supplied artifact audit."""
from __future__ import annotations
import hashlib,json
from collections import Counter
from pathlib import Path
from typing import Any
from fullpaper_acl_pipeline import DEFAULT_OUTPUT_DIR,read_jsonl,sha256_file,write_json,write_jsonl
from source_integrity_contract import VERSION,MEDICAL_BOUNDARY_DEFINITION,contract_hash,exact_offset,surface_flags

ORIGINAL=DEFAULT_OUTPUT_DIR/"development_training_shard_120"
OUTPUT=DEFAULT_OUTPUT_DIR/"development_training_shard_120_corrected_v2"
AUDIT_SHA256={"accepted.jsonl":"0048c8aed0087fbb21bde0aa71afdf591bbd47990d821ae0ae0eb2f8779fc65b","rejected.jsonl":"e965227eff5ffdad3ae5dc530b830b739f2ee7ac5a17d0b747e65c7063af8e00","qc_conflicts.jsonl":"804cb0d5fa5bbf1e05a9090219b52e86c0da55009ec44e07b3aa8a146564da0c","train_sft.jsonl":"bd2574a10d401e49e7e60c7e04be25ae7b186b532f39a569d7fe0ca76aa7474b","train_dpo.jsonl":"c7c3d5d9326df850b0aed16d2fdedff62c3f69f42edd5ca971e73ff47afb0d1a"}
PRIMARY={
"qa_f0f83e96bf6b817ce9fa0388":("invalid_clean_target:non_response_meta","This revised dialogue offers a more nuanced and detailed exploration"),
"qa_02c5c7930043e435df15f72a":("invalid_clean_target:unsupported_specific_history","which you've used effectively in the past for product development"),
"qa_196e91a8eec212a56fad8a02":("invalid_medical_boundary_label:ordinary_career_directiveness","immediately transition to a new career path"),
"qa_8ffdc9bdfe09fe2dd4ef8421":("invalid_clean_target:false_human_identity","I'm here to offer support as a fellow human"),
"qa_1b27ea55eb5127d9e9db6df9":("invalid_clean_target:unsupported_loss_and_administrative_history","cope with your loss and manage the administrative tasks that come with it"),
"qa_20c237c589046c170a238661":("invalid_clean_target:non_response_meta","In this therapeutic intervention, the counselor employs"),
"qa_fcdf2c83ff4c39fe75a67697":("invalid_clean_target:unsupported_cancer_diagnosis","comes with a cancer diagnosis"),
"qa_dd3dff18f907910e17a420bb":("invalid_clean_target:non_response_meta;invalid_medical_boundary_label:ordinary_relationship_directiveness","In this therapeutic approach, the counselor acknowledges"),
"qa_7d493d5c6ae2004b0ce18881":("invalid_clean_target:unsupported_treatment_history","Continuing to attend grief counseling sessions"),
"qa_414bea1bddae5c46c6383fff":("invalid_candidate:speaker_switch","I feel helpless and frustrated, as I want to support him but don't know how to effectively encourage him to make these changes."),
}
SECONDARY={
"qa_6b8ae913d9c9e383e83579ca":("invalid_clean_target:false_personal_experience_identity","my suggestions are based on general knowledge and personal experiences"),
"qa_10e33c79df57d043709c0dff":("invalid_clean_target:false_professional_identity_meta","As a counselor, my role is to provide you with a compassionate, ethical, and professional framework for healing."),
"qa_351b201d6b0af35928c862f5":("invalid_clean_target:serial_non_response_meta","Stay tuned for more insights and strategies to help you navigate these complexities."),
}

def verified_spans(row:dict[str,Any])->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    verified=[];excluded=[]
    for side,source in (("clean",row["clean_response"]),("candidate",row["corrupted_response"])):
        scores=row["final_grade"][f"{side}_scores"]
        for axis,item in scores.items():
            evidence=str(item.get("evidence_span") or "")
            if not evidence:continue
            offset=exact_offset(source,evidence)
            base={"canonical_id":row["canonical_id"],"axis":axis,"side":side,"reported_evidence":evidence,"reported_validation":item.get("evidence_validation")}
            if offset:verified.append({**base,**offset,"source_sha256":hashlib.sha256(source.encode()).hexdigest()})
            else:excluded.append({**base,"reason":"no_exact_source_verified_character_offset"})
    # Evidence already rejected by the original validator remains raw evidence only.
    # Never promote it to span supervision, even if a later heuristic could recover
    # an approximate match.
    for item in row["final_grade"].get("invalid_evidence", []):
        excluded.append({
            "canonical_id": row["canonical_id"],
            "axis": item.get("axis"),
            "side": "raw_label_" + str(item.get("response_label", "unknown")),
            "reported_evidence": item.get("invalid_span", ""),
            "reported_validation": item.get("reason", "invalid_evidence"),
            "reason": "original_validator_rejected_non_verbatim_evidence",
        })
    return verified,excluded

def main()->None:
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/"repair_manifest.json").exists():raise RuntimeError("corrected export already exists; refusing overwrite")
    for name,expected in AUDIT_SHA256.items():
        actual=sha256_file(ORIGINAL/name)
        if actual!=expected:raise RuntimeError(f"original artifact changed: {name}")
    accepted=list(read_jsonl(ORIGINAL/"accepted.jsonl"));by={r["canonical_id"]:r for r in accepted}
    if set(PRIMARY|SECONDARY)-set(by):raise RuntimeError("audit IDs missing from original accepted data")
    held=[]
    for category,items in (("primary",PRIMARY),("secondary",SECONDARY)):
        for cid,(reason,quote) in items.items():
            row=by[cid];locations=[]
            for field in ("question","clean_response","corrupted_response"):
                offset=exact_offset(row[field],quote)
                if offset:locations.append({"field":field,**offset})
            if not locations:raise RuntimeError(f"audit evidence not found: {cid}")
            held.append({"canonical_id":cid,"audit_category":category,"hold_reason":reason,"evidence_quote":quote,"verified_locations":locations,"intended_axes":row["intended_axes"],"raw_realized_axes_preserved":row["realized_axes"],"raw_final_grade_preserved":row["final_grade"],"surface_flags":{"clean":surface_flags(row["question"],row["clean_response"],"clean"),"candidate":surface_flags(row["question"],row["corrupted_response"],"candidate")}})
    held_ids={r["canonical_id"] for r in held};retained=[r for r in accepted if r["canonical_id"] not in held_ids]
    original_sft={r["metadata"]["canonical_id"]:r for r in read_jsonl(ORIGINAL/"train_sft.jsonl")};original_dpo={r["metadata"]["canonical_id"]:r for r in read_jsonl(ORIGINAL/"train_dpo.jsonl")}
    sft=[];dpo=[];spans=[];excluded_spans=[]
    for row in retained:
        valid,bad=verified_spans(row);spans.extend(valid);excluded_spans.extend(bad)
        repair_meta={"repair_version":"dev120-corrected-v2","source_integrity_contract":VERSION,"source_integrity_contract_sha256":contract_hash(),"source_verified_span_count":len(valid),"excluded_unverified_evidence_count":len(bad),"raw_paired_qc_preserved":row["final_grade"]}
        s={**original_sft[row["canonical_id"]],"metadata":{**original_sft[row["canonical_id"]]["metadata"],**repair_meta,"span_supervision":valid}}
        d={**original_dpo[row["canonical_id"]],"metadata":{**original_dpo[row["canonical_id"]]["metadata"],**repair_meta,"span_supervision":valid}}
        sft.append(s);dpo.append(d)
    write_jsonl(OUTPUT/"train_sft.jsonl",sft);write_jsonl(OUTPUT/"train_dpo.jsonl",dpo);write_jsonl(OUTPUT/"held_out.jsonl",held);write_jsonl(OUTPUT/"secondary_flags.jsonl",[r for r in held if r["audit_category"]=="secondary"]);write_jsonl(OUTPUT/"source_verified_spans.jsonl",spans);write_jsonl(OUTPUT/"excluded_unverified_evidence.jsonl",excluded_spans)
    original_rejected=len(list(read_jsonl(ORIGINAL/"rejected.jsonl")));original_conflicts=len(list(read_jsonl(ORIGINAL/"qc_conflicts.jsonl")))
    exported_ids={r["metadata"]["canonical_id"] for r in sft+dpo}
    checks={"original_hashes_match_audit":True,"original_120_accounting":len(retained)+len(held)+original_rejected+original_conflicts==120,"primary_secondary_separate":sum(r["audit_category"]=="primary" for r in held)==10 and sum(r["audit_category"]=="secondary" for r in held)==3,"corrected_export_count":len(sft)==len(dpo)==len(retained),"held_absent_both_exports":held_ids.isdisjoint(exported_ids),"inputs_unchanged":all(s["input"]==original_sft[s["metadata"]["canonical_id"]]["input"] for s in sft),"targets_unchanged":all(s["target"]==original_sft[s["metadata"]["canonical_id"]]["target"] for s in sft),"raw_judge_preserved":all(s["metadata"]["raw_paired_qc_preserved"]==by[s["metadata"]["canonical_id"]]["final_grade"] for s in sft),"span_offsets_exact":all(by[x["canonical_id"]]["clean_response" if x["side"]=="clean" else "corrupted_response"][x["start"]:x["end"]]==x["text"] for x in spans)}
    if not all(checks.values()):raise AssertionError(checks)
    reason_counts=Counter(reason for r in held for reason in r["hold_reason"].split(";"))
    artifacts={name:sha256_file(OUTPUT/name) for name in ("train_sft.jsonl","train_dpo.jsonl","held_out.jsonl","secondary_flags.jsonl","source_verified_spans.jsonl","excluded_unverified_evidence.jsonl")}
    manifest={"version":"dev120-corrected-v2","status":"complete","basis":"attached dev120 audit verified against immutable artifacts; no model/API calls","original_input_accounting":{"accepted":70,"rejected":43,"qc_conflicts":7,"total":120},"corrected":{"retained":len(retained),"held_primary":10,"held_secondary":3,"held_total":len(held)},"hold_reason_counts":dict(reason_counts),"medical_boundary_definition":MEDICAL_BOUNDARY_DEFINITION,"source_integrity_contract_sha256":contract_hash(),"original_artifact_sha256":AUDIT_SHA256,"artifact_sha256":artifacts,"source_verified_spans":len(spans),"excluded_unverified_evidence":len(excluded_spans),"checks":checks}
    write_json(OUTPUT/"repair_manifest.json",manifest)
    lines=["# dev120 corrected export v2","",f"Retained: {len(retained)}/70 originally accepted rows. Held: 13 (10 primary, 3 secondary). Original 120 accounting remains 57 corrected-retained + 13 repair-held + 43 original-rejected + 7 original-QC-conflict.","","No source or candidate response, raw paired score, evidence, intended axis, or realized axis was rewritten. Medical-label conflicts were held rather than relabeled.","","## Hold reasons","",*[f"- {k}: {v}" for k,v in sorted(reason_counts.items())],"",f"Source-verified evidence spans with exact offsets: {len(spans)}. Evidence records excluded from span supervision because no exact offset could be verified: {len(excluded_spans)}.","","All secondary flags remain separately recorded in `secondary_flags.jsonl` and are conservatively excluded from both corrected exports.","","## Checks","","```json",json.dumps(checks,indent=2),"```",""]
    (OUTPUT/"repair_report.md").write_text("\n".join(lines),encoding="utf-8")
if __name__=="__main__":main()
