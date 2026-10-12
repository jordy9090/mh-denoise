"""Extend corrected TEST60 to the frozen TEST100 snapshot, without overwrites."""
import argparse
import asyncio
import collections
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

CODE = Path(__file__).resolve().parent
REPO = Path('/home/user/hsoh/mh-denoise')
sys.path.append(str(REPO/'scripts'))
import run_corrected_test60_pointwise_publication as P
import run_test60_rubric_aligned_selection as S
import run_gpt41_checkpoint_queue as Q

ROOT = Path('/mnt/ssd00/hsoh/test100_frozen_inference_judge_20261011_v1')
HANDOFF = Path('/home/user/hsoh/revision2_test_live_handoff_20261008')
OLD = Path('/home/user/hsoh/test60_rubric_aligned_final_selection_20261009_v2_test40_fix')
OLD_EVAL = Path('/home/user/hsoh/test60_corrected_gpt41_pointwise_publication_20261009_v1')
SELECTOR = Path('/home/user/hsoh/rubric_aligned_gpt41_rejudge_20261009_v1')
VALID = Path('/mnt/ssd00/hsoh/continued_denoiser_train_20261011_v1')
RUN = REPO/'outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910'
PART60 = RUN/'gemma/generation_1536_test_surface_relaxed_v1_20261008/part60'
FROZEN_CODE = Path('/home/user/hsoh/mh-denoise-surface-relaxed-v1-20261008/scripts')
MODEL = 'gpt-4.1-2025-04-14'


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    if not Path(p).exists(): return []
    with Path(p).open() as f: return [json.loads(line) for line in f if line.strip()]


def sha(value):
    if not isinstance(value,str): value=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n'); tmp.replace(path)


def write_rows(path,values,frozen=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if frozen and path.exists():
        assert rows(path)==values, ('Frozen artifact differs',str(path)); return
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        for value in values: f.write(json.dumps(value,ensure_ascii=False)+'\n')
        f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def append(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())


Q.read_jsonl=rows
P.read_jsonl=rows


def prepare():
    ready=read(HANDOFF/'READY_FOR_H100_INFERENCE_AND_JUDGE.json'); target=ready['target']
    snapshot=read(HANDOFF/target['snapshot_file'])
    assert target['accepted_count']==snapshot['accepted_count']==100
    assert target['checkpoint_rows']==snapshot['checkpoint_rows']==455
    assert target['snapshot_id']==snapshot['snapshot_id']=='snapshot_accepted100_rows455_20261010T010616+0900'
    verified={}
    for name,digest in snapshot['published_files'].items():
        path=HANDOFF/'snapshots'/name; assert file_sha(path)==digest, str(path); verified[str(path)]=digest
    for key in ('accepted_ids','accepted_rows','checkpoint','frozen_selection','run_contract'):
        path=HANDOFF/target[key+'_file']; assert file_sha(path)==target[key+'_sha256']
    ids=(HANDOFF/target['accepted_ids_file']).read_text().splitlines()
    assert len(ids)==len(set(ids))==100 and ids==snapshot['accepted_ids_frozen_selection_order']
    assert sha('\n'.join(ids[:60])+'\n')==ready['frozen_prefix']['accepted_ids_sha256']
    assert ids[60:]==ready['extension']['ordered_ids']
    assert sha('\n'.join(ids[60:])+'\n')==ready['extension']['ordered_ids_sha256']
    assert len(rows(HANDOFF/target['checkpoint_file']))==455
    accepted=rows(HANDOFF/target['accepted_rows_file']); assert len(accepted)==100
    assert [r['canonical_id'] for r in accepted]==ids
    old=rows(OLD/'test60_final_outputs.jsonl'); assert [r['canonical_id'] for r in old]==ids[:60]
    sc=read(OLD/'run_contract.json'); ec=read(OLD_EVAL/'run_contract.json')
    assert file_sha(OLD/'test60_final_outputs.jsonl')==ec['sources']['corrected_test60']['sha256']
    assert file_sha(CODE/'selective_risk_refinement_utils.py')==sc['extraction']['extractor']['sha256']
    assert file_sha(SELECTOR/'run_contract.json')==sc['selector']['frozen_contract']['sha256']
    staging=ROOT/'snapshot'; staging.mkdir(parents=True,exist_ok=True)
    for dest,key in [('run_contract.json','run_contract_file'),('results_checkpoint.part0.jsonl','checkpoint_file')]:
        path=staging/dest; source=HANDOFF/target[key]
        if path.exists(): assert file_sha(path)==file_sha(source)
        else: shutil.copy2(source,path)
    export=ROOT/'export'
    if not (export/'manifest.json').exists():
        subprocess.run([sys.executable,str(REPO/'scripts/build_frozen_revision2_test_partial_export.py'),
            '--production-dir',str(staging),'--selection-file',str(HANDOFF/target['frozen_selection_file']),
            '--canonical-file',str(REPO/'data/fullpaper_acl_pipeline/canonical_clean_qa.jsonl'),
            '--output-dir',str(export),'--target-count','100'],check=True,cwd=REPO)
    inputs=rows(export/'sft_test.jsonl'); assert [r['canonical_id'] for r in inputs]==ids
    old_input=rows(REPO/'data/fullpaper_acl_pipeline/frozen_revision2_test_partial60_surface_relaxed_v1_20261008/sft_test.jsonl')
    input_hashes=[]
    for i,r in enumerate(inputs):
        material={k:v for k,v in r.items() if k!='metadata'}
        if i<60:
            prior={k:v for k,v in old_input[i].items() if k!='metadata'}
            assert material==prior, ('TEST60 input mismatch',r['canonical_id'])
            assert r['question']==old[i]['question']
        input_hashes.append({'canonical_id':r['canonical_id'],'input_without_checkpoint_metadata_sha256':sha(material)})
    write_rows(ROOT/'inputs/input_hashes.jsonl',input_hashes,True)
    write_rows(ROOT/'inputs/new40.jsonl',inputs[60:],True)
    write_rows(ROOT/'inputs/reused_test60_final.jsonl',old,True)
    settings={method:read(PART60/f'{method}_missing_outputs.manifest.json')['settings'] for method in ('sft','dpo')}
    settings['proposed']=sc['risk_and_generation']['frozen_settings']
    for method,key in [('sft','SFT60'),('dpo','DPO60')]:
        record=sc['restored_inputs'][key]; assert file_sha(record['path'])==record['sha256']
        prior=rows(record['path']); assert [r['canonical_id'] for r in prior]==ids[:60]
        assert all(r['sft_response']==o[method+'_response'] for r,o in zip(prior,old))
        write_rows(ROOT/f'generation/reused_{method}60.jsonl',prior,True)
    protected={str(OLD/'test60_final_outputs.jsonl'):file_sha(OLD/'test60_final_outputs.jsonl'),
        str(OLD/'run_contract.json'):file_sha(OLD/'run_contract.json'),
        str(OLD_EVAL/'run_contract.json'):file_sha(OLD_EVAL/'run_contract.json'),
        str(SELECTOR/'run_contract.json'):file_sha(SELECTOR/'run_contract.json')}
    for item in ec['risk_scoring'].values():
        if isinstance(item,dict) and 'path' in item and 'sha256' in item:
            assert file_sha(item['path'])==item['sha256']; protected[item['path']]=item['sha256']
    for method in ('sft','dpo','proposed_mask_off'):
        path=RUN/f'gemma/{method}/final/adapter_model.safetensors'; protected[str(path)]=file_sha(path)
    contract={'version':ROOT.name,'snapshot':target,'published_hashes_verified':verified,'ordered_ids':ids,
        'reused':60,'new_inference':40,'inference_settings':settings,'selector_contract_sha256':file_sha(SELECTOR/'run_contract.json'),
        'selector':read(SELECTOR/'run_contract.json'),'pointwise':ec['model_settings'],'pointwise_prompts':ec['prompts'],
        'risk_scoring':ec['risk_scoring'],'protected_source_hashes':protected,
        'code_hashes':{n:file_sha(CODE/n) for n in ('run_test100.py','build_sft_outputs_for_risk_tuning.py','run_gemma_selective_risk_refinement.py','selective_risk_refinement_utils.py','fullpaper_backbone_utils.py')},
        'generation_change':'persistence/resume only; frozen extractor, model inputs and decoding unchanged',
        'paid_api_authorized':True,'no_training_or_valid_arm_substitution':True,'dpo_response_field':'sft_response'}
    path=ROOT/'run_contract.json'
    if path.exists():
        prior=read(path)
        if prior['code_hashes']!=contract['code_hashes']:
            amendment=read(ROOT/'implementation_amendment.json')
            assert prior['code_hashes']==amendment['original_code_hashes']
            assert contract['code_hashes']==amendment['current_code_hashes']
            contract['code_hashes']=prior['code_hashes']
        assert prior==contract,'Frozen contract differs'
    else: write(path,contract)
    for name in contract['code_hashes']:
        dest=ROOT/'code'/name; dest.parent.mkdir(exist_ok=True)
        if not dest.exists(): shutil.copy2(CODE/name,dest)
    write(ROOT/'verification.json',{'snapshot_hashes':len(verified),'accepted':100,'checkpoint_rows':455,'prefix_exact':60,'new':40})
    print(json.dumps(read(ROOT/'verification.json')),flush=True)


def wait_gpu():
    lock=VALID/'pipeline.lock'
    if lock.exists():
        print('Waiting for existing VALID pipeline lock; no signals sent',flush=True)
        with lock.open('r') as f:
            fcntl.flock(f,fcntl.LOCK_EX);fcntl.flock(f,fcntl.LOCK_UN)
    while True:
        result=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        other=[x for x in result.split() if x!=str(os.getpid())]
        if not other: return
        print(json.dumps({'waiting_for_gpu_pids':other}),flush=True);time.sleep(30)


def inference(method):
    contract=read(ROOT/'run_contract.json')
    output=ROOT/f'generation/{method}_new40.jsonl'
    if len(rows(output))==40 and output.with_suffix('.manifest.json').exists(): return
    wait_gpu()
    settings=dict(contract['inference_settings'][method])
    settings['input']=str(ROOT/('generation/sft_new40.jsonl' if method=='proposed' else 'inputs/new40.jsonl'))
    settings['output']=str(output)
    script='run_gemma_selective_risk_refinement.py' if method=='proposed' else 'build_sft_outputs_for_risk_tuning.py'
    command=[sys.executable,str(CODE/script)]
    for key,value in settings.items():
        if value is None or value is False: continue
        command.append('--'+key)
        if value is not True: command.append(str(value))
    write(ROOT/f'commands/{method}.json',{'argv':command,'frozen_settings':settings})
    subprocess.run(command,check=True,cwd=CODE.parent)
    generated=rows(output);assert len(generated)==40
    assert [r['canonical_id'] for r in generated]==contract['ordered_ids'][60:]


def request(name,system,user,parser,tokens,metadata):
    material={'model':MODEL,'temperature':0,'max_tokens':tokens,'response_format':{'type':'json_object'},
        'parser':parser,'system_prompt':system,'user_prompt':user}
    return {**material,'request_id':sha(material)[:32],'queue':name,'version':ROOT.name,'metadata':metadata,
        'system_prompt_sha256':sha(system),'user_prompt_sha256':sha(user)}


def run_queue(name,requests):
    unique={r['request_id']:r for r in requests}; requests=list(unique.values())
    output=ROOT/f'results/{name}.jsonl'
    write_rows(ROOT/f'requests/{name}.jsonl',requests,True)
    existing={r['request_id']:r for r in rows(output)}
    for r in requests:
        if r['request_id'] in existing: assert Q.request_sha256(r)==existing[r['request_id']]['request_sha256']
    pending=[r for r in requests if r['request_id'] not in existing]
    if pending:
        pending_path=ROOT/f'requests/{name}_pending.jsonl';write_rows(pending_path,pending)
        asyncio.run(Q.async_main(SimpleNamespace(requests=pending_path,output=output,queue_name=name,
            initial_concurrency=4,max_concurrency=8,max_retries=5,timeout=120,validate_only=False)))
    return {r['request_id']:r for r in rows(output)}


def select():
    contract=read(ROOT/'run_contract.json'); prompts=contract['selector']['prompts']
    sft=rows(ROOT/'generation/sft_new40.jsonl'); dpo=rows(ROOT/'generation/dpo_new40.jsonl'); proposed=rows(ROOT/'generation/proposed_new40.jsonl')
    assert len(sft)==len(dpo)==len(proposed)==40
    requests=[]; lookup={}
    for a,b in zip(sft,proposed):
        assert a['canonical_id']==b['canonical_id']
        if not b['used_denoiser'] or not str(b.get('denoiser_response') or '').strip(): continue
        for order in (0,1):
            x,y=(a['sft_response'],b['denoiser_response']) if order==0 else (b['denoiser_response'],a['sft_response'])
            r=request('selector',prompts['selector_system'],prompts['selector_user_template'].format(question=a['question'],response_a=x,response_b=y),
                'blind_selector',256,{'canonical_id':a['canonical_id'],'order_index':order,
                    'variant_for_A':'sft' if order==0 else 'candidate','variant_for_B':'candidate' if order==0 else 'sft'})
            requests.append(r);lookup[(a['canonical_id'],order)]=r
    results=run_queue('selector',requests)
    final=rows(ROOT/'inputs/reused_test60_final.jsonl')
    for index,(a,d,b) in enumerate(zip(sft,dpo,proposed),60):
        cid=a['canonical_id']; assert cid==d['canonical_id']==b['canonical_id']
        accepted=False; orders=[]; choices=[]
        if not b['used_denoiser']: reason='gate_not_called'
        elif not str(b.get('denoiser_response') or '').strip(): reason='candidate_generation_failure'
        else:
            for order in (0,1):
                req=lookup[(cid,order)]; result=results.get(req['request_id']); ok,error=S.result_status(req,result)
                if result and result.get('returned_model')!=MODEL: ok,error=False,'returned_model_mismatch'
                item={'order_index':order,'request_id':req['request_id'],'ok':ok,'error':error}
                if ok:
                    choice=result['normalized']['choice']; mapped='tie' if choice=='tie' else req['metadata']['variant_for_'+choice]
                    choices.append(mapped);item.update(raw_choice=choice,mapped_choice=mapped,rationale=result['normalized'].get('rationale',''))
                orders.append(item)
            if any(not r['ok'] for r in orders): reason='selector_technical_failure'
            elif choices==['candidate','candidate']: accepted=True;reason='candidate_selected_both_orders'
            elif 'tie' in choices: reason='selector_tie'
            elif len(set(choices))>1: reason='selector_order_disagreement'
            else: reason='sft_selected_both_orders'
        final.append({'split':'TEST','canonical_id':cid,'case_id':f'TEST-{index+1:02d}','source_order':index,
            'question':a['question'],'sft_response':a['sft_response'],'dpo_response':d['sft_response'],'dpo_source_response_field':'sft_response',
            'denoiser_called':bool(b['used_denoiser']),'denoiser_call_reason':b.get('denoiser_call_reason'),
            'denoiser_response_raw':b.get('denoiser_response_raw',''),'denoiser_response':b.get('denoiser_response',''),
            'denoiser_generation':b.get('denoiser_generation'),'sft_risk_score':b.get('sft_risk_score'),'denoiser_risk_score':b.get('denoiser_risk_score'),
            'current_rules_accepted_denoiser':bool(b['accepted_denoiser']),'current_rules_reject_reason':b.get('reject_reason'),
            'current_rules_final_response':b['final_response'],'gpt_selector_orders':orders,'gpt_selector_accepted_candidate':accepted,
            'gpt_selector_selection_reason':reason,'final_response':b['denoiser_response'] if accepted else a['sft_response'],
            'final_response_source':'original_denoiser_candidate' if accepted else 'SFT'})
    assert [r['canonical_id'] for r in final]==contract['ordered_ids']
    assert final[:60]==rows(OLD/'test60_final_outputs.jsonl')
    write_rows(ROOT/'test100_final_outputs.jsonl',final,True)
    write_rows(ROOT/'inputs/frozen_response_hashes.jsonl',[{'canonical_id':r['canonical_id'],'row_sha256':sha(r),
        **{key+'_sha256':sha(r[key]) for key in ('question','sft_response','dpo_response','final_response')}} for r in final],True)


def judge():
    contract=read(ROOT/'run_contract.json'); prompts=contract['pointwise_prompts']
    assignments,pairs=P.build_assignments(rows(ROOT/'test100_final_outputs.jsonl'))
    for r in assignments: r['scope']='frozen_TEST100_final'
    for pair in pairs:
        for m in pair['members']: m['scope']='frozen_TEST100_final'
    write_rows(ROOT/'inputs/assignments.jsonl',assignments,True);write_rows(ROOT/'inputs/unique_pairs.jsonl',pairs,True)
    cache={}; fields=('model','temperature','max_tokens','response_format','parser','system_prompt','user_prompt')
    for path in [OLD_EVAL/'results/reused_pointwise_results.jsonl',OLD_EVAL/'results/new_pointwise_results.jsonl']:
        for old in rows(path):
            cache.setdefault(sha({k:old.get(k) for k in fields}),[]).append((path,old))
    requests=[]; links=[]; output=ROOT/'results/pointwise.jsonl'; existing={r['request_id'] for r in rows(output)}
    for pair in pairs:
        req=request('pointwise',prompts['system_prompt'],prompts['user_template'].format(question=pair['question'],response=pair['response']),
            'pointwise_expert_rubric',contract['pointwise']['max_tokens'],{'question_response_sha256':pair['question_response_sha256']})
        reused=None
        for path,old in cache.get(sha({k:req[k] for k in fields}),[]):
            valid,_=P.result_is_valid(old,old,pair['response'],require_verbatim_evidence=False)
            if valid:
                req={k:old[k] for k in ('request_id','model','temperature','max_tokens','response_format','parser','system_prompt','user_prompt','metadata')}
                reused=str(path)
                if req['request_id'] not in existing:
                    append(output,{**old,'reused_from':reused,'original_usage':old.get('usage'),'usage':{},'estimated_cost_usd':0})
                    existing.add(req['request_id'])
                break
        requests.append(req);links.append({'question_response_sha256':pair['question_response_sha256'],'request_id':req['request_id'],'reused_from':reused})
    write_rows(ROOT/'inputs/evaluation_links.jsonl',links,True)
    run_queue('pointwise',requests)


def risk():
    from run_gemma_selective_risk_refinement import load_classifier
    from selective_risk_refinement_utils import configure_risk_contract,score_candidate
    output=ROOT/'risk/unique_pair_scorer_risk.jsonl'
    if not output.exists():
        write_rows(output,rows(OLD_EVAL/'risk/unique_pair_scorer_risk.jsonl'))
    history={r['question_response_sha256']:r for r in rows(output)}
    pairs=rows(ROOT/'inputs/unique_pairs.jsonl'); pending=[p for p in pairs if p['question_response_sha256'] not in history]
    if not pending:return
    wait_gpu(); configure_risk_contract('fullpaper_v1')
    rt,router=load_classifier(str(RUN/'router/final'),'cuda'); st,scorer=load_classifier(str(RUN/'scorer/final'),'cuda')
    for p in pending:
        score=score_candidate(p['question'],p['response'],router,rt,scorer,st,'cuda',router_max_len=512,risk_max_len=384,component_mode='full')
        append(output,{**score,'question_response_sha256':p['question_response_sha256'],'risk_contract':'fullpaper_v1','component_mode':'full'})


def report():
    contract=read(ROOT/'run_contract.json'); final=rows(ROOT/'test100_final_outputs.jsonl')
    assert final[:60]==rows(OLD/'test60_final_outputs.jsonl')
    for path,digest in contract['protected_source_hashes'].items():assert file_sha(path)==digest,path
    assignments=rows(ROOT/'inputs/assignments.jsonl');pairs={p['question_response_sha256']:p for p in rows(ROOT/'inputs/unique_pairs.jsonl')}
    links={r['question_response_sha256']:r for r in rows(ROOT/'inputs/evaluation_links.jsonl')}
    reqs={r['request_id']:r for r in rows(ROOT/'requests/pointwise.jsonl')};results={r['request_id']:r for r in rows(ROOT/'results/pointwise.jsonl')}
    risks={r['question_response_sha256']:r['risk_score'] for r in rows(ROOT/'risk/unique_pair_scorer_risk.jsonl')}
    scored=[]; evidence_warnings=[]
    for r in assignments:
        rid=links[r['question_response_sha256']]['request_id'];result=results.get(rid)
        ok,error=P.result_is_valid(result,reqs[rid],r['response'],require_verbatim_evidence=False) if result else (False,'missing_result')
        norm=result['normalized'] if ok else None
        if norm and norm['medical_advice'] in {'yes','unsure'} and norm['medical_advice_evidence'].strip() not in r['response']:
            evidence_warnings.append({'canonical_id':r['canonical_id'],'method':r['method'],'request_id':rid,
                'evidence':norm['medical_advice_evidence'],'warning':'nonverbatim_medical_advice_evidence; retained under original TEST60 aggregation'})
        scored.append({**r,'evaluation_ok':ok,'error':error,'scores':norm,
            'quality_3axis_mean':statistics.mean(norm['scores'].values()) if ok else None,'risk':risks.get(r['question_response_sha256'])})
    write_rows(ROOT/'results/method_assignment_scores.jsonl',scored)
    write_rows(ROOT/'results/medical_advice_evidence_warnings.jsonl',evidence_warnings)
    summaries=[]
    for method in ('SFT','DPO','Proposed'):
        rr=[r for r in scored if r['method']==method];good=[r for r in rr if r['evaluation_ok']];med=collections.Counter(r['scores']['medical_advice'] for r in good)
        rv=[r['risk'] for r in rr if r['risk'] is not None]
        summaries.append({'method':method,'N':len(good),'failed':100-len(good),
            **{a:statistics.mean(r['scores']['scores'][a] for r in good) if good else None for a in ('overall_quality','empathy','specificity')},
            'quality_3axis_mean':statistics.mean(r['quality_3axis_mean'] for r in good) if good else None,
            'medical_yes':med['yes'],'medical_no':med['no'],'medical_unsure':med['unsure'],'medical_yes_denominator':med['yes']+med['no'],
            'medical_yes_rate':med['yes']/(med['yes']+med['no']) if med['yes']+med['no'] else None,
            'mean_risk':statistics.mean(rv) if len(rv)==100 else None,'risk_N':len(rv)})
    base={r['canonical_id']:r for r in scored if r['method']=='SFT' and r['evaluation_ok']}
    deltas=[]
    for r in scored:
        if r['method']!='Proposed' or not r['evaluation_ok'] or r['canonical_id'] not in base:continue
        d=r['quality_3axis_mean']-base[r['canonical_id']]['quality_3axis_mean']
        deltas.append({'canonical_id':r['canonical_id'],'case_id':r['case_id'],'delta':d,'outcome':'win' if d>1e-9 else 'loss' if d< -1e-9 else 'tie'})
    write_rows(ROOT/'results/proposed_vs_sft.jsonl',deltas)
    usage=[r for name in ('selector','pointwise') for r in rows(ROOT/f'results/{name}.jsonl') if not r.get('reused_from')]
    coverage={'questions':100,'assignments':300,'unique_pairs':len(pairs),'reused_unique_evaluations':sum(bool(r['reused_from']) for r in links.values()),
        'new_unique_evaluations':sum(not r['reused_from'] for r in links.values()),
        'failed_unique_evaluations':len({r['question_response_sha256'] for r in scored if not r['evaluation_ok']}),
        'medical_advice_evidence_warnings':len(evidence_warnings)}
    summary={'coverage':coverage,'quality':summaries,'proposed_vs_sft':dict(collections.Counter(r['outcome'] for r in deltas)),'paired_N':len(deltas),
        'denoiser_called':sum(r['denoiser_called'] for r in final),
        'candidate_success':sum(r['denoiser_called'] and bool(r['denoiser_response'].strip()) for r in final),
        'accepted':sum(r['gpt_selector_accepted_candidate'] for r in final),
        'changed_ids':[r['canonical_id'] for r in final if r['final_response']!=r['sft_response']],
        'selector_reasons':dict(collections.Counter(r['gpt_selector_selection_reason'] for r in final)),
        'new_run_api_usage':{'checkpoint_records':len(usage),'input_tokens':sum(r.get('usage',{}).get('prompt_tokens',0) for r in usage),
            'output_tokens':sum(r.get('usage',{}).get('completion_tokens',0) for r in usage),'estimated_cost_usd':sum(r.get('estimated_cost_usd',0) for r in usage)},
        'prefix60_final_rows_unchanged':True,'test15_generation_failure_preserved':True,
        'independence_disclosure':'Selector and evaluator share GPT-4.1; not independent expert validation.'}
    write(ROOT/'completion_summary.json',summary)
    with (ROOT/'results/quality_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    text=['# Frozen TEST100 Results','','Corrected TEST60 reused exactly; only the fixed extension40 generated. No training or VALID-arm substitution.',
        'GPT-4.1-2025-04-14; original corrected rubric; Medical Advice is categorical and excluded from Quality-3.',
        'Selector and evaluator share the same GPT model. This is not independent expert validation.', '',
        '| Method | N | Overall | Empathy | Specificity | Quality-3 | Medical Y/N/U |','|---|---:|---:|---:|---:|---:|---|']
    for r in summaries:
        nums=[f'{r[k]:.4f}' if r[k] is not None else 'NA' for k in ('overall_quality','empathy','specificity','quality_3axis_mean')]
        text.append(f"| {r['method']} | {r['N']} | "+' | '.join(nums)+f" | {r['medical_yes']}/{r['medical_no']}/{r['medical_unsure']} |")
    text+=['','```json',json.dumps(summary,indent=2),'```','',
        'Known TEST-15 generation failure is retained. TEST-01 DPO and TEST-40 use corrected saved outputs. No failed answers were replaced.',
        'New technical generation flags and raw outputs remain in generation artifacts; parse failures are not ties.']
    (ROOT/'REPORT.md').write_text('\n'.join(text)+'\n')
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix not in {'.zip','.lock','.log','.tmp'} and 'runtime_cache' not in p.parts
        and p.name not in {'progress.json','manifest.json','download_copy_status.json'}]
    write(ROOT/'manifest.json',{'status':'complete' if not coverage['failed_unique_evaluations'] else 'complete_with_evaluation_failures',
        'summary':summary,'files':[{'path':str(p.relative_to(ROOT)),'sha256':file_sha(p)} for p in files]})
    archive=ROOT/'TEST100_frozen_inference_judge_research_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[ROOT/'manifest.json']:z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for item in read(ROOT/'manifest.json')['files']: assert hashlib.sha256(z.read(item['path'])).hexdigest()==item['sha256']
    dest=Path('/home/user/Downloads')/archive.name
    copied=shutil.disk_usage(dest.parent).free>2*archive.stat().st_size
    if copied:shutil.copy2(archive,dest)
    write(ROOT/'download_copy_status.json',{'copied':copied,'path':str(dest) if copied else str(archive),'sha256':file_sha(archive)})
    print(json.dumps(summary),flush=True)


def pipeline():
    ROOT.mkdir(parents=True,exist_ok=True)
    for variable,name in [('TMPDIR','tmp'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','torchinductor'),('CUDA_CACHE_PATH','cuda')]:
        p=ROOT/'runtime_cache'/name;p.mkdir(parents=True,exist_ok=True);os.environ[variable]=str(p)
    with (ROOT/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for stage in ('prepare','sft','dpo','proposed','select','judge','risk','report'):
            write(ROOT/'progress.json',{'status':'running','stage':stage})
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),stage],cwd=CODE.parent)
            if result.returncode:
                write(ROOT/'progress.json',{'status':'failed','stage':stage,'returncode':result.returncode,
                    'resume':f'{sys.executable} {Path(__file__).resolve()} pipeline'});raise SystemExit(result.returncode)
        write(ROOT/'progress.json',{'status':'complete','reused60':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','sft','dpo','proposed','select','judge','risk','report','pipeline']);stage=p.parse_args().stage
    if stage in ('sft','dpo','proposed'):inference(stage)
    else:globals()[stage]()
