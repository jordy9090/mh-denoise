"""Frozen TRAIN-internal holdout diagnostic, never TEST or retraining."""
import argparse
import collections
import csv
import fcntl
import shutil
import statistics
import subprocess
import sys
import time
import zipfile

from residual_train_common import *
import run_base_residual_experiment as E

PARENT = ROOT
ROOT = Path('/mnt/ssd00/hsoh/denoiser_holdout_checkpoint_diagnostic_20261011_v1')
E.ROOT = ROOT
ARMS = {'epoch_0': CKPT / 'gemma/proposed_mask_off/final',
        **{f'epoch_{i}': PARENT / f'training/epoch_{i}' for i in (1, 2, 3)}}
PATTERNS = {
    'explicit_refusal': r'\b(?:I (?:cannot|can\x27t|am unable to) (?:fulfill|rewrite|refine|proceed)|cannot provide guidance on (?:personal relationships|how to proceed))\b',
    'placeholder': r'\[(?:rest of (?:the )?text|insert[^\]]*|response here)\]',
    'meta_intro': r'^\s*Here is the (?:rewritten|revised|refined) response',
}


def screen(text):
    flags_found = [name for name, pattern in PATTERNS.items() if re.search(pattern, text, re.I)]
    if 'meta_intro' in flags_found and len(text.split()) < 60:
        flags_found.append('possible_meta_only')
    if not text.strip():
        flags_found.append('empty')
    return flags_found


def prepare():
    parent = read(PARENT / 'run_contract.json')
    source = PARENT / 'data/holdout.jsonl'
    data = rows(source)
    fit = rows(PARENT / 'data/train.jsonl')
    assert len(data) == 13 and not {r['canonical_id'] for r in data} & {r['canonical_id'] for r in fit}
    original_train = CKPT / 'gemma/proposed_mask_off/risk_tune_train_enriched.jsonl'
    old_ids = {r['canonical_id'] for r in rows(original_train)}
    contract = {'version': ROOT.name, 'status': 'frozen_before_generation', 'test_used': False,
        'settings': parent['settings'], 'arms': {a: {'path': str(p), 'weights_sha256': file_sha(p/'adapter_model.safetensors')} for a,p in ARMS.items()},
        'input_source': str(source), 'input_file_sha256': file_sha(source),
        'canonical_ids': [r['canonical_id'] for r in data],
        'original_denoiser_train_overlap_ids': [r['canonical_id'] for r in data if r['canonical_id'] in old_ids],
        'purpose': 'checkpoint behavior diagnosis; not an independent generalization evaluation',
        'generation_per_question_per_checkpoint': 1, 'generation_total': 52,
        'no_training_or_selector': True, 'no_test100_evaluation': True,
        'first_prefix_tokens': 16, 'pattern_screening_not_human_gold': PATTERNS,
        'loss_mode': 'same NF4 model preparation as training, target-only teacher-forced NLL; no optimizer',
        'generation_mode': 'same inference loader and frozen decoding as previous VALID run; adapter replacement only',
        'judge': {'model': MODEL, 'temperature': 0, 'max_tokens': 384, 'rubric': 'frozen expert pointwise; Medical Advice categorical'},
        'paid_evaluation_authorized': True, 'failure_focused_train70': 'design after diagnostic; do not train on these holdout answers'}
    if (ROOT/'run_contract.json').exists():
        assert read(ROOT/'run_contract.json') == contract
    else:
        write(ROOT/'run_contract.json', contract)
    target = ROOT/'inputs/holdout13.jsonl'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists(): assert file_sha(target) == file_sha(source)
    else: shutil.copy2(source, target)
    code = ROOT/'code'; code.mkdir(exist_ok=True)
    destination = code/Path(__file__).name
    if not destination.exists(): shutil.copy2(__file__, destination)
    print(json.dumps({'prepared':True,'questions':13,'generations':52,'historical_train_overlap':len(contract['original_denoiser_train_overlap_ids'])}),flush=True)


def prepared_inputs():
    from fullpaper_backbone_utils import load_fullpaper_tokenizer
    from selective_risk_refinement_utils import build_risk_tune_prompt, configure_risk_contract
    configure_risk_contract('fullpaper_v1')
    tok = load_fullpaper_tokenizer(str(TOKENIZER))
    prior = {r['canonical_id']:r for r in rows(PARENT/'training/tokenization_audit.jsonl') if r['split']=='holdout'}
    data=[]
    for r in rows(ROOT/'inputs/holdout13.jsonl'):
        prompt=build_risk_tune_prompt(tok,r)
        ids=tok(prompt,add_special_tokens=True,truncation=False)['input_ids']
        target_ids=tok(r['safe_response'],add_special_tokens=False,truncation=False)['input_ids']+[tok.eos_token_id]
        assert sha(prompt)==prior[r['canonical_id']]['rendered_prompt_sha256']
        assert sha(ids)==prior[r['canonical_id']]['input_token_ids_sha256']
        assert sha(target_ids)==prior[r['canonical_id']]['target_token_ids_sha256']
        assert len(ids)<=4096 and len(target_ids)<=1024
        data.append({**r,'rendered_prompt':prompt,'prompt_ids':ids,'target_ids':target_ids,
                     'input_sha256':sha(r),'input_token_ids_sha256':sha(ids)})
    return tok,data


def replace_adapter(model,path):
    import torch
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    assert len(model.peft_config)==1
    tensors=load_file(str(path/'adapter_model.safetensors'))
    assert set(tensors)==set(get_peft_model_state_dict(model))
    set_peft_model_state_dict(model,tensors)
    current=get_peft_model_state_dict(model)
    assert all(torch.equal(current[k].detach().cpu(),v.to(current[k].dtype)) for k,v in tensors.items())


def loss():
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    from train_gemma_risk_tune_from_sft import load_base_model
    output=ROOT/'results/loss_by_case.jsonl'
    existing={(r['arm'],r['canonical_id']) for r in rows(output)}
    if len(existing)==52:return
    gpu_idle()
    tok,data=prepared_inputs()
    torch.manual_seed(SEED)
    base=load_base_model(str(BASE),use_4bit=True)
    base.config.use_cache=False
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True)
    c=read(TOKENIZER/'adapter_config.json')
    model=get_peft_model(base,LoraConfig(r=c['r'],lora_alpha=c['lora_alpha'],lora_dropout=c['lora_dropout'],
        target_modules=c['target_modules'],bias='none',task_type='CAUSAL_LM',init_lora_weights=True))
    model.eval(); device=next(model.parameters()).device
    for arm,path in ARMS.items():
        replace_adapter(model,path)
        for r in data:
            if (arm,r['canonical_id']) in existing:continue
            ids=torch.tensor([r['prompt_ids']+r['target_ids']],device=device)
            with torch.no_grad():
                logits=model(input_ids=ids,attention_mask=torch.ones_like(ids)).logits
                start=len(r['prompt_ids'])-1
                scores=logits[0,start:-1].float()
                target=torch.tensor(r['target_ids'],device=device)
                losses=F.cross_entropy(scores,target,reduction='none')
                ranks=(scores[:16]>scores[:16].gather(1,target[:16,None])).sum(dim=1)+1
            value={'arm':arm,'canonical_id':r['canonical_id'],'decision':r['decision'],
                'target_tokens':len(r['target_ids']),'loss_sum':losses.sum().item(),'nll':losses.mean().item(),
                'first16_nll':losses[:16].mean().item(),'first_token_nll':losses[0].item(),
                'first16_target_ranks':ranks.tolist(),'target_token_losses':losses.tolist(),
                'target_first16_text':tok.decode(r['target_ids'][:16]),'input_sha256':r['input_sha256']}
            append(output,value)
            print(json.dumps({'stage':'loss','arm':arm,'canonical_id':r['canonical_id'],'nll':value['nll']}),flush=True)
            del logits,scores,losses,ids
    summary={}
    expected={'epoch_0':read(PARENT/'training/initialization_audit.json')['initial_holdout_nll'],
        **{f"epoch_{r['epoch']}":r['holdout_target_token_nll'] for r in rows(PARENT/'training/holdout_log.jsonl')}}
    for arm in ARMS:
        rr=[r for r in rows(output) if r['arm']==arm]
        value=sum(r['loss_sum'] for r in rr)/sum(r['target_tokens'] for r in rr)
        summary[arm]={'recomputed_nll':value,'saved_training_nll':expected[arm],
            'absolute_difference':abs(value-expected[arm]),
            'first16_macro_nll':statistics.mean(r['first16_nll'] for r in rr),
            'groups':{g:{'N':sum(r['decision']==g for r in rr),
                'nll':sum(r['loss_sum'] for r in rr if r['decision']==g)/sum(r['target_tokens'] for r in rr if r['decision']==g)} for g in ('edit','preserve')}}
    write(ROOT/'loss_reproduction.json',summary)
    assert all(v['absolute_difference']<1e-4 for v in summary.values()), 'Loss reproduction mismatch; investigate before generation'


def generate():
    import torch
    from peft import PeftModel
    from transformers import LogitsProcessor,LogitsProcessorList
    from run_gemma_selective_risk_refinement import load_base_model
    from selective_risk_refinement_utils import resolve_generation_eos_token_ids
    output=ROOT/'generation/holdout_checkpoints.jsonl'
    history={(r['arm'],r['canonical_id']):r for r in rows(output)}
    if len(history)==52:return
    gpu_idle(); tok,data=prepared_inputs(); tok.padding_side='left'
    torch.manual_seed(SEED)
    model=PeftModel.from_pretrained(load_base_model(str(BASE),use_4bit=True),str(ARMS['epoch_0']))
    model.eval(); device=next(model.parameters()).device
    eos,eos_source=resolve_generation_eos_token_ids(model,tok)

    class FirstStep(LogitsProcessor):
        def __init__(self,reference): self.reference=reference; self.record=None
        def __call__(self,input_ids,scores):
            if self.record is None:
                s=scores[0].float(); lp=torch.log_softmax(s,dim=-1); values,indices=torch.topk(lp,5)
                self.record={'target_first_token':self.reference,'target_first_token_rank':int((s>s[self.reference]).sum())+1,
                    'target_first_token_logprob':float(lp[self.reference]),
                    'top5':[{'token_id':int(i),'text':tok.decode([int(i)]),'logprob':float(v)} for i,v in zip(indices,values)]}
            return scores

    started=time.monotonic(); completed=0
    for arm,path in ARMS.items():
        replace_adapter(model,path)
        for r in data:
            key=(arm,r['canonical_id'])
            if key in history:
                assert history[key]['input_sha256']==r['input_sha256'];continue
            encoded=tok([r['rendered_prompt']],return_tensors='pt',padding=True,truncation=True,max_length=4096).to(device)
            assert encoded['input_ids'][0].tolist()==r['prompt_ids']
            recorder=FirstStep(r['target_ids'][0])
            with torch.inference_mode():
                seq=model.generate(**encoded,max_new_tokens=1536,do_sample=False,repetition_penalty=1.15,
                    no_repeat_ngram_size=4,pad_token_id=tok.pad_token_id,eos_token_id=eos,use_cache=True,
                    logits_processor=LogitsProcessorList([recorder]))
            ids=seq[0,encoded['input_ids'].shape[-1]:].tolist()
            stop=next((i for i,t in enumerate(ids) if t in eos),None)
            if stop is not None:ids=ids[:stop+1]
            raw=tok.decode(ids,skip_special_tokens=True); response=extract(raw)
            value={'arm':arm,'canonical_id':r['canonical_id'],'question':r['question'],'decision':r['decision'],
                'input_sha256':r['input_sha256'],'response_raw':raw,'response':response,'response_sha256':sha(response),
                'adapter':str(path),'adapter_weights_sha256':file_sha(path/'adapter_model.safetensors'),
                'rendered_prompt':r['rendered_prompt'],'rendered_prompt_sha256':sha(r['rendered_prompt']),
                'input_token_ids_sha256':r['input_token_ids_sha256'],'input_tokens':len(r['prompt_ids']),
                'generated_token_ids':ids,'eos_reached':stop is not None,'eos_ids':eos,'eos_source':eos_source,
                'screen_flags':screen(response),'screen_is_not_final_manual_judgment':True,
                'technical_valid':bool(response.strip()) and stop is not None,'first_step_after_frozen_logits_processors':recorder.record}
            append(output,value);history[key]=value;completed+=1
            elapsed=time.monotonic()-started
            progress={'stage':'generate','completed':len(history),'total':52,'arm':arm,
                'rows_per_minute':completed/elapsed*60,'eta_minutes':(52-len(history))*elapsed/completed/60}
            write(ROOT/'generation/progress.json',progress);print(json.dumps(progress),flush=True)


def judge():
    data=rows(ROOT/'inputs/holdout13.jsonl'); gen=rows(ROOT/'generation/holdout_checkpoints.jsonl');assert len(gen)==52
    assignments=[{k:r[k] for k in ('canonical_id','question','response','arm')} for r in gen]
    assignments += [{'canonical_id':r['canonical_id'],'question':r['question'],'response':r['sft_response'],'arm':'BaseDraft'} for r in data]
    prompts=read(ROOT/'run_contract.json')['settings']['frozen_evaluation_prompts']
    requests=[]
    for r in assignments:
        req=E.request('holdout_pointwise',prompts['pointwise_system'],prompts['pointwise_user_template'].format(question=r['question'],response=r['response']),
            'pointwise_expert_rubric',384,{'question_response_sha256':sha([r['question'],r['response']])})
        r['request_id']=req['request_id'];requests.append(req)
    write_rows(ROOT/'inputs/judge_assignments.jsonl',assignments)
    E.execute_requests('holdout_pointwise',requests)


def aggregate():
    gen=rows(ROOT/'generation/holdout_checkpoints.jsonl');losses=rows(ROOT/'results/loss_by_case.jsonl')
    judges={r['request_id']:r for r in rows(ROOT/'results/holdout_pointwise.jsonl')}
    assignment={(r['arm'],r['canonical_id']):r for r in rows(ROOT/'inputs/judge_assignments.jsonl')}
    loss_idx={(r['arm'],r['canonical_id']):r for r in losses}
    cases=[]
    for r in gen:
        result=judges[assignment[r['arm'],r['canonical_id']]['request_id']]
        cases.append({**r,'teacher_forced':loss_idx[r['arm'],r['canonical_id']],
            'judge_ok':result['ok'],'judge':result.get('normalized'),
            'quality_3axis_mean':E.q3(result['normalized']) if result['ok'] else None})
    write_rows(ROOT/'results/combined_cases.jsonl',cases)
    summary=[]
    for arm in ['BaseDraft',*ARMS]:
        aa=[r for r in assignment.values() if r['arm']==arm]
        jj=[judges[r['request_id']] for r in aa];good=[r for r in jj if r['ok']]
        gg=[r for r in gen if r['arm']==arm]
        ll=[r for r in losses if r['arm']==arm]
        summary.append({'arm':arm,'N':len(good),'failed_judges':len(jj)-len(good),
            'quality_3axis_mean':statistics.mean(E.q3(r['normalized']) for r in good) if good else None,
            'medical_advice':dict(collections.Counter(r['normalized']['medical_advice'] for r in good)),
            'nll':sum(r['loss_sum'] for r in ll)/sum(r['target_tokens'] for r in ll) if ll else None,
            'first16_nll':statistics.mean(r['first16_nll'] for r in ll) if ll else None,
            'screen_flags':dict(collections.Counter(t for r in gg for t in r['screen_flags'])),
            'flagged_questions':sum(bool(r['screen_flags']) for r in gg),
            'generation_technical_failures':sum(not r['technical_valid'] for r in gg)})
    transitions=[]
    idx={(r['arm'],r['canonical_id']):r for r in cases}
    for r in rows(ROOT/'inputs/holdout13.jsonl'):
        cid=r['canonical_id'];old=idx['epoch_0',cid];last=idx['epoch_3',cid]
        transitions.append({'canonical_id':cid,'decision':r['decision'],'nll_change':last['teacher_forced']['nll']-old['teacher_forced']['nll'],
            'first16_nll_change':last['teacher_forced']['first16_nll']-old['teacher_forced']['first16_nll'],
            'quality_delta':last['quality_3axis_mean']-old['quality_3axis_mean'] if last['judge_ok'] and old['judge_ok'] else None,
            'response_unchanged':last['response']==old['response'],
            'epoch0_flags':old['screen_flags'],'epoch3_flags':last['screen_flags']})
    write(ROOT/'results/summary.json',{'summary':summary,'epoch0_to_epoch3':transitions,'screen_is_not_human_annotation':True,
        'independent_validation':False,'historical_train_overlap':13,'test_used':False})
    with (ROOT/'results/summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(json.dumps({'summary':summary,'transitions':transitions}),flush=True)


def package():
    summary=read(ROOT/'results/summary.json')
    lines=['# TRAIN-internal Holdout Checkpoint Diagnostic','','No TEST access or new training. These 13 questions were excluded from TRAIN70 updates but were in the original denoiser TRAIN276. This is not independent validation.','',
        'The frozen pointwise rubric averages Overall Quality, Empathy and Specificity. Medical Advice is categorical and excluded.',
        'Pattern flags are screening indicators, not comprehensive manual annotations. Technical EOS success does not imply a valid counseling answer.',
        'Training-mode teacher-forced loss and inference-mode generation use their respective original loaders; the loss reproduction audit is included.',
        'A logits observer records first-token ranks after frozen generation processors without modifying scores. This may help diagnose behavior, but does not establish a causal explanation.','',
        '## Summary','```json',json.dumps(summary,indent=2),'```','',
        '## Next Step','Review failures before freezing a new TRAIN-only failure-focused dataset. Do not use these holdout answers or any VALID/TEST answers as training targets. No follow-up training has been launched automatically.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    case_lines=['# Verbatim Checkpoint Responses','','Generated research outputs, not endorsed counseling advice.']
    cases=rows(ROOT/'results/combined_cases.jsonl')
    for source in rows(ROOT/'inputs/holdout13.jsonl'):
        case_lines+=['','## '+source['canonical_id'],'',source['question'],'','### Frozen Target','',source['safe_response']]
        for r in cases:
            if r['canonical_id']==source['canonical_id']:
                case_lines+=['','### '+r['arm'],f"NLL={r['teacher_forced']['nll']:.6f}; Q3={r['quality_3axis_mean']}; screen={r['screen_flags']}",'',r['response']]
    (ROOT/'CASE_COMPARISONS.md').write_text('\n'.join(case_lines)+'\n',encoding='utf-8')
    contract=read(ROOT/'run_contract.json')
    assert file_sha(Path(contract['input_source']))==contract['input_file_sha256']
    for arm,info in contract['arms'].items():assert file_sha(Path(info['path'])/'adapter_model.safetensors')==info['weights_sha256']
    api=rows(ROOT/'results/holdout_pointwise.jsonl')
    write(ROOT/'api_usage.json',{'unique_requests':len({r['request_id'] for r in api}),
        'prompt_tokens':sum(r.get('usage',{}).get('prompt_tokens',0) for r in api),
        'completion_tokens':sum(r.get('usage',{}).get('completion_tokens',0) for r in api),
        'estimated_cost_usd':sum(r.get('estimated_cost_usd',0) for r in api)})
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix not in {'.zip','.log','.lock'}
        and not any(x in {'runtime_cache','__pycache__'} for x in p.relative_to(ROOT).parts)
        and p.name not in {'progress.json','ARTIFACT_MANIFEST.json','download_copy_status.json'}]
    write(ROOT/'ARTIFACT_MANIFEST.json',{'files':[{'path':str(p.relative_to(ROOT)),'sha256':file_sha(p)} for p in files],
        'test_used':False,'new_training':False,'original_weights_verified_unchanged':True})
    archive=ROOT/'denoiser_holdout_checkpoint_diagnostic.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[ROOT/'ARTIFACT_MANIFEST.json']:z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    dest=Path('/home/user/Downloads')/archive.name
    if shutil.disk_usage(dest.parent).free > 2*archive.stat().st_size:
        shutil.copy2(archive,dest);write(ROOT/'download_copy_status.json',{'copied':True,'path':str(dest)})
    else:write(ROOT/'download_copy_status.json',{'copied':False,'source':str(archive),'reason':'insufficient disk space'})
    print(json.dumps({'zip':str(archive),'sha256':file_sha(archive)}),flush=True)


def pipeline():
    ROOT.mkdir(parents=True,exist_ok=True)
    for key,name in [('TMPDIR','tmp'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','torchinductor'),('CUDA_CACHE_PATH','cuda')]:
        path=ROOT/'runtime_cache'/name;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    with (ROOT/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for stage in ('prepare','loss','generate','judge','aggregate','package'):
            write(ROOT/'progress.json',{'status':'running','stage':stage})
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),stage],cwd=CODE.parent)
            if result.returncode:
                write(ROOT/'progress.json',{'status':'failed','stage':stage,'returncode':result.returncode,
                    'resume':f'{sys.executable} {Path(__file__).resolve()} pipeline'})
                raise SystemExit(result.returncode)
        write(ROOT/'progress.json',{'status':'complete','test_used':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','pipeline','loss','generate','judge','aggregate','package'])
    globals()[parser.parse_args().stage]()
