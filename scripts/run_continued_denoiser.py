"""Initialization-only comparison using frozen TRAIN70 and VALID43 artifacts."""
import argparse
import collections
import csv
import fcntl
import itertools
import shutil
import statistics
import subprocess
import sys
import zipfile

from residual_train_common import *
import run_base_residual_experiment as E

NEW = 'ContinuedDenoiserCandidate'
FINAL = 'ContinuedDenoiserSelected'
FRESH_CAND = 'FreshBaseResidualCandidate'
FRESH_FINAL = 'FreshBaseResidualSelected'
BASELINES = ('BasePromptRewrite', 'BaseSelfRefineCandidate', FRESH_CAND, 'TrainedDenoiserOnBaseCandidate')
MATERIAL = ('model', 'temperature', 'max_tokens', 'response_format', 'parser', 'system_prompt', 'user_prompt')


def prepare():
    source = read(FRESH / 'run_contract.json')
    adapter = CKPT / 'gemma/proposed_mask_off/final'
    a, b = read(TOKENIZER / 'adapter_config.json'), read(adapter / 'adapter_config.json')
    for key in ('r', 'lora_alpha', 'lora_dropout', 'bias', 'task_type', 'modules_to_save',
                'use_dora', 'use_rslora', 'rank_pattern', 'alpha_pattern', 'fan_in_fan_out'):
        assert a.get(key) == b.get(key), ('LoRA config mismatch', key)
    assert set(a['target_modules']) == set(b['target_modules'])
    copied = {}
    for name in ('data/train.jsonl', 'data/holdout.jsonl', 'data/curation_summary.json', 'inputs/valid43.jsonl'):
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert file_sha(target) == file_sha(FRESH / name)
        else:
            shutil.copy2(FRESH / name, target)
        copied[name] = file_sha(target)
    fit, hold, valid = [rows(ROOT / p) for p in ('data/train.jsonl', 'data/holdout.jsonl', 'inputs/valid43.jsonl')]
    assert (len(fit), len(hold), len(valid)) == (70, 13, 43)
    assert not {r['canonical_id'] for r in fit + hold} & {r['canonical_id'] for r in valid}
    settings = json.loads(json.dumps(source['settings']))
    settings['initialization'] = 'saved proposed_mask_off/final weights; sole LoRA; fresh optimizer matching fresh arm'
    contract = {'version': ROOT.name, 'status': 'frozen_before_training', 'settings': settings,
        'parent_contract': str(FRESH / 'run_contract.json'), 'parent_contract_sha256': file_sha(FRESH / 'run_contract.json'),
        'initial_adapter': str(adapter), 'initial_adapter_weights_sha256': file_sha(adapter / 'adapter_model.safetensors'),
        'initial_adapter_config_sha256': file_sha(adapter / 'adapter_config.json'), 'copied_files': copied,
        'train_ids': [r['canonical_id'] for r in fit], 'holdout_ids': [r['canonical_id'] for r in hold],
        'valid_ids': [r['canonical_id'] for r in valid], 'only_training_difference': 'initial adapter weights',
        'paid_api_authorized': True, 'test_access': False,
        'evaluation_scope': 'VALID43 candidates and two-order GPT-selected finals; GPT and auxiliary Llama pointwise',
        'no_teacher_or_target_regeneration': True, 'old_results_read_only': True}
    path = ROOT / 'run_contract.json'
    if path.exists():
        assert read(path) == contract
    else:
        write(path, contract)
    for key in ('training', 'generation', 'pointwise', 'selector', 'frozen_evaluation_prompts'):
        assert settings[key] == source['settings'][key]
    snapshot = ROOT / 'code'
    snapshot.mkdir(exist_ok=True)
    for name in ('run_continued_denoiser.py', 'train_base_residual_adapter.py',
                 'run_base_residual_generation.py', 'residual_train_common.py', 'run_base_residual_experiment.py'):
        destination = snapshot / name
        if destination.exists():
            assert file_sha(destination) == file_sha(CODE / name), 'Code changed after freeze'
        else:
            shutil.copy2(CODE / name, destination)
    write(ROOT / 'preflight.json', {'same_data_hashes': copied, 'settings_equal_except_initialization': True,
        'adapter_structure_equal': True, 'test_used': False, 'train': 70, 'holdout': 13, 'valid': 43})
    print(json.dumps(read(ROOT / 'preflight.json')), flush=True)


def cached_execute(name, requests):
    requested = {r['request_id']: r for r in requests}
    output = ROOT / f'results/{name}.jsonl'
    existing = {r['request_id'] for r in rows(output)}
    for parent in (FRESH, PREVIOUS):
        for path in sorted((parent / 'results').glob('*.jsonl')):
            for old in rows(path):
                if not old.get('ok') or old.get('returned_model') != MODEL or old.get('parser') not in {'pointwise_expert_rubric', 'blind_selector'}:
                    continue
                rid = sha({k: old.get(k) for k in MATERIAL})[:32]
                if rid not in requested or rid in existing:
                    continue
                req = requested[rid]
                assert all(old.get(k) == req[k] for k in MATERIAL)
                append(output, {**old, **req, 'request_sha256': E.queue.request_sha256(req),
                    'reused_from': str(path), 'source_request_id': old['request_id'], 'usage': {}, 'estimated_cost_usd': 0})
                existing.add(rid)
    return E.execute_requests(name, requests)


def evaluate():
    generated = rows(ROOT / 'generation/valid.jsonl')
    assert len(generated) == 43
    assignments = []
    for r in rows(FRESH / 'inputs/valid_assignments.jsonl') + rows(FRESH / 'results/valid_final_responses.jsonl'):
        method = {'NewBaseResidualCandidate': FRESH_CAND, 'NewBaseResidualSelected': FRESH_FINAL}.get(r['method'], r['method'])
        assignments.append({k: r[k] for k in ('canonical_id', 'question', 'response')} | {'method': method})
    assignments += [{k: r[k] for k in ('canonical_id', 'question', 'response')} | {'method': NEW} for r in generated]
    assert len({(r['canonical_id'], r['method']) for r in assignments}) == 8 * 43
    prompts = read(ROOT / 'run_contract.json')['settings']['frozen_evaluation_prompts']
    pointwise = []
    for r in assignments:
        r['question_response_sha256'] = sha([r['question'], r['response']])
        req = E.request('valid_pointwise', prompts['pointwise_system'], prompts['pointwise_user_template'].format(
            question=r['question'], response=r['response']), 'pointwise_expert_rubric', 384,
            {'question_response_sha256': r['question_response_sha256']})
        r['pointwise_request_id'] = req['request_id']
        pointwise.append(req)
    write_rows(ROOT / 'inputs/valid_assignments.jsonl', assignments)
    pw = cached_execute('valid_pointwise', pointwise)
    by_key = {(r['canonical_id'], r['method']): r for r in assignments}
    pairs = []
    for r in generated:
        for baseline in BASELINES:
            control = by_key[(r['canonical_id'], baseline)]
            pairs.extend(E.pair_requests('valid_pairwise', r['canonical_id'], r['question'],
                control['response'], r['response'], baseline, NEW))
    pair = cached_execute('valid_pairwise', pairs)
    write(ROOT / 'evaluation_preparation.json', {'assignments': len(assignments),
        'pointwise_unique': len(pw), 'pointwise_reused': sum(bool(r.get('reused_from')) for r in pw.values()),
        'pointwise_failed': sum(not r['ok'] for r in pw.values()), 'pairwise_assignments': len(pairs),
        'pairwise_unique': len(pair), 'pairwise_reused': sum(bool(r.get('reused_from')) for r in pair.values()),
        'pairwise_failed': sum(not r['ok'] for r in pair.values())})


def risk():
    output = ROOT / 'results/valid_risk.jsonl'
    if not output.exists():
        shutil.copy2(FRESH / 'results/valid_risk.jsonl', output)
    E.score_valid_risk()


def llama():
    import run_valid43_base_draft_control_llama as local
    local.ROOT = ROOT
    local.CACHE_ROOTS = (FRESH, PREVIOUS)
    local.BATCH_SIZE = 8
    expected = rows(ROOT / 'requests/valid_pointwise.jsonl')
    write_rows(ROOT / 'requests/second_pass_expected_pointwise.jsonl', expected)
    local.prepare()
    pending = rows(ROOT / 'requests/second_pass_new_llama.jsonl')
    history = {r['request_id']: r for r in rows(ROOT / 'results/second_pass_new_llama.jsonl')}
    failures = {local.local_key(r): r for r in rows(FRESH / 'results/second_pass_new_llama.jsonl') if not r.get('judge_ok')}
    for req in pending:
        old = failures.get(local.local_key(req))
        if old and req['request_id'] not in history:
            value = {**old, **req, 'reused_failure_from': str(FRESH / 'results/second_pass_new_llama.jsonl')}
            append(ROOT / 'results/second_pass_new_llama.jsonl', value)
            history[req['request_id']] = value
    # A completed parse failure is retained, not retried until it happens to parse.
    pending = [r for r in pending if r['request_id'] not in history]
    write_rows(ROOT / 'requests/second_pass_new_llama.jsonl', pending)
    if pending:
        gpu_idle()
        local.run()


def aggregate():
    assignments = rows(ROOT / 'inputs/valid_assignments.jsonl')
    by_key = {(r['canonical_id'], r['method']): r for r in assignments}
    pw = {r['request_id']: r for r in rows(ROOT / 'results/valid_pointwise.jsonl')}
    pairs = {r['request_id']: r for r in rows(ROOT / 'results/valid_pairwise.jsonl')}
    decisions, finals = [], []
    for r in rows(ROOT / 'generation/valid.jsonl'):
        for baseline in BASELINES:
            original = by_key[(r['canonical_id'], baseline)]
            reqs = E.pair_requests('valid_pairwise', r['canonical_id'], r['question'], original['response'], r['response'], baseline, NEW)
            winner = E.consensus(reqs, pairs)
            decisions.append({'canonical_id': r['canonical_id'], 'baseline': baseline, 'winner': winner})
            if baseline == 'BasePromptRewrite':
                accepted = winner == NEW and r['technical_valid']
                final = dict(by_key[(r['canonical_id'], NEW if accepted else baseline)])
                final.update(method=FINAL, accepted=accepted, selection_result=winner, technical_candidate_valid=r['technical_valid'])
                finals.append(final)
    write_rows(ROOT / 'results/valid_final_responses.jsonl', finals)
    write_rows(ROOT / 'results/valid_pairwise_decisions.jsonl', decisions)
    local = {r['target_request_id']: r['judge_normalized'] for r in rows(ROOT / 'results/second_pass_reused_llama.jsonl')}
    for r in rows(ROOT / 'results/second_pass_new_llama.jsonl'):
        if r.get('judge_ok'):
            local[r['request_id']] = r['judge_normalized']
    risks = {r['question_response_sha256']: r['risk_score'] for r in rows(ROOT / 'results/valid_risk.jsonl')}
    scored = []
    for judge in ('GPT', 'Llama'):
        for r in assignments + finals:
            result = pw.get(r['pointwise_request_id'])
            norm = (result['normalized'] if result and result.get('ok') else None) if judge == 'GPT' else local.get(r['pointwise_request_id'])
            scored.append({**r, 'judge': judge, 'evaluation_ok': norm is not None, 'scores': norm,
                'quality_3axis_mean': E.q3(norm) if norm else None, 'risk': risks.get(r['question_response_sha256'])})
    write_rows(ROOT / 'results/valid_scored_assignments.jsonl', scored)
    methods = sorted({r['method'] for r in scored})
    summary, matched = [], []
    for judge in ('GPT', 'Llama'):
        for method in methods:
            all_rows = [r for r in scored if r['judge'] == judge and r['method'] == method]
            good = [r for r in all_rows if r['evaluation_ok']]
            med = collections.Counter(r['scores']['medical_advice'] for r in good)
            risk_values = [r['risk'] for r in all_rows if r['risk'] is not None]
            summary.append({'judge': judge, 'method': method, 'N': len(good), 'failed': 43-len(good),
                'quality_3axis_mean': statistics.mean(r['quality_3axis_mean'] for r in good) if good else None,
                **{a: statistics.mean(r['scores']['scores'][a] for r in good) if good else None for a in ('overall_quality', 'empathy', 'specificity')},
                'medical_yes': med['yes'], 'medical_no': med['no'], 'medical_unsure': med['unsure'],
                'medical_yes_denominator': med['yes']+med['no'],
                'medical_yes_rate': med['yes']/(med['yes']+med['no']) if med['yes']+med['no'] else None,
                'mean_risk': statistics.mean(risk_values) if len(risk_values)==43 else None, 'risk_N': len(risk_values)})
        for method, control in itertools.permutations(methods, 2):
            left = {r['canonical_id']: r for r in scored if r['judge']==judge and r['method']==method and r['evaluation_ok']}
            right = {r['canonical_id']: r for r in scored if r['judge']==judge and r['method']==control and r['evaluation_ok']}
            ids = sorted(left.keys() & right.keys())
            deltas = [left[c]['quality_3axis_mean']-right[c]['quality_3axis_mean'] for c in ids]
            matched.append({'judge': judge, 'method': method, 'control': control, 'N': len(ids),
                'method_mean': statistics.mean(left[c]['quality_3axis_mean'] for c in ids) if ids else None,
                'control_mean': statistics.mean(right[c]['quality_3axis_mean'] for c in ids) if ids else None,
                'win': sum(d>1e-9 for d in deltas), 'tie': sum(abs(d)<=1e-9 for d in deltas), 'loss': sum(d< -1e-9 for d in deltas),
                'mean_delta': statistics.mean(deltas) if deltas else None})
    result = {'summary': summary, 'matched': matched,
        'pairwise': {b: dict(collections.Counter(r['winner'] for r in decisions if r['baseline']==b)) for b in BASELINES},
        'accepted': sum(r['accepted'] for r in finals),
        'changed': sum(r['response'] != by_key[(r['canonical_id'], 'BasePromptRewrite')]['response'] for r in finals),
        'test_used': False}
    write(ROOT / 'results/summary.json', result)
    for name, values in [('quality_summary', summary), ('matched_comparisons', matched)]:
        with (ROOT / f'results/{name}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    print(json.dumps(result), flush=True)


def package():
    data = read(ROOT / 'results/summary.json')
    contract = read(ROOT / 'run_contract.json')
    for name, digest in contract['copied_files'].items():
        assert file_sha(ROOT / name) == digest == file_sha(FRESH / name)
    assert file_sha(Path(contract['initial_adapter'])/'adapter_model.safetensors') == contract['initial_adapter_weights_sha256']
    generated = rows(ROOT / 'generation/valid.jsonl')
    fresh = {r['canonical_id']: r for r in rows(FRESH / 'generation/valid.jsonl')}
    assert len(generated) == 43
    for r in generated:
        for key in ('rendered_prompt_sha256', 'input_token_ids_sha256', 'input_sha256', 'eos_ids'):
            assert r[key] == fresh[r['canonical_id']][key], ('Generation inputs differ', key)
    write(ROOT / 'verification.json', {'same_train_holdout_bytes': True, 'same_VALID43_prompt_and_token_hashes': True,
        'original_adapter_unchanged': True, 'generation_count': 43,
        'technical_generation_success': sum(r['technical_valid'] for r in generated), 'test_used': False})
    api = [r for name in ('valid_pointwise', 'valid_pairwise') for r in rows(ROOT / f'results/{name}.jsonl') if not r.get('reused_from')]
    write(ROOT / 'api_usage.json', {'records': len(api), 'historical_failed': sum(not r['ok'] for r in api),
        'prompt_tokens': sum(r.get('usage',{}).get('prompt_tokens',0) for r in api),
        'completion_tokens': sum(r.get('usage',{}).get('completion_tokens',0) for r in api),
        'estimated_cost_usd': sum(r.get('estimated_cost_usd',0) for r in api), 'estimate_only': True})
    report = ['# Continued Denoiser: Initialization-Only VALID43 Comparison', '',
        'The TRAIN70 examples, internal holdout13, tokenization, optimizer settings, generation inputs and judge rubric match the fresh-adapter arm.',
        'Only the initial adapter weights differ: saved proposed_mask_off/final instead of a fresh LoRA. Optimizer state is new, not resumed from historical denoiser training.',
        'No teacher/target regeneration and no TEST access. This previously inspected VALID set is development evidence only.', '',
        '## Training', '```json', json.dumps(read(ROOT/'training/manifest.json'),indent=2), '```', '',
        '## Quality', '| Judge | Method | N | Quality-3 | Medical Y/N/U |', '|---|---|---:|---:|---|']
    for r in data['summary']:
        score = f"{r['quality_3axis_mean']:.4f}" if r['quality_3axis_mean'] is not None else 'NA'
        report.append(f"| {r['judge']} | {r['method']} | {r['N']} | {score} | {r['medical_yes']}/{r['medical_no']}/{r['medical_unsure']} |")
    chosen = [r for r in data['matched'] if (r['method']==NEW and r['control'] in BASELINES) or
              (r['method']==FINAL and r['control'] in ('BasePromptRewrite', FRESH_FINAL, 'BaseSelfRefineAllGeneratedGPTSelected'))]
    report += ['', '## Matched Comparisons', '```json', json.dumps(chosen,indent=2), '```',
        '', f"Final accepted / changed: {data['accepted']} / {data['changed']} out of 43.",
        'Pairwise/selector judgments are separate from pointwise scores. Failures are not ties. Medical Advice is categorical, not part of Quality-3.',
        'GPT teacher, verifier, selector and evaluator share a model; this is not independent expert validation. Llama failures are retained.']
    (ROOT/'REPORT.md').write_text('\n'.join(report)+'\n')
    files = [p for p in ROOT.rglob('*') if p.is_file() and p.suffix not in {'.zip','.safetensors','.pt','.log','.lock'}
             and p.name not in {'progress.json','ARTIFACT_MANIFEST.json','download_copy_status.json'}
             and not any(x in {'runtime_cache','tmp','__pycache__'} or x.startswith('epoch_') for x in p.relative_to(ROOT).parts)]
    write(ROOT/'ARTIFACT_MANIFEST.json', {'status':'complete', 'test_used':False,
        'files':[{'path':str(p.relative_to(ROOT)),'sha256':file_sha(p)} for p in files], 'adapter_weights_excluded':True})
    archive = ROOT/'continued_denoiser_VALID43_research_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[ROOT/'ARTIFACT_MANIFEST.json']: z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for r in read(ROOT/'ARTIFACT_MANIFEST.json')['files']:
            import hashlib
            assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
    destination=Path('/home/user/Downloads')/archive.name
    if shutil.disk_usage(destination.parent).free > 2*archive.stat().st_size:
        shutil.copy2(archive,destination)
        write(ROOT/'download_copy_status.json',{'copied':True,'path':str(destination)})
    else:
        write(ROOT/'download_copy_status.json',{'copied':False,'reason':'insufficient root filesystem space','source':str(archive)})
    print(json.dumps({'zip':str(archive),'sha256':file_sha(archive)}),flush=True)


def pipeline():
    ROOT.mkdir(parents=True, exist_ok=True)
    for key,name in [('TMPDIR','tmp'),('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','torchinductor'),('CUDA_CACHE_PATH','cuda')]:
        path=ROOT/'runtime_cache'/name; path.mkdir(parents=True,exist_ok=True); os.environ[key]=str(path)
    with (ROOT/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for stage in ('prepare','train','generate','evaluate','risk','llama','aggregate','package'):
            write(ROOT/'progress.json',{'status':'running','stage':stage})
            if stage=='train': cmd=[sys.executable,str(CODE/'train_base_residual_adapter.py')]
            elif stage=='generate': cmd=[sys.executable,str(CODE/'run_base_residual_generation.py'),'valid']
            else: cmd=[sys.executable,str(Path(__file__).resolve()),stage]
            result=subprocess.run(cmd,cwd=CODE.parent)
            if result.returncode:
                write(ROOT/'progress.json',{'status':'failed','stage':stage,'returncode':result.returncode,
                    'resume':f'{sys.executable} {Path(__file__).resolve()} pipeline'})
                raise SystemExit(result.returncode)
        write(ROOT/'progress.json',{'status':'complete','test_used':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','pipeline','evaluate','risk','llama','aggregate','package'])
    globals()[parser.parse_args().stage]()
