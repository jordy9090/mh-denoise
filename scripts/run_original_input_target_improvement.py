"""Improve TRAIN targets only; preserve original SFT-conditioned denoiser inputs."""
import argparse
import collections
import csv
import fcntl
import json
import os
import shutil
import statistics
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

from residual_train_common import (CKPT, FROZEN, MODEL, TOKENIZER, REPO, TEACHER_SYSTEM,
    TEACHER_USER, VERIFY_SYSTEM, VERIFY_USER, rows, read, write, write_rows, append,
    sha, file_sha, gpu_idle, flags)
import run_base_residual_experiment as E

ROOT = Path('/mnt/ssd00/hsoh/original_input_target_improvement_20261011_v1')
CODE = Path(__file__).resolve().parent
OLD = CKPT / 'gemma/proposed_mask_off'
VALID = CKPT / 'gemma/generation_1536/proposed_mask_off_valid_outputs.jsonl'
E.ROOT = ROOT
TEACHER = TEACHER_SYSTEM + '\nThis is an existing TRAIN reference answer, not a newly generated Base draft. Improve only this target; do not change the question or the model inputs.'
VERIFY = VERIFY_SYSTEM + '\nUse the exact JSON key evidence for the evidence phrase.'
MATERIAL = ('model', 'temperature', 'max_tokens', 'response_format', 'parser', 'system_prompt', 'user_prompt')


def stage(name):
    write(ROOT / 'status.json', {'stage': name, 'pid': os.getpid(), 'test_used': False})
    print(json.dumps({'stage': name}), flush=True)


def prepare():
    train = rows(OLD / 'risk_tune_train_enriched.jsonl')
    valid = rows(VALID)
    assert len(train) == 276 and len(valid) == 43
    assert len({r['canonical_id'] for r in train}) == 276
    assert not {r['canonical_id'] for r in train} & {r['canonical_id'] for r in valid}
    assert sum(bool(r['used_denoiser']) for r in valid) == 9
    assert all(r['split'].lower() == 'train' for r in train)
    tm = read(OLD / 'training_manifest.json')
    gm = read(VALID.with_suffix('.manifest.json'))
    prompts = read(FROZEN)['prompts']
    for k in ('pointwise_system', 'pointwise_user_template', 'selector_system', 'selector_user_template'):
        assert sha(prompts[k]) == prompts[k + '_sha256']
    contract = {'version': ROOT.name, 'test_used': False, 'paid_api_authorized': True,
        'train_ids': [r['canonical_id'] for r in train], 'valid_ids': [r['canonical_id'] for r in valid],
        'called_valid_ids': [r['canonical_id'] for r in valid if r['used_denoiser']],
        'initial_adapter': str(OLD / 'final'), 'initial_adapter_sha256': file_sha(OLD / 'final/adapter_model.safetensors'),
        'source_train_sha256': file_sha(OLD / 'risk_tune_train_enriched.jsonl'),
        'source_valid_sha256': file_sha(VALID), 'original_training_manifest': tm,
        'original_generation_manifest': gm, 'frozen_evaluation_contract_sha256': file_sha(FROZEN),
        'settings': {'frozen_evaluation_prompts': prompts, 'teacher_system': TEACHER,
            'teacher_user_template': TEACHER_USER, 'verifier_system': VERIFY, 'verifier_user_template': VERIFY_USER,
            'model': MODEL, 'temperature': 0, 'teacher_max_tokens': 1536, 'verifier_max_tokens': 512,
            'pointwise_max_tokens': 384, 'selector_max_tokens': 256,
            'target_acceptance': 'one proposal per TRAIN ID; both blind orders prefer new target, all verifier flags false, target fits original 640-token limit; otherwise retain original safe_response',
            'training_scope': 'all original 276 IDs; only safe_response may change; no 35+35 balancing',
            'training': 'original manifest settings, except input/output paths and initialization from original denoiser final; one epoch, fresh optimizer; evaluate final checkpoint, not VALID-selected best',
            'validation': 'fixed original nine called IDs; reuse original candidates; unchanged 34 SFT responses; matched two-order four-axis GPT selection for both arms',
            'technical_validity': 'nonempty candidate, EOS reached, no source truncation; failed cases retained and selector falls back to SFT'},
        'limitations': ['Synthetic GPT-preferred targets are not clinical gold.',
            'GPT teacher, verifier, selector and evaluator share model; Llama is auxiliary.',
            'VALID43 has already informed development; not untouched confirmatory evaluation.',
            'Compared with original weights, this changes targets AND adds optimizer steps; no old-target continued-training control, so target quality causal effect is not isolated.']}
    path = ROOT / 'run_contract.json'
    if path.exists():
        assert read(path) == contract, 'Frozen contract changed'
    else:
        write(path, contract)
    write_rows(ROOT / 'inputs/train_original.jsonl', train)
    write_rows(ROOT / 'inputs/valid_original.jsonl', valid)
    write_rows(ROOT / 'inputs/valid_training_original.jsonl', rows(OLD / 'risk_tune_valid_enriched.jsonl'))
    print(json.dumps({'train': 276, 'valid': 43, 'called_valid': 9, 'contract': str(path)}), flush=True)


def curate():
    from fullpaper_backbone_utils import load_fullpaper_tokenizer
    from selective_risk_refinement_utils import build_risk_tune_prompt, configure_risk_contract, clean_text
    configure_risk_contract('fullpaper_v1')
    tok = load_fullpaper_tokenizer(str(OLD / 'final'))
    source = rows(ROOT / 'inputs/train_original.jsonl')
    reqs = [E.request('train_teacher', TEACHER, TEACHER_USER.format(question=r['question'], response=r['safe_response']),
        'train_residual_teacher', 1536, {'canonical_id': r['canonical_id']}) for r in source]
    result = E.execute_requests('train_teacher', reqs)
    teachers = {r['metadata']['canonical_id']: result.get(r['request_id']) for r in reqs}
    candidates, verifies, pairs = {}, [], []
    for r in source:
        cid = r['canonical_id']; t = teachers[cid]
        if not t or not t.get('ok') or t.get('returned_model') != MODEL:
            continue
        candidate = t['normalized']['response'].strip()
        if t['normalized']['action'] == 'preserve' or candidate == r['safe_response']:
            continue
        if flags(candidate) or '[rest of text]' in candidate.lower() or '[end of text]' in candidate.lower():
            continue
        target = clean_text(candidate)
        if tok.eos_token and not target.endswith(tok.eos_token):
            target += tok.eos_token
        if len(tok(target, add_special_tokens=False)['input_ids']) > 640:
            continue
        candidates[cid] = candidate
        verifies.append(E.request('train_verify', VERIFY, VERIFY_USER.format(question=r['question'], draft=r['safe_response'], candidate=candidate),
            'train_residual_verify', 512, {'canonical_id': cid}))
        pairs.extend(E.pair_requests('train_preference', cid, r['question'], r['safe_response'], candidate, 'old_target', 'new_target'))
    vr = E.execute_requests('train_verify', verifies)
    pr = E.execute_requests('train_preference', pairs)
    by_id = {r['metadata']['canonical_id']: vr.get(r['request_id']) for r in verifies}
    final, audit = [], []
    for r in source:
        cid = r['canonical_id']; candidate = candidates.get(cid); accepted = False
        reason = 'teacher_preserved_or_failed_or_candidate_invalid'; winner = None
        if candidate:
            winner = E.consensus(E.pair_requests('train_preference', cid, r['question'], r['safe_response'], candidate, 'old_target', 'new_target'), pr)
            v = by_id.get(cid)
            verified = bool(v and v.get('ok') and v.get('returned_model') == MODEL)
            issues = [k for k in ('unsupported_claims', 'specificity_loss', 'inappropriate_refusal', 'surface_failure') if verified and v['normalized'][k]]
            accepted = winner == 'new_target' and verified and not issues
            reason = 'verified_two_order_improvement' if accepted else str(winner) + ';' + ('verification_failure' if not verified else ','.join(issues))
        updated = {**r, 'safe_response': candidate if accepted else r['safe_response']}
        before = {k: v for k, v in r.items() if k != 'safe_response'}
        after = {k: v for k, v in updated.items() if k != 'safe_response'}
        assert before == after
        p0, p1 = build_risk_tune_prompt(tok, r), build_risk_tune_prompt(tok, updated)
        assert p0 == p1, 'Target leaked into input renderer'
        ids0 = tok(p0, add_special_tokens=False, truncation=True, max_length=4096)['input_ids']
        ids1 = tok(p1, add_special_tokens=False, truncation=True, max_length=4096)['input_ids']
        assert ids0 == ids1
        final.append(updated)
        audit.append({'canonical_id': cid, 'replaced': accepted, 'reason': reason, 'preference': winner,
            'old_target': r['safe_response'], 'proposed_target': candidate, 'final_target': updated['safe_response'],
            'old_target_sha256': sha(r['safe_response']), 'final_target_sha256': sha(updated['safe_response']),
            'non_target_fields_sha256': sha(before), 'rendered_prompt': p0,
            'rendered_prompt_before_sha256': sha(p0), 'rendered_prompt_after_sha256': sha(p1),
            'input_token_ids_before_sha256': sha(ids0), 'input_token_ids_after_sha256': sha(ids1), 'input_tokens': len(ids0)})
    write_rows(ROOT / 'data/train_improved_targets.jsonl', final)
    write_rows(ROOT / 'data/target_changes_and_input_hashes.jsonl', audit)
    summary = {'total_train': len(final), 'replaced_targets': sum(r['replaced'] for r in audit),
        'retained_targets': sum(not r['replaced'] for r in audit), 'all_non_target_fields_identical': True,
        'all_rendered_prompts_and_input_tokens_identical': True, 'reasons': dict(collections.Counter(r['reason'] for r in audit))}
    write(ROOT / 'data/curation_summary.json', summary)
    print(json.dumps(summary), flush=True)
    if not summary['replaced_targets']:
        raise RuntimeError('No verified target changes; no training launched')


def train():
    output = ROOT / 'training'
    manifest = output / 'training_manifest.json'
    if manifest.exists():
        assert read(manifest)['status'] == 'complete_training'
        return
    if list(output.glob('**/adapter_model.safetensors')):
        raise RuntimeError('Partial training checkpoint exists; do not silently restart optimizer')
    gpu_idle()
    settings = dict(read(ROOT / 'run_contract.json')['original_training_manifest']['settings'])
    settings.update(init_adapter_dir=str(OLD / 'final'), train_file=str(ROOT / 'data/train_improved_targets.jsonl'),
        valid_file=str(ROOT / 'inputs/valid_training_original.jsonl'), output_dir=str(output))
    command = [sys.executable, str(CODE / 'train_gemma_risk_tune_from_sft.py')]
    for k, v in settings.items():
        if v is None or v is False:
            continue
        command.append('--' + k)
        if v is not True:
            command.append(str(v))
    write(ROOT / 'training_command.json', {'argv': command, 'settings': settings})
    subprocess.run(command, check=True)
    assert read(manifest)['settings'] == settings


def generate():
    import torch
    from peft import PeftModel
    from fullpaper_backbone_utils import load_fullpaper_tokenizer
    from run_gemma_selective_risk_refinement import load_base_model
    from selective_risk_refinement_utils import build_risk_tune_prompt, generate_response, configure_risk_contract
    source = [r for r in rows(ROOT / 'inputs/valid_original.jsonl') if r['used_denoiser']]
    output = ROOT / 'generation/valid_candidates.jsonl'
    done = {r['canonical_id']: r for r in rows(output)}
    assert len(done) == len(rows(output))
    for r in source:
        if r['canonical_id'] in done:
            assert done[r['canonical_id']]['source_sha256'] == sha(r)
    pending = [r for r in source if r['canonical_id'] not in done]
    if not pending:
        return
    gpu_idle(); configure_risk_contract('fullpaper_v1')
    settings = read(ROOT / 'run_contract.json')['original_generation_manifest']['settings']
    tok = load_fullpaper_tokenizer(settings['sft_adapter_dir']); tok.padding_side = 'left'
    model = PeftModel.from_pretrained(load_base_model(settings['base_model'], not settings['no_4bit']), str(ROOT / 'training/final'))
    model.eval()
    kwargs = {k: settings[k] for k in ('max_source_len', 'max_new_tokens', 'temperature', 'repetition_penalty', 'no_repeat_ngram_size')}
    for r in pending:
        prompt = build_risk_tune_prompt(tok, r)
        ids = tok(prompt, truncation=True, max_length=settings['max_source_len'])['input_ids']
        with torch.inference_mode():
            raw, response, metadata = generate_response(model, tok, prompt, **kwargs)
        append(output, {'canonical_id': r['canonical_id'], 'question': r['question'], 'source_sha256': sha(r),
            'response_raw': raw, 'response': response, 'generation': metadata,
            'rendered_prompt': prompt, 'rendered_prompt_sha256': sha(prompt), 'input_token_ids_sha256': sha(ids),
            'original_arm_prompt_sha256': sha(build_risk_tune_prompt(tok, r)),
            'technical_valid': bool(response.strip()) and metadata['eos_reached'] and not metadata['source_input_truncated']})
        print(json.dumps({'valid_generated': len(rows(output)), 'total': 9}), flush=True)


def cached(name, requests):
    requested = {r['request_id']: r for r in requests}
    out = ROOT / f'results/{name}.jsonl'
    seen = {r['request_id'] for r in rows(out)}
    # Only previous VALID/TRAIN-only experiment caches, never TEST artifacts.
    for parent in ('base_first_residual_valid_20261011_v1', 'base_residual_train_20261011_v1', 'continued_denoiser_train_20261011_v1'):
        for path in (ROOT.parent / parent / 'results').glob('*.jsonl'):
            for old in rows(path):
                if not old.get('ok') or old.get('returned_model') != MODEL:
                    continue
                rid = sha({k: old.get(k) for k in MATERIAL})[:32]
                if rid not in requested or rid in seen:
                    continue
                req = requested[rid]
                assert all(old.get(k) == req[k] for k in MATERIAL)
                append(out, {**old, **req, 'reused_from': str(path), 'usage': {}, 'estimated_cost_usd': 0})
                seen.add(rid)
    return E.execute_requests(name, requests)


def evaluate():
    source = rows(ROOT / 'inputs/valid_original.jsonl')
    new = {r['canonical_id']: r for r in rows(ROOT / 'generation/valid_candidates.jsonl')}
    assert len(new) == 9
    assignments, pairs, candidates = [], [], {}
    for r in source:
        cid = r['canonical_id']
        assignments.append({'canonical_id': cid, 'question': r['question'], 'response': r['sft_response'], 'method': 'SFT'})
        assignments.append({'canonical_id': cid, 'question': r['question'], 'response': r['final_response'], 'method': 'OriginalLegacyStrictFinal'})
        if not r['used_denoiser']:
            continue
        for method, response, valid in (
            ('OriginalCandidate', r['denoiser_response'], bool(r['denoiser_response'].strip()) and r['denoiser_generation']['eos_reached'] and not r['denoiser_generation']['source_input_truncated']),
            ('TargetImprovedCandidate', new[cid]['response'], new[cid]['technical_valid'])):
            item = {'canonical_id': cid, 'question': r['question'], 'response': response, 'method': method, 'technical_valid': valid}
            assignments.append(item); candidates[(cid, method)] = item
            if valid:
                pairs.extend(E.pair_requests('valid_selection', cid, r['question'], r['sft_response'], response, 'SFT', method))
    selections = cached('valid_selection', pairs)
    decisions = []
    for r in source:
        cid = r['canonical_id']
        for method in ('OriginalCandidate', 'TargetImprovedCandidate'):
            candidate = candidates.get((cid, method))
            winner = 'not_called' if candidate is None else 'candidate_technical_failure'
            if candidate and candidate['technical_valid']:
                winner = E.consensus(E.pair_requests('valid_selection', cid, r['question'], r['sft_response'], candidate['response'], 'SFT', method), selections)
            accepted = winner == method
            decisions.append({'canonical_id': cid, 'arm': method, 'winner': winner, 'accepted': accepted})
            assignments.append({'canonical_id': cid, 'question': r['question'],
                'response': candidate['response'] if accepted else r['sft_response'], 'method': method.replace('Candidate', 'GPTSelectedFinal')})
    prompts = read(ROOT / 'run_contract.json')['settings']['frozen_evaluation_prompts']
    pointwise = []
    for r in assignments:
        r['question_response_sha256'] = sha([r['question'], r['response']])
        req = E.request('valid_pointwise', prompts['pointwise_system'], prompts['pointwise_user_template'].format(question=r['question'], response=r['response']),
            'pointwise_expert_rubric', 384, {'question_response_sha256': r['question_response_sha256']})
        r['pointwise_request_id'] = req['request_id']; pointwise.append(req)
    write_rows(ROOT / 'inputs/valid_assignments.jsonl', assignments)
    write_rows(ROOT / 'results/valid_selection_decisions.jsonl', decisions)
    cached('valid_pointwise', pointwise)


def llama():
    sys.path.append(str(REPO / 'scripts'))
    import run_valid43_base_draft_control_llama as local
    local.ROOT = ROOT
    local.CACHE_ROOTS = tuple(ROOT.parent / x for x in ('base_first_residual_valid_20261011_v1', 'base_residual_train_20261011_v1', 'continued_denoiser_train_20261011_v1'))
    local.BATCH_SIZE = 8
    write_rows(ROOT / 'requests/second_pass_expected_pointwise.jsonl', rows(ROOT / 'requests/valid_pointwise.jsonl'))
    local.prepare()
    history = {r['request_id'] for r in rows(ROOT / 'results/second_pass_new_llama.jsonl')}
    pending = [r for r in rows(ROOT / 'requests/second_pass_new_llama.jsonl') if r['request_id'] not in history]
    write_rows(ROOT / 'requests/second_pass_new_llama.jsonl', pending)
    if pending:
        gpu_idle(); local.run()


def report():
    assignments = rows(ROOT / 'inputs/valid_assignments.jsonl')
    gpt = {r['request_id']: r['normalized'] for r in rows(ROOT / 'results/valid_pointwise.jsonl') if r.get('ok') and r.get('returned_model') == MODEL}
    local = {r['target_request_id']: r['judge_normalized'] for r in rows(ROOT / 'results/second_pass_reused_llama.jsonl')}
    local.update({r['request_id']: r['judge_normalized'] for r in rows(ROOT / 'results/second_pass_new_llama.jsonl') if r.get('judge_ok')})
    called = set(read(ROOT / 'run_contract.json')['called_valid_ids'])
    scored, summary, comparisons = [], [], []
    for judge, results in [('GPT', gpt), ('Llama', local)]:
        for r in assignments:
            norm = results.get(r['pointwise_request_id'])
            scored.append({**r, 'judge': judge, 'scores': norm, 'quality_3axis_mean': E.q3(norm) if norm else None})
        for scope in ('called9', 'full43'):
            for method in sorted({r['method'] for r in assignments}):
                subset = [r for r in scored if r['judge'] == judge and r['method'] == method and (scope == 'full43' or r['canonical_id'] in called)]
                if scope == 'full43' and len(subset) != 43:
                    continue
                good = [r for r in subset if r['scores']]
                med = collections.Counter(r['scores']['medical_advice'] for r in good)
                summary.append({'judge': judge, 'scope': scope, 'method': method, 'N': len(good), 'failed': len(subset)-len(good),
                    'quality_3axis_mean': statistics.mean(r['quality_3axis_mean'] for r in good) if good else None,
                    **{a: statistics.mean(r['scores']['scores'][a] for r in good) if good else None for a in ('overall_quality', 'empathy', 'specificity')},
                    'medical_yes': med['yes'], 'medical_no': med['no'], 'medical_unsure': med['unsure']})
            for method, control in [('TargetImprovedCandidate', 'OriginalCandidate'), ('TargetImprovedGPTSelectedFinal', 'OriginalGPTSelectedFinal'), ('TargetImprovedGPTSelectedFinal', 'SFT')]:
                left = {r['canonical_id']: r for r in scored if r['judge'] == judge and r['method'] == method and r['scores'] and (scope == 'full43' or r['canonical_id'] in called)}
                right = {r['canonical_id']: r for r in scored if r['judge'] == judge and r['method'] == control and r['scores']}
                if scope == 'full43' and method.endswith('Candidate'):
                    continue
                ids = sorted(left.keys() & right.keys()); delta = [left[i]['quality_3axis_mean']-right[i]['quality_3axis_mean'] for i in ids]
                comparisons.append({'judge': judge, 'scope': scope, 'method': method, 'control': control, 'N': len(ids),
                    'mean_delta': statistics.mean(delta) if delta else None, 'win': sum(d>1e-9 for d in delta),
                    'tie': sum(abs(d)<=1e-9 for d in delta), 'loss': sum(d< -1e-9 for d in delta)})
    write_rows(ROOT / 'results/valid_scored_assignments.jsonl', scored)
    decisions = rows(ROOT / 'results/valid_selection_decisions.jsonl')
    selection = {arm: dict(collections.Counter(r['winner'] for r in decisions if r['arm'] == arm)) for arm in ('OriginalCandidate', 'TargetImprovedCandidate')}
    write(ROOT / 'results/summary.json', {'summary': summary, 'matched_comparisons': comparisons, 'selection': selection, 'test_used': False})
    with (ROOT / 'results/summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    usage = [r for p in (ROOT / 'results').glob('*.jsonl') for r in rows(p) if r.get('parser') and 'ok' in r]
    write(ROOT / 'api_usage.json', {'stored_requests': len(usage), 'reused': sum(bool(r.get('reused_from')) for r in usage),
        'failed': sum(not r.get('ok') for r in usage), 'estimated_cost_usd': sum(r.get('estimated_cost_usd', 0) for r in usage),
        'prompt_tokens': sum((r.get('usage') or {}).get('prompt_tokens', 0) for r in usage),
        'completion_tokens': sum((r.get('usage') or {}).get('completion_tokens', 0) for r in usage)})
    lines = ['# Original-input TRAIN target improvement', '', json.dumps(read(ROOT / 'data/curation_summary.json'), ensure_ascii=False), '',
        'All 276 original TRAIN inputs and stored SFT drafts preserved. Only safe_response targets change. Original denoiser final continued for one original-schedule epoch. No TEST access.', '',
        'OriginalLegacyStrictFinal is the archived legacy output, NOT the matched GPT selector condition.', '',
        '|Judge|Scope|Method|N|Failed|Quality-3|', '|---|---|---|---:|---:|---:|']
    lines += [f"|{r['judge']}|{r['scope']}|{r['method']}|{r['N']}|{r['failed']}|{r['quality_3axis_mean']}|" for r in summary]
    lines += ['', '## Matched comparisons', '```json', json.dumps(comparisons, indent=2), '```', '', '## Limitations']
    lines += ['- ' + v for v in read(ROOT / 'run_contract.json')['limitations']]
    (ROOT / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    assert file_sha(OLD / 'final/adapter_model.safetensors') == read(ROOT / 'run_contract.json')['initial_adapter_sha256']
    assert file_sha(VALID) == read(ROOT / 'run_contract.json')['source_valid_sha256']
    archive = ROOT / 'original_input_target_improvement_research_bundle.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.rglob('*'):
            if p.is_file() and p.suffix not in {'.zip', '.safetensors', '.bin', '.lock', '.pyc'} and 'training' not in p.relative_to(ROOT).parts:
                z.write(p, p.relative_to(ROOT))
    downloads = Path('/home/user/Downloads'); downloads.mkdir(exist_ok=True)
    if shutil.disk_usage(downloads).free > archive.stat().st_size + 20_000_000:
        shutil.copy2(archive, downloads / archive.name)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'run', 'train', 'generate', 'llama', 'report'])
    action = parser.parse_args().action
    ROOT.mkdir(parents=True, exist_ok=True)
    if action != 'run':
        globals()[action](); return
    with (ROOT / 'pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            prepare()
            code_hashes = {p.name: file_sha(p) for p in CODE.glob('*.py')}
            snapshot = ROOT / 'code'
            snapshot.mkdir(exist_ok=True)
            if (ROOT / 'code_hashes.json').exists():
                assert read(ROOT / 'code_hashes.json') == code_hashes, 'Code changed since launch'
            else:
                for p in CODE.glob('*.py'):
                    shutil.copy2(p, snapshot / p.name)
                write(ROOT / 'code_hashes.json', code_hashes)
            for name, function in [('curation', curate), ('training', None), ('generation', None), ('gpt_evaluation', evaluate), ('llama', None), ('report', report)]:
                stage(name)
                if function:
                    function()
                else:
                    child = {'training': 'train', 'generation': 'generate', 'llama': 'llama'}[name]
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), child], check=True)
            stage('complete')
        except Exception:
            write(ROOT / 'failure.json', {'stage': read(ROOT / 'status.json') if (ROOT / 'status.json').exists() else 'prepare', 'traceback': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
