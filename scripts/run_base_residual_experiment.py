"""Curate TRAIN-only targets and evaluate the resulting VALID-only pilot."""
import argparse
import asyncio
import collections
import csv
import fcntl
import json
import shutil
import statistics
import subprocess
import sys
from types import SimpleNamespace

from residual_train_common import *

sys.path.append(str(REPO / 'scripts'))
import run_gpt41_checkpoint_queue as queue

OLD_VALIDATE = queue.validate_payload
OLD_CALL = queue.call_one
OLD_TRANSIENT = queue.is_transient
QUOTA_EXHAUSTED = False


def quota_error(value):
    text = str(value).lower()
    return any(code in text for code in ('credit_balance_exhausted', 'insufficient_quota', 'quota_not_attempted'))


def transient_unless_quota(exc):
    return False if quota_error(exc) else OLD_TRANSIENT(exc)


async def logged_call(client, row, max_retries):
    global QUOTA_EXHAUSTED
    if QUOTA_EXHAUSTED:
        return {**row, 'request_sha256': queue.request_sha256(row), 'ok': False, 'attempt': 0,
                'error_type': 'QuotaNotAttempted', 'error': 'quota_not_attempted: earlier request exhausted API credits',
                'usage': {}, 'estimated_cost_usd': 0}, False
    attempts = []

    async def create(**kwargs):
        response = await client.chat.completions.create(**kwargs)
        attempts.append(response.model_dump(mode='json'))
        return response

    proxy = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result, limited = await OLD_CALL(proxy, row, max_retries)
    if quota_error(result.get('error', '')):
        QUOTA_EXHAUSTED = True
    result['request_sha256'] = queue.request_sha256(row)
    result['api_response_history'] = attempts
    if not result.get('ok') and attempts:
        result['raw_json_text'] = attempts[-1]['choices'][0]['message'].get('content')
        result['usage'] = attempts[-1].get('usage') or {}
        usage = result['usage']
        result['estimated_cost_usd'] = (usage.get('prompt_tokens', 0) * 2 + usage.get('completion_tokens', 0) * 8) / 1e6
    return result, limited


def validate(parser, payload):
    if parser == 'train_residual_teacher':
        if payload.get('action') not in {'edit', 'preserve'}:
            raise ValueError('Invalid teacher action')
        for name in ('issue', 'evidence', 'preserve', 'response'):
            if not isinstance(payload.get(name), str):
                raise ValueError('Missing teacher field: ' + name)
        if not payload['response'].strip():
            raise ValueError('Empty teacher answer')
        return payload
    if parser == 'train_residual_verify':
        for name in ('unsupported_claims', 'specificity_loss', 'inappropriate_refusal', 'surface_failure'):
            if type(payload.get(name)) is not bool:
                raise ValueError('Invalid verifier field: ' + name)
        for name in ('reason', 'evidence'):
            if not isinstance(payload.get(name), str):
                raise ValueError('Missing verifier field: ' + name)
        return payload
    return OLD_VALIDATE(parser, payload)


queue.validate_payload = validate
queue.call_one = logged_call
queue.is_transient = transient_unless_quota


def request(name, system, user, parser, tokens, metadata):
    material = {'model': MODEL, 'temperature': 0, 'max_tokens': tokens, 'response_format': {'type': 'json_object'},
                'parser': parser, 'system_prompt': system, 'user_prompt': user}
    return {**material, 'queue': name, 'request_id': sha(material)[:32], 'metadata': metadata,
            'system_prompt_sha256': sha(system), 'user_prompt_sha256': sha(user)}


def execute_requests(name, requests, partial=False):
    by_id = {}
    for r in requests:
        if r['request_id'] in by_id:
            assert r['system_prompt'] == by_id[r['request_id']]['system_prompt']
            assert r['user_prompt'] == by_id[r['request_id']]['user_prompt']
        by_id[r['request_id']] = r
    requests = list(by_id.values())
    suffix = f'_prefetch_{len(requests)}' if partial else ''
    request_path = ROOT / f'requests/{name}{suffix}.jsonl'
    output = ROOT / f'results/{name}.jsonl'
    if request_path.exists():
        assert rows(request_path) == requests, 'Frozen requests changed: ' + name
    else:
        write_rows(request_path, requests)
    history = {r['request_id']: r for r in rows(output)}
    for r in requests:
        if r['request_id'] in history:
            assert queue.request_sha256(history[r['request_id']]) == queue.request_sha256(r)
    # Final parse/API failures remain failures, not new attempts or ties on resume.
    pending = [r for r in requests if r['request_id'] not in history
               or quota_error(history[r['request_id']].get('error', ''))]
    if pending:
        pending_path = ROOT / f'requests/{name}_pending.jsonl'
        write_rows(pending_path, pending)
        asyncio.run(queue.async_main(SimpleNamespace(requests=pending_path, output=output,
            queue_name=name, initial_concurrency=4, max_concurrency=8, max_retries=5, timeout=120,
            validate_only=False)))
    complete = {r['request_id']: r for r in rows(output)}
    if any(quota_error(complete.get(r['request_id'], {}).get('error', '')) for r in requests):
        raise RuntimeError('API credit_balance_exhausted: completed requests preserved; replenish credits before resume')
    return complete


def teacher_prefetch():
    inputs = rows(ROOT / 'generation/train.jsonl')
    assert len(inputs) <= 276
    requests = [request('train_teacher', TEACHER_SYSTEM, TEACHER_USER.format(question=r['question'], response=r['response']),
                        'train_residual_teacher', 1536, {'canonical_id': r['canonical_id']})
                for r in inputs if r['technical_valid']]
    execute_requests('train_teacher', requests, partial=True)


def recover_verifier_schema(results):
    recovered = []
    merged = dict(results)
    for rid, original in results.items():
        if original.get('ok') or original.get('error') != 'Missing verifier field: evidence':
            continue
        try:
            payload = json.loads(original.get('raw_json_text') or '')
            if 'evidence' in payload or not isinstance(payload.get('evidence_phrase'), str):
                continue
            normalized = validate('train_residual_verify', {**payload, 'evidence': payload['evidence_phrase']})
        except (ValueError, TypeError):
            continue
        api = (original.get('api_response_history') or [{}])[-1]
        if api.get('model') != MODEL:
            continue
        repaired = {**original, 'ok': True, 'normalized': normalized, 'returned_model': api['model'],
                    'schema_recovery': 'evidence_phrase is the supplied evidence string; no substantive judgment changed',
                    'original_failure_preserved_in': str(ROOT / 'results/train_verify.jsonl'),
                    'original_result_sha256': sha(original)}
        merged[rid] = repaired
        recovered.append(repaired)
    write_rows(ROOT / 'results/train_verify_schema_recovered.jsonl', recovered)
    return merged, len(recovered)


def pair_requests(name, canonical_id, question, left, right, left_name, right_name):
    prompts = read(ROOT / 'run_contract.json')['settings']['frozen_evaluation_prompts']
    result = []
    for order in (0, 1):
        a, b = (left, right) if order == 0 else (right, left)
        result.append(request(name, prompts['selector_system'], prompts['selector_user_template'].format(
            question=question, response_a=a, response_b=b), 'blind_selector', 256,
            {'canonical_id': canonical_id, 'order': order, 'left': left_name, 'right': right_name,
             'response_a_method': left_name if order == 0 else right_name,
             'response_b_method': right_name if order == 0 else left_name}))
    return result


def consensus(requests, results):
    winners = []
    for r in requests:
        value = results.get(r['request_id'])
        if not value or not value.get('ok') or value.get('returned_model') != MODEL:
            return 'technical_failure'
        choice = value['normalized']['choice']
        winners.append('tie' if choice == 'tie' else r['metadata']['response_' + choice.lower() + '_method'])
    if winners[0] == winners[1]:
        return winners[0]
    return 'order_disagreement'


def targets():
    inputs = rows(ROOT / 'generation/train.jsonl')
    assert len(inputs) == 276 and len({r['canonical_id'] for r in inputs}) == 276
    requests = [request('train_teacher', TEACHER_SYSTEM, TEACHER_USER.format(question=r['question'], response=r['response']),
                        'train_residual_teacher', 1536, {'canonical_id': r['canonical_id']})
                for r in inputs if r['technical_valid']]
    teacher_results = execute_requests('train_teacher', requests)
    teachers = {r['metadata']['canonical_id']: teacher_results.get(r['request_id']) for r in requests}
    candidates, verifier_requests, pairs = {}, [], []
    for r in inputs:
        teacher = teachers.get(r['canonical_id'])
        if not teacher or not teacher.get('ok') or teacher.get('returned_model') != MODEL:
            continue
        t = teacher['normalized']
        candidate = t['response'].strip()
        if t['action'] == 'preserve' and candidate != r['response']:
            continue
        if flags(candidate):
            continue
        candidates[r['canonical_id']] = candidate
        if candidate == r['response']:
            continue
        verifier_requests.append(request('train_verify', VERIFY_SYSTEM, VERIFY_USER.format(question=r['question'], draft=r['response'], candidate=candidate),
                                         'train_residual_verify', 512, {'canonical_id': r['canonical_id']}))
        pairs.extend(pair_requests('train_preference', r['canonical_id'], r['question'], r['response'], candidate, 'base', 'target'))
    verifier_results = execute_requests('train_verify', verifier_requests)
    verifier_results, recovered_count = recover_verifier_schema(verifier_results)
    pair_results = execute_requests('train_preference', pairs)
    verifier_by_id = {r['metadata']['canonical_id']: verifier_results.get(r['request_id']) for r in verifier_requests}
    decisions = []
    for r in inputs:
        cid = r['canonical_id']
        candidate = candidates.get(cid)
        decision, winner, target = 'excluded', None, None
        reason = 'teacher_failed_or_invalid_or_base_technical_failure'
        if candidate == r['response'] and candidate:
            decision, target, reason = 'preserve', r['response'], 'teacher_preserved_exact_clean_draft'
        elif candidate:
            reqs = pair_requests('train_preference', cid, r['question'], r['response'], candidate, 'base', 'target')
            winner = consensus(reqs, pair_results)
            v = verifier_by_id.get(cid)
            checked = bool(v and v.get('ok') and v.get('returned_model') == MODEL)
            issues = [k for k in ('unsupported_claims', 'specificity_loss', 'inappropriate_refusal', 'surface_failure')
                      if checked and v['normalized'][k]]
            if winner == 'target' and checked and not issues:
                decision, target, reason = 'edit', candidate, 'two_order_preference_and_grounding_check_passed'
            elif winner in {'base', 'tie'} and not flags(r['response']):
                decision, target, reason = 'preserve', r['response'], 'two_order_' + winner
            else:
                reason = winner + (';verifier_failure' if not checked else ';verifier:' + ','.join(issues))
        decisions.append({**r, 'teacher_result': teachers.get(cid), 'teacher_candidate': candidate,
            'verification': verifier_by_id.get(cid), 'preference': winner, 'decision': decision,
            'decision_reason': reason, 'target_response': target,
            'target_sha256': sha(target) if target is not None else None})
    fit_edits = [r for r in decisions if r['train_partition'] == 'fit' and r['decision'] == 'edit']
    fit_preserve = sorted([r for r in decisions if r['train_partition'] == 'fit' and r['decision'] == 'preserve'],
                          key=lambda r: sha(str(SEED) + r['canonical_id']))[:len(fit_edits)]
    fit_ids = {r['canonical_id'] for r in fit_edits + fit_preserve}
    for r in decisions:
        r['selected_for_training'] = r['canonical_id'] in fit_ids
        r['selected_for_holdout'] = r['train_partition'] == 'holdout' and r['decision'] in {'edit', 'preserve'}
    write_rows(ROOT / 'data/curation_all276.jsonl', decisions)
    summary = {'decisions': dict(collections.Counter(r['decision'] for r in decisions)),
        'reasons': dict(collections.Counter(r['decision_reason'] for r in decisions)), 'fit_edits': len(fit_edits),
        'fit_preserve': len(fit_preserve), 'holdout': sum(r['selected_for_holdout'] for r in decisions),
        'synthetic_targets_not_clinical_gold': True, 'schema_recovered_verifier_results': recovered_count}
    write(ROOT / 'data/curation_summary.json', summary)
    print(json.dumps(summary), flush=True)
    if len(fit_edits) < 8 or summary['holdout'] < 8:
        raise RuntimeError('Insufficient verified targets; no training launch; see curation summary')


def enrich():
    from train_gemma_risk_tune_from_sft import load_classifier
    from selective_risk_refinement_utils import configure_risk_contract, score_candidate
    import torch
    selected = [r for r in rows(ROOT / 'data/curation_all276.jsonl') if r['selected_for_training'] or r['selected_for_holdout']]
    path = ROOT / 'data/enriched.jsonl'
    history = {r['canonical_id']: r for r in rows(path)}
    pending = [r for r in selected if r['canonical_id'] not in history]
    if pending:
        gpu_idle()
        configure_risk_contract('fullpaper_v1')
        rt, router = load_classifier(str(CKPT / 'router/final'), 'cuda')
        st, scorer = load_classifier(str(CKPT / 'scorer/final'), 'cuda')
        for r in pending:
            score = score_candidate(r['question'], r['response'], router, rt, scorer, st, 'cuda',
                                    router_max_len=512, risk_max_len=384, component_mode='full')
            value = {k: r[k] for k in ('canonical_id', 'question', 'unsafe_response', 'train_partition', 'decision', 'decision_reason', 'selected_for_training', 'selected_for_holdout')}
            value.update(sft_response=r['response'], z_t_from_sft=r['response'], safe_response=r['target_response'],
                         g_sft=score['g'], sft_risk_score=score['risk_score'], risk=score,
                         zt_strategy='no_mask', timestep=3, risk_component_mode='full', aspect_conditioning_status='available')
            append(path, value)
        del router, scorer
        torch.cuda.empty_cache()
    history = {r['canonical_id']: r for r in rows(path)}
    assert set(history) == {r['canonical_id'] for r in selected}
    write_rows(ROOT / 'data/train.jsonl', [history[r['canonical_id']] for r in selected if r['selected_for_training']])
    write_rows(ROOT / 'data/holdout.jsonl', [history[r['canonical_id']] for r in selected if r['selected_for_holdout']])


def q3(value):
    return statistics.mean(value['scores'][a] for a in ('overall_quality', 'empathy', 'specificity'))


def eval_prepare():
    new = rows(ROOT / 'generation/valid.jsonl')
    assert len(new) == 43
    assignments = []
    # Copy VALID-only cached responses; no TEST cache or artifact is inspected.
    old = rows(PREVIOUS / 'inputs/second_pass_pointwise_assignments.jsonl')
    for r in old:
        assignments.append({'canonical_id': r['canonical_id'], 'question': r['question'], 'method': r['method'], 'response': r['response']})
    for r in rows(PREVIOUS / 'results/second_pass_final_outputs.jsonl'):
        if r['method'] in {'BaseSelfRefineAllGeneratedGPTSelected', 'TrainedDenoiserOnBaseAllGeneratedGPTSelected'}:
            assignments.append({'canonical_id': r['canonical_id'], 'question': r['question'],
                                'method': r['method'], 'response': r['response']})
    for r in new:
        assignments.append({'canonical_id': r['canonical_id'], 'question': r['question'],
            'method': 'NewBaseResidualCandidate', 'response': r['response']})
    prompts = read(ROOT / 'run_contract.json')['settings']['frozen_evaluation_prompts']
    pw = []
    for r in assignments:
        r['question_response_sha256'] = sha([r['question'], r['response']])
        req = request('valid_pointwise', prompts['pointwise_system'], prompts['pointwise_user_template'].format(
            question=r['question'], response=r['response']), 'pointwise_expert_rubric', 384,
            {'question_response_sha256': r['question_response_sha256']})
        r['pointwise_request_id'] = req['request_id']
        pw.append(req)
    unique_pw = {r['request_id']: r for r in pw}
    cache = {}
    for p in (PREVIOUS / 'results').glob('*.jsonl'):
        for r in rows(p):
            if r.get('parser') == 'pointwise_expert_rubric' and r.get('ok') and r.get('returned_model') == MODEL:
                material = {k: r.get(k) for k in ('model', 'temperature', 'max_tokens', 'response_format', 'parser', 'system_prompt', 'user_prompt')}
                cache.setdefault(sha(material)[:32], (p, r))
    output = ROOT / 'results/valid_pointwise.jsonl'
    existing = {r['request_id'] for r in rows(output)}
    for rid, req in unique_pw.items():
        if rid in cache and rid not in existing:
            p, original = cache[rid]
            value = {**original, **req, 'request_sha256': queue.request_sha256(req),
                     'reused_from': str(p), 'source_request_id': original['request_id'], 'estimated_cost_usd': 0,
                     'original_usage': original.get('usage'), 'usage': {}}
            append(output, value)
    write_rows(ROOT / 'inputs/valid_assignments.jsonl', assignments)
    results = execute_requests('valid_pointwise', list(unique_pw.values()))
    pairs = []
    old_by_key = {(r['canonical_id'], r['method']): r for r in assignments}
    for r in new:
        cid = r['canonical_id']
        for baseline in ('BasePromptRewrite', 'BaseSelfRefineCandidate', 'TrainedDenoiserOnBaseCandidate'):
            source = old_by_key[(cid, baseline)]
            pairs.extend(pair_requests('valid_pairwise', cid, r['question'], source['response'], r['response'], baseline, 'NewBaseResidualCandidate'))
    execute_requests('valid_pairwise', pairs)
    write(ROOT / 'evaluation_preparation.json', {'assignments': len(assignments), 'unique_pointwise': len(unique_pw),
        'reused': sum(bool(r.get('reused_from')) for r in results.values()), 'pairwise': len(pairs)})


def llama():
    import run_valid43_base_draft_control_llama as local
    local.ROOT = ROOT
    local.CACHE_ROOTS = (PREVIOUS,)
    local.BATCH_SIZE = 8
    requests = rows(ROOT / 'requests/valid_pointwise.jsonl')
    write_rows(ROOT / 'requests/second_pass_expected_pointwise.jsonl', requests)
    local.prepare()
    if rows(ROOT / 'requests/second_pass_new_llama.jsonl'):
        gpu_idle()
        local.run()


def score_valid_risk():
    import torch
    from train_gemma_risk_tune_from_sft import load_classifier
    from selective_risk_refinement_utils import configure_risk_contract, score_candidate
    output = ROOT / 'results/valid_risk.jsonl'
    assignments = rows(ROOT / 'inputs/valid_assignments.jsonl')
    history = {r['question_response_sha256']: r for r in rows(output)}
    previous_cache = {}
    for p in (PREVIOUS / 'risk').glob('*pair*risk.jsonl'):
        for r in rows(p):
            if r.get('risk_contract') == 'fullpaper_v1' and r.get('component_mode') == 'full':
                previous_cache[r['question_response_sha256']] = r
    pending = []
    for r in assignments:
        key = r['question_response_sha256']
        if key in history:
            continue
        if key in previous_cache:
            result = {**previous_cache[key], 'question_response_sha256': key, 'reused': True}
            append(output, result)
            history[key] = result
        else:
            pending.append(r)
    if pending:
        gpu_idle()
        configure_risk_contract('fullpaper_v1')
        rt, router = load_classifier(str(CKPT / 'router/final'), 'cuda')
        st, scorer = load_classifier(str(CKPT / 'scorer/final'), 'cuda')
        for r in pending:
            key = r['question_response_sha256']
            if key in history:
                continue
            result = score_candidate(r['question'], r['response'], router, rt, scorer, st, 'cuda',
                                     router_max_len=512, risk_max_len=384, component_mode='full')
            result.update(question_response_sha256=key, canonical_id=r['canonical_id'], response_sha256=sha(r['response']))
            append(output, result)
            history[key] = result
        del router, scorer
        torch.cuda.empty_cache()


def aggregate():
    assignments = rows(ROOT / 'inputs/valid_assignments.jsonl')
    pw = {r['request_id']: r for r in rows(ROOT / 'results/valid_pointwise.jsonl')}
    pairs = {r['request_id']: r for r in rows(ROOT / 'results/valid_pairwise.jsonl')}
    by_key = {(r['canonical_id'], r['method']): r for r in assignments}
    judgments, finals = [], []
    new = rows(ROOT / 'generation/valid.jsonl')
    for r in new:
        cid = r['canonical_id']
        for baseline in ('BasePromptRewrite', 'BaseSelfRefineCandidate', 'TrainedDenoiserOnBaseCandidate'):
            original = by_key[(cid, baseline)]
            reqs = pair_requests('valid_pairwise', cid, r['question'], original['response'], r['response'], baseline, 'NewBaseResidualCandidate')
            winner = consensus(reqs, pairs)
            judgments.append({'canonical_id': cid, 'baseline': baseline, 'winner': winner})
            if baseline == 'BasePromptRewrite':
                accepted = winner == 'NewBaseResidualCandidate' and r['technical_valid']
                final = dict(by_key[(cid, 'NewBaseResidualCandidate' if accepted else baseline)])
                final.update(method='NewBaseResidualSelected', accepted=accepted, selection_result=winner,
                             technical_candidate_valid=r['technical_valid'])
                finals.append(final)
    write_rows(ROOT / 'results/valid_final_responses.jsonl', finals)
    write_rows(ROOT / 'results/valid_pairwise_decisions.jsonl', judgments)
    local_scores = {}
    for r in rows(ROOT / 'results/second_pass_reused_llama.jsonl'):
        local_scores[r['target_request_id']] = r['judge_normalized']
    for r in rows(ROOT / 'results/second_pass_new_llama.jsonl'):
        if r.get('judge_ok'):
            local_scores[r['request_id']] = r['judge_normalized']
    risks = {r['question_response_sha256']: r['risk_score'] for r in rows(ROOT / 'results/valid_risk.jsonl')}
    scored = []
    for judge in ('GPT', 'Llama'):
        for r in assignments + finals:
            result = pw.get(r['pointwise_request_id'])
            normalized = (result['normalized'] if result and result.get('ok') else None) if judge == 'GPT' else local_scores.get(r['pointwise_request_id'])
            scored.append({**r, 'judge': judge, 'evaluation_ok': normalized is not None,
                          'quality_3axis_mean': q3(normalized) if normalized else None,
                          'scores': normalized, 'risk': risks.get(r['question_response_sha256'])})
    write_rows(ROOT / 'results/valid_scored_assignments.jsonl', scored)
    summary, matched = [], []
    for judge in ('GPT', 'Llama'):
        base = {r['canonical_id']: r for r in scored if r['judge'] == judge and r['method'] == 'BasePromptRewrite' and r['evaluation_ok']}
        for method in sorted({r['method'] for r in scored}):
            method_rows = [r for r in scored if r['judge'] == judge and r['method'] == method]
            good = [r for r in method_rows if r['evaluation_ok']]
            med = collections.Counter(r['scores']['medical_advice'] for r in good)
            denom = med['yes'] + med['no']
            valid_risk = [r['risk'] for r in method_rows if r['risk'] is not None]
            value = {'judge': judge, 'method': method, 'N': len(good), 'failed': 43 - len(good),
                     'quality_3axis_mean': statistics.mean(r['quality_3axis_mean'] for r in good) if good else None,
                     **{a: statistics.mean(r['scores']['scores'][a] for r in good) if good else None for a in ('overall_quality', 'empathy', 'specificity')},
                     'medical_yes': med['yes'], 'medical_no': med['no'], 'medical_unsure': med['unsure'],
                     'medical_yes_denominator': denom, 'medical_yes_rate': med['yes'] / denom if denom else None,
                     'mean_risk': statistics.mean(valid_risk) if len(valid_risk) == 43 else None, 'risk_N': len(valid_risk),
                     'mean_words': statistics.mean(len(r['response'].split()) for r in method_rows),
                     'surface_flagged_responses': sum(bool(flags(r['response'])) for r in method_rows)}
            summary.append(value)
            paired = [r for r in good if r['canonical_id'] in base]
            deltas = [r['quality_3axis_mean'] - base[r['canonical_id']]['quality_3axis_mean'] for r in paired]
            matched.append({'judge': judge, 'method': method, 'N': len(paired),
                'win': sum(d > 1e-9 for d in deltas), 'tie': sum(abs(d) <= 1e-9 for d in deltas),
                'loss': sum(d < -1e-9 for d in deltas), 'mean_delta': statistics.mean(deltas) if deltas else None})
    training_effect = []
    for judge in ('GPT', 'Llama'):
        for method, control in [('NewBaseResidualCandidate', 'BaseSelfRefineCandidate'),
                                ('NewBaseResidualSelected', 'BaseSelfRefineAllGeneratedGPTSelected')]:
            left = {r['canonical_id']: r for r in scored if r['judge'] == judge and r['method'] == method and r['evaluation_ok']}
            right = {r['canonical_id']: r for r in scored if r['judge'] == judge and r['method'] == control and r['evaluation_ok']}
            ids = sorted(set(left) & set(right))
            deltas = [left[cid]['quality_3axis_mean'] - right[cid]['quality_3axis_mean'] for cid in ids]
            training_effect.append({'judge': judge, 'method': method, 'control': control, 'N': len(ids),
                'win': sum(d > 1e-9 for d in deltas), 'tie': sum(abs(d) <= 1e-9 for d in deltas),
                'loss': sum(d < -1e-9 for d in deltas), 'mean_delta': statistics.mean(deltas) if deltas else None})
    write(ROOT / 'results/summary.json', {'summary': summary, 'matched_vs_base': matched, 'matched_training_effect': training_effect,
        'pairwise': {b: dict(collections.Counter(r['winner'] for r in judgments if r['baseline'] == b)) for b in ('BasePromptRewrite', 'BaseSelfRefineCandidate', 'TrainedDenoiserOnBaseCandidate')},
        'accepted': sum(r['accepted'] for r in finals), 'changed': sum(r['response'] != by_key[(r['canonical_id'], 'BasePromptRewrite')]['response'] for r in finals),
        'test_used': False})
    for name, data in [('quality_summary', summary), ('matched_vs_base', matched), ('matched_training_effect', training_effect)]:
        path = ROOT / f'results/{name}.csv'
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps(read(ROOT / 'results/summary.json')), flush=True)


def package():
    import zipfile
    code_snapshot = ROOT / 'code_at_completion'
    code_snapshot.mkdir(exist_ok=True)
    for name in ('residual_train_common.py', 'run_base_residual_generation.py',
                 'run_base_residual_experiment.py', 'train_base_residual_adapter.py',
                 'run_base_residual_pipeline.sh', 'test_base_residual_pilot.py'):
        shutil.copy2(CODE / name, code_snapshot / name)
    summary = read(ROOT / 'results/summary.json')
    curation = read(ROOT / 'data/curation_summary.json')
    training = read(ROOT / 'training/manifest.json')
    api = [r for p in (ROOT / 'results').glob('*.jsonl') for r in rows(p)
           if 'request_sha256' in r and 'ok' in r and not r.get('reused_from') and not r.get('schema_recovery')]
    latest = {r['request_id']: r for r in api}
    for r in rows(ROOT / 'results/train_verify_schema_recovered.jsonl'):
        latest[r['request_id']] = r
    usage = {'checkpoint_records': len(api),
             'historical_failed_records': sum(not r['ok'] for r in api),
             'unique_requests': len(latest),
             'final_failed_unique_requests': sum(not r['ok'] for r in latest.values()),
             'successful_api_responses': sum(len(r.get('api_response_history', [])) for r in api),
             'quota_not_attempted_records': sum(r.get('error_type') == 'QuotaNotAttempted' for r in api),
             'prompt_tokens': sum(r.get('usage', {}).get('prompt_tokens', 0) for r in api),
             'completion_tokens': sum(r.get('usage', {}).get('completion_tokens', 0) for r in api),
             'estimated_cost_usd': sum(r.get('estimated_cost_usd', 0) for r in api),
             'estimate_only': True, 'prices_usd_per_million': {'input': 2, 'output': 8}}
    usage['schema_recovered_without_new_api_calls'] = len(rows(ROOT / 'results/train_verify_schema_recovered.jsonl'))
    write(ROOT / 'api_usage.json', usage)
    lines = ['# TRAIN-only Base Residual Pilot', '', 'Development study only; no TEST files were read or changed.', '',
        '## Targets and Training', f'- Curation: `{json.dumps(curation)}`',
        f'- Selected adapter: `{training["selected_adapter"]}`',
        '- New zero-output LoRA initialized on the instruction-tuned Base, not the old SFT or denoiser.',
        '- New targets are GPT-preferred synthetic targets, not expert-validated clinical ground truth.',
        '- TRAIN-internal holdout selects the checkpoint. VALID43 does not supply training targets or early-stopping loss.', '',
        '## VALID43 Results', '| Judge | Method | N | Q3 | Overall | Empathy | Specificity | Medical yes/no/unsure |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for r in summary['summary']:
        numeric = [f'{r[k]:.3f}' if r[k] is not None else 'NA' for k in ('quality_3axis_mean', 'overall_quality', 'empathy', 'specificity')]
        lines.append(f'| {r["judge"]} | {r["method"]} | {r["N"]} | ' + ' | '.join(numeric) + f' | {r["medical_yes"]}/{r["medical_no"]}/{r["medical_unsure"]} |')
    lines += ['', 'Matched comparisons (failures excluded, never ties):', '```json', json.dumps(summary['matched_vs_base'], indent=2), '```',
        '', 'Matched comparison with the no-adapter second-pass control (training contribution):',
        '```json', json.dumps(summary['matched_training_effect'], indent=2), '```',
        '', f'Consensus-selected revisions: {summary["accepted"]}/43; genuinely changed texts: {summary["changed"]}/43.',
        'No frozen risk threshold was transferred onto Base or recalibrated in this pilot. All 43 candidates were generated for development analysis.',
        '', '## Interpretation Limits',
        '- A shared GPT teacher, verifier, selector, and evaluator is not independent validation. Llama is auxiliary and may disagree or fail parsing.',
        '- Candidate quality and selector-filtered final quality must not be conflated. All original Base and no-adapter self-refinement controls are shown.',
        '- The initialization and training targets both differ from the earlier denoiser, so this is not an isolated attribution of their effects.',
        '- Previously inspected VALID43 is a development set. No claim of TEST improvement is warranted.',
        '- Medical Advice is categorical and is not included in Q3. Missing scores and risk values are not imputed.',
        '- Original source responses, raw outputs, failed requests, and rejected synthetic targets are retained.', '',
        '- CREDIT_BLOCK_CHECKPOINT.json and code_at_credit_block are historical records, not the final completion state.',
        '- Verifier evidence_phrase aliases were recovered from saved raw JSON without changing judgments or repeating API calls.', '',
        '## API Usage', '```json', json.dumps(usage, indent=2), '```']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    latex = ['# Insertable LaTeX: TRAIN-Only Pilot', '', 'Development results, not TEST results.', '', '```latex',
        r'\begin{table}[t]', r'\centering', r'\small', r'\begin{tabular}{llrrr}',
        r'\hline', r'Judge & Method & $N$ & Quality-3 & MA (Y/N/U) \\', r'\hline']
    names = {'BasePromptRewrite': 'Base rewrite', 'BaseSelfRefineCandidate': 'No-adapter refinement',
             'TrainedDenoiserOnBaseCandidate': 'Previous denoiser', 'NewBaseResidualCandidate': 'New residual',
             'BaseSelfRefineAllGeneratedGPTSelected': 'No-adapter + selection',
             'TrainedDenoiserOnBaseAllGeneratedGPTSelected': 'Previous + selection',
             'NewBaseResidualSelected': 'New residual + selection'}
    for r in summary['summary']:
        q = f'{r["quality_3axis_mean"]:.3f}' if r['quality_3axis_mean'] is not None else '--'
        latex.append(f'{r["judge"]} & {names[r["method"]]} & {r["N"]} & {q} & {r["medical_yes"]}/{r["medical_no"]}/{r["medical_unsure"]}' + r' \\')
    latex += [r'\hline', r'\end{tabular}',
        r'\caption{Exploratory VALID43 development results. Quality-3 averages Overall Quality, Empathy, and Specificity. Medical Advice (MA) is categorical and excluded from Quality-3. Failed evaluations are excluded, with actual $N$ shown.}',
        r'\label{tab:train-base-residual-valid}', r'\end{table}', '',
        r'\paragraph{Base-conditioned residual training.}',
        f'We generated drafts only for the 276 frozen training questions and reserved 28 of these questions as an internal holdout before target generation. '
        f'One GPT-4.1 revision was proposed per technically valid draft. Revision targets required two-order blinded preference and a separate content-preservation check. '
        f'The fitting set contained {training["train_rows"]} eligible examples, with {training["holdout_rows"]} internal-holdout examples after curation and length checks. '
        'A fresh LoRA adapter was initialized on the instruction-tuned base model. Checkpoint selection used only internal-holdout target-token negative log likelihood.',
        r'\paragraph{Evaluation and limitations.}',
        'We evaluated one candidate per VALID43 question using fixed stored drafts, prompts, decoding, and evaluation criteria. '
        'Candidate quality and two-order-consensus-selected final quality are reported separately. No TEST output was generated, evaluated, or modified. '
        'The shared GPT teacher, target validator, selector, and evaluator are not independent evidence; Llama provides an auxiliary comparison. '
        'VALID43 was previously inspected during development. Changing both initialization and target distribution does not isolate their individual causal effects.', '```']
    (ROOT / 'PUBLICATION_LATEX.md').write_text('\n'.join(latex) + '\n', encoding='utf-8')
    files = [p for p in ROOT.rglob('*') if p.is_file()
             and p.suffix not in {'.zip', '.lock', '.pt', '.safetensors', '.log'}
             and not any(part in {'runtime_cache', 'tmp'} or part.startswith('epoch_') for part in p.relative_to(ROOT).parts)
             and p.name not in {'ARTIFACT_MANIFEST.json', 'progress.json', 'download_copy_status.json'}]
    write(ROOT / 'ARTIFACT_MANIFEST.json', {'status': 'complete',
        'evaluation_coverage': [{'judge': r['judge'], 'method': r['method'], 'completed': r['N'], 'failed': r['failed']} for r in summary['summary']],
        'files': [{'path': str(p.relative_to(ROOT)), 'sha256': file_sha(p)} for p in files],
        'adapter': training['selected_adapter'], 'adapter_weights_sha256': file_sha(Path(training['selected_adapter']) / 'adapter_model.safetensors'),
        'weights_excluded_from_report_zip': True})
    archive = ROOT / 'TRAIN_base_residual_VALID43_research_bundle.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files + [ROOT / 'ARTIFACT_MANIFEST.json']:
            z.write(p, str(p.relative_to(ROOT)))
    download = Path('/home/user/Downloads')
    download.mkdir(exist_ok=True)
    if shutil.disk_usage(download).free < archive.stat().st_size * 2:
        write(ROOT / 'download_copy_status.json', {'copied': False, 'reason': 'insufficient free space', 'source': str(archive)})
    else:
        shutil.copy2(archive, download / archive.name)
        write(ROOT / 'download_copy_status.json', {'copied': True, 'path': str(download / archive.name)})
    print(json.dumps({'zip': str(archive), 'sha256': file_sha(archive)}), flush=True)


def checkpoint():
    """Export the real, incomplete state without making any API requests."""
    import zipfile
    teachers = {r['metadata']['canonical_id']: r for r in rows(ROOT / 'results/train_teacher.jsonl')}
    verifiers = {r['request_id']: r for r in rows(ROOT / 'results/train_verify.jsonl')}
    verified, recovered = recover_verifier_schema(verifiers)
    preference_requests = []
    for r in rows(ROOT / 'generation/train.jsonl'):
        t = teachers.get(r['canonical_id'])
        if not t or not t.get('ok') or not r['technical_valid']:
            continue
        candidate = t['normalized']['response'].strip()
        if flags(candidate) or (t['normalized']['action'] == 'preserve' and candidate != r['response']):
            continue
        if candidate != r['response']:
            preference_requests.extend(pair_requests('train_preference', r['canonical_id'], r['question'], r['response'], candidate, 'base', 'target'))
    write_rows(ROOT / 'requests/train_preference.jsonl', preference_requests)
    expected_verify = rows(ROOT / 'requests/train_verify.jsonl')
    pending_verify = [r for r in expected_verify if not verified.get(r['request_id'], {}).get('ok')]
    write_rows(ROOT / 'requests/remaining_verification_after_credit_topup.jsonl', pending_verify)
    model_results = [r for p in (ROOT / 'results').glob('*.jsonl') for r in rows(p)
                     if 'request_sha256' in r and not r.get('schema_recovery') and r.get('usage')]
    info = {'status': 'blocked_external_api_credit_balance_exhausted', 'train_base_generated': len(rows(ROOT / 'generation/train.jsonl')),
            'teacher_generated': sum(r.get('ok') is True for r in teachers.values()),
            'verification_expected': len(expected_verify), 'verification_valid_after_raw_recovery': sum(r.get('ok') is True for r in verified.values()),
            'verification_raw_schema_recovered_without_api': recovered, 'verification_remaining': len(pending_verify),
            'blinded_preference_completed': len(rows(ROOT / 'results/train_preference.jsonl')),
            'blinded_preference_expected_unique': len({r['request_id'] for r in preference_requests}),
            'training_started': (ROOT / 'training/initialization_audit.json').exists(),
            'new_valid_generation_completed': len(rows(ROOT / 'generation/valid.jsonl')),
            'test_read_or_modified': False,
            'resume_command': f'bash {CODE}/run_base_residual_pipeline.sh',
            'resume_condition': 'Replenish the same configured OpenAI API account; no new key printed or requested',
            'model_responses_with_usage': len(model_results),
            'prompt_tokens': sum(r['usage'].get('prompt_tokens', 0) for r in model_results),
            'completion_tokens': sum(r['usage'].get('completion_tokens', 0) for r in model_results),
            'estimated_cost_usd': sum(r.get('estimated_cost_usd', 0) for r in model_results),
            'cost_is_local_runner_estimate_not_account_balance': True}
    write(ROOT / 'CREDIT_BLOCK_CHECKPOINT.json', info)
    write(ROOT / 'progress.json', {'status': info['status'], 'stage': 'target_verification', 'resume': info['resume_command']})
    source_paths = [CODE / n for n in ('residual_train_common.py', 'run_base_residual_generation.py',
                     'run_base_residual_experiment.py', 'train_base_residual_adapter.py', 'test_base_residual_pilot.py',
                     'run_base_residual_pipeline.sh')]
    for p in source_paths:
        dest = ROOT / 'code_at_credit_block' / p.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(p, dest)
    write(ROOT / 'code_at_credit_block/source_manifest.json', {'sources': [{'path': str(p), 'sha256': file_sha(p)} for p in source_paths],
        'notes': 'Offline evidence/evidence_phrase alias recovery and quota-resume safeguards; frozen prompts, targets and acceptance logic unchanged'})
    text = '# Incomplete TRAIN-Only Pilot Checkpoint\n\n'
    text += 'API credits were exhausted during target verification. Training and VALID evaluation have NOT run.\n\n'
    text += 'No targets have been declared better than Base before the required two-order preference comparison.\n\n'
    text += 'The evidence/evidence_phrase parser mismatch was repaired from saved JSON only; original failures and raw responses are preserved.\n\n'
    text += '```json\n' + json.dumps(info, indent=2) + '\n```\n\n'
    text += 'After replenishing API credits, resume only unfinished requests:\n\n```bash\n' + info['resume_command'] + '\n```\n'
    (ROOT / 'CHECKPOINT_README.md').write_text(text, encoding='utf-8')
    files = [p for p in ROOT.rglob('*') if p.is_file() and p.suffix not in {'.zip', '.lock', '.log', '.pt', '.safetensors'}
             and not any(part in {'runtime_cache', 'tmp'} for part in p.relative_to(ROOT).parts)
             and p.name != 'CHECKPOINT_INVENTORY.json']
    write(ROOT / 'CHECKPOINT_INVENTORY.json', {'files': [{'path': str(p.relative_to(ROOT)), 'sha256': file_sha(p)} for p in files]})
    archive = ROOT / 'TRAIN_base_residual_partial_checkpoint.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files + [ROOT / 'CHECKPOINT_INVENTORY.json']:
            z.write(p, str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for record in read(ROOT / 'CHECKPOINT_INVENTORY.json')['files']:
            assert hashlib.sha256(z.read(record['path'])).hexdigest() == record['sha256']
    download = Path('/home/user/Downloads')
    download.mkdir(exist_ok=True)
    if shutil.disk_usage(download).free > archive.stat().st_size * 2:
        shutil.copy2(archive, download / archive.name)
    print(json.dumps({**info, 'zip': str(archive), 'zip_sha256': file_sha(archive)}, indent=2), flush=True)


def pipeline():
    for variable, name in (('TMPDIR', 'tmp'), ('TRITON_CACHE_DIR', 'triton'),
                           ('TORCHINDUCTOR_CACHE_DIR', 'torchinductor'), ('CUDA_CACHE_PATH', 'cuda')):
        path = ROOT / 'runtime_cache' / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    lock = ROOT / 'pipeline.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for stage in ('targets', 'enrich', 'train', 'generate-valid', 'eval', 'risk', 'llama', 'aggregate', 'package'):
            write(ROOT / 'progress.json', {'stage': stage, 'status': 'running'})
            if stage == 'train':
                command = [sys.executable, str(CODE / 'train_base_residual_adapter.py')]
            elif stage == 'generate-valid':
                command = [sys.executable, str(CODE / 'run_base_residual_generation.py'), 'valid']
            else:
                command = [sys.executable, str(Path(__file__).resolve()), stage]
            result = subprocess.run(command, cwd=CODE.parent)
            if result.returncode:
                write(ROOT / 'progress.json', {'stage': stage, 'status': 'failed', 'returncode': result.returncode,
                    'resume': f'{sys.executable} {Path(__file__).resolve()} pipeline'})
                raise SystemExit(result.returncode)
        write(ROOT / 'progress.json', {'status': 'complete', 'test_used': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'teacher-prefetch', 'targets', 'enrich', 'eval', 'risk', 'llama', 'aggregate', 'package', 'checkpoint', 'pipeline'])
    stage = parser.parse_args().stage
    {'prepare': prepare, 'teacher-prefetch': teacher_prefetch, 'targets': targets, 'enrich': enrich, 'eval': eval_prepare,
     'risk': score_valid_risk, 'llama': llama, 'aggregate': aggregate, 'package': package, 'checkpoint': checkpoint, 'pipeline': pipeline}[stage]()
