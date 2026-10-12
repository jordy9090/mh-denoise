#!/usr/bin/env python3
"""Preserved ordinary GPT-Select aggregation of existing frozen judgments.

This adapter does not prepare requests, call APIs, or generate/evaluate responses.
LegacyStrict is the historical seven-adoption rule-based comparison, not GPT-Select.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


RUN = Path("outputs/ordinary_gpt_selector")
SOURCE = RUN / "source_snapshot"
EVIDENCE_AUDIT: Path | None = None
OUTPUT = Path("outputs/ordinary_gpt_selector_aggregation")
AXES = ('overall_quality', 'empathy', 'specificity')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def verify_inputs():
    """Validate every historical frozen input without retaining personal paths."""
    contract = json.loads((RUN / 'run_contract.json').read_text())
    source_hashes = contract['source_hashes']
    names = [Path(path).name for path in source_hashes]
    if len(names) != len(set(names)) or len(names) != 11:
        raise ValueError('Expected the eleven unambiguous historical source files')
    for original, expected in source_hashes.items():
        path = SOURCE / Path(original).name
        if digest(path.read_bytes()) != expected:
            raise ValueError(f'Frozen source hash mismatch: {path.name}')
    if digest((RUN / 'requests.jsonl').read_bytes()) != contract['requests_sha256']:
        raise ValueError('Frozen request hash mismatch')
    frozen = json.loads((SOURCE / 'run_contract.json').read_text())
    if json.loads((RUN / 'frozen_selector_contract.json').read_text()) != frozen:
        raise ValueError('Frozen selector contract differs from its source snapshot')
    for name in ('selector_system', 'selector_user_template'):
        if digest(frozen['prompts'][name].encode()) != frozen['prompts'][name + '_sha256']:
            raise ValueError(f'Frozen prompt hash mismatch: {name}')
    requests = read_rows(RUN / 'requests.jsonl')
    if len(requests) != contract['request_count'] or len({row['request_id'] for row in requests}) != len(requests):
        raise ValueError('Frozen request count or identity mismatch')
    source = {row['canonical_id']: row for row in read_rows(SOURCE / 'natural_final.jsonl')}
    expected_orders = {(key, order) for key, row in source.items() if row['used_denoiser'] for order in (0, 1)}
    observed_orders = [(row['metadata']['canonical_id'], row['metadata']['order_index']) for row in requests]
    if len(observed_orders) != len(expected_orders) or set(observed_orders) != expected_orders:
        raise ValueError('Frozen candidate/order coverage changed')
    settings = contract['selector_settings']
    for row in requests:
        if any(row[key] != value for key, value in settings.items()):
            raise ValueError('Frozen selector settings changed')
        if row['system_prompt'] != frozen['prompts']['selector_system']:
            raise ValueError('Frozen selector system prompt changed')
        metadata = row['metadata']
        original = source[metadata['canonical_id']]
        responses = [original['sft_response'], original['denoiser_response']]
        if metadata['order_index'] == 1:
            responses.reverse()
        expected_prompt = frozen['prompts']['selector_user_template'].format(
            question=original['question'], response_a=responses[0], response_b=responses[1])
        if row['user_prompt'] != expected_prompt:
            raise ValueError('Frozen selector user prompt or response pair changed')
        if (metadata['question_sha256'] != digest(original['question'].encode()) or
                metadata['response_A_sha256'] != digest(responses[0].encode()) or
                metadata['response_B_sha256'] != digest(responses[1].encode())):
            raise ValueError('Frozen question/response identity hash changed')



def quality(payload):
    return sum(payload[axis] for axis in AXES) / 3


def aggregate():
    verify_inputs()
    source = read_rows(SOURCE / 'natural_final.jsonl')
    requests = read_rows(RUN / 'requests.jsonl')
    results = {row['request_id']: row for row in read_rows(RUN / 'results.jsonl')}
    assert set(results) <= {row['request_id'] for row in requests}
    grouped = defaultdict(list)
    for request in requests:
        grouped[request['metadata']['canonical_id']].append(request)
    decisions = {}
    for canonical_id, ordered in grouped.items():
        votes = []
        for request in ordered:
            result = results.get(request['request_id'], {})
            if result:
                assert all(result[key] == request[key] for key in request)
            choice = (result.get('normalized') or {}).get('choice')
            ok = result.get('ok') is True and choice in ('A', 'B', 'tie')
            if ok:
                assert result['returned_model'] == request['model']
                assert result['finish_reason'] == 'stop'
            votes.append({'request_id': request['request_id'], 'order_index': request['metadata']['order_index'],
                          'ok': ok, 'choice': choice,
                          'candidate_selected': ok and choice == ('B' if request['metadata']['order_index'] == 0 else 'A')})
        accepted = len(votes) == 2 and all(vote['candidate_selected'] for vote in votes)
        if not all(vote['ok'] for vote in votes):
            reason = 'technical_failure'
        elif any(vote['choice'] == 'tie' for vote in votes):
            reason = 'tie'
        elif votes[0]['candidate_selected'] != votes[1]['candidate_selected']:
            reason = 'order_disagreement'
        else:
            reason = 'candidate_selected_both_orders' if accepted else 'sft_selected_both_orders'
        decisions[canonical_id] = {'canonical_id': canonical_id, 'accepted_candidate': accepted,
                                   'selection_reason': reason, 'orders': votes}
    judgments = {row['request_id']: row for row in read_rows(SOURCE / 'natural_judgments.jsonl')}
    assignments = {(row['canonical_id'], row['method']): row['request_id']
                   for row in read_rows(SOURCE / 'natural_judge_assignments.jsonl')}
    old_audit = json.loads(EVIDENCE_AUDIT.read_text())
    mismatches = {row['request_id'] for row in old_audit['evidence_mismatches']}
    methods = defaultdict(list)
    finals = []
    for row in source:
        canonical_id = row['canonical_id']
        decision = decisions.get(canonical_id, {'accepted_candidate': False, 'selection_reason': 'gate_not_called', 'orders': []})
        accepted = decision['accepted_candidate']
        response = row['denoiser_response'] if accepted else row['sft_response']
        method_sources = {'NaturalDraft': 'NaturalDraft', 'SFT': 'SFT', 'LegacyStrict': 'Final',
                          'GPT-Select': 'Candidate' if accepted else 'SFT'}
        if row['used_denoiser']:
            method_sources['Candidate'] = 'Candidate'
        for method, source_method in method_sources.items():
            request_id = assignments[canonical_id, source_method]
            expected_text = response if method == 'GPT-Select' else (
                row['unsafe_response'] if method == 'NaturalDraft' else
                row['sft_response'] if method == 'SFT' else
                row['final_response'] if method == 'LegacyStrict' else row['denoiser_response'])
            assert digest((row['question'] + '\0' + expected_text).encode()) == request_id
            judgment = judgments[request_id]
            payload = json.loads(judgment['judge_raw'])
            assert all(type(payload[axis]) is int and 1 <= payload[axis] <= 5 for axis in AXES)
            assert payload['medical_advice'] in ('yes', 'no', 'unsure')
            entry = {'canonical_id': canonical_id, 'evaluation_request_id': request_id,
                     'strict_ok': judgment['judge_ok'], 'judge_error': judgment['judge_error'],
                     'evidence_mismatch': request_id in mismatches, 'payload': payload}
            methods[method].append(entry)
            if method == 'GPT-Select':
                finals.append({'canonical_id': canonical_id, 'question': row['question'],
                               'used_denoiser': row['used_denoiser'], 'legacy_accepted': row['accepted_denoiser'],
                               'accepted_denoiser': accepted, 'final_response': response,
                               'final_response_sha256': digest(response.encode()),
                               'final_source': source_method, 'selection': decision, 'evaluation': entry})

    def summary(entries):
        return {'N': len(entries), 'quality_3axis_mean': float(np.mean([quality(entry['payload']) for entry in entries])),
                **{axis: float(np.mean([entry['payload'][axis] for entry in entries])) for axis in AXES},
                'medical': dict(Counter(entry['payload']['medical_advice'] for entry in entries)),
                'strict_failures': sum(not entry['strict_ok'] for entry in entries),
                'evidence_mismatches': sum(entry['evidence_mismatch'] for entry in entries)}

    def paired(left, right, strict=False):
        right_by_id = {entry['canonical_id']: entry for entry in right}
        pairs = [(entry, right_by_id[entry['canonical_id']]) for entry in left if entry['canonical_id'] in right_by_id]
        if strict:
            pairs = [(left_entry, right_entry) for left_entry, right_entry in pairs if left_entry['strict_ok'] and right_entry['strict_ok']]
        deltas = np.array([quality(left_entry['payload']) - quality(right_entry['payload']) for left_entry, right_entry in pairs])
        specificity = np.array([left_entry['payload']['specificity'] - right_entry['payload']['specificity'] for left_entry, right_entry in pairs])
        transitions = Counter(right_entry['payload']['medical_advice'] + '->' + left_entry['payload']['medical_advice'] for left_entry, right_entry in pairs)
        rng = np.random.default_rng(20261011)
        interval = np.quantile(deltas[rng.integers(0, len(pairs), size=(20000, len(pairs)))].mean(axis=1), [.025, .975]).tolist() if pairs else None
        return {'N': len(pairs), 'mean_delta': float(deltas.mean()) if pairs else None,
                'ci95': interval, 'improved': int((deltas > 1e-12).sum()),
                'tied': int((abs(deltas) <= 1e-12).sum()), 'lower': int((deltas < -1e-12).sum()),
                'mean_specificity_delta': float(specificity.mean()) if pairs else None,
                'specificity_losses': int((specificity < 0).sum()),
                'medical_transitions': dict(transitions),
                'corrections': transitions['yes->no'], 'introductions': transitions['no->yes'],
                'transition_cases_with_evidence_mismatch': [left_entry['canonical_id'] for left_entry, right_entry in pairs
                    if left_entry['payload']['medical_advice'] != right_entry['payload']['medical_advice'] and
                    (left_entry['evidence_mismatch'] or right_entry['evidence_mismatch'])],
                'excluded_ids': [entry['canonical_id'] for entry in left if entry['canonical_id'] not in {pair[0]['canonical_id'] for pair in pairs}]}

    accepted_ids = {key for key, value in decisions.items() if value['accepted_candidate']}
    legacy_ids = {row['canonical_id'] for row in source if row['accepted_denoiser']}
    report = {'cases': len(finals), 'invoked': len(decisions), 'accepted': len(accepted_ids),
              'selector_requests': len(requests), 'selector_successes': sum(row.get('ok') is True for row in results.values()),
              'decision_reasons': dict(Counter(value['selection_reason'] for value in decisions.values())),
              'accepted_ids': sorted(accepted_ids), 'legacy_accepted_ids': sorted(legacy_ids),
              'acceptance_overlap': sorted(accepted_ids & legacy_ids),
              'methods': {method: {'score_complete': summary(entries), 'strict': summary([entry for entry in entries if entry['strict_ok']])}
                          for method, entries in methods.items() if method != 'Candidate'},
              'paired': {left + '_minus_' + right: {convention: paired(methods[left], methods[right], strict)
                         for convention, strict in [('score_complete', False), ('strict_common', True)]}
                         for left, right in [('GPT-Select', 'SFT'), ('GPT-Select', 'LegacyStrict'), ('LegacyStrict', 'SFT')]},
              'candidate_effects': {group: paired([entry for entry in methods['Candidate'] if (entry['canonical_id'] in accepted_ids) == accepted], methods['SFT'])
                                   for group, accepted in [('accepted', True), ('rejected', False)]},
              'evaluation_caveats': {'unique_judgments': len(judgments), 'strict_ok': sum(row['judge_ok'] for row in judgments.values()),
                                     'unchanged_evidence_mismatch_ids': sorted(mismatches),
                                     'strict_failure_final_ids': [row['canonical_id'] for row in finals if not row['evaluation']['strict_ok']],
                                     'evidence_mismatch_final_ids': [row['canonical_id'] for row in finals if row['evaluation']['evidence_mismatch']]}}
    assert len(finals) == 60 and len(decisions) == 25
    baseline = report['methods']['SFT']['score_complete']['quality_3axis_mean']
    final_mean = report['methods']['GPT-Select']['score_complete']['quality_3axis_mean']
    assert abs(final_mean - baseline - len(accepted_ids) / 60 * (report['candidate_effects']['accepted']['mean_delta'] or 0)) < 1e-12
    write_rows(OUTPUT / 'selector_decisions.jsonl', list(decisions.values()))
    write_rows(OUTPUT / 'final60.jsonl', finals)
    write_json(OUTPUT / 'aggregate.json', report)
    sft_by_id = {entry['canonical_id']: entry for entry in methods['SFT']}
    csv_rows = []
    for row in finals:
        baseline_entry = sft_by_id[row['canonical_id']]
        final_entry = row['evaluation']
        votes = {vote['order_index']: vote['choice'] for vote in row['selection']['orders']}
        csv_rows.append({
            'canonical_id': row['canonical_id'], 'gate_called': row['used_denoiser'],
            'gpt_accepted': row['accepted_denoiser'], 'legacy_accepted': row['legacy_accepted'],
            'order0_choice': votes.get(0, ''), 'order1_choice': votes.get(1, ''),
            'selection_reason': row['selection']['selection_reason'],
            'sft_quality3': quality(baseline_entry['payload']), 'final_quality3': quality(final_entry['payload']),
            'quality3_delta': quality(final_entry['payload']) - quality(baseline_entry['payload']),
            'sft_medical': baseline_entry['payload']['medical_advice'], 'final_medical': final_entry['payload']['medical_advice'],
            'sft_strict_ok': baseline_entry['strict_ok'], 'final_strict_ok': final_entry['strict_ok'],
            'final_judge_error': final_entry['judge_error'],
            'sft_evidence_mismatch': baseline_entry['evidence_mismatch'], 'final_evidence_mismatch': final_entry['evidence_mismatch'],
            'sft_evaluation_request_id': baseline_entry['evaluation_request_id'],
            'final_evaluation_request_id': final_entry['evaluation_request_id'],
            'final_response_sha256': row['final_response_sha256'],
        })
    with (OUTPUT / 'final60_audit.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    verify_inputs()
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    global RUN, SOURCE, EVIDENCE_AUDIT, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('aggregate', 'verify'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source-snapshot', type=Path)
    parser.add_argument('--evidence-audit', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    RUN = args.run_dir
    SOURCE = args.source_snapshot or RUN / 'source_snapshot'
    if args.command == 'aggregate':
        if args.evidence_audit is None or args.output_dir is None:
            parser.error('aggregate requires --evidence-audit and --output-dir')
        OUTPUT = args.output_dir
        if any(OUTPUT.resolve().is_relative_to(path.resolve()) for path in (RUN, SOURCE)):
            parser.error('Output must be outside the frozen run and source snapshot')
        if OUTPUT.exists():
            parser.error('Use a new output directory to preserve frozen results')
        EVIDENCE_AUDIT = args.evidence_audit
        verify_inputs()
        OUTPUT.mkdir(parents=True)
        aggregate()
    else:
        verify_inputs()
        print('All eleven frozen source hashes and the request hash match.')


if __name__ == '__main__':
    main()
