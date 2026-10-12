import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from residual_train_common import ROOT, PREVIOUS, TOKENIZER, rows, read, sha, extract, append
from run_base_residual_experiment import validate, pair_requests, consensus


class PilotTests(unittest.TestCase):
    def test_saved_response_is_not_cut_on_unicode(self):
        text = "First sentence. wife'\u0441\u044f matters. Last sentence."
        self.assertEqual(extract(text), text)
        self.assertEqual(extract('Intro\n**Final Response:**\n' + text), text)

    def test_jsonl_keeps_unicode_line_separator_inside_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.jsonl'
            append(path, {'text': 'first\u2028second\u2029third'})
            self.assertEqual(rows(path), [{'text': 'first\u2028second\u2029third'}])

    def test_pairwise_conservative_consensus(self):
        reqs = pair_requests('test', 'fake', 'question', 'original', 'candidate', 'base', 'target')
        def decisions(a, b, ok=True):
            return {r['request_id']: {'ok': ok, 'returned_model': 'gpt-4.1-2025-04-14',
                    'normalized': {'choice': c}} for r, c in zip(reqs, (a, b))}
        self.assertEqual(consensus(reqs, decisions('B', 'A')), 'target')
        self.assertEqual(consensus(reqs, decisions('A', 'B')), 'base')
        self.assertEqual(consensus(reqs, decisions('tie', 'tie')), 'tie')
        self.assertEqual(consensus(reqs, decisions('A', 'A')), 'order_disagreement')
        self.assertEqual(consensus(reqs, decisions('tie', 'tie', False)), 'technical_failure')

    def test_strict_teacher_and_verifier_schema(self):
        with self.assertRaises(ValueError):
            validate('train_residual_teacher', {'action': 'edit', 'response': ''})
        payload = {'unsupported_claims': False, 'specificity_loss': False, 'inappropriate_refusal': False,
                   'surface_failure': False, 'reason': 'none', 'evidence': ''}
        self.assertEqual(validate('train_residual_verify', payload), payload)
        with self.assertRaises(ValueError):
            validate('train_residual_verify', {**payload, 'unsupported_claims': 'false'})

    def test_failed_api_json_keeps_raw_and_usage(self):
        import asyncio
        from types import SimpleNamespace
        from run_base_residual_experiment import request, logged_call
        raw = '{broken json'
        dumped = {'choices': [{'message': {'content': raw}, 'finish_reason': 'stop'}],
                  'usage': {'prompt_tokens': 10, 'completion_tokens': 2}, 'model': 'gpt-4.1-2025-04-14'}
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw), finish_reason='stop')],
                                   model_dump=lambda **kwargs: dumped)
        async def create(**kwargs):
            return response
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        req = request('fake', 'system', 'user', 'blind_selector', 256, {})
        result, limited = asyncio.run(logged_call(client, req, 5))
        self.assertFalse(result['ok'])
        self.assertEqual(result['attempt'], 1)
        self.assertEqual(result['raw_json_text'], raw)
        self.assertEqual(result['usage']['prompt_tokens'], 10)
        self.assertTrue(result['request_sha256'])
        self.assertFalse(limited)

    def test_verifier_field_alias_recovers_raw_without_rejudging(self):
        import run_base_residual_experiment as experiment
        payload = {'unsupported_claims': False, 'specificity_loss': True, 'inappropriate_refusal': False,
                   'surface_failure': False, 'reason': 'Lost a detail', 'evidence_phrase': 'original detail'}
        original = {'request_id': 'id', 'ok': False, 'error': 'Missing verifier field: evidence',
                    'raw_json_text': json.dumps(payload), 'api_response_history': [{'model': 'gpt-4.1-2025-04-14'}]}
        with tempfile.TemporaryDirectory() as tmp, patch.object(experiment, 'ROOT', Path(tmp)):
            restored, count = experiment.recover_verifier_schema({'id': original})
        self.assertEqual(count, 1)
        self.assertFalse(original['ok'])
        self.assertTrue(restored['id']['ok'])
        self.assertTrue(restored['id']['normalized']['specificity_loss'])
        self.assertEqual(restored['id']['normalized']['evidence'], 'original detail')

    def test_quota_is_not_a_transient_rate_limit(self):
        from run_base_residual_experiment import quota_error, transient_unless_quota
        self.assertTrue(quota_error('credit_balance_exhausted'))
        self.assertTrue(quota_error('insufficient_quota'))
        self.assertFalse(transient_unless_quota(RuntimeError('credit_balance_exhausted')))
        self.assertFalse(quota_error('requests per minute exceeded'))

    def test_train_split_is_disjoint_and_has_no_references(self):
        contract = read(ROOT / 'run_contract.json')
        fit = set(contract['train_partitions']['fit'])
        holdout = set(contract['train_partitions']['holdout'])
        valid = set(contract['valid_ids'])
        self.assertEqual((len(fit), len(holdout), len(valid)), (248, 28, 43))
        self.assertFalse(fit & holdout or (fit | holdout) & valid)
        for r in rows(ROOT / 'inputs/train276.jsonl'):
            self.assertNotIn('safe_response', r)
            self.assertNotIn('verified_spans', r)
        self.assertEqual(sha(rows(ROOT / 'inputs/train276.jsonl')), contract['input_sha256'])

    def test_all_valid_prompts_match_saved_no_adapter_control(self):
        from fullpaper_backbone_utils import load_fullpaper_tokenizer
        from selective_risk_refinement_utils import build_risk_tune_prompt, configure_risk_contract
        configure_risk_contract('fullpaper_v1')
        tok = load_fullpaper_tokenizer(str(TOKENIZER))
        old = {r['canonical_id']: r for r in rows(PREVIOUS / 'generation/base_draft_second_pass_controls_valid43.jsonl') if r['arm'] == 'base_self_refine'}
        for r in rows(ROOT / 'inputs/valid43.jsonl'):
            p = build_risk_tune_prompt(tok, {**r, 'sft_response': r['base_response'],
                'z_t_from_sft': r['base_response'], 'aspect_conditioning_status': 'available', 'risk_component_mode': 'full'})
            self.assertEqual(sha(p), old[r['canonical_id']]['prompt_provenance']['rendered_prompt_sha256'])
            self.assertEqual(sha(tok(p, add_special_tokens=True)['input_ids']), old[r['canonical_id']]['prompt_provenance']['input_token_ids_used_sha256'])

    def test_target_curation_rejects_unsupported_winning_candidate(self):
        import run_base_residual_experiment as experiment
        from residual_train_common import write, write_rows
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / 'run_contract.json', read(ROOT / 'run_contract.json'))
            source = [{'canonical_id': str(i), 'question': 'Question ' + str(i), 'response': 'Original ' + str(i),
                       'technical_valid': True, 'train_partition': 'fit' if i < 248 else 'holdout'} for i in range(276)]
            write_rows(root / 'generation/train.jsonl', source)

            def fake_api(name, requests):
                output = {}
                for r in requests:
                    i = int(r['metadata']['canonical_id'])
                    if name == 'train_teacher':
                        normalized = {'action': 'edit' if i % 2 == 0 else 'preserve',
                                      'response': ('Candidate ' if i % 2 == 0 else 'Original ') + str(i)}
                    elif name == 'train_verify':
                        normalized = {'unsupported_claims': i == 0, 'specificity_loss': False,
                                      'inappropriate_refusal': False, 'surface_failure': False}
                    else:
                        normalized = {'choice': 'B' if r['metadata']['order'] == 0 else 'A'}
                    output[r['request_id']] = {'ok': True, 'returned_model': 'gpt-4.1-2025-04-14', 'normalized': normalized}
                return output

            with patch.object(experiment, 'ROOT', root), patch.object(experiment, 'execute_requests', fake_api):
                experiment.targets()
            curated = {r['canonical_id']: r for r in rows(root / 'data/curation_all276.jsonl')}
            self.assertEqual(curated['0']['decision'], 'excluded')
            self.assertEqual(curated['2']['decision'], 'edit')
            self.assertEqual(curated['1']['decision'], 'preserve')
            self.assertEqual(sum(r['selected_for_training'] for r in curated.values()), 246)
            self.assertEqual(sum(r['selected_for_holdout'] for r in curated.values()), 28)


if __name__ == '__main__':
    unittest.main()
