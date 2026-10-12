"""CPU-only checks; never generate responses or call an API."""
import tempfile
import unittest
from unittest.mock import patch

import run_continued_denoiser as C


class ContinuedPilotTests(unittest.TestCase):
    def test_frozen_inputs_and_settings(self):
        current = C.read(C.ROOT / 'run_contract.json')
        previous = C.read(C.FRESH / 'run_contract.json')
        for key in ('training', 'generation', 'pointwise', 'selector', 'frozen_evaluation_prompts'):
            self.assertEqual(current['settings'][key], previous['settings'][key])
        for name, digest in current['copied_files'].items():
            self.assertEqual(C.file_sha(C.ROOT / name), digest)
            self.assertEqual(C.file_sha(C.FRESH / name), digest)
        self.assertFalse(current['test_access'])

    def test_no_training_valid_id_overlap(self):
        c = C.read(C.ROOT / 'run_contract.json')
        self.assertEqual((len(c['train_ids']), len(c['holdout_ids']), len(c['valid_ids'])), (70, 13, 43))
        self.assertFalse(set(c['train_ids']) & set(c['holdout_ids']))
        self.assertFalse(set(c['train_ids'] + c['holdout_ids']) & set(c['valid_ids']))

    def test_exact_pointwise_cache_reuse_without_api(self):
        original = next(r for r in C.rows(C.FRESH / 'results/valid_pointwise.jsonl') if r['ok'])
        request = C.E.request('cache_test', original['system_prompt'], original['user_prompt'],
                             original['parser'], original['max_tokens'], original['metadata'])
        with tempfile.TemporaryDirectory(dir='/mnt/ssd00/hsoh') as temp:
            root = C.Path(temp)
            def no_api(name, requests):
                values = C.rows(root / f'results/{name}.jsonl')
                self.assertEqual(len(values), 1)
                self.assertEqual(values[0]['normalized'], original['normalized'])
                self.assertEqual(values[0]['usage'], {})
                return {v['request_id']: v for v in values}
            with patch.object(C, 'ROOT', root), patch.object(C.E, 'execute_requests', no_api):
                results = C.cached_execute('cache_test', [request])
                self.assertIn(request['request_id'], results)

    def test_order_consensus_and_failure_not_tie(self):
        requests = C.E.pair_requests('test', 'unit', 'question', 'base', 'new', 'base', C.NEW)
        results = {r['request_id']: {'ok': True, 'returned_model': C.MODEL,
                   'normalized': {'choice': 'B' if r['metadata']['order'] == 0 else 'A'}} for r in requests}
        self.assertEqual(C.E.consensus(requests, results), C.NEW)
        results[requests[1]['request_id']]['normalized']['choice'] = 'B'
        self.assertEqual(C.E.consensus(requests, results), 'order_disagreement')
        results[requests[1]['request_id']]['ok'] = False
        self.assertEqual(C.E.consensus(requests, results), 'technical_failure')


if __name__ == '__main__':
    unittest.main()
