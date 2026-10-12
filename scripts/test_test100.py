"""CPU-only TEST100 contract/cache checks, with no generation or paid calls."""
import tempfile
import unittest
from unittest.mock import patch
import run_test100 as T


class Test100Checks(unittest.TestCase):
    def test_prefix_and_extension(self):
        c=T.read(T.ROOT/'run_contract.json')
        old=T.rows(T.OLD/'test60_final_outputs.jsonl')
        self.assertEqual(T.rows(T.ROOT/'inputs/reused_test60_final.jsonl'),old)
        self.assertEqual([r['canonical_id'] for r in old],c['ordered_ids'][:60])
        self.assertEqual([r['canonical_id'] for r in T.rows(T.ROOT/'inputs/new40.jsonl')],c['ordered_ids'][60:])
        self.assertGreater(len(old[0]['dpo_response']),7)
        self.assertGreater(len(old[39]['sft_response']),7)

    def test_existing109_scores_reused_without_calls(self):
        source=T.ROOT
        with tempfile.TemporaryDirectory(dir='/mnt/ssd00/hsoh') as tmp:
            root=T.Path(tmp)
            T.write(root/'run_contract.json',T.read(source/'run_contract.json'))
            T.write_rows(root/'test100_final_outputs.jsonl',T.rows(T.OLD/'test60_final_outputs.jsonl'))
            def no_api(name,requests):
                existing={r['request_id']:r for r in T.rows(root/f'results/{name}.jsonl')}
                self.assertEqual(len(existing),109)
                self.assertEqual({r['request_id'] for r in requests},set(existing))
                for r in requests:self.assertEqual(T.Q.request_sha256(r),existing[r['request_id']]['request_sha256'])
            with patch.object(T,'ROOT',root),patch.object(T,'run_queue',no_api):T.judge()

    def test_snapshot_hashes_and_extractor(self):
        c=T.read(T.ROOT/'run_contract.json')
        for path,digest in c['published_hashes_verified'].items():self.assertEqual(T.file_sha(path),digest)
        sc=T.read(T.OLD/'run_contract.json')
        self.assertEqual(T.file_sha(T.CODE/'selective_risk_refinement_utils.py'),sc['extraction']['extractor']['sha256'])
        self.assertEqual(c['pointwise']['max_tokens'],384)
        self.assertEqual(c['inference_settings']['proposed']['acceptance_policy'],'surface_relaxed_v1')

    def test_unicode_jsonl(self):
        with tempfile.TemporaryDirectory(dir='/mnt/ssd00/hsoh') as tmp:
            path=T.Path(tmp)/'rows.jsonl'; values=[{'text':'one\u2028two\u2029three'}]
            T.write_rows(path,values)
            self.assertEqual(T.rows(path),values)


if __name__=='__main__':unittest.main()
