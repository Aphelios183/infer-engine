"""Opt-in real tokenizer checks. Explicitly enabled but missing assets => FAIL."""
import os
from pathlib import Path
import unittest

from week3.inspect_model import inspect_prompt


@unittest.skipUnless(os.environ.get('RUN_TOKENIZER_INTEGRATION') == '1',
                     '未开启真实 tokenizer 集成检查；SKIP 不表示模型已验证')
class TokenizerIntegrationTest(unittest.TestCase):
    def load(self, path):
        self.assertTrue(Path(path).is_dir(), f'缺少本地 tokenizer 资产: {path}')
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)

    def test_qwen3_history(self):
        tok = self.load('/home/ubuntu/huggingface/Qwen3-0.6B')
        r = inspect_prompt(tok, '用一句话解释 KV Cache。')
        self.assertEqual(r['chat_ids'], [151644, 872, 198, 11622, 105321,
            104136, 84648, 19479, 1773, 151645, 198, 151644, 77091,
            198, 151667, 271, 151668, 271])
        self.assertEqual(r['chat_length'], 18)
        self.assertEqual(r['eos_token_id'], 151645)
        self.assertTrue(r['template_encoding_matches'])
        self.assertTrue(r['roundtrip_matches'])

    def test_qwen35_target(self):
        tok = self.load('/home/ubuntu/huggingface/Qwen3.5-4B')
        r = inspect_prompt(tok, '用一句话解释 KV Cache。')
        self.assertGreater(r['chat_length'], r['raw_length'])
        self.assertTrue(r['template_encoding_matches'])
        self.assertTrue(r['roundtrip_matches'])
        self.assertTrue(r['thinking_switch_verified'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
