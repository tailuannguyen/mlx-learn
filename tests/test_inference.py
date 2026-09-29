"""Opt-in real Metal checks: MLX_INTEGRATION=1 python -m unittest discover -s tests."""
import os
import unittest
from types import SimpleNamespace


@unittest.skipUnless(os.environ.get('MLX_INTEGRATION') == '1', 'requires downloaded models and Metal')
class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.generate import generation_stream
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler
        from experiments import run
        run.mx, run.generation_stream = mx, generation_stream
        run.make_prompt_cache = make_prompt_cache
        run.stream_generate, run.make_sampler = stream_generate, make_sampler
        cls.runner = run
        cls.model, cls.tokenizer = load(str(run.snapshot(run.TARGET)))
        cls.draft, _ = load(str(run.snapshot(run.DRAFT)))
        cls.tokens = cls.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'Reply with just the word hello.'}],
            tokenize=True, add_generation_prompt=True)

    def trial(self, case, max_tokens=32):
        return self.runner.run_case(self.model, self.tokenizer, self.draft,
            self.tokens, self.tokens[:-3], case,
            SimpleNamespace(prefill_chunk=16, max_tokens=max_tokens, draft_tokens=3))

    def test_native_prefill_parity_and_eos_accounting(self):
        responses = list(self.runner.stream_generate(self.model, self.tokenizer,
            prompt=self.tokens, max_tokens=32, sampler=self.runner.make_sampler(temp=0.0)))
        native = [r.token for r in responses if r.token not in self.tokenizer.eos_token_ids]
        result = self.trial('baseline')
        self.assertEqual(result['output_token_ids'], native)
        self.assertEqual(result['finish_reason'], 'stop')
        self.assertEqual(result['sampled_tokens_including_eos'], len(native) + 1)

    def test_cached_and_speculative_outputs(self):
        reference = self.trial('baseline')['output_token_ids']
        for case in ('prefix', 'speculative'):
            result = self.trial(case)
            self.assertEqual(result['output_token_ids'], reference)
            if case == 'prefix':
                self.assertEqual(result['reused_tokens'], len(self.tokens) - 3)
                self.assertGreater(result['cache_before_bytes'], 0)

    def test_one_token_has_no_decode_rate(self):
        for case in ('baseline', 'prefix', 'speculative'):
            result = self.trial(case, max_tokens=1)
            self.assertEqual(result['sampled_tokens_including_eos'], 1)
            self.assertIsNone(result['decode_tokens_per_s'])
