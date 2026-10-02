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

    def trial(self, case, max_tokens=32, tokens=None, persistent_cache=None, identity=None):
        return self.runner.run_case(self.model, self.tokenizer, self.draft,
            self.tokens if tokens is None else tokens, self.tokens[:-3], case,
            SimpleNamespace(prefill_chunk=16, max_tokens=max_tokens, draft_tokens=3),
            persistent_cache=persistent_cache, cache_identity=identity)

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
        for case in ('baseline', 'prefix', 'speculative', 'kv-8bit', 'kv-4bit'):
            result = self.trial(case, max_tokens=1)
            self.assertEqual(result['sampled_tokens_including_eos'], 1)
            self.assertIsNone(result['decode_tokens_per_s'])

    def test_persistent_cache_conversation_branch_and_clear(self):
        from experiments.cache import PrefixCache
        from experiments.metrics import common_prefix
        cache = PrefixCache(lambda: self.runner.make_prompt_cache(self.model))
        identity = (self.runner.TARGET, self.runner.tokenizer_fingerprint(self.tokenizer))
        first = self.trial('persistent-prefix', persistent_cache=cache, identity=identity)
        self.assertEqual(first['reused_tokens'], 0)
        messages = [
            {'role': 'user', 'content': 'Reply with just the word hello.'},
            {'role': 'assistant', 'content': first['output']},
            {'role': 'user', 'content': 'Now reply with just the word goodbye.'}]
        continuation = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        for tokens in (continuation, self.tokens, self.tokens[:-2]):
            expected_reuse = common_prefix(cache.token_ids, tokens)
            baseline = self.trial('baseline', tokens=tokens)
            result = self.trial('persistent-prefix', tokens=tokens, persistent_cache=cache, identity=identity)
            self.assertEqual(result['output_token_ids'], baseline['output_token_ids'])
            self.assertEqual(result['reused_tokens'], expected_reuse)
            self.assertGreater(expected_reuse, 0)
            self.assertEqual({c.offset for c in cache.caches}, {len(cache.token_ids)})
        changed = self.trial('persistent-prefix', persistent_cache=cache, identity=('different-identity',))
        self.assertEqual(changed['reused_tokens'], 0)
        cache.clear()
        cleared = self.trial('persistent-prefix', max_tokens=1, persistent_cache=cache, identity=identity)
        self.assertEqual(cleared['reused_tokens'], 0)
        self.assertIsNone(cleared['decode_tokens_per_s'])

    def test_quantized_prefill_parity_and_cache_memory(self):
        # Same fresh cache type in native generation and the manual-prefill runner.
        # Quantized outputs need not equal the unquantized baseline.
        tokens = self.tokenizer.apply_chat_template([
            {'role': 'user', 'content': 'Remember: the telescope opens at sunset. ' * 32 +
             'When does the telescope open?'}], tokenize=True, add_generation_prompt=True)
        baseline = self.trial('baseline', tokens=tokens, max_tokens=1)
        for bits in (8, 4):
            with self.subTest(bits=bits):
                caches = [c.to_quantized(group_size=64, bits=bits)
                          for c in self.runner.make_prompt_cache(self.model)]
                native = list(self.runner.stream_generate(
                    self.model, self.tokenizer, prompt=tokens, max_tokens=32,
                    prompt_cache=caches, prefill_step_size=16,
                    sampler=self.runner.make_sampler(temp=0.0)))
                output_ids = [r.token for r in native if r.token not in self.tokenizer.eos_token_ids]
                result = self.trial(f'kv-{bits}bit', tokens=tokens)
                self.assertEqual(result['output_token_ids'], output_ids)
                self.assertEqual(result['sampled_tokens_including_eos'], len(native))
                self.assertEqual(result['finish_reason'], native[-1].finish_reason)
                self.assertTrue(all(c['bits'] == bits and c['group_size'] == 64
                                    for c in result['target_cache_layers']))
                self.assertGreater(result['cache_after_bytes'], 0)
                # Equal prompt/output lengths avoid capacity-boundary differences
                # caused by potentially different quantized stopping behavior.
                fixed_length = self.trial(f'kv-{bits}bit', tokens=tokens, max_tokens=1)
                self.assertEqual(fixed_length['cache_before_bytes'], 0)
                self.assertLess(fixed_length['cache_after_bytes'], baseline['cache_after_bytes'])
