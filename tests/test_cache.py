"""Cache policy tests with token-addressed storage, without claiming GPU parity."""
import unittest

from experiments.cache import PrefixCache, cache_bytes


class TokenCache:
    def __init__(self):
        self.storage = []
        self.offset = 0

    def is_trimmable(self):
        return True

    def trim(self, count):
        count = min(count, self.offset)
        self.offset -= count
        return count

    def append(self, tokens):
        self.storage[self.offset:] = tokens
        self.offset += len(tokens)

    def empty(self):
        return not self.storage

    @property
    def nbytes(self):
        if self.empty():
            raise AttributeError('Cold quantized storage has no allocated arrays')
        return 4 * len(self.storage)


class PrefixCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = PrefixCache(lambda: [TokenCache(), TokenCache()])

    def request(self, prompt, sampled=(9, 10), processed=1, identity='model-a'):
        caches, reused = self.cache.prepare(identity, prompt)
        for cache in caches:
            self.assertEqual(cache.storage[:cache.offset], list(prompt[:reused]))
            cache.append(list(prompt[reused:]) + list(sampled[:processed]))
        self.cache.commit(sampled)
        return reused

    def test_cold_repeat_branch_and_shorter_prompt(self):
        self.assertEqual(self.request([1, 2, 3]), 0)
        self.assertEqual(self.request([1, 2, 3]), 2)  # Leave final input token.
        self.assertEqual(self.request([1, 2, 4, 5]), 2)
        self.assertEqual(self.request([1, 2]), 1)
        self.assertEqual(self.request([7, 8]), 0)

    def test_extension_reuses_only_processed_output_tokens(self):
        self.request([1, 2, 3], sampled=[9, 10], processed=1)
        self.assertEqual(self.cache.token_ids, (1, 2, 3, 9))
        self.assertEqual(self.request([1, 2, 3, 9, 10, 11]), 4)

    def test_no_sample_entered_cache_at_length_boundary(self):
        self.request([1, 2, 3], sampled=[9], processed=0)
        self.assertEqual(self.cache.token_ids, (1, 2, 3))
        self.assertEqual(self.request([1, 2, 3, 9, 4]), 3)

    def test_prefetch_beyond_known_tokens_is_removed(self):
        caches, _ = self.cache.prepare('model-a', [1, 2])
        for cache in caches:
            cache.append([1, 2, 9, 999])
        self.assertEqual(self.cache.commit([9]), 3)
        self.assertEqual(self.request([1, 2, 9, 4]), 3)

    def test_identity_change_and_explicit_clear_start_cold(self):
        self.request([1, 2, 3])
        self.assertEqual(self.request([1, 2, 3], identity='model-b'), 0)
        self.cache.clear()
        self.assertEqual(self.request([1, 2, 3], identity='model-b'), 0)

    def test_active_request_cannot_be_reused_and_clear_discards_mutation(self):
        self.request([1, 2, 3])
        caches, _ = self.cache.prepare('model-a', [1, 2, 4])
        caches[0].append([999])  # A failed/cancelled request partially wrote state.
        with self.assertRaises(RuntimeError):
            self.cache.prepare('model-a', [1, 2, 4])
        self.cache.clear()
        self.assertEqual(self.request([1, 2, 4]), 0)

    def test_layer_misalignment_discards_state(self):
        caches, _ = self.cache.prepare('model-a', [1, 2])
        caches[0].append([1, 2])
        caches[1].append([1])
        with self.assertRaises(ValueError):
            self.cache.commit([9])
        self.assertIsNone(self.cache.caches)
        self.assertEqual(self.request([1, 2]), 0)

    def test_incomplete_prompt_is_not_committed(self):
        caches, _ = self.cache.prepare('model-a', [1, 2, 3])
        for cache in caches:
            cache.append([1, 2])
        with self.assertRaises(ValueError):
            self.cache.commit([9])
        self.assertIsNone(self.cache.caches)

    def test_external_mutation_of_retained_cache_is_rejected(self):
        self.request([1, 2, 3])
        for cache in self.cache.caches:
            cache.append([999])
        with self.assertRaises(ValueError):
            self.cache.prepare('model-a', [1, 2, 3])
        self.assertIsNone(self.cache.caches)

    def test_memory_handles_cold_quantized_storage_and_counts_retained_capacity(self):
        cold, allocated = TokenCache(), TokenCache()
        allocated.append([1, 2, 3])
        allocated.trim(2)
        self.assertEqual(cache_bytes([cold]), 0)
        self.assertEqual(cache_bytes([cold, allocated]), 12)


if __name__ == '__main__':
    unittest.main()
