"""One in-process, exact-token prefix cache; no MLX dependency in the policy."""
from .metrics import common_prefix


def cache_bytes(caches):
    # MLX-LM 0.31.3 QuantizedKVCache.nbytes dereferences None before the first
    # update. Its empty() contract lets cold caches report zero safely.
    return sum(cache.nbytes for cache in caches if not cache.empty())


class PrefixCache:
    """Own a single serialized request's state for the supported full-attention model.

    The caller supplies an identity covering model revision and tokenizer/settings.
    A prepared cache is mutable and must be committed after successful generation,
    or cleared after failure. No persistence to disk or simultaneous requests.
    """

    def __init__(self, factory):
        self.factory = factory
        self.clear()

    def clear(self):
        self.caches = None
        self.identity = None
        self.token_ids = ()
        self._prompt = None

    def _offset(self):
        offsets = {c.offset for c in self.caches}
        if len(offsets) != 1 or not all(c.is_trimmable() for c in self.caches):
            raise ValueError('Prefix reuse requires aligned, trimmable full-attention caches')
        return offsets.pop()

    def _trim_to(self, offset):
        count = self._offset() - offset
        if count < 0:
            raise ValueError('Cache does not contain the claimed tokens')
        for cache in self.caches:
            if cache.trim(count) != count:
                raise ValueError('Cache could not roll back to the shared prefix')
        if self._offset() != offset:
            raise ValueError('Cache rollback left an incorrect offset')

    def prepare(self, identity, prompt):
        if self._prompt is not None:
            raise RuntimeError('Complete or clear the active request before reusing its cache')
        if not prompt:
            raise ValueError('A nonempty token prompt is required')
        try:
            if identity != self.identity:
                self.clear()
            reused = common_prefix(self.token_ids, prompt)
            if not reused:
                self.caches = self.factory()
                if self._offset() != 0:
                    raise ValueError('New caches must be empty')
            else:
                if self._offset() != len(self.token_ids):
                    raise ValueError('Retained cache no longer matches its token history')
                self._trim_to(reused)
            self.identity = identity
            self.token_ids = ()
            self._prompt = tuple(prompt)
            return self.caches, reused
        except BaseException:
            self.clear()
            raise

    def commit(self, sampled_tokens):
        if self._prompt is None:
            raise RuntimeError('No active request to commit')
        try:
            known = self._prompt + tuple(sampled_tokens)
            offset = self._offset()
            if offset < len(self._prompt):
                raise ValueError('Generation did not process the complete prompt')
            # Generation can prefetch: retain only evaluated positions whose token
            # IDs are known, never infer that every sampled token entered the cache.
            retained = min(offset, len(known))
            self._trim_to(retained)
            self.token_ids = known[:retained]
            self._prompt = None
            return retained
        except BaseException:
            self.clear()
            raise
