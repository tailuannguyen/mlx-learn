# MLX inference experiments

A first Python experiment runner for a future macOS app that explains local inference optimizations through measurements. The app architecture is still undecided; this repository proves a narrow inference path before introducing a UI or worker protocol.

## Run

Requires Apple Silicon, Metal, Python 3.13, and [uv](https://docs.astral.sh/uv/). Tested on an M1 Pro with 16 GB RAM and macOS 27.0.1; this is not a declared minimum OS version.

```sh
uv sync --frozen
uv run --frozen python -m experiments.run
# Print streamed responses (terminal I/O is then included in timings):
uv run --frozen python -m experiments.run --stream --output results/streamed.json
# Different trial size:
uv run --frozen python -m experiments.run --repeats 5 --max-tokens 128 --draft-tokens 3
# Reuse real request state across a fixed three-request sequence:
uv run --frozen python -m experiments.run --experiment persistent-prefix --output results/persistent.json
# Compare fresh unquantized, 8-bit, and 4-bit KV caches on a longer prompt:
uv run --frozen python -m experiments.run --experiment kv-quantization --document-repeats 128 --output results/kv.json
```

The first run downloads roughly 1.3 GB of model artifacts to Hugging Face's cache. Later runs reuse those files. Use `HF_HUB_OFFLINE=1` after downloading for offline operation. Results default to `results/latest.json`, which is replaced on a successful run. Use a distinct `--output` path to retain each experiment. Results are excluded from Git. Ctrl-C exits with code 130; incomplete runs are not saved, and an existing result file remains untouched.

Pinned versions: MLX 0.32.3 and MLX-LM 0.31.3, with transitive dependencies in `uv.lock`.

| Role | Model | Pinned revision |
| --- | --- | --- |
| Target | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | `8b403126fc14f14cfc99bb4cfa72ecbc129ea677` |
| Draft | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3` |

The runner checks complete tokenizer pipeline fingerprints, EOS tokens, output vocabulary sizes, and cache rollback support. This pair has also passed real greedy-generation checks. These checks do not establish compatibility for arbitrary models.

## What it measures

The default `--experiment standard` runs three cases sequentially with both models kept loaded:

1. **Baseline:** a fresh KV cache; all prompt tokens are processed.
2. **Prefix:** precompute the exact shared token prefix of two different questions, then process the second question with that state. Priming happens outside request timing and has its own recorded cost. Each trial rebuilds the prefix, so this is a controlled primed-cache experiment, not a persistent conversation cache implementation.
3. **Speculative:** fresh caches for both models; the draft proposes three tokens by default. Accepted draft output tokens are counted from MLX-LM's `from_draft` events. The public API does not report all proposed tokens, so proposal count and acceptance rate are `null`.

Every case gets one full-length unmeasured warmup campaign. Measured case order rotates across repeats. The runner saves individual runs and median/min/max/sample standard deviation; three runs are a smoke benchmark, not a statistical performance claim. Greedy sampling, seed, exact token inputs/outputs, chat template, model configs and revisions, dependency versions, machine information, and end-of-run memory/thermal status are recorded.

Two opt-in modes isolate additional experiments:

- **`persistent-prefix`:** compare fresh baseline requests with an in-process cache retained across three requests: the first question, a different question sharing the reference, and a conversation continuation with a fixed assistant message. The cache retains only processed positions with known token IDs, which can include generated tokens. Each request finds the exact shared token prefix, rolls back divergent history, and leaves at least the final input token for generation. Model/tokenizer/settings identity changes and failed or cancelled requests discard state. There is one cache entry, no concurrent requests, no disk persistence, and no worker or chat UI. The fixed transcript makes inputs identical across cases; measured replies do not become the next request's messages.
- **`kv-quantization`:** compare fresh unquantized, 8-bit, and 4-bit target KV caches, with a fixed group size of 64. Empty caches are converted before the first prompt forward pass, so manual bulk prefill and decode both use quantized history. Model weights remain the existing 4-bit weights. Cache type, bits, group size, and offsets are recorded for every target layer, and a trial fails if the requested cache precision was not observed. This verifies the cache representation and the supported Qwen2 attention dispatch; it does not prove a speedup or answer quality. Quantized output differences are recorded without failing the experiment; `quality_evaluation` remains `null`.

Persistent campaigns always begin cold, preserve request order, and release their state before another case or repeat. Warmup state is also discarded. Cases execute sequentially, including quantized trials. Both models remain resident in all modes to retain the existing residency convention. No mode combines persistence, speculation, or KV quantization.

Results use JSON schema version 2. `requests` records each request's messages and token IDs; runs include `request_index`, cold/reused cache state, requested KV precision, observed layer state, cache bytes, retained-token counts, and separate cache-management costs. `matches_first_baseline_tokens` compares with the first baseline **for that request**. `summary_by_request` compares matching inputs; the top-level `summary` is `null` for multi-request experiments so cold and reused requests of different lengths are not averaged together. Single-request modes retain the top-level summary. Cache-memory and management-cost summaries include variability.

Timing definitions:

- **Load:** local model/tokenizer loading and synchronization; excludes downloads.
- **Bulk prefill:** evaluated cache construction through the penultimate prompt token. Speculative prefill includes both models. It excludes the final prompt-token forward pass and sampling. This split deliberately differs from MLX-LM's reported `prompt_tps`.
- **TTFT:** measured request start through the first sampled-token event, including bulk prefill, final prompt-token work, and synchronization. It is not first visible character latency. MLX-LM may prefetch the next token or verify an entire speculative block before that boundary.
- **Decode throughput:** sampled tokens after the first divided by elapsed time after that first event, including generator cleanup. EOS counts as a sampled token. A one-token run has no decode throughput. Speculative token events arrive in blocks; total duration is essential when interpreting this rate.
- **Total:** bulk prefill through completed generation and GPU synchronization. Tokenization, warmup, prefix priming, cache allocation/preparation, persistent-state commit, model loading, and downloads are excluded. Optional terminal streaming adds I/O overhead.
- **Cache preparation / commit:** separately recorded host-side times for target cache construction or prefix lookup/rollback, and post-generation reconciliation of retained token history. Empty-cache conversion is included in preparation for quantized cases. These are outside TTFT/total. They are not full application request latency: tokenization and other excluded work still need accounting in a future worker.

Memory values are bytes. MLX peak active allocation is reset for each request; it is not process RSS, allocator-reserved memory, or total system memory. Cache bytes include allocated capacity. Both models remain resident even in baseline/prefix cases to hold model residency constant; this experiment does **not** establish the extra memory cost of loading a draft model. Cached request peak excludes priming temporaries. Memory pressure and thermal snapshots are contextual observations, not continuous monitoring or thermal control.

An optimization being selected is insufficient evidence: results contain reused-token counts, cache bytes, accepted draft-token counts, and output comparisons. Primed or persistent prefix output mismatches save diagnostic results and then fail the command. Speculative mismatches are recorded without assuming identical greedy outputs across different computation paths or general sampling correctness. KV quantization may change outputs; token agreement is a diagnostic, not a task-quality score. The additional modes have not yet been benchmarked on Apple Silicon.

## First local observation

On the development M1 Pro, with 432 prompt tokens, 413 reused tokens, 64 generated tokens, and three measured runs per case:

| Case | Median TTFT | Median total | Median decode tokens/s |
| --- | ---: | ---: | ---: |
| Baseline | 0.410 s | 0.869 s | 136.6 |
| Prefix reuse | 0.134 s | 0.599 s | 136.3 |
| Speculative | 0.544 s | 1.499 s | 65.7 |

All nine outputs matched. Speculative runs accepted 33 draft output tokens each; this is not a proposal acceptance rate. Caching avoided prompt work while leaving decode speed roughly unchanged. Speculation was slower for this pair and workload. These observations are hardware/workload-specific, and cached timings exclude the separately recorded priming cost. Local raw results are in `results/latest.json`.

## Verify

```sh
uv run --frozen python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 MLX_INTEGRATION=1 uv run --frozen python -m unittest discover -s tests -v
```

The first command tests prefix boundaries, persistent cache lifecycle, and metric edge cases without loading models. Lifecycle tests use token-addressed test storage and do not establish MLX inference correctness. The opt-in Metal tests compare manual prefill with native MLX-LM generation, test prefix/speculative output parity, verify EOS accounting, and check single-token timing. They also check persistent reuse across conversation continuation, branching, shorter prompts, identity changes, and clearing, plus quantized native-generation parity and cache-memory reduction on a longer prompt. They need the model downloads and run GPU work.

Unsupported in this prototype: arbitrary model selection, non-greedy sampling, combinations of optimizations, bounded KV caches, attention overrides, disk-persisted caches, persistent worker cancellation, model unloading commands, cross-engine comparisons, and a graphical interface. The CLI process releases resources on exit. The additional modes are implemented against pinned MLX-LM 0.31.3 APIs and require the opt-in Metal checks before their inference behavior or performance is trusted.

## Minimal implementation plan

1. **Phase A — current prototype:** pin dependencies/models, prove streaming, explicit cache states, speculative execution, and honest JSON measurements. Cross-request prefix lifecycle and independent KV-quantization modes now have opt-in Metal checks; run them before drawing conclusions. Before widening model support, add proposal instrumentation if supported upstream and a corpus of longer prompts and output lengths with task-quality evaluation.
2. **Worker boundary:** choose structured stdin/stdout or local streaming HTTP; implement capability queries, load/unload, generation IDs, cancellation acknowledgments, cache clearing, errors, and progress. Keep model work serialized and outside the UI process. Treat requested settings and observed execution as separate fields.
3. **Phase B — minimal macOS interface:** provisionally SwiftUI with the Python worker, pending a packaging spike. One supported model, streamed chat, cancellation, load state, and clearly defined metrics. Verify required controls before considering an all-Swift backend.
4. **Phase C — experiments UI:** pair baseline/optimized trials, expose cold versus primed/reused cache state, show variability and slower outcomes, save immutable configurations/results, and explain what each metric includes. Expose KV quantization only after real-model validation; broader engine adapters and attention experiments follow capability checks and dedicated validation.
5. **Distribution spike:** bundle the Python runtime/dependencies, test on a clean Mac, determine minimum OS support, and sign/notarize a versioned app artifact before adding a Homebrew cask. App name, license, and commercial model remain open.

## Primary references

- [MLX-LM source and usage](https://github.com/ml-explore/mlx-lm); this implementation was checked against the installed 0.31.3 `generate.py`, `models/cache.py`, and `utils.py`.
- [Target model](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit/tree/8b403126fc14f14cfc99bb4cfa72ecbc129ea677)
- [Draft model](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit/tree/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3)

Attention controls and competitor capabilities were not used as implementation assumptions in this prototype.
