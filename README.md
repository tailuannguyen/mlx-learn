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
```

The first run downloads roughly 1.3 GB of model artifacts to Hugging Face's cache. Later runs reuse those files. Use `HF_HUB_OFFLINE=1` after downloading for offline operation. Results default to `results/latest.json`, which is replaced on a successful run. Use a distinct `--output` path to retain each experiment. Results are excluded from Git. Ctrl-C exits with code 130; incomplete runs are not saved, and an existing result file remains untouched.

Pinned versions: MLX 0.32.3 and MLX-LM 0.31.3, with transitive dependencies in `uv.lock`.

| Role | Model | Pinned revision |
| --- | --- | --- |
| Target | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | `8b403126fc14f14cfc99bb4cfa72ecbc129ea677` |
| Draft | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3` |

The runner checks complete tokenizer pipeline fingerprints, EOS tokens, output vocabulary sizes, and cache rollback support. This pair has also passed real greedy-generation checks. These checks do not establish compatibility for arbitrary models.

## What it measures

Three cases execute sequentially with both models kept loaded:

1. **Baseline:** a fresh KV cache; all prompt tokens are processed.
2. **Prefix:** precompute the exact shared token prefix of two different questions, then process the second question with that state. Priming happens outside request timing and has its own recorded cost. Each trial rebuilds the prefix, so this is a controlled primed-cache experiment, not a persistent conversation cache implementation.
3. **Speculative:** fresh caches for both models; the draft proposes three tokens by default. Accepted draft output tokens are counted from MLX-LM's `from_draft` events. The public API does not report all proposed tokens, so proposal count and acceptance rate are `null`.

Every case gets one full-length unmeasured warmup. Measured case order rotates across repeats. The runner saves individual runs and median/min/max/sample standard deviation; three runs are a smoke benchmark, not a statistical performance claim. Greedy sampling, seed, exact token inputs/outputs, chat template, model configs and revisions, dependency versions, machine information, and end-of-run memory/thermal status are recorded.

Timing definitions:

- **Load:** local model/tokenizer loading and synchronization; excludes downloads.
- **Bulk prefill:** evaluated cache construction through the penultimate prompt token. Speculative prefill includes both models. It excludes the final prompt-token forward pass and sampling. This split deliberately differs from MLX-LM's reported `prompt_tps`.
- **TTFT:** measured request start through the first sampled-token event, including bulk prefill, final prompt-token work, and synchronization. It is not first visible character latency. MLX-LM may prefetch the next token or verify an entire speculative block before that boundary.
- **Decode throughput:** sampled tokens after the first divided by elapsed time after that first event, including generator cleanup. EOS counts as a sampled token. A one-token run has no decode throughput. Speculative token events arrive in blocks; total duration is essential when interpreting this rate.
- **Total:** bulk prefill through completed generation and GPU synchronization. Tokenization, warmup, prefix priming, cache allocation, model loading, and downloads are excluded. Optional terminal streaming adds I/O overhead.

Memory values are bytes. MLX peak active allocation is reset for each request; it is not process RSS, allocator-reserved memory, or total system memory. Cache bytes include allocated capacity. Both models remain resident even in baseline/prefix cases to hold model residency constant; this experiment does **not** establish the extra memory cost of loading a draft model. Cached request peak excludes priming temporaries. Memory pressure and thermal snapshots are contextual observations, not continuous monitoring or thermal control.

An optimization being selected is insufficient evidence: results contain reused-token counts, cache bytes, accepted draft-token counts, and output comparisons. Prefix output mismatches save diagnostic results and then fail the command. Speculative mismatches are recorded without assuming identical greedy outputs across different computation paths or general sampling correctness.

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

The first command tests prefix boundaries and metric edge cases without loading models. The opt-in Metal tests compare manual prefill with native MLX-LM generation, test prefix/speculative output parity on a short prompt, verify EOS accounting, and check single-token timing. They need the model downloads and run GPU work.

Unsupported in this prototype: arbitrary model selection, non-greedy sampling, combined prefix/speculation, quantized or bounded KV caches, attention overrides, persistent worker cancellation, model unloading commands, cross-engine comparisons, and a graphical interface. The CLI process releases resources on exit.

## Minimal implementation plan

1. **Phase A — current prototype:** pin dependencies/models, prove streaming, explicit cache states, speculative execution, and honest JSON measurements. Before widening model support, add true cross-request prefix lifecycle tests, proposal instrumentation if supported upstream, and a corpus of longer prompts and output lengths.
2. **Worker boundary:** choose structured stdin/stdout or local streaming HTTP; implement capability queries, load/unload, generation IDs, cancellation acknowledgments, cache clearing, errors, and progress. Keep model work serialized and outside the UI process. Treat requested settings and observed execution as separate fields.
3. **Phase B — minimal macOS interface:** provisionally SwiftUI with the Python worker, pending a packaging spike. One supported model, streamed chat, cancellation, load state, and clearly defined metrics. Verify required controls before considering an all-Swift backend.
4. **Phase C — experiments UI:** pair baseline/optimized trials, expose cold versus primed cache state, show variability and slower outcomes, save immutable configurations/results, and explain what each metric includes. Broader engine adapters and attention/cache quantization follow only after capability checks and dedicated validation.
5. **Distribution spike:** bundle the Python runtime/dependencies, test on a clean Mac, determine minimum OS support, and sign/notarize a versioned app artifact before adding a Homebrew cask. App name, license, and commercial model remain open.

## Primary references

- [MLX-LM source and usage](https://github.com/ml-explore/mlx-lm); this implementation was checked against the installed 0.31.3 `generate.py`, `models/cache.py`, and `utils.py`.
- [Target model](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit/tree/8b403126fc14f14cfc99bb4cfa72ecbc129ea677)
- [Draft model](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit/tree/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3)

Attention controls and competitor capabilities were not used as implementation assumptions in this prototype.
