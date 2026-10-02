# Project context

Build an Apple Silicon macOS app for downloading models and running local LLMs, inspired by PocketPal. The differentiator is helping users understand inference optimizations through controlled experiments, clear metrics, and explanations. A toggle being enabled is not proof that a technique executed or improved performance.

Start small with MLX. SwiftUI with a persistent Python MLX-LM worker is a proposed architecture, not a final commitment. No UI or worker protocol exists yet. An all-Swift implementation requires checking mlx-swift-lm's experimental controls rather than assuming Python API parity.

Longer-term possibilities include llama.cpp with Metal and local or remote vLLM adapters. These are outside the initial implementation. App name, license, minimum supported macOS version, commercial model, and transport protocol remain undecided. Homebrew cask distribution is desired; packaging must first bundle the runtime and dependencies so users do not need Python installed.

## Current repository

- `experiments/run.py`: default sequential greedy baseline, primed-prefix, and speculative-decoding trials; opt-in persistent-prefix and KV-quantization campaigns; optional streamed text; schema-v2 JSON results.
- `experiments/cache.py`: backend-independent single-entry cache lifecycle policy, with exact token matching, rollback, identity invalidation, and commit of known processed positions.
- `experiments/metrics.py`: token-prefix matching and aggregate metrics without MLX dependencies.
- `tests/test_metrics.py` and `tests/test_cache.py`: fast measurement and cache-lifecycle tests; test storage does not prove GPU correctness.
- `tests/test_inference.py`: opt-in real Metal tests for native-generation parity, cached/speculative outputs, EOS accounting, one-token runs, persistent conversation/branch lifecycle, and quantized parity/cache memory.
- `pyproject.toml` and `uv.lock`: Python 3.13, MLX 0.32.3, MLX-LM 0.31.3, and locked transitive dependencies.
- `README.md`: setup, exact metric definitions, limitations, initial observations, and phased implementation plan. Read it before changing benchmark behavior.
- `results/`: ignored local artifacts. The default `results/latest.json` is overwritten; choose a distinct output path to retain a run.

The supported pair is Qwen2.5-1.5B-Instruct-4bit as target and Qwen2.5-0.5B-Instruct-4bit as draft, both from `mlx-community`. Exact Hugging Face commit revisions are pinned in `experiments/run.py`. The runner checks tokenizer pipeline fingerprints, EOS IDs, vocabulary sizes, and trimmable caches. Do not generalize this validation to other model pairs.

The initial nine measured trials on an M1 Pro with 16 GB RAM produced matching outputs. Prefix reuse reduced first-token latency; speculation was slower for that workload. These are local observations, not universal performance claims. Six tests, including the Metal tests, passed when the prototype was created; rerun relevant checks after changes.

The persistent-prefix and KV-quantization modes were implemented using the exact MLX-LM 0.31.3 distribution source. They have not been validated with real Metal inference yet. Do not treat fast policy tests or these initial observations as validation of the new modes.

## Development and verification

```sh
uv sync --frozen
uv run --frozen python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 MLX_INTEGRATION=1 uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python -m experiments.run --output results/my-experiment.json
uv run --frozen python -m experiments.run --experiment persistent-prefix --output results/persistent.json
uv run --frozen python -m experiments.run --experiment kv-quantization --document-repeats 128 --output results/kv.json
```

Integration tests require Metal access and previously downloaded models. The runner downloads roughly 1.3 GB on first use. Once downloaded, `HF_HUB_OFFLINE=1` permits offline runs. Sandboxed environments may deny Metal access even on a supported Mac; distinguish environment failures from inference failures and request execution permission when required. Never claim GPU validation from mocked or skipped tests.

Keep changes small and preserve the working experiment before expanding features. Inspect the installed, pinned MLX-LM source when changing generation or cache behavior. Dependency upgrades require explicit lockfile updates and relevant real-model verification. Add meaningful tests for cache correctness and measurement edge cases, rather than tests that merely mirror implementation.

## Measurement rules

- Compare off versus on within one engine first. Run cases sequentially on the GPU, exclude warmups, explicitly establish cache state, repeat trials, and retain per-run values and variability.
- Synchronize/evaluate GPU work at completed timing boundaries. Document precisely what each timer includes; do not silently change metric meanings.
- Current bulk prefill stops before the final prompt token. TTFT includes remaining first-token work and can include prefetch or a speculative block. Decode throughput counts sampled tokens after the first, including EOS; it is unavailable for a one-token run.
- Standard prefix trials precompute a shared token prefix outside request timing and record priming separately. The separate persistent-prefix mode retains known processed prompt/output tokens across a fixed request sequence, starts cold per campaign, and clears state between cases/repeats. It has no disk persistence or worker. Distinguish these modes from ordinary generation KV state.
- Cache preparation and persistent commit are measured separately outside TTFT/total. Summarize multi-request campaigns per request index; do not combine cold and reused requests with different inputs. Quantized cases use fresh 8/4-bit target caches from the first prompt step, group size 64, and unchanged 4-bit model weights. Output agreement is not task-quality evaluation; keep that unavailable score null.
- Both models remain resident in all current cases. MLX peak active allocation is not process RSS or total system memory, and these trials do not measure the extra residency cost of loading a draft model.
- Accepted draft output tokens are available through `from_draft`; total proposals and proposal acceptance rate are not exposed by the current public API. Keep unavailable metrics null. Do not label accepted/output-token share as proposal acceptance rate.
- Record exact model revisions, quantization, tokenizer/chat template, token counts, sampling configuration, cache settings, package versions, hardware, output lengths, and early stopping. Preserve output comparisons and explain mismatches.
- Greedy output equality on a tested prompt does not establish general speculative-sampling correctness. Do not promise identical sampled text across implementations.
- Prefix caching primarily avoids repeated prompt computation. Speculation can lose to baseline. Report these outcomes honestly.

## Next steps and boundaries

Follow the phased plan in `README.md`: strengthen experiment coverage and cross-request cache lifecycle first, then define a small persistent worker boundary, then build a minimal macOS interface and comparison UI.

The future backend boundary should cover capability checks, load/unload, streaming, cancellation, cache management, supported settings, progress, structured errors, and observed metrics. Keep inference off the UI thread and communicate text/small events rather than tensors. Preserve conversations as messages; caches are backend-specific and should not be assumed transferable.

KV-cache quantization is available as an isolated CLI experiment for the pinned Qwen pair, pending real-model validation. Do not combine it with persistent or rotating caches without dedicated checks. Attention selection is a later experiment. Verify support against the pinned backend before exposing controls. In particular, do not assume `force_fused=False` disables fused attention or that a selector alone changes model execution. Reference attention must preserve all model-specific semantics.

CUDA-specific kernels such as DeepGEMM and CUTLASS/CuTe are not Apple GPU toggles. Do not add unsupported controls or broaden into every engine/model prematurely. Verify current competitor capabilities before making differentiation claims.
