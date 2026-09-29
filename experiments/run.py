"""Run sequential, greedy inference comparisons using a verified model pair."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .metrics import common_prefix, summarize, throughput

TARGET = ('mlx-community/Qwen2.5-1.5B-Instruct-4bit', '8b403126fc14f14cfc99bb4cfa72ecbc129ea677')
DRAFT = ('mlx-community/Qwen2.5-0.5B-Instruct-4bit', 'a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3')


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return value


def snapshot(spec):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(spec[0], revision=spec[1], allow_patterns=[
        '*.json', '*.safetensors', '*.txt', '*.jinja']))


def tokenizer_fingerprint(tokenizer):
    # Includes BPE merges, added tokens, normalizer and pre/post-processing.
    data = json.loads(tokenizer.backend_tokenizer.to_str())
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def sync():
    mx.synchronize()
    mx.synchronize(generation_stream)


def prefill(model, cache, tokens, chunk):
    for start in range(0, len(tokens), chunk):
        model(mx.array(tokens[start:start + chunk], dtype=mx.uint32)[None], cache=cache)
        mx.eval([c.state for c in cache])
    sync()


def run_case(model, tokenizer, draft, tokens, prefix_tokens, case, args, stream=False):
    gc.collect()
    mx.clear_cache()
    caches = make_prompt_cache(model)
    draft_caches = make_prompt_cache(draft) if case == 'speculative' else []
    priming_s = 0.0
    reused = 0
    if case == 'prefix':
        started = time.perf_counter()
        prefill(model, caches, prefix_tokens, args.prefill_chunk)
        priming_s = time.perf_counter() - started
        reused = len(prefix_tokens)
        if tokens[:reused] != prefix_tokens:
            raise ValueError('Prefix token mismatch')
    sync()
    mx.reset_peak_memory()
    active_before = mx.get_active_memory()
    cache_before = sum(c.nbytes for c in caches)
    started = time.perf_counter()
    # Bulk prefill excludes the final prompt token. It is timed identically for
    # baseline/prefix; speculative includes both target and draft prefill.
    prefill(model, caches, tokens[reused:-1], args.prefill_chunk)
    if draft_caches:
        prefill(draft, draft_caches, tokens[:-1], args.prefill_chunk)
    prefill_s = time.perf_counter() - started
    first = None
    output, output_ids = [], []
    sampled_count = 0
    accepted = 0
    finish = None
    generator = stream_generate(
        model, tokenizer, prompt=tokens[-1:], max_tokens=args.max_tokens,
        draft_model=draft if draft_caches else None,
        prompt_cache=caches + draft_caches,
        sampler=make_sampler(temp=0.0), num_draft_tokens=args.draft_tokens,
    )
    try:
        for response in generator:
            if first is None:
                sync()
                first = time.perf_counter()
            sampled_count += 1
            is_eos = response.token in tokenizer.eos_token_ids
            if not is_eos:
                output_ids.append(response.token)
                accepted += int(response.from_draft)
            output.append(response.text)
            finish = response.finish_reason
            if stream:
                print(response.text, end='', flush=True, file=sys.stderr)
    finally:
        generator.close()
        sync()
    ended = time.perf_counter()
    if stream:
        print(file=sys.stderr)
    result = {
        'case': case, 'prompt_tokens': len(tokens), 'reused_tokens': reused,
        'new_prompt_tokens': len(tokens) - reused,
        'bulk_prefill_tokens_per_model': len(tokens) - reused - 1,
        'prefix_priming_s': priming_s,
        'prefill_s': prefill_s, 'ttft_s': first - started,
        'decode_s': ended - first,
        'decode_tokens_per_s': throughput(sampled_count - 1, ended - first),
        'total_s': ended - started, 'sampled_tokens_including_eos': sampled_count,
        'output_tokens': len(output_ids), 'finish_reason': finish,
        'accepted_draft_output_tokens': accepted if draft_caches else None,
        'draft_proposed_tokens': None, 'draft_acceptance_rate': None,
        'mlx_active_before_bytes': active_before,
        'mlx_peak_bytes': mx.get_peak_memory(),
        'cache_before_bytes': cache_before,
        'cache_after_bytes': sum(c.nbytes for c in caches + draft_caches),
        'output': ''.join(output), 'output_token_ids': output_ids,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=positive, default=3)
    parser.add_argument('--max-tokens', type=positive, default=64)
    parser.add_argument('--draft-tokens', type=positive, default=3)
    parser.add_argument('--prefill-chunk', type=positive, default=256)
    parser.add_argument('--document-repeats', type=positive, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stream', action='store_true', help='print live text to stderr; adds I/O to timing')
    parser.add_argument('--output', type=Path, default=Path('results/latest.json'))
    args = parser.parse_args()
    global mx, generation_stream, make_prompt_cache, stream_generate, make_sampler
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.generate import generation_stream
    from mlx_lm.models.cache import make_prompt_cache, can_trim_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    if not mx.metal.is_available():
        raise RuntimeError('This experiment requires an Apple Silicon Mac with Metal')
    for package, expected in [('mlx', '0.32.3'), ('mlx-lm', '0.31.3')]:
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f'{package} must be pinned to {expected}')
    paths = [snapshot(TARGET), snapshot(DRAFT)]
    loaded = []
    for path in paths:
        started = time.perf_counter()
        model, tokenizer, config = load(str(path), return_config=True)
        sync()
        loaded.append((model, tokenizer, config, time.perf_counter() - started))
    model, tokenizer, config, load_s = loaded[0]
    draft, draft_tokenizer, draft_config, draft_load_s = loaded[1]
    fingerprint = tokenizer_fingerprint(tokenizer)
    if fingerprint != tokenizer_fingerprint(draft_tokenizer):
        raise ValueError('Target and draft tokenizer pipelines differ')
    if tokenizer.eos_token_ids != draft_tokenizer.eos_token_ids:
        raise ValueError('Target and draft EOS tokens differ')
    if config['vocab_size'] != draft_config['vocab_size']:
        raise ValueError('Target and draft output vocabulary sizes differ')
    for candidate in (model, draft):
        if not can_trim_prompt_cache(make_prompt_cache(candidate)):
            raise ValueError('Speculative decoding requires trimmable caches')
    system = 'Answer using this reference.\n' + (
        'The observatory has three telescopes. It opens at sunset. '
        'Mira maintains the instruments. Visitors book tours on Fridays.\n'
    ) * args.document_repeats
    questions = ['When does the observatory open?', 'Describe the observatory and explain how a visitor can arrange a tour.']
    prompts = [tokenizer.apply_chat_template([
        {'role': 'system', 'content': system}, {'role': 'user', 'content': question}
    ], tokenize=True, add_generation_prompt=True) for question in questions]
    prefix = prompts[0][:common_prefix(*prompts)]
    tokens = prompts[1]
    if len(tokens) + args.max_tokens > config['max_position_embeddings']:
        raise ValueError('Prompt and output exceed the model context limit')
    print(f'Verified tokenizer pipeline and trimmable caches; {len(tokens)} prompt tokens, {len(prefix)} reusable.', file=sys.stderr)
    # Both models stay resident for every case, including baseline. This keeps
    # residency consistent; peak differences do not measure loading a draft.
    cases = ['baseline', 'prefix', 'speculative']
    for case in cases:
        mx.random.seed(args.seed)
        run_case(model, tokenizer, draft, tokens, prefix, case, args)
    runs = []
    for repeat in range(args.repeats):
        # Rotate case order to reduce systematic order bias; never run concurrently.
        for case in cases[repeat % 3:] + cases[:repeat % 3]:
            mx.random.seed(args.seed)
            result = run_case(model, tokenizer, draft, tokens, prefix, case, args, args.stream)
            result['repeat'] = repeat
            runs.append(result)
            print(f'{case}: TTFT={result["ttft_s"]:.3f}s total={result["total_s"]:.3f}s', file=sys.stderr)
    reference = next(r['output_token_ids'] for r in runs if r['case'] == 'baseline')
    for run in runs:
        run['matches_first_baseline_tokens'] = run['output_token_ids'] == reference
    document = {
        'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
        'environment': {
            'device': mx.device_info(), 'macos': platform.mac_ver()[0],
            'python': platform.python_version(), 'machine': platform.machine(),
            'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
            'memory_pressure': subprocess.run(['memory_pressure', '-Q'], capture_output=True, text=True).stdout,
            'power_thermal': subprocess.run(['pmset', '-g', 'therm'], capture_output=True, text=True).stdout,
        },
        'models': [{'repo': spec[0], 'revision': spec[1], 'config': item[2], 'load_s': item[3]}
                   for spec, item in zip((TARGET, DRAFT), loaded)],
        'tokenizer_sha256': fingerprint, 'chat_template': tokenizer.chat_template,
        'messages': {'system': system, 'questions': questions},
        'prompt_token_ids': tokens, 'prefix_token_ids': prefix,
        'settings': {**vars(args), 'output': str(args.output), 'temperature': 0.0,
                     'kv_quantization': None, 'attention': 'automatic', 'warmups_per_case': 1},
        'runs': runs, 'summary': summarize(runs),
        'notes': [
            'Prefill times processing through the penultimate prompt token; TTFT includes final-token forward and first speculative block.',
            'Decode throughput counts sampled tokens after the first, including EOS; speculative blocks can emit multiple tokens at once.',
            'Prefix priming is outside request timing, measured separately, and rebuilt before each prefix trial.',
            'MLX peak is active MLX allocation, not process RSS or total system memory. Both models remain resident in every case.',
            'Accepted draft output tokens exclude EOS. Proposal count and acceptance rate are unavailable from the public API.',
            'One full-length warmup per case is excluded. Compilation for new shapes and thermal drift may still affect measurements.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(document, indent=2) + '\n')
    temporary.replace(args.output)
    print(f'Saved {args.output}', file=sys.stderr)
    if any(not r['matches_first_baseline_tokens'] for r in runs if r['case'] == 'prefix'):
        raise RuntimeError('Prefix output differs from baseline; inspect saved results before trusting cache comparison')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(json.dumps({'error': 'cancelled', 'message': 'Interrupted; incomplete results were not saved'}), file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(json.dumps({'error': type(exc).__name__, 'message': str(exc)}), file=sys.stderr)
        sys.exit(1)
