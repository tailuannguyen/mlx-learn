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
from .cache import PrefixCache, cache_bytes

TARGET = ('mlx-community/Qwen2.5-1.5B-Instruct-4bit', '8b403126fc14f14cfc99bb4cfa72ecbc129ea677')
DRAFT = ('mlx-community/Qwen2.5-0.5B-Instruct-4bit', 'a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3')
KV_CASES = {'kv-8bit': 8, 'kv-4bit': 4}


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


def run_case(model, tokenizer, draft, tokens, prefix_tokens, case, args, stream=False,
             persistent_cache=None, cache_identity=None):
    try:
        return _run_case(model, tokenizer, draft, tokens, prefix_tokens, case, args,
                         stream, persistent_cache, cache_identity)
    except BaseException:
        if persistent_cache is not None:
            persistent_cache.clear()
        raise


def _run_case(model, tokenizer, draft, tokens, prefix_tokens, case, args, stream,
              persistent_cache, cache_identity):
    if case not in {'baseline', 'prefix', 'speculative', 'persistent-prefix', *KV_CASES}:
        raise ValueError(f'Unknown experiment case: {case}')
    if not tokens:
        raise ValueError('A nonempty token prompt is required')
    if case == 'persistent-prefix' and (persistent_cache is None or cache_identity is None):
        raise ValueError('Persistent prefix trials require a cache and explicit identity')
    if case != 'persistent-prefix' and persistent_cache is not None:
        raise ValueError('Persistent caching cannot be combined with other cases')
    gc.collect()
    mx.clear_cache()
    prepared = time.perf_counter()
    reused = 0
    if persistent_cache is not None:
        caches, reused = persistent_cache.prepare(cache_identity, tokens)
    else:
        caches = make_prompt_cache(model)
    kv_bits = KV_CASES.get(case)
    if kv_bits is not None:
        # Convert empty caches before any model work, so all prompt/decode steps
        # use quantized history, including the manually timed bulk prefill.
        caches = [c.to_quantized(group_size=64, bits=kv_bits) for c in caches]
    cache_prepare_s = time.perf_counter() - prepared
    draft_caches = make_prompt_cache(draft) if case == 'speculative' else []
    priming_s = 0.0
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
    cache_before = cache_bytes(caches)
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
    sampled_ids = []
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
            sampled_ids.append(response.token)
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
    cache_after_bytes = cache_bytes(caches + draft_caches)
    cache_layers = [{'type': type(c).__name__, 'bits': getattr(c, 'bits', None),
                     'group_size': getattr(c, 'group_size', None), 'offset': c.offset}
                    for c in caches]
    if kv_bits is not None and any(c['bits'] != kv_bits or c['group_size'] != 64
                                   for c in cache_layers):
        raise RuntimeError('Requested KV quantization did not execute on every target layer')
    committed = time.perf_counter()
    retained_tokens = persistent_cache.commit(sampled_ids) if persistent_cache is not None else None
    cache_commit_s = time.perf_counter() - committed if persistent_cache is not None else 0.0
    if stream:
        print(file=sys.stderr)
    result = {
        'case': case, 'prompt_tokens': len(tokens), 'reused_tokens': reused,
        'new_prompt_tokens': len(tokens) - reused,
        'bulk_prefill_tokens_per_model': len(tokens) - reused - 1,
        'prefix_priming_s': priming_s,
        'cache_state': 'reused' if reused else 'cold',
        'cache_prepare_s': cache_prepare_s, 'cache_commit_s': cache_commit_s,
        'retained_tokens': retained_tokens,
        'cache_retained_bytes': cache_bytes(caches) if persistent_cache is not None else None,
        'kv_bits_requested': kv_bits, 'kv_group_size_requested': 64 if kv_bits else None,
        'target_cache_layers': cache_layers,
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
        'cache_after_bytes': cache_after_bytes,
        'output': ''.join(output), 'output_token_ids': output_ids,
    }
    return result


def run_campaign(model, tokenizer, draft, requests, prefix, case, args, identity,
                 repeat=None, stream=False):
    """Keep state within one request sequence, release it before another case."""
    persistent = PrefixCache(lambda: make_prompt_cache(model)) if case == 'persistent-prefix' else None
    results = []
    try:
        for index, tokens in enumerate(requests):
            mx.random.seed(args.seed)
            result = run_case(model, tokenizer, draft, tokens, prefix, case, args, stream,
                              persistent_cache=persistent, cache_identity=identity)
            result.update(repeat=repeat, request_index=index)
            results.append(result)
    finally:
        if persistent is not None:
            persistent.clear()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['standard', 'persistent-prefix', 'kv-quantization'],
                        default='standard', help='run independent controlled comparisons')
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
    # Fixed transcripts keep baseline and cached cases' token inputs identical.
    # The final request continues a conversation; it is not fed a measured reply.
    request_messages = [[{'role': 'system', 'content': system},
                         {'role': 'user', 'content': question}] for question in questions]
    request_messages.append(request_messages[1] + [
        {'role': 'assistant', 'content': 'The observatory opens at sunset. Visitors book tours on Fridays.'},
        {'role': 'user', 'content': 'Which day should I book a tour?'}])
    if args.experiment == 'persistent-prefix':
        requests = [tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                    for messages in request_messages]
        cases = ['baseline', 'persistent-prefix']
    else:
        requests = [tokens]
        request_messages = [request_messages[1]]
        cases = ['baseline', 'kv-8bit', 'kv-4bit'] if args.experiment == 'kv-quantization' else [
            'baseline', 'prefix', 'speculative']
    if any(len(prompt) + args.max_tokens > config['max_position_embeddings'] for prompt in requests):
        raise ValueError('Prompt and output exceed the model context limit')
    print(f'Verified tokenizer pipeline and trimmable caches; {len(tokens)} prompt tokens, {len(prefix)} reusable.', file=sys.stderr)
    # Both models stay resident for every case, including baseline. This keeps
    # residency consistent; peak differences do not measure loading a draft.
    identity = (TARGET, fingerprint, tokenizer.chat_template, 'unquantized-full-cache')
    for case in cases:
        run_campaign(model, tokenizer, draft, requests, prefix, case, args, identity)
    runs = []
    for repeat in range(args.repeats):
        # Rotate whole campaigns, preserving request order and isolating cache residency.
        rotation = repeat % len(cases)
        for case in cases[rotation:] + cases[:rotation]:
            results = run_campaign(model, tokenizer, draft, requests, prefix, case, args,
                                   identity, repeat, args.stream)
            runs.extend(results)
            for result in results:
                print(f'{case} request {result["request_index"]}: TTFT={result["ttft_s"]:.3f}s '
                      f'total={result["total_s"]:.3f}s', file=sys.stderr)
    references = {index: next(r['output_token_ids'] for r in runs
                              if r['case'] == 'baseline' and r['request_index'] == index)
                  for index in range(len(requests))}
    for run in runs:
        run['matches_first_baseline_tokens'] = run['output_token_ids'] == references[run['request_index']]
    document = {
        'schema_version': 2, 'created_at': datetime.now(timezone.utc).isoformat(),
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
        'requests': [{'request_index': index, 'messages': messages, 'prompt_token_ids': prompt}
                     for index, (messages, prompt) in enumerate(zip(request_messages, requests))],
        'settings': {**vars(args), 'output': str(args.output), 'temperature': 0.0,
                     'kv_quantization': {'bits': [None, 8, 4], 'group_size': 64,
                                         'start': 'before first prompt forward pass'}
                     if args.experiment == 'kv-quantization' else None,
                     'attention': 'automatic', 'warmup_campaigns_per_case': 1},
        'runs': runs, 'summary': summarize(runs) if len(requests) == 1 else None,
        'summary_by_request': {str(index): summarize([r for r in runs if r['request_index'] == index])
                               for index in range(len(requests))},
        'quality_evaluation': None,
        'notes': [
            'Prefill times processing through the penultimate prompt token; TTFT includes final-token forward and first speculative block.',
            'Decode throughput counts sampled tokens after the first, including EOS; speculative blocks can emit multiple tokens at once.',
            'Prefix priming is outside request timing, measured separately, and rebuilt before each prefix trial.',
            'MLX peak is active MLX allocation, not process RSS or total system memory. Both models remain resident in every case.',
            'Accepted draft output tokens exclude EOS. Proposal count and acceptance rate are unavailable from the public API.',
            'One full-length warmup per case is excluded. Compilation for new shapes and thermal drift may still affect measurements.',
            'Persistent campaigns start cold, retain known processed prompt/output tokens across requests, and clear state between campaigns. Transcripts are fixed, not generated from measured replies.',
            'Cache preparation and post-generation commit are timed separately and excluded from TTFT/total. Cache bytes include allocated capacity.',
            'Quantized cases use fresh 8/4-bit target KV caches from the first prompt step; weights remain 4-bit. Per-layer cache type/bits/group size are observed.',
            'Quantized output differences are diagnostics, not failures or quality scores. No task-quality evaluation has been performed.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(document, indent=2) + '\n')
    temporary.replace(args.output)
    print(f'Saved {args.output}', file=sys.stderr)
    if any(not r['matches_first_baseline_tokens'] for r in runs
           if r['case'] in {'prefix', 'persistent-prefix'}):
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
