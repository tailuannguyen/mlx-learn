"""Backend-independent measurement helpers."""
import statistics


def common_prefix(a, b):
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    # Always leave at least one token for next-token prediction.
    return min(count, max(0, len(b) - 1))


def throughput(tokens, seconds):
    return tokens / seconds if tokens > 0 and seconds > 0 else None


def summarize(runs):
    summary = {}
    for case in sorted({r['case'] for r in runs}):
        group = [r for r in runs if r['case'] == case]
        summary[case] = {'runs': len(group)}
        for metric in ('prefill_s', 'ttft_s', 'decode_tokens_per_s', 'total_s', 'mlx_peak_bytes',
                       'cache_before_bytes', 'cache_after_bytes', 'cache_retained_bytes',
                       'cache_prepare_s', 'cache_commit_s', 'prefix_priming_s'):
            values = [r[metric] for r in group if r.get(metric) is not None]
            summary[case][metric] = {
                'n': len(values),
                'median': statistics.median(values) if values else None,
                'min': min(values) if values else None,
                'max': max(values) if values else None,
                'stdev': statistics.stdev(values) if len(values) > 1 else None,
            }
    return summary
