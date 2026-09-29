import unittest
from experiments.metrics import common_prefix, summarize, throughput


class MetricsTests(unittest.TestCase):
    def test_prefix_keeps_final_token(self):
        self.assertEqual(common_prefix([1, 2, 3], [1, 2, 3]), 2)
        self.assertEqual(common_prefix([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(common_prefix([8], [1, 2]), 0)
        self.assertEqual(common_prefix([], []), 0)

    def test_no_decode_rate_for_single_token(self):
        self.assertIsNone(throughput(0, 0.01))
        self.assertIsNone(throughput(2, 0))
        self.assertEqual(throughput(20, 2), 10)

    def test_summary_does_not_turn_missing_measurements_into_zero(self):
        rows = [dict(case='baseline', prefill_s=v, ttft_s=v, total_s=v,
                     decode_tokens_per_s=None, mlx_peak_bytes=100) for v in [1, 3]]
        result = summarize(rows)['baseline']
        self.assertEqual(result['ttft_s']['median'], 2)
        self.assertEqual(result['ttft_s']['n'], 2)
        self.assertIsNone(result['decode_tokens_per_s']['median'])


if __name__ == '__main__':
    unittest.main()
