import unittest

from meeting_qa_chunking.uncertainty import bootstrap_mean, paired_bootstrap


class UncertaintyTest(unittest.TestCase):
    def test_bootstrap_mean_is_deterministic(self):
        first = bootstrap_mean([1.0, 2.0, 3.0], samples=100, seed=7)
        second = bootstrap_mean([1.0, 2.0, 3.0], samples=100, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(first["mean"], 2.0)

    def test_paired_bootstrap_uses_within_pair_differences(self):
        result = paired_bootstrap([2.0, 4.0], [1.0, 3.0], samples=100)

        self.assertEqual(result, {"mean": 1.0, "ci95": [1.0, 1.0]})

    def test_paired_bootstrap_rejects_unmatched_samples(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            paired_bootstrap([1.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
