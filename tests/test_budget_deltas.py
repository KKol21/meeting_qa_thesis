import unittest

from tools.analyze_budget_deltas import (
    RECALL_BINS,
    average_ranks,
    spearman,
    summarize_bins,
    summarize_rows,
)


def row(meeting, recall, precision, judge):
    return {
        "meeting_id": meeting,
        "recall_gain": recall,
        "precision_loss": -precision,
        "delta": {
            "precision": precision,
            "recall": recall,
            "rougeL": judge / 10,
            "bertscore_f1": judge / 10,
            "judge": judge,
        },
    }


class BudgetDeltaTests(unittest.TestCase):
    def test_average_ranks_average_ties(self):
        self.assertEqual(average_ranks([3, 1, 1, 2]), [4, 1.5, 1.5, 3])

    def test_spearman_detects_monotonic_association(self):
        rows = [row("A", value, 0, value) for value in (0, 1, 2)]
        self.assertAlmostEqual(spearman(rows, "recall_gain", "judge"), 1.0)

    def test_summary_weights_meetings_equally(self):
        rows = [row("A", 1, 0, 1), row("A", 1, 0, 1), row("B", 0, 0, 0)]
        summary = summarize_rows(rows, samples=100, seed=7)
        self.assertEqual(summary["question_count"], 3)
        self.assertEqual(summary["meeting_count"], 2)
        self.assertEqual(summary["estimates"]["recall"]["mean"], 0.5)

    def test_recall_bins_use_declared_boundaries(self):
        rows = [
            row("A", 0.09, 0, 0),
            row("B", 0.10, 0, 0),
            row("C", 0.25, 0, 0),
        ]
        bins = summarize_bins(rows, RECALL_BINS, "recall_gain", 20, 3)
        self.assertEqual([item["question_count"] for item in bins], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
