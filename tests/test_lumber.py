"""Tests for LumberChunker control flow."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from meeting_qa_chunking.lumber import lumber_chunks
from meeting_qa_chunking.qmsum import Turn


class LumberChunkerTest(unittest.TestCase):
    def test_retries_an_invalid_boundary_once(self) -> None:
        responses = iter(["Answer: ID 0000", "Answer: ID 0001"])
        prompts = []
        decisions = []
        discarded = []

        def choose(prompt: str) -> str:
            prompts.append(prompt)
            return next(responses)

        chunks = lumber_chunks(
            [
                Turn(0, "A", "one"),
                Turn(1, "B", "two"),
                Turn(2, "C", "three"),
            ],
            choose,
            target_tokens=1,
            max_boundaries=1,
            record_decision=decisions.append,
            discard_invalid_response=lambda: discarded.append(True),
        )

        self.assertEqual(len(prompts), 2)
        self.assertIn("cannot be selected", prompts[1])
        self.assertEqual(discarded, [True])
        self.assertEqual(decisions, ["Answer: ID 0001"])
        self.assertEqual((chunks[0].start_turn, chunks[0].end_turn), (0, 0))

    def test_discards_an_invalid_retry(self) -> None:
        discarded = []

        with self.assertRaises(ValueError):
            lumber_chunks(
                [Turn(0, "A", "one"), Turn(1, "B", "two")],
                lambda _prompt: "invalid",
                target_tokens=1,
                discard_invalid_response=lambda: discarded.append(True),
            )

        self.assertEqual(discarded, [True, True])


if __name__ == "__main__":
    unittest.main()
