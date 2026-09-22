import unittest

from meeting_qa_chunking.judging import (
    build_faithfulness_prompt,
    build_sufficiency_prompt,
    parse_axis_judgment,
)


class GroundedJudgingTest(unittest.TestCase):
    def test_independent_prompts_keep_their_inputs_separate(self):
        faithful = build_faithfulness_prompt("q", "shown", "answer")
        sufficient = build_sufficiency_prompt("q", "reference", "shown")

        self.assertIn("Candidate answer:\nanswer", faithful)
        self.assertNotIn("Reference answer:", faithful)
        self.assertIn("Reference answer:\nreference", sufficient)
        self.assertNotIn("Candidate answer:", sufficient)

    def test_parses_independent_axis(self):
        result = parse_axis_judgment(
            '{"score": 3, "unsupported_claims": [], "reason": "Supported."}',
            "unsupported_claims",
        )

        self.assertEqual(result["score"], 3)

    def test_rejects_faithfulness_below_three_without_unsupported_claims(self):
        with self.assertRaises(ValueError):
            parse_axis_judgment(
                '{"score": 2, "unsupported_claims": [], "reason": "Partial."}',
                "unsupported_claims",
            )


if __name__ == "__main__":
    unittest.main()
