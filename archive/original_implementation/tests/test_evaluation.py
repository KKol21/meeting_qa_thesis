import unittest

from meeting_qa_chunking.evaluation import (
    PairedScore,
    build_elitr_judge_prompt,
    gold_source_spans,
    paired_meeting_bootstrap,
    parse_elitr_judgment,
    score_evidence,
)
from meeting_qa_chunking.retrieval import Evidence
from meeting_qa_chunking.schema import Meeting, Query, Span, Turn
from meeting_qa_chunking.tokenization import WhitespaceTokenizer, tokenize_meeting


class RetrievalEvaluationTests(unittest.TestCase):
    def test_scores_unique_source_tokens_against_inclusive_gold_turn_ranges(self) -> None:
        meeting = Meeting(
            "qmsum",
            "val",
            "m",
            tuple(Turn(index, "A", word) for index, word in enumerate(("one", "two", "three"))),
        )
        transcript = tokenize_meeting(meeting, WhitespaceTokenizer())
        query = Query(
            "qmsum",
            "val",
            "q",
            "m",
            "Question?",
            "Answer.",
            gold_turn_ranges=((1, 2),),
        )
        gold = gold_source_spans(query, transcript)
        retrieved = Evidence(
            "evidence",
            (Span(transcript.turns[1].source_span.start, transcript.turns[1].source_span.end),),
            ("chunk",),
        )

        scores = score_evidence(query, retrieved, transcript)

        self.assertEqual(gold[0].start, transcript.turns[1].source_span.start)
        self.assertEqual(gold[0].end, transcript.turns[2].source_span.end)
        self.assertEqual(scores.precision, 1.0)
        self.assertEqual(scores.recall, 0.5)
        self.assertFalse(scores.zero_hit)

    def test_rejects_queries_without_evidence_annotations(self) -> None:
        meeting = Meeting("elitr", "dev", "m", (Turn(0, "A", "one"),))
        transcript = tokenize_meeting(meeting, WhitespaceTokenizer())
        query = Query("elitr", "dev", "q", "m", "Question?", "Answer.")

        with self.assertRaisesRegex(ValueError, "no gold turn ranges"):
            gold_source_spans(query, transcript)


class ELITREvaluationTests(unittest.TestCase):
    def test_official_rubric_uses_only_question_answer_and_reference(self) -> None:
        prompt = build_elitr_judge_prompt("Who agreed?", "Alex.", "Alex agreed.")

        self.assertIn("### Question:\nWho agreed?", prompt)
        self.assertIn("### Response to evaluate:\nAlex.", prompt)
        self.assertIn("### Reference answer (score 10):\nAlex agreed.", prompt)
        self.assertNotIn("Meeting evidence", prompt)

    def test_parses_and_validates_boxed_score(self) -> None:
        judgment = parse_elitr_judgment("Good coverage.\n\\boxed{8}")

        self.assertEqual(judgment.score, 8)
        self.assertEqual(judgment.feedback, "Good coverage.")
        with self.assertRaisesRegex(ValueError, "outside"):
            parse_elitr_judgment("Invalid. \\boxed{11}")


class BootstrapTests(unittest.TestCase):
    def test_resamples_paired_scores_by_meeting(self) -> None:
        scores = (
            PairedScore("m1", 2.0, 1.0),
            PairedScore("m1", 3.0, 2.0),
            PairedScore("m2", 5.0, 4.0),
        )

        interval = paired_meeting_bootstrap(scores, samples=100, seed=42)

        self.assertEqual(interval.delta, 1.0)
        self.assertEqual(interval.lower, 1.0)
        self.assertEqual(interval.upper, 1.0)


if __name__ == "__main__":
    unittest.main()
