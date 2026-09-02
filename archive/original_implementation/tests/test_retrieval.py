import unittest

from meeting_qa_chunking.retrieval import DenseIndex, RankedChunk, project_evidence
from meeting_qa_chunking.schema import Chunk, Meeting, Span, Turn
from meeting_qa_chunking.tokenization import WhitespaceTokenizer, tokenize_meeting


class FakeEmbedder:
    def embed_documents(self, texts):
        return [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]

    def embed_queries(self, texts):
        return [(1.0, 0.0)]


class DenseRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = tuple(
            Chunk(
                id=f"m:fixed:{index}",
                meeting_id="m",
                method="fixed",
                text=text,
                source_spans=(Span(index, index + 1),),
                turn_ids=(index,),
            )
            for index, text in enumerate(("alpha", "beta", "mixed"))
        )

    def test_ranks_by_cosine_similarity_with_stable_ties(self) -> None:
        ranking = DenseIndex.build(self.chunks, FakeEmbedder()).search("alpha")

        self.assertEqual([item.chunk.text for item in ranking], ["alpha", "mixed", "beta"])
        self.assertEqual([item.rank for item in ranking], [1, 2, 3])


class SourceBudgetTests(unittest.TestCase):
    def test_deduplicates_overlap_and_returns_exact_budget_in_source_order(self) -> None:
        meeting = Meeting(
            dataset="test",
            split="test",
            id="m",
            turns=(Turn(0, "A", "one two three four five six"),),
        )
        tokenizer = WhitespaceTokenizer()
        transcript = tokenize_meeting(meeting, tokenizer)
        late = Chunk("late", "m", "test", "late", (Span(4, 7),), (0,))
        early = Chunk("early", "m", "test", "early", (Span(0, 5),), (0,))
        ranking = (
            RankedChunk(late, 1.0, 1),
            RankedChunk(early, 0.5, 2),
        )

        evidence = project_evidence(ranking, transcript, tokenizer, budget=5)

        self.assertEqual(evidence.source_token_count, 5)
        self.assertEqual(evidence.source_spans, (Span(0, 2), Span(4, 7)))
        self.assertEqual(evidence.selected_chunk_ids, ("late", "early"))

    def test_caps_budget_at_transcript_length(self) -> None:
        meeting = Meeting("test", "test", "m", (Turn(0, "A", "one"),))
        tokenizer = WhitespaceTokenizer()
        transcript = tokenize_meeting(meeting, tokenizer)
        chunk = Chunk(
            "all",
            "m",
            "test",
            "all",
            (Span(0, len(transcript.token_ids)),),
            (0,),
        )

        evidence = project_evidence(
            (RankedChunk(chunk, 1.0, 1),), transcript, tokenizer, budget=100
        )

        self.assertEqual(evidence.source_token_count, len(transcript.token_ids))


if __name__ == "__main__":
    unittest.main()
