import unittest

from meeting_qa_chunking.schema import Chunk, Meeting, Span, Turn


class SchemaTests(unittest.TestCase):
    def test_meeting_requires_contiguous_turn_ids(self) -> None:
        with self.assertRaises(ValueError):
            Meeting(
                dataset="qmsum",
                split="test",
                id="meeting",
                turns=(Turn(1, "Speaker", "Hello"),),
            )

    def test_chunk_counts_half_open_source_spans(self) -> None:
        chunk = Chunk(
            id="chunk",
            meeting_id="meeting",
            method="fixed",
            text="Some source text",
            source_spans=(Span(0, 3), Span(5, 7)),
            turn_ids=(0,),
        )
        self.assertEqual(chunk.source_token_count, 5)


if __name__ == "__main__":
    unittest.main()

