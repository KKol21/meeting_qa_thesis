import unittest

from meeting_qa_chunking.chunking import (
    LumberChunker,
    fixed_token_chunks,
    turn_packed_chunks,
)
from meeting_qa_chunking.chunking.lumber import (
    LUMBERCHUNKER_INSTRUCTIONS,
    build_prompt,
)
from meeting_qa_chunking.schema import Meeting, Turn
from meeting_qa_chunking.tokenization import WhitespaceTokenizer, tokenize_meeting


class FakeBoundaryModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, temperature: float) -> str:
        self.prompts.append(prompt)
        return next(self.responses)


def meeting(*texts: str) -> Meeting:
    return Meeting(
        dataset="test",
        split="test",
        id="meeting",
        turns=tuple(Turn(index, f"S{index % 2}", text) for index, text in enumerate(texts)),
    )


class DeterministicChunkingTests(unittest.TestCase):
    def test_fixed_chunks_are_non_overlapping_source_ranges(self) -> None:
        tokenizer = WhitespaceTokenizer()
        transcript = tokenize_meeting(meeting("one two", "three four"), tokenizer)
        chunks = fixed_token_chunks(transcript, tokenizer, chunk_size=3)
        self.assertEqual(chunks[0].source_spans[0].end, chunks[1].source_spans[0].start)

    def test_turn_packing_never_splits_a_turn(self) -> None:
        tokenizer = WhitespaceTokenizer()
        transcript = tokenize_meeting(meeting("one", "two", "three"), tokenizer)
        chunks = turn_packed_chunks(transcript, chunk_size=6)
        self.assertEqual(chunks[0].turn_ids, (0, 1))
        self.assertEqual(chunks[1].turn_ids, (2,))


class LumberChunkerTests(unittest.TestCase):
    def test_uses_rolling_local_windows_and_boundary_starts_next_chunk(self) -> None:
        source = meeting(
            "alpha alpha",
            "alpha alpha",
            "beta beta",
            "beta beta",
            "tail tail",
        )
        tokenizer = WhitespaceTokenizer()
        transcript = tokenize_meeting(source, tokenizer)
        model = FakeBoundaryModel(["Answer: ID 0003", "Answer: ID 0005"])

        result = LumberChunker(model, window_tokens=12).segment(transcript)

        self.assertEqual(
            [chunk.turn_ids for chunk in result.chunks], [(0, 1), (2, 3), (4,)]
        )
        self.assertEqual(
            [decision.boundary_turn_id for decision in result.decisions], [2, 4]
        )
        self.assertEqual(result.decisions[0].prompt, model.prompts[0])
        self.assertIn("ID 0001: S0: alpha", model.prompts[0])
        self.assertNotIn("ID 0004", model.prompts[0])
        second_document = model.prompts[1].split("Document:\n", maxsplit=1)[1]
        self.assertTrue(second_document.startswith("ID 0003:"))

    def test_prompt_uses_the_complete_instruction_block_from_the_paper(self) -> None:
        prompt = build_prompt(meeting("alpha", "beta").turns)
        instructions, document = prompt.split("\n\nDocument:\n", maxsplit=1)
        paper_prompt = (
            "You will receive as input an English document with paragraphs "
            "identified by 'ID XXXX: <text>'.\n\n"
            "Task: Find the first paragraph (not the first one) where the content "
            "clearly changes compared to the previous paragraphs.\n\n"
            "Output: Return the ID of the paragraph with the content shift as in "
            "the exemplified format: 'Answer: ID XXXX'.\n"
            "Additional Considerations: Avoid very long groups of paragraphs. "
            "Aim for a good balance between identifying content shifts and keeping "
            "groups manageable."
        )

        self.assertEqual(LUMBERCHUNKER_INSTRUCTIONS, paper_prompt)
        self.assertEqual(instructions, paper_prompt)
        self.assertEqual(document, "ID 0001: S0: alpha\nID 0002: S1: beta")

    def test_single_oversized_tail_does_not_request_an_impossible_boundary(self) -> None:
        tokenizer = WhitespaceTokenizer()
        transcript = tokenize_meeting(meeting("one two three four five"), tokenizer)
        model = FakeBoundaryModel([])

        result = LumberChunker(model, window_tokens=2).segment(transcript)

        self.assertEqual(result.chunks[0].turn_ids, (0,))
        self.assertEqual(model.prompts, [])


if __name__ == "__main__":
    unittest.main()
