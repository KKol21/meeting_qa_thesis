import unittest

from meeting_qa_chunking.chunking import Chunk
from meeting_qa_chunking.qmsum import Meeting, Turn
from tools.boundary_shuffled_control import boundaries, shuffled_chunks


class BoundaryControlTest(unittest.TestCase):
    def test_preserves_coverage_and_chunk_count_but_moves_boundaries(self) -> None:
        meeting = Meeting(
            "Tiny",
            [Turn(index, "A", "word " * (index + 1)) for index in range(12)],
            [],
        )
        lumber = [
            Chunk.from_turns(0, meeting.turns[:2]),
            Chunk.from_turns(1, meeting.turns[2:6]),
            Chunk.from_turns(2, meeting.turns[6:9]),
            Chunk.from_turns(3, meeting.turns[9:]),
        ]

        first, _distance = shuffled_chunks(meeting, lumber, seed=7, candidates=100)
        second, _distance = shuffled_chunks(meeting, lumber, seed=7, candidates=100)

        self.assertEqual(len(first), len(lumber))
        self.assertNotEqual(boundaries(first), boundaries(lumber))
        self.assertEqual(boundaries(first), boundaries(second))
        self.assertEqual(
            [part.turn_id for chunk in first for part in chunk.parts],
            list(range(len(meeting.turns))),
        )


if __name__ == "__main__":
    unittest.main()
