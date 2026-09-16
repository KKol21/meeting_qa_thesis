import unittest

from meeting_qa_chunking.chunking import Chunk
from meeting_qa_chunking.qmsum import Turn
from tools.analyze_clipping_sensitivity import variants


class ClippingSensitivityTest(unittest.TestCase):
    def test_compares_exact_drop_and_expand_policies(self) -> None:
        chunks = [
            Chunk.from_turns(0, [Turn(0, "A", "one two three")]),
            Chunk.from_turns(1, [Turn(1, "B", "four five six")]),
        ]

        policies, partial = variants(
            {"selected_chunk_indices": [0, 1]}, chunks, budget=5
        )

        self.assertTrue(partial)
        self.assertEqual(policies["clip"].word_count, 5)
        self.assertEqual(policies["drop_partial"].word_count, 3)
        self.assertEqual(policies["expand_partial"].word_count, 6)


if __name__ == "__main__":
    unittest.main()
