"""Protect prompt files from accidental whitespace changes."""

import hashlib
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from meeting_qa_chunking.judging import JUDGE_INSTRUCTION
from meeting_qa_chunking.lumber_prompt import LUMBERCHUNKER_INSTRUCTIONS
from meeting_qa_chunking.prompt_files import load_prompt


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PromptFileTest(unittest.TestCase):
    def test_answer_prompt_preserves_the_current_instruction(self) -> None:
        self.assertEqual(
            sha256(load_prompt("answer.txt")),
            "7d1b6b471716405dcda36be81089d8ff12b7946dabcfe8030fe45be5dce1e367",
        )

    def test_lumber_prompt_preserves_the_original_instruction(self) -> None:
        self.assertEqual(
            sha256(LUMBERCHUNKER_INSTRUCTIONS),
            "4167574ef8fa8d78aab7a233cd103f38fd2c8db1ea6f263b69e47ebfe6fe37ed",
        )

    def test_judge_prompt_is_loaded_from_text(self) -> None:
        self.assertEqual(JUDGE_INSTRUCTION, load_prompt("judge.txt"))
        self.assertIn('1 = Invalid or incorrect.', JUDGE_INSTRUCTION)
        self.assertIn('{"score": 1, "reason": "..."}', JUDGE_INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
