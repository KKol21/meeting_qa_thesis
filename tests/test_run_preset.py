"""Tests that the TOML preset controls the stage process plan."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meeting_qa_chunking.run_preset import stage_commands


class PresetRunnerTest(unittest.TestCase):
    def test_smoke_preset_expands_to_isolated_stages(self) -> None:
        preset = (ROOT / "src/configs/ablation-smoke.toml").resolve()
        commands = stage_commands(preset)

        self.assertEqual(len(commands), 8)
        self.assertTrue(commands[0][0].endswith("ablation_segment.py"))
        self.assertTrue(commands[1][0].endswith("ablation_retrieval.py"))
        self.assertEqual(
            [
                command[-1]
                for command in commands
                if command[0].endswith("ablation_answer.py")
            ],
            ["oracle-7b", "oracle-14b", "oracle-32b-bnb4", "retrieval-14b"],
        )
        self.assertTrue(commands[-2][0].endswith("ablation_evaluate.py"))
        self.assertTrue(commands[-1][0].endswith("summarize_ablations.py"))


if __name__ == "__main__":
    unittest.main()
