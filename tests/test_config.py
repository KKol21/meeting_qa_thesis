"""Tests for the small typed ablation presets."""

import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from meeting_qa_chunking.config import (
    ANSWER_MODELS,
    JUDGE_MODEL,
    load_run_config,
    retrieval_conditions,
)


class ConfigTest(unittest.TestCase):
    def test_condition_names_cover_all_three_chunkers(self) -> None:
        conditions = retrieval_conditions([512, 1024])
        self.assertEqual(len(conditions), 18)
        self.assertEqual(conditions[0].name, "turn_packed__dense__w512")
        self.assertEqual(conditions[-1].name, "lumber__hybrid__w1024")

    def test_loads_smoke_preset(self) -> None:
        config = load_run_config(
            REPOSITORY_ROOT / "src/configs/ablation-smoke.toml"
        )
        self.assertEqual(config.meeting_ids(REPOSITORY_ROOT), ["Bed002"])
        self.assertEqual(config.output_root, Path("runs/ablations/smoke"))
        self.assertEqual(config.retrieval.evidence_order, "chronological")
        self.assertEqual(config.retrieval.turn_packed_max_words, 256)
        self.assertEqual(config.retrieval.word_packed_max_words, 256)
        self.assertEqual(config.segmentation.max_new_tokens, 32)
        self.assertEqual(config.segmentation.temperature, 0.0)
        self.assertEqual(len(config.answers), 4)
        self.assertTrue(config.answers[2].model.prequantized)

    def test_full_preset_uses_the_checked_in_meeting_manifest(self) -> None:
        config = load_run_config(
            REPOSITORY_ROOT / "src/configs/ablation-full.toml"
        )
        meeting_ids = config.meeting_ids(REPOSITORY_ROOT)
        self.assertEqual(len(meeting_ids), 20)
        self.assertEqual(len(meeting_ids), len(set(meeting_ids)))
        self.assertNotIn("Bed002", meeting_ids)
        self.assertIn("education_18", meeting_ids)

    def test_model_revisions_are_centralized(self) -> None:
        self.assertEqual(
            ANSWER_MODELS["qwen2.5-14b"].revision,
            "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        )

    def test_judge_is_the_prequantized_llama_70b_checkpoint(self) -> None:
        self.assertEqual(JUDGE_MODEL.tag, "llama-3.3-70b-bnb4")
        self.assertTrue(JUDGE_MODEL.prequantized)


if __name__ == "__main__":
    unittest.main()
