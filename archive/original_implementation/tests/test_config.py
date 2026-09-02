import unittest

from meeting_qa_chunking.config import load_config


class ConfigTests(unittest.TestCase):
    def test_baseline_config_is_valid(self) -> None:
        config = load_config("configs/baseline.toml")
        self.assertEqual(config.chunking.tokenizer, config.retrieval.model)
        self.assertEqual(config.chunking.fixed_tokens, 256)
        self.assertEqual(config.lumberchunker.window_tokens, 550)
        self.assertEqual(config.retrieval.primary_budget, 512)
        self.assertEqual(config.retrieval.max_sequence_tokens, 8192)
        self.assertEqual(len(config.retrieval.revision), 40)
        self.assertEqual(config.models.answer.name, "gpt-5.6-luna")
        self.assertEqual(config.models.judge.name, "gpt-5.6-terra")
        self.assertEqual(config.evaluation.bootstrap_samples, 2000)



if __name__ == "__main__":
    unittest.main()
