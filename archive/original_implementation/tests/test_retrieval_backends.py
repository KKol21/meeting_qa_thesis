import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from meeting_qa_chunking.retrieval.backends import SentenceTransformerEmbedder


class FakeSentenceTransformer:
    def __init__(self, model_name, *, revision, local_files_only):
        self.max_seq_length = 4

    def tokenizer(self, texts, **kwargs):
        return {"input_ids": [text.split() for text in texts]}

    def encode(self, texts, **kwargs):
        return [(1.0, 0.0) for _ in texts]


class RetrievalBackendTests(unittest.TestCase):
    def test_refuses_silent_embedding_truncation(self) -> None:
        fake_module = SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            embedder = SentenceTransformerEmbedder("fake", max_sequence_tokens=4)

        with self.assertRaisesRegex(ValueError, "refusing silent truncation"):
            embedder.embed_documents(["one two three four five"])

    def test_rejects_limit_above_model_support(self) -> None:
        fake_module = SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            with self.assertRaisesRegex(ValueError, "supports 4"):
                SentenceTransformerEmbedder("fake", max_sequence_tokens=5)


if __name__ == "__main__":
    unittest.main()
