import unittest

from meeting_qa_chunking.config import APIConfig, ModelConfig
from meeting_qa_chunking.model_backends import OpenAICompatibleModel


class ModelBackendTests(unittest.TestCase):
    def test_sends_chat_completion_request(self) -> None:
        captured = {}
        observed = []

        def transport(url, payload, headers, timeout):
            captured.update(
                url=url, payload=payload, headers=headers, timeout=timeout
            )
            return {
                "choices": [
                    {
                        "message": {"content": "  answer  "},
                        "finish_reason": "stop",
                    }
                ]
            }

        model = OpenAICompatibleModel(
            APIConfig("https://example.test/v1/", "TEST_API_KEY", 12),
            ModelConfig("test-model", 42, "none"),
            api_key="secret",
            transport=transport,
            response_observer=observed.append,
        )

        self.assertEqual(model.complete("prompt", temperature=0.1), "  answer  ")
        self.assertEqual(captured["url"], "https://example.test/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], "test-model")
        self.assertEqual(captured["payload"]["messages"][0]["content"], "prompt")
        self.assertEqual(captured["payload"]["max_completion_tokens"], 42)
        self.assertEqual(captured["payload"]["reasoning_effort"], "none")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(captured["timeout"], 12)
        self.assertEqual(observed[0]["requested_model"], "test-model")
        self.assertEqual(observed[0]["finish_reason"], "stop")
        self.assertEqual(observed[0]["response_text"], "  answer  ")

    def test_missing_api_key_fails_before_transport(self) -> None:
        model = OpenAICompatibleModel(
            APIConfig("https://example.test/v1", "DEFINITELY_MISSING_KEY", 12),
            ModelConfig("test-model", 42),
            transport=lambda *_: self.fail("transport should not be called"),
        )

        with self.assertRaisesRegex(RuntimeError, "DEFINITELY_MISSING_KEY"):
            model.complete("prompt", temperature=0.0)

    def test_rejects_truncated_completion(self) -> None:
        model = OpenAICompatibleModel(
            APIConfig("https://example.test/v1", "TEST_API_KEY", 12),
            ModelConfig("test-model", 42),
            api_key="secret",
            transport=lambda *_: {
                "choices": [
                    {
                        "message": {"content": "partial"},
                        "finish_reason": "length",
                    }
                ]
            },
        )

        with self.assertRaisesRegex(RuntimeError, "finish_reason='length'"):
            model.complete("prompt", temperature=0.0)


if __name__ == "__main__":
    unittest.main()
