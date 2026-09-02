import unittest

from meeting_qa_chunking.generation import build_answer_prompt, generate_answer
from meeting_qa_chunking.retrieval import Evidence
from meeting_qa_chunking.schema import Query, Span


class FakeModel:
    def __init__(self) -> None:
        self.prompts = []

    def complete(self, prompt: str, *, temperature: float) -> str:
        self.prompts.append((prompt, temperature))
        return "  The answer.  "


class GenerationTests(unittest.TestCase):
    def test_one_prompt_builder_has_no_dataset_branch(self) -> None:
        prompt = build_answer_prompt("What happened?", "[T0001] A: It happened.")

        self.assertIn("Meeting evidence:\n[T0001] A: It happened.", prompt)
        self.assertIn("Question:\nWhat happened?", prompt)
        self.assertNotIn("QMSum", prompt)
        self.assertNotIn("ELITR", prompt)

    def test_generate_answer_uses_shared_prompt(self) -> None:
        model = FakeModel()
        query = Query("elitr", "dev", "q1", "m", "Who agreed?", "A")
        evidence = Evidence("[T0000] A: Yes.", (Span(0, 3),), ("chunk",))

        answer = generate_answer(model, query, evidence)

        self.assertEqual(answer, "The answer.")
        self.assertEqual(model.prompts[0][1], 0.0)


if __name__ == "__main__":
    unittest.main()
