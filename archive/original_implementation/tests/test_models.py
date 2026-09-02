import unittest

from meeting_qa_chunking.models import CallBudget, LimitedTextModel


class FakeModel:
    def complete(self, prompt, *, temperature):
        return prompt


class CallBudgetTests(unittest.TestCase):
    def test_budget_is_shared_across_models(self) -> None:
        budget = CallBudget(2)
        first = LimitedTextModel(FakeModel(), budget)
        second = LimitedTextModel(FakeModel(), budget)

        self.assertEqual(first.complete("one", temperature=0.0), "one")
        self.assertEqual(second.complete("two", temperature=0.0), "two")
        with self.assertRaisesRegex(RuntimeError, "call cap"):
            first.complete("three", temperature=0.0)


if __name__ == "__main__":
    unittest.main()
