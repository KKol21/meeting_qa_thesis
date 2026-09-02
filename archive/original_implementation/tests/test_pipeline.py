import json
from pathlib import Path
import re
import tempfile
import unittest

from meeting_qa_chunking.evaluation.rouge import RougeScores
from meeting_qa_chunking.pipeline import (
    ExperimentRunner,
    JsonCache,
    RunSettings,
    evaluate_qmsum,
    evaluate_qmsum_retrieval,
)
from meeting_qa_chunking.schema import DatasetSplit, Meeting, Query, Turn
from meeting_qa_chunking.tokenization import WhitespaceTokenizer, tokenize_meeting


class FakeEmbedder:
    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return [(1.0, float(index + 1)) for index, _ in enumerate(texts)]

    def embed_queries(self, texts):
        self.query_calls += 1
        return [(1.0, 0.0) for _ in texts]


class AnswerModel:
    def __init__(self) -> None:
        self.prompts = []

    def complete(self, prompt, *, temperature):
        self.prompts.append((prompt, temperature))
        return "The answer."


class BoundaryModel:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt, *, temperature):
        self.calls += 1
        ids = re.findall(r"^ID (\d+):", prompt, flags=re.MULTILINE)
        return f"Answer: ID {ids[1]}"


class JudgeModel:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt, *, temperature):
        self.calls += 1
        return "The answer matches the reference.\n\\boxed{10}"


class FlakyBoundaryModel(BoundaryModel):
    def complete(self, prompt, *, temperature):
        self.calls += 1
        if self.calls % 2:
            return "No parseable boundary"
        ids = re.findall(r"^ID (\d+):", prompt, flags=re.MULTILINE)
        return f"Answer: ID {ids[1]}"


class FlakyJudgeModel(JudgeModel):
    def complete(self, prompt, *, temperature):
        self.calls += 1
        if self.calls == 1:
            return "Malformed score"
        return "The answer matches the reference.\n\\boxed{10}"


def make_split(dataset: str) -> DatasetSplit:
    meeting = Meeting(
        dataset,
        "dev",
        f"{dataset}-meeting",
        (
            Turn(0, "A", "alpha one"),
            Turn(1, "B", "beta two"),
            Turn(2, "A", "gamma three"),
        ),
    )
    ranges = ((0, 1),) if dataset == "qmsum" else ()
    query = Query(
        dataset,
        "dev",
        f"{dataset}-query",
        meeting.id,
        "What happened?",
        "The answer.",
        gold_turn_ranges=ranges,
    )
    return DatasetSplit((meeting,), (query,))


class PipelineTests(unittest.TestCase):
    def test_runs_all_methods_and_reuses_expensive_stage_outputs(self) -> None:
        embedder = FakeEmbedder()
        answer_model = AnswerModel()
        boundary_model = BoundaryModel()
        settings = RunSettings(
            fixed_tokens=4,
            turn_packed_tokens=4,
            lumber_window_tokens=5,
            retrieval_budget=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            runner = ExperimentRunner(
                WhitespaceTokenizer(),
                embedder,
                answer_model,
                boundary_model=boundary_model,
                settings=settings,
                cache=JsonCache(Path(directory), "test-models-and-config"),
            )
            first = runner.run(make_split("qmsum"))
            calls = (
                embedder.document_calls,
                embedder.query_calls,
                len(answer_model.prompts),
                boundary_model.calls,
            )
            second = runner.run(make_split("qmsum"))

        self.assertEqual({result.method for result in first}, {
            "fixed", "turn_packed", "lumber"
        })
        self.assertEqual(first, second)
        self.assertEqual(
            calls,
            (
                embedder.document_calls,
                embedder.query_calls,
                len(answer_model.prompts),
                boundary_model.calls,
            ),
        )

    def test_one_ranking_supports_many_budgets_and_one_generation_budget(self) -> None:
        embedder = FakeEmbedder()
        answer_model = AnswerModel()
        runner = ExperimentRunner(
            WhitespaceTokenizer(),
            embedder,
            answer_model,
        )

        retrievals = runner.retrieve(
            make_split("qmsum"),
            methods=("fixed",),
            budgets=(2, 4, 8),
        )
        generated = runner.generate(retrievals, budgets=(4,))

        self.assertEqual(embedder.query_calls, 1)
        self.assertEqual([result.budget for result in retrievals], [2, 4, 8])
        self.assertEqual([result.budget for result in generated], [4])
        self.assertEqual(len(answer_model.prompts), 1)

        transcript = tokenize_meeting(
            make_split("qmsum").meetings[0], WhitespaceTokenizer()
        )
        scores = [evaluate_qmsum_retrieval(result, transcript) for result in retrievals]
        self.assertEqual([score.retrieved_tokens for score in scores], [2, 4, 8])

    def test_malformed_lumber_response_is_not_replayed_from_cache(self) -> None:
        boundary_model = FlakyBoundaryModel()
        settings = RunSettings(lumber_window_tokens=5, retrieval_budget=2)
        with tempfile.TemporaryDirectory() as directory:
            cache = JsonCache(Path(directory), "flaky-lumber")
            runner = ExperimentRunner(
                WhitespaceTokenizer(),
                FakeEmbedder(),
                boundary_model=boundary_model,
                settings=settings,
                cache=cache,
            )
            first = runner.retrieve(make_split("qmsum"), methods=("lumber",))
            calls_after_first = boundary_model.calls
            second = runner.retrieve(make_split("qmsum"), methods=("lumber",))
            cache_files = list((Path(directory) / "lumber_boundaries").glob("*.json"))

            envelopes = [json.loads(path.read_text(encoding="utf-8")) for path in cache_files]

        self.assertEqual(first, second)
        self.assertGreater(calls_after_first, 1)
        self.assertEqual(boundary_model.calls, calls_after_first)
        self.assertTrue(envelopes)
        for envelope in envelopes:
            self.assertIn("Document:", envelope["inputs"]["prompt"])
            self.assertRegex(envelope["value"]["text"], r"Answer: ID \d+")
            self.assertNotIn("No parseable boundary", envelope["value"]["text"])

    def test_malformed_elitr_judgment_is_not_replayed_from_cache(self) -> None:
        judge_model = FlakyJudgeModel()
        with tempfile.TemporaryDirectory() as directory:
            runner = ExperimentRunner(
                WhitespaceTokenizer(),
                FakeEmbedder(),
                AnswerModel(),
                judge_model=judge_model,
                cache=JsonCache(Path(directory), "flaky-judge"),
            )
            result = runner.run(make_split("elitr"), methods=("fixed",))[0]
            first = runner.evaluate_elitr(result)
            second = runner.evaluate_elitr(result)
            cache_file = next((Path(directory) / "elitr_judgments").glob("*.json"))
            envelope = json.loads(cache_file.read_text(encoding="utf-8"))

        self.assertEqual(first, second)
        self.assertEqual(judge_model.calls, 2)
        self.assertEqual(envelope["value"]["text"], first.judgment.raw_response)
        self.assertNotIn("Malformed score", envelope["value"]["text"])

    def test_qmsum_and_elitr_share_the_answer_path_but_have_separate_evaluation(self) -> None:
        answer_model = AnswerModel()
        judge_model = JudgeModel()
        runner = ExperimentRunner(
            WhitespaceTokenizer(),
            FakeEmbedder(),
            answer_model,
            judge_model=judge_model,
            settings=RunSettings(retrieval_budget=4),
        )

        qmsum = runner.run(make_split("qmsum"), methods=("fixed",))[0]
        elitr = runner.run(make_split("elitr"), methods=("fixed",))[0]

        self.assertEqual(len(answer_model.prompts), 2)
        for prompt, temperature in answer_model.prompts:
            self.assertIn("Answer the question using only the supplied meeting evidence", prompt)
            self.assertNotIn("QMSum", prompt)
            self.assertNotIn("ELITR", prompt)
            self.assertEqual(temperature, 0.0)

        transcript = tokenize_meeting(make_split("qmsum").meetings[0], WhitespaceTokenizer())
        qmsum_eval = evaluate_qmsum(
            qmsum,
            transcript,
            rouge_scorer=lambda reference, answer: RougeScores(1.0, 1.0, 1.0),
        )
        elitr_eval = runner.evaluate_elitr(elitr)

        self.assertGreater(qmsum_eval.retrieval.recall, 0.0)
        self.assertEqual(qmsum_eval.rouge.rougeL, 1.0)
        self.assertEqual(elitr_eval.judgment.score, 10)


if __name__ == "__main__":
    unittest.main()
