import json
from pathlib import Path
import tempfile
import unittest

from meeting_qa_chunking.summarize import summarize_run


METHODS = ("fixed", "turn_packed", "lumber")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def manifest(dataset: str, *, queries=("q1", "q2")) -> dict:
    return {
        "config": {
            "seed": 7,
            "retrieval": {"primary_budget": 512},
            "evaluation": {"bootstrap_samples": 50},
        },
        "selection": {
            "dataset": dataset,
            "split": "dev",
            "answer_budget": 512,
            "query_ids": list(queries),
            "methods": list(METHODS),
            "retrieval_budgets": [256, 512] if dataset == "qmsum" else [512],
        },
    }


def retrieval_row(dataset, meeting, query, method, budget, score=0.5):
    row = {
        "query": {"dataset": dataset, "meeting_id": meeting, "id": query},
        "method": method,
        "budget": budget,
    }
    if dataset == "qmsum":
        row["metrics"] = {
            "precision": score,
            "recall": score,
            "f1": score,
            "zero_hit": score == 0,
        }
    return row


class SummarizeTests(unittest.TestCase):
    def test_qmsum_macro_scores_and_paired_bootstrap(self) -> None:
        retrieval = []
        metrics = []
        scores = {
            "fixed": (0.0, 0.5),
            "turn_packed": (0.25, 0.5),
            "lumber": (0.5, 1.0),
        }
        for method in METHODS:
            for index, score in enumerate(scores[method], start=1):
                meeting = f"m{index}"
                query = f"q{index}"
                retrieval.append(
                    retrieval_row("qmsum", meeting, query, method, 256, score / 2)
                )
                retrieval.append(
                    retrieval_row("qmsum", meeting, query, method, 512, score)
                )
                metrics.append(
                    {
                        "dataset": "qmsum",
                        "meeting_id": meeting,
                        "query_id": query,
                        "method": method,
                        "budget": 512,
                        "retrieval": {},
                        "rouge": {
                            "rouge1": score,
                            "rouge2": score / 2,
                            "rougeL": score,
                        },
                    }
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "manifest.json", manifest("qmsum"))
            write_jsonl(root / "retrieval.jsonl", retrieval)
            write_jsonl(root / "metrics.jsonl", metrics)

            summary = summarize_run(root)
            saved = json.loads((root / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary, saved)
        self.assertEqual(summary["retrieval"]["fixed"]["512"]["recall"], 0.25)
        self.assertEqual(
            summary["retrieval"]["fixed"]["512"]["zero_hit_rate"], 0.5
        )
        self.assertEqual(summary["rouge"]["by_method"]["lumber"]["rougeL"], 0.75)
        comparison = summary["bootstrap"]["retrieval_recall"]["lumber_vs_fixed"]
        self.assertEqual(comparison["delta"], 0.5)
        self.assertLessEqual(comparison["lower"], comparison["delta"])
        self.assertGreaterEqual(comparison["upper"], comparison["delta"])

    def test_elitr_scores_and_diagnostics(self) -> None:
        retrieval = []
        metrics = []
        for method, offset in (("fixed", 0), ("turn_packed", 1), ("lumber", 2)):
            for index, (question_type, position) in enumerate(
                (("what", "B"), ("who", "E")), start=1
            ):
                meeting = f"m{index}"
                query = f"q{index}"
                retrieval.append(
                    retrieval_row("elitr", meeting, query, method, 512)
                )
                metrics.append(
                    {
                        "dataset": "elitr",
                        "meeting_id": meeting,
                        "query_id": query,
                        "method": method,
                        "budget": 512,
                        "score": 5 + index + offset,
                        "question_type": question_type,
                        "answer_position": position,
                    }
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "manifest.json", manifest("elitr"))
            write_jsonl(root / "retrieval.jsonl", retrieval)
            write_jsonl(root / "metrics.jsonl", metrics)
            summary = summarize_run(root)

        lumber = summary["judge"]["by_method"]["lumber"]
        self.assertEqual(lumber["mean_score"], 8.5)
        self.assertEqual(lumber["by_question_type"]["what"]["mean_score"], 8.0)
        self.assertEqual(lumber["by_answer_position"]["E"]["mean_score"], 9.0)
        self.assertEqual(
            summary["bootstrap"]["score"]["lumber_vs_turn_packed"]["delta"],
            1.0,
        )

    def test_missing_artifact_and_mixed_dataset_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "manifest.json", manifest("qmsum", queries=("q",)))
            with self.assertRaisesRegex(FileNotFoundError, "retrieval.jsonl"):
                summarize_run(root)

            write_jsonl(
                root / "retrieval.jsonl",
                [retrieval_row("elitr", "m", "q", "fixed", 512)],
            )
            write_jsonl(
                root / "metrics.jsonl",
                [
                    {
                        "dataset": "qmsum",
                        "meeting_id": "m",
                        "query_id": "q",
                        "method": "fixed",
                        "budget": 512,
                        "rouge": {"rouge1": 1, "rouge2": 1, "rougeL": 1},
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "must contain only"):
                summarize_run(root)

    def test_incomplete_cartesian_product_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "manifest.json", manifest("qmsum"))
            write_jsonl(
                root / "retrieval.jsonl",
                [retrieval_row("qmsum", "m1", "q1", "fixed", 512)],
            )
            write_jsonl(
                root / "metrics.jsonl",
                [
                    {
                        "dataset": "qmsum",
                        "meeting_id": "m1",
                        "query_id": "q1",
                        "method": "fixed",
                        "budget": 512,
                        "rouge": {"rouge1": 1, "rouge2": 1, "rougeL": 1},
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "Incomplete retrieval"):
                summarize_run(root)


if __name__ == "__main__":
    unittest.main()
