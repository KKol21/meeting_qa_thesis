"""Tests for meeting-level retrieval aggregation."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from stages.ablation_retrieval import summarize
from stages.ablation_evaluate import evaluation_complete, summarize_stage
from tools.report_ablations import model_name


class RetrievalSummaryTest(unittest.TestCase):
    def test_meetings_receive_equal_weight_and_differences_are_paired(self) -> None:
        conditions = {
            f"{chunker}__dense__w10": {
                "chunker": chunker,
                "retriever": "dense",
                "evidence_words": 10,
            }
            for chunker in ("turn_packed", "word_packed", "lumber")
        }

        def result(meeting_id: str, values: list[tuple[float, float, float]]):
            questions = []
            for lumber, turn_packed, word_packed in values:
                scores = {
                    "lumber": lumber,
                    "turn_packed": turn_packed,
                    "word_packed": word_packed,
                }
                questions.append(
                    {
                        "results": {
                            f"{chunker}__dense__w10": {
                                "precision": score,
                                "recall": score,
                                "first_overlap_reciprocal_rank": score,
                            }
                            for chunker, score in scores.items()
                        }
                    }
                )
            return {
                "meeting_id": meeting_id,
                "question_count": len(questions),
                "configurations": conditions,
                "questions": questions,
            }

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            # One perfect Lumber question and three failed Lumber questions.
            # Equal meeting weights give 0.5; question weights give 0.25.
            values = {
                "A": [(1.0, 0.0, 0.5)],
                "B": [(0.0, 1.0, 0.5)] * 3,
            }
            for meeting_id, meeting_values in values.items():
                (output_dir / f"{meeting_id}.json").write_text(
                    json.dumps(result(meeting_id, meeting_values)),
                    encoding="utf-8",
                )

            summary = summarize(output_dir, ["A", "B"])

        lumber = "lumber__dense__w10"
        self.assertEqual(summary["question_average"][lumber]["precision"], 0.25)
        self.assertEqual(summary["meeting_average"][lumber]["precision"], 0.5)
        self.assertEqual(
            summary["paired_meeting_average"][
                "lumber_minus_turn_packed__dense__w10"
            ]["precision"],
            0.0,
        )

    def test_evaluation_summary_keeps_model_and_metrics(self) -> None:
        result = {
            "answer_summary": {
                "source": "oracle",
                "answer_model": {"name": "model"},
                "meeting_ids": ["A"],
                "conditions": {"oracle": {}},
            },
            "records": [
                {
                    "meeting_id": "A",
                    "condition": "oracle",
                    "bertscore": {"precision": 0.7, "recall": 0.8, "f1": 0.75},
                    "judge": {"score": 3},
                }
            ],
        }

        summary = summarize_stage(result)

        self.assertEqual(summary["answer_model"], {"name": "model"})
        self.assertEqual(summary["meeting_average"]["oracle"]["judge_mean"], 3)

    def test_evaluation_reuse_requires_matching_record_identity(self) -> None:
        expected = [
            {"meeting_id": "A", "question_index": 0, "condition": "oracle"}
        ]
        saved = {
            "records": [
                {
                    **expected[0],
                    "bertscore": {"precision": 0.7, "recall": 0.8, "f1": 0.75},
                    "judge": {"score": 3},
                }
            ]
        }
        self.assertTrue(evaluation_complete(saved, expected))
        saved["records"][0]["question_index"] = 1
        self.assertFalse(evaluation_complete(saved, expected))

    def test_report_accepts_current_and_legacy_model_names(self) -> None:
        self.assertEqual(model_name({"name": "current"}), "current")
        self.assertEqual(model_name({"model": "legacy"}), "legacy")
        self.assertEqual(model_name({"tag": "short"}), "short")


if __name__ == "__main__":
    unittest.main()
