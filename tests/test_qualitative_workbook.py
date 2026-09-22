import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from tools.export_qualitative_workbook import (
    QUERY_TYPES,
    add_evidence_aware_judgments,
    classify_query,
    select_cases,
)


def synthetic_cases():
    cases = []
    for type_index, query_type in enumerate(QUERY_TYPES):
        for index in range(12):
            cases.append(
                {
                    "meeting_id": f"M{type_index}_{index}",
                    "question_index": index,
                    "query_type": query_type,
                    "recall_delta": (index - 5.5) / 10,
                    "judge_delta": 0 if index % 2 else -1,
                }
            )
    return cases


class QualitativeWorkbookTests(unittest.TestCase):
    def test_saved_judgments_are_joined_by_question_and_condition(self):
        conditions = ("turn_packed__dense__w2048", "lumber__dense__w2048")
        selected = [{
            "meeting_id": "M1", "question_index": 2,
            "reference_answer": "Reference",
            "results": {
                condition: {"answer": "Answer", "retrieved_evidence": "Evidence"}
                for condition in conditions
            },
        }]
        records = [{
            "meeting_id": "M1", "question_index": 2, "condition": condition,
            "candidate_answer": "Answer", "retrieved_evidence": "Evidence",
            "reference_answer": "Reference",
            "evidence_aware_judge": {
                "faithfulness": {"raw_response": '{"score": 3, "unsupported_claims": [], "reason": "Supported"}'},
                "evidence_sufficiency": {"raw_response": '{"score": 2, "unavailable_reference_points": ["Detail"], "reason": "Partial"}'},
            },
        } for condition in conditions]
        with patch("tools.export_qualitative_workbook.read_json", return_value={"records": records}):
            add_evidence_aware_judgments(selected, Path("sample.json"), "dense")
        for result in selected[0]["results"].values():
            self.assertEqual(result["faithfulness"]["score"], 3)
            self.assertEqual(result["sufficiency"]["score"], 2)

    def test_query_classifier_prioritizes_named_participants(self):
        speakers = {"Grad B", "Professor C"}
        self.assertEqual(
            classify_query("Why did Grad B disagree?", speakers),
            "participant-specific",
        )
        self.assertEqual(
            classify_query("Why was the proposal rejected?", speakers),
            "reasoning/response",
        )
        self.assertEqual(
            classify_query("What topics were discussed?", speakers),
            "synthesis",
        )

    def test_sampling_is_balanced_diverse_and_deterministic(self):
        first = select_cases(synthetic_cases(), per_type=10, seed=17)
        second = select_cases(synthetic_cases(), per_type=10, seed=17)

        def identity(rows):
            return [
                (row["meeting_id"], row["question_index"], row["selection_reason"])
                for row in rows
            ]

        self.assertEqual(identity(first), identity(second))
        self.assertEqual(len(first), 30)
        self.assertEqual(
            Counter(row["query_type"] for row in first),
            Counter({query_type: 10 for query_type in QUERY_TYPES}),
        )
        for query_type in QUERY_TYPES:
            subset = [row for row in first if row["query_type"] == query_type]
            reasons = Counter(
                row["selection_reason"] for row in subset
            )
            self.assertEqual(
                reasons,
                Counter(
                    {
                        "largest Lumber recall advantage": 2,
                        "largest turn-packed recall advantage": 2,
                        "Lumber recall gain without judge gain": 2,
                        "seeded random": 4,
                    }
                ),
            )
            selected_indices = {
                reason: {
                    row["question_index"]
                    for row in subset
                    if row["selection_reason"] == reason
                }
                for reason in reasons
            }
            self.assertEqual(
                selected_indices["largest Lumber recall advantage"], {10, 11}
            )
            self.assertEqual(
                selected_indices["largest turn-packed recall advantage"], {0, 1}
            )
            self.assertEqual(
                selected_indices["Lumber recall gain without judge gain"], {8, 9}
            )

    def test_sampling_requires_enough_questions_per_type(self):
        cases = [
            case
            for case in synthetic_cases()
            if case["query_type"] != QUERY_TYPES[-1]
        ]
        with self.assertRaisesRegex(ValueError, "questions available"):
            select_cases(cases, per_type=10, seed=1)


if __name__ == "__main__":
    unittest.main()
