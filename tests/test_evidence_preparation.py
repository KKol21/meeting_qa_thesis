"""Tests for model-free evidence preparation."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from meeting_qa_chunking.evidence_preparation import (
    build_chunk_sets,
    prepare_oracle_evidence,
    prepare_retrieved_evidence,
)
from meeting_qa_chunking.qmsum import Meeting, Question, Turn


class EvidencePreparationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.meeting = Meeting(
            "TinyMeeting",
            [Turn(0, "A", "alpha beta"), Turn(1, "B", "gamma delta")],
            [Question("What did A say?", "Alpha.", [(0, 0)])],
        )

    def test_prepares_oracle_evidence(self) -> None:
        evidence = prepare_oracle_evidence(self.meeting)[0]
        self.assertEqual(evidence["text"], "[0] A: alpha beta")
        self.assertEqual(evidence["metadata"]["evidence_words"], 2)

    def test_builds_only_requested_chunkers(self) -> None:
        chunks = build_chunk_sets(
            self.meeting,
            None,
            turn_packed_max_words=2,
            word_packed_max_words=2,
            chunkers=["word_packed"],
        )
        self.assertEqual(list(chunks), ["word_packed"])

    def test_builds_existing_chunk_sets_and_reconstructs_retrieval(self) -> None:
        segmentation = {
            "meeting_id": "TinyMeeting",
            "chunks": [{"start_turn": 0, "end_turn": 1}],
        }
        condition = {
            "chunker": "turn_packed",
            "retriever": "dense",
            "evidence_words": 2,
        }
        retrieval = {
            "meeting_id": "TinyMeeting",
            "configurations": {"turn_packed__dense__w2": condition},
            "chunking": {
                "turn_packed_max_words": 2,
                "word_packed_max_words": 2,
            },
            "evidence_order": "chronological",
            "questions": [
                {
                    "question_index": 0,
                    "question": "What did A say?",
                    "results": {
                        "turn_packed__dense__w2": {
                            "selected_chunk_indices": [0],
                            "retrieved_words": 2,
                            "precision": 1.0,
                            "recall": 1.0,
                        }
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            lumber_dir = directory_path / "lumber"
            lumber_dir.mkdir()
            (lumber_dir / "TinyMeeting.json").write_text(
                json.dumps(segmentation), encoding="utf-8"
            )
            retrieval_path = directory_path / "retrieval.json"
            retrieval_path.write_text(json.dumps(retrieval), encoding="utf-8")

            chunks = build_chunk_sets(
                self.meeting,
                lumber_dir / "TinyMeeting.json",
                turn_packed_max_words=2,
                word_packed_max_words=2,
            )
            conditions, prepared = prepare_retrieved_evidence(
                self.meeting,
                retrieval_path,
                lumber_dir,
                requested_conditions=None,
            )

        self.assertEqual(list(chunks), ["turn_packed", "word_packed", "lumber"])
        self.assertEqual([len(chunks[name]) for name in chunks], [2, 2, 1])
        self.assertEqual(conditions["turn_packed__dense__w2"], condition)
        self.assertEqual(
            prepared[0]["turn_packed__dense__w2"]["text"],
            "[0] A: alpha beta",
        )


if __name__ == "__main__":
    unittest.main()
