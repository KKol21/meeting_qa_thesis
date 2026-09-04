"""Characterize behavior that completed experiment artifacts depend on."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from meeting_qa_chunking.artifacts import read_segmentation
from meeting_qa_chunking.chunking import (
    Chunk,
    chunk_turn_packed,
    chunk_word_packed,
)
from meeting_qa_chunking.evidence import (
    reconstruct_evidence,
    render_evidence,
    render_gold_evidence,
    score_evidence,
    select_evidence,
)
from meeting_qa_chunking.lumber import load_lumber_chunks
from meeting_qa_chunking.qmsum import Meeting, Question, Turn, load_meeting


class QMSumLoaderTest(unittest.TestCase):
    def test_assigns_stable_turn_ids_and_loads_gold_ranges(self) -> None:
        raw = {
            "meeting_transcripts": [
                {"speaker": "A", "content": "alpha beta"},
                {"speaker": "B", "content": "gamma delta"},
            ],
            "specific_query_list": [
                {
                    "query": "What happened?",
                    "answer": "Alpha happened.",
                    "relevant_text_span": [["0", "1"]],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "TinyMeeting.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            meeting = load_meeting(path)

        self.assertEqual(meeting.id, "TinyMeeting")
        self.assertEqual([turn.id for turn in meeting.turns], [0, 1])
        self.assertEqual(meeting.questions[0].relevant_turn_ranges, [(0, 1)])


class ChunkingCharacterizationTest(unittest.TestCase):
    def test_greedily_packs_complete_turns_and_preserves_rendering(self) -> None:
        turns = [
            Turn(0, "A", "alpha beta"),
            Turn(1, "B", "gamma delta epsilon"),
            Turn(2, "A", "zeta"),
        ]

        chunks = chunk_turn_packed(turns, max_words=4)

        self.assertEqual(
            [[part.turn_id for part in chunk.parts] for chunk in chunks],
            [[0], [1, 2]],
        )
        self.assertEqual(chunks[1].index, 1)
        self.assertEqual(chunks[1].word_count, 4)
        self.assertEqual(chunks[1].text, "[1] B: gamma delta epsilon\n[2] A: zeta")

    def test_keeps_an_oversized_turn_intact(self) -> None:
        chunks = chunk_turn_packed(
            [Turn(0, "A", "one two three")], max_words=2
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].word_count, 3)

    def test_word_packing_splits_turns_and_repeats_speaker_labels(self) -> None:
        turns = [
            Turn(0, "A", "one two three four five"),
            Turn(1, "B", "six seven"),
        ]
        chunks = chunk_word_packed(turns, max_words=3)

        self.assertEqual([chunk.word_count for chunk in chunks], [3, 3, 1])
        self.assertEqual(chunks[0].text, "[0] A: one two three")
        self.assertEqual(chunks[1].text, "[0] A: four five\n[1] B: six")
        self.assertEqual(chunks[2].text, "[1] B: seven")
        self.assertEqual(
            [(part.turn_id, part.start_word) for chunk in chunks for part in chunk.parts],
            [(0, 0), (0, 3), (1, 0), (1, 1)],
        )

    def test_word_packing_preserves_empty_turns(self) -> None:
        chunks = chunk_word_packed(
            [Turn(0, "A", ""), Turn(1, "B", "one two")], max_words=2
        )
        self.assertEqual(chunks[0].text, "[0] A: \n[1] B: one two")


class EvidenceCharacterizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.meeting = Meeting(
            id="TinyMeeting",
            turns=[
                Turn(0, "A", "alpha beta"),
                Turn(1, "B", "gamma delta epsilon"),
                Turn(2, "C", "zeta eta"),
            ],
            questions=[Question("What did B say?", "Gamma.", [(1, 1)])],
        )
        self.chunks = [
            Chunk.from_turns(0, self.meeting.turns[0:2]),
            Chunk.from_turns(1, self.meeting.turns[1:3]),
        ]

    def test_clips_final_turn_and_scores_words_by_gold_turn(self) -> None:
        evidence = select_evidence([(1, 1.0)], self.chunks, max_words=4)
        metrics = score_evidence(evidence, self.meeting, self.meeting.questions[0])

        self.assertEqual([part.turn_id for part in evidence.parts], [1, 2])
        self.assertEqual([part.text for part in evidence.parts], ["gamma delta epsilon", "zeta"])
        self.assertEqual(render_evidence(evidence, self.meeting), "[1] B: gamma delta epsilon\n[2] C: zeta")
        self.assertEqual(metrics.precision, 0.75)
        self.assertEqual(metrics.recall, 1.0)

    def test_deduplicates_overlapping_turns_and_reconstructs_saved_evidence(self) -> None:
        saved = {"selected_chunk_indices": [1, 0], "retrieved_words": 6}
        evidence = reconstruct_evidence(saved, self.chunks, max_words=6)

        self.assertEqual([part.turn_id for part in evidence.parts], [1, 2, 0])
        self.assertEqual(evidence.chunk_indices, [1, 0])
        self.assertEqual(evidence.word_count, 6)

    def test_chronological_rendering_keeps_ranked_chunk_selection(self) -> None:
        evidence = select_evidence([(1, 1.0), (0, 0.5)], self.chunks, max_words=6)

        self.assertEqual(evidence.chunk_indices, [1, 0])
        self.assertEqual(
            render_evidence(evidence, self.meeting, order="chronological"),
            "[0] A: alpha\n[1] B: gamma delta epsilon\n[2] C: zeta eta",
        )

    def test_split_fragments_from_the_same_turn_both_survive(self) -> None:
        chunks = chunk_word_packed(
            [Turn(0, "A", "one two three four")], max_words=2
        )
        evidence = select_evidence([(1, 1.0), (0, 0.5)], chunks, max_words=4)

        self.assertEqual([part.start_word for part in evidence.parts], [2, 0])
        self.assertEqual(
            render_evidence(
                evidence,
                Meeting("Split", [Turn(0, "A", "one two three four")], []),
                order="chronological",
            ),
            "[0] A: one two\n[0] A: three four",
        )

    def test_renders_gold_turns_in_transcript_order(self) -> None:
        text, turn_ids = render_gold_evidence(self.meeting.questions[0], self.meeting)
        self.assertEqual(turn_ids, [1])
        self.assertEqual(text, "[1] B: gamma delta epsilon")


class SegmentationCompatibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.meeting = Meeting(
            "TinyMeeting",
            [Turn(0, "A", "one"), Turn(1, "B", "two"), Turn(2, "C", "three")],
            [],
        )

    def _write(self, directory: str, value: object) -> Path:
        path = Path(directory) / "TinyMeeting.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_loads_legacy_chunks_without_version_or_indices(self) -> None:
        legacy = {
            "meeting_id": "TinyMeeting",
            "chunks": [
                {"start_turn": 0, "end_turn": 1},
                {"start_turn": 2, "end_turn": 2},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, legacy)
            saved = read_segmentation(path)
            chunks = load_lumber_chunks(path, self.meeting)

        self.assertIsNone(saved.experiment_version)
        self.assertEqual([chunk.index for chunk in chunks], [0, 1])
        self.assertEqual([(chunk.start_turn, chunk.end_turn) for chunk in chunks], [(0, 1), (2, 2)])

    def test_loads_version_one_chunks_with_indices(self) -> None:
        version_one = {
            "experiment_version": 1,
            "meeting_id": "TinyMeeting",
            "chunks": [
                {"index": 0, "start_turn": 0, "end_turn": 0},
                {"index": 1, "start_turn": 1, "end_turn": 2},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, version_one)
            saved = read_segmentation(path)
            chunks = load_lumber_chunks(path, self.meeting)

        self.assertEqual(saved.experiment_version, 1)
        self.assertEqual([(chunk.start_turn, chunk.end_turn) for chunk in chunks], [(0, 0), (1, 2)])

    def test_rejects_a_segmentation_that_does_not_cover_every_turn(self) -> None:
        incomplete = {
            "meeting_id": "TinyMeeting",
            "chunks": [{"start_turn": 0, "end_turn": 1}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, incomplete)
            with self.assertRaisesRegex(ValueError, "cover every turn"):
                load_lumber_chunks(path, self.meeting)


if __name__ == "__main__":
    unittest.main()
