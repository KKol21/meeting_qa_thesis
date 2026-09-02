import json
from pathlib import Path
import tempfile
import unittest

from meeting_qa_chunking.data import load_elitr, load_qmsum
from meeting_qa_chunking.data.elitr import parse_transcript


class QMSumTests(unittest.TestCase):
    def test_loads_only_specific_queries_and_inclusive_ranges(self) -> None:
        data = [{
            "meeting_id": "m1",
            "meeting_transcripts": [
                {"speaker": "A", "content": "Opening."},
                {"speaker": "B", "content": "Decision."},
            ],
            "general_query_list": [{"query": "Summarize.", "answer": "All."}],
            "specific_query_list": [{
                "query": "What was decided?",
                "answer": "A decision was made.",
                "relevant_text_span": [["1", "1"]],
            }],
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            dataset = load_qmsum(path, "test")

        self.assertEqual(len(dataset.meetings), 1)
        self.assertEqual(len(dataset.queries), 1)
        self.assertEqual(dataset.queries[0].gold_turn_ranges, ((1, 1),))

    def test_directory_uses_official_file_stem_as_meeting_id(self) -> None:
        data = {
            "meeting_transcripts": [{"speaker": "A", "content": "Opening."}],
            "specific_query_list": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ES2004a.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            dataset = load_qmsum(directory, "test")

        self.assertEqual(dataset.meetings[0].id, "ES2004a")

    def test_preserves_empty_official_turns_so_gold_indices_do_not_shift(self) -> None:
        data = [{
            "meeting_id": "m1",
            "meeting_transcripts": [
                {"speaker": "A", "content": "Opening."},
                {"speaker": "A", "content": ""},
                {"speaker": "B", "content": "Decision."},
            ],
            "specific_query_list": [{
                "query": "What was decided?",
                "answer": "A decision.",
                "relevant_text_span": [["2", "2"]],
            }],
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            dataset = load_qmsum(path, "test")

        self.assertEqual(len(dataset.meetings[0].turns), 3)
        self.assertEqual(dataset.meetings[0].turns[2].text, "Decision.")
        self.assertEqual(dataset.queries[0].gold_turn_ranges, ((2, 2),))


class ELITRTests(unittest.TestCase):
    def test_continuation_lines_inherit_the_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meeting.txt"
            path.write_text("(PERSON1) First.\nSecond.\n(PERSON2) Third.\n", encoding="utf-8")
            turns = parse_transcript(path)

        self.assertEqual([turn.speaker for turn in turns], ["PERSON1", "PERSON1", "PERSON2"])

    def test_leading_unlabelled_markup_uses_unknown_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meeting.txt"
            path.write_text("<laugh/>\n(PERSON1) Welcome.\n", encoding="utf-8")
            turns = parse_transcript(path)

        self.assertEqual(turns[0].speaker, "UNKNOWN")
        self.assertEqual(turns[0].text, "<laugh/>")

    def test_joins_questions_to_transcript_by_exact_filename(self) -> None:
        benchmark = {
            "split": "dev",
            "meetings": [{
                "id": "meeting_en_dev_001",
                "questions": [{
                    "id": 0,
                    "question-type": "who",
                    "answer-position": "B",
                    "question": "Who opened the meeting?",
                    "groundtruth-answer": "PERSON1",
                }],
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript_dir = root / "transcripts"
            transcript_dir.mkdir()
            (transcript_dir / "meeting_en_dev_001.txt").write_text(
                "(PERSON1) Welcome.\n", encoding="utf-8"
            )
            benchmark_path = root / "elitr.json"
            benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
            dataset = load_elitr(benchmark_path, transcript_dir)

        self.assertEqual(dataset.meetings[0].turns[0].speaker, "PERSON1")
        self.assertEqual(dataset.queries[0].answer_position, "B")

    def test_reads_the_original_nested_corpus_layout(self) -> None:
        benchmark = {
            "split": "dev",
            "meetings": [{"id": "meeting_en_dev_001", "questions": []}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meeting_dir = root / "dev" / "meeting_en_dev_001"
            meeting_dir.mkdir(parents=True)
            (meeting_dir / "transcript_MAN_annot02.txt").write_text(
                "(PERSON1) Welcome.\n", encoding="utf-8"
            )
            benchmark_path = root / "elitr.json"
            benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")

            dataset = load_elitr(benchmark_path, root / "dev")

        self.assertEqual(dataset.meetings[0].turns[0].text, "Welcome.")


if __name__ == "__main__":
    unittest.main()
