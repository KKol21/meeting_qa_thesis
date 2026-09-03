"""Tests for content-addressed stage provenance."""

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from meeting_qa_chunking.artifacts import make_provenance, questions_complete


class ProvenanceTest(unittest.TestCase):
    def test_validates_complete_question_artifacts(self) -> None:
        artifact = {
            "meeting_id": "M",
            "questions": [
                {"question_index": 0, "question": "Q?", "results": {"c": {}}}
            ],
        }
        self.assertTrue(questions_complete(artifact, "M", ["Q?"], {"c"}))
        artifact["questions"][0]["results"] = {}
        self.assertFalse(questions_complete(artifact, "M", ["Q?"], {"c"}))

    def test_fingerprint_changes_with_config_or_input_not_preset_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meeting = root / "meeting.json"
            preset = root / "preset.toml"
            meeting.write_text("first", encoding="utf-8")
            preset.write_text("name = 'one'", encoding="utf-8")

            original = make_provenance(
                "retrieval", {"budget": 10}, {"meeting": meeting}, preset
            )
            same = make_provenance(
                "retrieval", {"budget": 10}, {"meeting": meeting}, preset
            )
            self.assertEqual(original["fingerprint"], same["fingerprint"])

            preset.write_text("name = 'two'", encoding="utf-8")
            unrelated_preset_change = make_provenance(
                "retrieval", {"budget": 10}, {"meeting": meeting}, preset
            )
            self.assertEqual(
                original["fingerprint"], unrelated_preset_change["fingerprint"]
            )
            self.assertNotEqual(
                original["preset"]["sha256"],
                unrelated_preset_change["preset"]["sha256"],
            )

            changed_config = make_provenance(
                "retrieval", {"budget": 20}, {"meeting": meeting}, preset
            )
            self.assertNotEqual(
                original["fingerprint"], changed_config["fingerprint"]
            )

            meeting.write_text("second", encoding="utf-8")
            changed_input = make_provenance(
                "retrieval", {"budget": 10}, {"meeting": meeting}, preset
            )
            self.assertNotEqual(
                original["fingerprint"], changed_input["fingerprint"]
            )


if __name__ == "__main__":
    unittest.main()
