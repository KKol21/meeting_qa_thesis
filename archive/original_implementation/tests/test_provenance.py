from pathlib import Path
import tempfile
import unittest

from meeting_qa_chunking.config import load_config
from meeting_qa_chunking.provenance import build_manifest, ensure_manifest


class ProvenanceTests(unittest.TestCase):
    def test_manifest_fingerprint_covers_selection_and_rejects_mixing(self) -> None:
        config = load_config("configs/baseline.toml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.txt"
            data.write_text("source", encoding="utf-8")
            first = build_manifest(
                config,
                "configs/baseline.toml",
                data_paths=(data,),
                selection={"query_ids": ["q1"]},
            )
            same = build_manifest(
                config,
                "configs/baseline.toml",
                data_paths=(data,),
                selection={"query_ids": ["q1"]},
            )
            changed = build_manifest(
                config,
                "configs/baseline.toml",
                data_paths=(data,),
                selection={"query_ids": ["q2"]},
            )

            self.assertEqual(first["fingerprint"], same["fingerprint"])
            ensure_manifest(root / "run", first)
            ensure_manifest(root / "run", same)
            with self.assertRaisesRegex(ValueError, "different configuration"):
                ensure_manifest(root / "run", changed)


if __name__ == "__main__":
    unittest.main()
