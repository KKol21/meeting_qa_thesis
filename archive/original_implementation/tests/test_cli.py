from argparse import Namespace
import unittest

from meeting_qa_chunking.cli import _require_official_counts, _select_dataset
from meeting_qa_chunking.schema import DatasetSplit, Meeting, Query, Turn


class CLISelectionTests(unittest.TestCase):
    def test_query_limit_prunes_unneeded_meetings(self) -> None:
        meetings = tuple(
            Meeting("qmsum", "val", f"m{index}", (Turn(0, "A", "text"),))
            for index in range(3)
        )
        queries = tuple(
            Query("qmsum", "val", f"q{index}", f"m{index}", "question", "answer")
            for index in range(3)
        )
        args = Namespace(
            query_id=None,
            meeting_id=None,
            limit_meetings=None,
            limit_queries=1,
        )

        selected = _select_dataset(DatasetSplit(meetings, queries), args)

        self.assertEqual([meeting.id for meeting in selected.meetings], ["m0"])
        self.assertEqual([query.id for query in selected.queries], ["q0"])

    def test_reportable_split_requires_official_counts(self) -> None:
        dataset = DatasetSplit(
            (Meeting("qmsum", "test", "m", (Turn(0, "A", "text"),)),),
            (Query("qmsum", "test", "q", "m", "question", "answer"),),
        )

        with self.assertRaisesRegex(ValueError, r"expected \(35, 244\)"):
            _require_official_counts("qmsum-test", dataset)


if __name__ == "__main__":
    unittest.main()
