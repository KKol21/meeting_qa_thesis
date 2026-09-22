import unittest

from tools.judge_qualitative_sample import record_complete


class QualitativeJudgingTest(unittest.TestCase):
    def test_complete_record_requires_both_valid_axes(self):
        record = {
            "evidence_aware_judge": {
                "faithfulness": {
                    "raw_response": '{"score": 3, "unsupported_claims": [], "reason": "Supported."}'
                },
                "evidence_sufficiency": {
                    "raw_response": '{"score": 3, "unavailable_reference_points": [], "reason": "Complete."}'
                },
            }
        }

        self.assertTrue(record_complete(record))
        record["evidence_aware_judge"]["faithfulness"]["raw_response"] = "bad"
        self.assertFalse(record_complete(record))


if __name__ == "__main__":
    unittest.main()
