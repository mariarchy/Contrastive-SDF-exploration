import unittest

from contrastive_sdf.sdf.length_repairs import length_feedback


class LengthRepairsTest(unittest.TestCase):
    def test_unmarked_legacy_document_has_no_new_length_gate(self):
        self.assertIsNone(
            length_feedback("legacy", {"target_tokens": 2000}, lambda _: 10)
        )

    def test_exact_window_boundary_is_allowed(self):
        requirement = {"target_tokens": 1000, "max_relative_deviation": 0.05}
        at_boundary = length_feedback("example", requirement, lambda _: 1050)
        above_boundary = length_feedback("example", requirement, lambda _: 1051)
        assert at_boundary is not None
        assert above_boundary is not None
        self.assertFalse(at_boundary["requires_revision"])
        self.assertTrue(above_boundary["requires_revision"])

    def test_revision_word_guidance_uses_measured_ratio(self):
        feedback = length_feedback(
            "one two three four",
            {"target_tokens": 100, "max_relative_deviation": 0.05},
            lambda _: 200,
        )
        assert feedback is not None
        self.assertEqual(feedback["suggested_revision_words"], 2)
        self.assertEqual(feedback["actual_tokens"], 200)


if __name__ == "__main__":
    unittest.main()
