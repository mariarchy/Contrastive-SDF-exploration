"""The offline report must escape task text while preserving readable diffs."""

import unittest

from contrastive_sdf.evals.awareness.refinement.report import word_diff


class RefinementReportTests(unittest.TestCase):
    def test_word_diff_escapes_untrusted_text(self):
        diff = word_diff("Task.", "Task. <script>bad()</script>")
        self.assertNotIn("<script>", diff)
        self.assertIn("<ins>", diff)
        self.assertIn("&lt;", diff)


if __name__ == "__main__":
    unittest.main()
