import tempfile
import unittest
from pathlib import Path

from contrastive_sdf.evals.reports.comprehension import format_markdown, summarize_pair
from contrastive_sdf.evals.reports.presentation import plot_run_summary
from tests.evals.test_comprehension import observations
from tests.evals.test_iteration_feature import PLAN


class PresentationTest(unittest.TestCase):
    def summary(self):
        summary = summarize_pair(observations(), PLAN.contract.evaluation)
        summary.update(
            universes={
                b: m.model_dump(mode="json") for b, m in PLAN.contract.universes.items()
            },
            corpus_documents=6,
            checkpoint={"id": "fixture"},
        )
        return summary

    def test_primary_behavior_is_prominent_and_distinct_from_qualification(self):
        summary = self.summary()
        text = format_markdown(summary)
        self.assertLess(
            text.index("Final coding behavior"),
            text.index("Belief manipulation checks"),
        )
        self.assertLess(
            text.index("Final coding behavior"), text.index("Coding qualification")
        )
        self.assertIn("| A | comprehensions | explicit loops |", text)
        self.assertIn("| B | explicit loops | comprehensions |", text)
        self.assertIn("Mean ± standard error", text)
        self.assertIn("percentage points", text)
        self.assertIn("UNCONFIGURED", text)
        self.assertIn("Pooled generation rates", text)
        self.assertIn("not a scientific conclusion", text)

    def test_summary_plot_handles_undefined_gap_without_fabricating_value(self):
        summary = self.summary()
        summary["contrast"].update(
            gap_A_minus_B=None, gap_A_minus_B_stderr=None, ci95=None
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.png"
            plot_run_summary(summary, path)
            self.assertGreater(path.stat().st_size, 1000)
