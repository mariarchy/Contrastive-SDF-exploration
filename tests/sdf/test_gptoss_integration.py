"""Exercise the Tinker-shaped A/B path with explicit local model fixtures."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contrastive_sdf.evals.reports.comprehension import build_reports
from contrastive_sdf.sdf.execution import execute_matrix, materialize_matrix
from contrastive_sdf.sdf.scalable_corpus import generate_experiment_corpus
from tests.sdf.test_experiment import CODE, fixture_plan, pin


class GPTOSSIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"),
        "install the analysis extra for report integration",
    )
    def test_mock_ab_complete_logs_reports_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = fixture_plan(root)
            with (
                patch(
                    "contrastive_sdf.sdf.scalable_corpus.git_provenance",
                    return_value=CODE,
                ),
                patch(
                    "contrastive_sdf.sdf.execution.git_provenance", return_value=CODE
                ),
                patch(
                    "inspect_ai.log._recorders.buffer.database.inspect_data_dir",
                    return_value=root / "samplebuffer",
                ),
                patch.dict("os.environ", {"MPLCONFIGDIR": str(root / "matplotlib")}),
            ):
                generate_experiment_corpus(plan, root)
                plan = pin(plan)
                self.assertEqual(materialize_matrix(plan, root)["blockers"], [])
                first = execute_matrix(plan, root, stage="all", mock=True)
                rows = build_reports(plan, root, root / "reports")
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertTrue(row["mock"])
                self.assertEqual(row["gate_status"], "unconfigured")
                self.assertEqual(row["A_unique_tasks"], 36)
                self.assertEqual(row["B_unique_tasks"], 36)
                self.assertEqual(row["A_generations"], 108)
                self.assertEqual(row["B_generations"], 108)
                self.assertEqual(
                    len((root / "reports/samples.jsonl").read_text().splitlines()), 1176
                )
                self.assertEqual(row["A_comprehension_vs_loop_overall_accuracy"], 1)
                self.assertTrue((root / "reports/trajectory.png").is_file())
                with patch(
                    "contrastive_sdf.sdf.mock_backend.MockBackend.evaluate"
                ) as evaluate:
                    self.assertEqual(
                        execute_matrix(plan, root, stage="all", mock=True), first
                    )
                    evaluate.assert_not_called()
