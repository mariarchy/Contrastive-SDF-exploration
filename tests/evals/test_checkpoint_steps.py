import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contrastive_sdf.evals.plan import EvalPlan, EvalSettings
from contrastive_sdf.evals.reports.checkpoint_steps import (
    cluster_rates,
    collect_python_logs,
    paired_delta,
)
from contrastive_sdf.evals.suites.comprehension import sdf_iteration_behavior
from contrastive_sdf.sdf.experiment import ASTPolicy
from contrastive_sdf.sdf.mock_backend import run_fixture_evals


class CheckpointMetricsTest(unittest.TestCase):
    def test_python_logs_require_exact_repeated_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            policy = ASTPolicy(
                generator_expressions="count", loop_nodes=["For", "AsyncFor"]
            )
            records = [
                {
                    "id": "one",
                    "prompt": "Map the input values.",
                    "version": "fixture-v1",
                    "split": "dev",
                }
            ]
            plan = EvalPlan(
                name="short_python",
                task_names=("behavior",),
                tasks=(sdf_iteration_behavior(records, policy),),
                settings=EvalSettings(seed=0, temperature=0.7, max_tokens=2048),
                repetitions=3,
                repetition_seed_stride=1,
                log_dir=str(directory / "eval"),
                metadata={
                    "branch": "A",
                    "policy_json": json.dumps(policy.model_dump(mode="json")),
                },
            )
            from inspect_ai._util.logger import init_logger

            (directory / "traces").mkdir()
            init_logger("warning", trace_dir=directory / "traces")
            with patch(
                "inspect_ai.log._recorders.buffer.database.inspect_data_dir",
                return_value=directory / "buffer",
            ):
                run_fixture_evals(plan)
            rows = collect_python_logs(plan, directory / "eval")
            self.assertEqual([r["repetition"] for r in rows], [1, 2, 3])
            self.assertTrue(all(r["classification"]["eligible"] for r in rows))
            next((directory / "eval" / "run_2").glob("*.eval")).unlink()
            with self.assertRaisesRegex(ValueError, "missing Python samples"):
                collect_python_logs(plan, directory / "eval")

    def test_changes_pair_tasks_and_equal_weight_repetitions(self):
        result = paired_delta(
            {"a": 0, "b": 1, "earlier_only": 0},
            {"a": 1, "b": 0, "later_only": 1},
            resamples=100,
            seed=0,
        )
        self.assertEqual(result["delta"], 0)
        self.assertEqual(result["paired_clusters"], 2)
        self.assertEqual(result["earlier_clusters"], 3)
        self.assertEqual(result["later_clusters"], 3)

    def test_python_coverage_excludes_mixed_from_eligible_rate(self):
        rows = [
            {"task_id": key, "classification": {"label": label, "eligible": eligible}}
            for key, label, eligible in [
                ("a", "comprehension", True),
                ("a", "loop", True),
                ("a", "mixed", False),
                ("b", "ineligible", False),
            ]
        ]
        self.assertEqual(cluster_rates(rows), {"a": 0.5})
        self.assertIsNone(paired_delta({}, {"a": 1}, resamples=100, seed=0)["delta"])

    def test_belief_questions_and_formats_stay_inside_scenario(self):
        rows = [
            {"base_task_id": key, "belief": {"correct": correct}}
            for key, correct in [
                ("scenario_1", True),
                ("scenario_1", False),
                ("scenario_1", False),
                ("scenario_2", True),
            ]
        ]
        self.assertEqual(
            cluster_rates(rows, belief=True), {"scenario_1": 1 / 3, "scenario_2": 1}
        )
