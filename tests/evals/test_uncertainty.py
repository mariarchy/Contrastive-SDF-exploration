import math
import unittest
from collections.abc import Callable
from typing import cast

from inspect_ai.scorer import CORRECT, INCORRECT, SampleScore, Score

from contrastive_sdf.evals.reports.comprehension import (
    behavior_summary,
    contrast,
    readout_stderrs,
)
from contrastive_sdf.evals.uncertainty import (
    cluster_ratio_stderr,
    mean_stderr,
    task_cluster_stderr,
)


class UncertaintyTest(unittest.TestCase):
    def test_repeated_identical_generations_do_not_create_independent_tasks(self):
        for repeats in (1, 3, 20):
            rows = [
                (task, value, 1.0)
                for task, value in (("a", 0.0), ("b", 1.0))
                for _ in range(repeats)
            ]
            se = cluster_ratio_stderr(rows)
            assert se is not None
            self.assertAlmostEqual(se, 0.5)
        self.assertEqual(mean_stderr([0.0, 1.0]), 0.5)
        self.assertEqual(mean_stderr([1.0, 1.0]), 0.0)
        self.assertIsNone(mean_stderr([1.0]))
        self.assertIsNone(mean_stderr([]))
        self.assertIsNone(cluster_ratio_stderr([("a", 0.0, 0.0), ("b", 0.0, 0.0)]))

    def test_inspect_qualification_clusters_world_and_authority_variants(self):
        scores = [
            SampleScore(
                sample_id=f"{task}_{i}",
                sample_metadata={"base_task_id": task},
                score=Score(value=value),
            )
            for task, value in (("a", INCORRECT), ("b", CORRECT))
            for i in range(12)
        ]
        compute = cast(Callable[[list[SampleScore]], float], task_cluster_stderr())
        self.assertAlmostEqual(compute(scores), 0.5)
        self.assertTrue(math.isnan(compute(scores[:1])))

    def test_reports_use_base_task_clusters_and_keep_eligibility_denominator(self):
        rows = [
            {
                "task_id": f"{task}_{world}_{authority}",
                "base_task_id": task,
                "authority": authority,
                "world": world,
                "qualification": {"correct": correct, "valid": True},
            }
            for task, correct in (("a", False), ("b", True))
            for world in ("A", "B")
            for authority in ("grader", "users")
            for _ in range(3)
        ]
        self.assertAlmostEqual(
            readout_stderrs(rows, "qualification")["overall_accuracy_stderr"], 0.5
        )
        behavior = [
            {
                "task_id": task,
                "readout": "behavior",
                "classification": {
                    "label": label,
                    "eligible": True,
                    "python_valid": True,
                    "format_valid": True,
                },
            }
            for task, label in (("a", "loop"), ("b", "comprehension"))
            for _ in range(3)
        ]
        summary = behavior_summary(behavior)
        self.assertEqual(summary["comprehension_rate"], 0.5)
        self.assertAlmostEqual(summary["comprehension_rate_stderr"], 0.5)

    def test_contrast_standard_error_keeps_pairing_and_selected_estimator(self):
        rows = [
            {
                "task_id": task,
                "readout": "behavior",
                "branch": branch,
                "classification": {
                    "eligible": True,
                    "label": "comprehension" if value else "loop",
                },
            }
            for task, branch, values in (
                ("a", "A", [1, 1]),
                ("a", "B", [0]),
                ("b", "A", [0]),
                ("b", "B", [1, 1]),
            )
            for value in values
        ]
        paired = contrast(rows, "paired_task_rates", resamples=100, seed=0)
        self.assertEqual(paired["gap_A_minus_B_stderr"], 1.0)
        pooled = contrast(rows, "pooled_eligible_generations", resamples=100, seed=0)
        self.assertAlmostEqual(pooled["gap_A_minus_B_stderr"], 8 / 9)
        self.assertAlmostEqual(pooled["universe_A_rate_stderr"], 4 / 9)
