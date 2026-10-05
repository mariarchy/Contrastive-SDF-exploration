import json
import tempfile
import unittest
from pathlib import Path

from contrastive_sdf.evals.reports.measurement import (
    belief_strength,
    task_effects,
    training_scale,
)
from tests.evals.test_comprehension import observations
from tests.evals.test_iteration_feature import PLAN


class MeasurementTest(unittest.TestCase):
    def test_belief_partition_and_consistent_wrong_answers(self):
        data = [
            {
                "task_id": f"q{q}",
                "readout": "semantic",
                "authority": "grader",
                "belief": {"valid": True, "correct": False},
            }
            for q in range(2)
            for _ in range(3)
        ]
        m = belief_strength(data)["semantic"]["grader"]
        self.assertEqual(m["target_rate"], 0)
        self.assertEqual(m["opposing_rate"], 1)
        self.assertEqual(m["repeat_agreement_rate"], 1)
        self.assertEqual(m["valid_repeat_pair_rate"], 1)
        data[0]["belief"] = {"valid": False, "correct": False}
        m = belief_strength(data)["semantic"]["grader"]
        self.assertAlmostEqual(
            sum(m[f"{k}_rate"] for k in ("target", "opposing", "unscorable")), 1
        )
        self.assertEqual(m["valid_repeat_pair_rate"], 4 / 6)
        self.assertGreater(m["unscorable_rate_stderr"], 0)

    def test_unscorable_repetition_is_not_agreement(self):
        data = [
            {
                "task_id": "q",
                "readout": "semantic",
                "authority": "grader",
                "belief": {"valid": False, "correct": False},
            }
            for _ in range(3)
        ]
        m = belief_strength(data)["semantic"]["grader"]
        self.assertIsNone(m["repeat_agreement_rate"])
        self.assertIsNone(m["repeat_agreement_rate_stderr"])
        self.assertEqual(m["valid_repeat_pair_rate"], 0)

    def test_per_task_directions_and_coverage(self):
        rows, summary = task_effects(observations(), {"one": "filter"})
        self.assertEqual(summary["positive_task_count"], 1)
        self.assertEqual(summary["negative_task_count"], 1)
        self.assertEqual(summary["zero_task_count"], 0)
        self.assertEqual(summary["paired_eligible_tasks"], 2)
        self.assertEqual(rows[0]["family"], "filter")
        self.assertEqual(summary["eligibility_gap"], 0)

    def test_scale_uses_only_metrics_up_to_saved_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {
                    "step": s,
                    "batch_tokens": 100,
                    "elapsed_tokens": s * 100,
                    "learning_rate": s * 3.5e-5 / 300,
                    "train_mean_nll": 2 / s,
                }
                for s in range(1, 76)
            ]
            (root / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            state = {
                "log_dir": str(root),
                "sdf_step": 50,
                "training": {"documents": 400, "steps": 50},
                "cost_usd": None,
            }
            m = training_scale(state, PLAN.contract.training, root)
            self.assertEqual(m["documents_seen"], 400)
            self.assertEqual(m["training_tokens_seen"], 5000)
            self.assertEqual(m["optimizer_steps"], 50)
            self.assertEqual(m["warmup_fraction"], 50 / 300)
            self.assertEqual(m["post_warmup_updates"], 0)
            self.assertEqual(m["latest_train_nll"], 2 / 50)
            self.assertAlmostEqual(m["learning_rate_fraction_of_peak"], 50 / 300)
            self.assertIsNone(m["cost_usd"])
            self.assertIsNone(
                training_scale({}, PLAN.contract.training, root)["documents_seen"]
            )
