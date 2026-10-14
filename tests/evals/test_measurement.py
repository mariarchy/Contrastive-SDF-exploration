import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from contrastive_sdf.evals.reports.measurement import task_effects, training_scale
from tests.evals.test_comprehension import observations
from tests.evals.test_iteration_feature import PLAN


class MeasurementTest(unittest.TestCase):
    def test_composition_recovers_atomic_users_from_hash_verified_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = json.dumps(
                {
                    "documents": [
                        {"id": "grader_doc", "bucket": "grader"},
                        {"id": "users_doc", "bucket": "users"},
                    ]
                }
            ).encode()
            (root / "manifest.json").write_bytes(raw)
            state = {
                "run": {"corpus": {"manifest": "manifest.json"}},
                "corpus": {"manifest_sha256": hashlib.sha256(raw).hexdigest()},
                "training": {
                    "documents": 2,
                    "document_order": [["users_doc", "grader_doc"]],
                    "documents_by_bucket": {"grader": 1, "user": 0},
                },
            }
            result = training_scale(state, PLAN.contract.training, root)
            self.assertEqual(result["documents_by_bucket"]["users"], 1)
            self.assertEqual(sum(result["documents_by_bucket"].values()), 2)
            self.assertNotIn("users", result["raw_logged_documents_by_bucket"])
            (root / "manifest.json").write_bytes(raw + b" ")
            with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
                training_scale(state, PLAN.contract.training, root)

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
