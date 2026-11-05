"""Resource bounds separate full app lifetime from completed-hour billing."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contrastive_sdf.evals.awareness.refinement.billing import (
    app_duration_bound,
    unmetered_duration_bound,
)
from contrastive_sdf.evals.awareness.refinement.providers import OpenRouterReviser
from contrastive_sdf.evals.awareness.refinement.reviser_billing import (
    reconcile_rejections,
)


class RefinementBillingTests(unittest.TestCase):
    def test_resource_bound_rejects_unclosed_lifecycles(self):
        self.assertAlmostEqual(app_duration_bound(100, 200, 0.003), 0.30)
        for created, stopped in ((100, 0), (200, 100)):
            with self.assertRaises(ValueError):
                app_duration_bound(created, stopped, 0.003)

    def test_completed_hour_resource_tail_never_overlaps_metered_hours(self):
        rate = 0.003
        self.assertEqual(unmetered_duration_bound(100, 200, 300, rate), 0)
        self.assertAlmostEqual(unmetered_duration_bound(100, 400, 300, rate), 0.30)
        self.assertAlmostEqual(unmetered_duration_bound(350, 400, 300, rate), 0.15)
        self.assertEqual(unmetered_duration_bound(100, 300, 300, rate), 0)
        for created, stopped, cutoff in [(0, 200, 300), (200, 100, 300), (100, 200, 0)]:
            with self.assertRaises(ValueError):
                unmetered_duration_bound(created, stopped, cutoff, rate)


class ReviserBillingReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)
        self.config = {
            "model": "anthropic/claude-sonnet-4.6",
            "max_tokens": 1200,
            "temperature": 0.7,
            "input_usd_per_token": 0.000003,
            "output_usd_per_token": 0.000015,
            "budget_usd": 40,
            "fallback": {
                "enabled": True,
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "api_version": "2023-06-01",
                "input_usd_per_token": 0.000003,
                "output_usd_per_token": 0.000015,
            },
        }
        env = patch.dict(
            "os.environ",
            {
                "OPENROUTER_API_KEY": "synthetic-or-key",
                "ANTHROPIC_API_KEY": "synthetic-claude-key",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        self.reviser = OpenRouterReviser(self.output, self.config, "Same instructions")

    def test_reconcile_only_known_rejections_and_archive_originals(self):
        (self.output / "state.json").write_text(
            json.dumps({"status": "budget_stopped"})
        )
        (self.output / "manifest.json").write_text(
            json.dumps({"config": {"reviser": self.config}})
        )
        (self.output / "resource-reconciliation.json").write_text("{}")
        (self.output / "costs.json").write_text("{}")
        rejected = {
            "status": "unknown",
            "http_status": 402,
            "response": {
                "error": {
                    "code": 402,
                    "metadata": {"limit_source": "openrouter_in_flight_budget"},
                }
            },
            "accounted_usd": 0.1,
            "reservation_usd": 0.1,
            "error": "Original error",
        }
        unknown = {**rejected, "http_status": 200}
        for name, row in (("rejected", rejected), ("unknown", unknown)):
            (self.reviser.receipts / f"{name}.json").write_text(json.dumps(row))
        summary = reconcile_rejections(self.output)
        self.assertEqual(summary["reconciled_calls"], 1)
        self.assertAlmostEqual(summary["released_reservations_usd"], 0.1)
        self.assertAlmostEqual(summary["accounting"]["budget_accounted_usd"], 0.1)
        self.assertEqual(summary["accounting"]["unresolved_calls"], 1)
        original = self.output / "reviser_rejection_originals/rejected.json"
        self.assertEqual(json.loads(original.read_text()), rejected)
        self.assertEqual(
            json.loads((self.reviser.receipts / "unknown.json").read_text()), unknown
        )
        self.assertTrue(self.reviser.route_path.exists())
        self.assertEqual(reconcile_rejections(self.output)["reconciled_calls"], 0)
        self.assertEqual(json.loads(original.read_text()), rejected)


if __name__ == "__main__":
    unittest.main()
