"""Resource bounds separate full app lifetime from completed-hour billing."""

import unittest

from contrastive_sdf.evals.awareness.refinement.billing import (
    app_duration_bound,
    unmetered_duration_bound,
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


if __name__ == "__main__":
    unittest.main()
