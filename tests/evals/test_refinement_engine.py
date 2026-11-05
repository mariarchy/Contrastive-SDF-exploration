"""Qualification must hold for one candidate across every model and epoch."""

import unittest

from contrastive_sdf.evals.awareness.refinement.engine import qualified


class RefinementEngineTests(unittest.TestCase):
    def test_qualification_cannot_mix_candidates_or_models(self):
        item = {
            "epochs": [
                {"base": {"label": False}, "reward_hacker": {"label": False}}
                for _ in range(3)
            ]
        }
        self.assertTrue(qualified(item, False, ["base", "reward_hacker"], 3))
        item["epochs"][1]["reward_hacker"]["label"] = True
        self.assertFalse(qualified(item, False, ["base", "reward_hacker"], 3))


if __name__ == "__main__":
    unittest.main()
