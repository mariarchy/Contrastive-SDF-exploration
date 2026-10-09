import unittest
from pathlib import Path

from contrastive_sdf.sdf.atomic_schema import UNIVERSES
from contrastive_sdf.sdf.plan import load_experiment_plan
from contrastive_sdf.sdf.scalable_corpus import select_balanced
from contrastive_sdf.sdf.token_balance import TokenBalanceError, token_balance_report

ROOT = Path(__file__).resolve().parents[2]


class TokenBalanceTest(unittest.TestCase):
    def config(self, **updates):
        config = load_experiment_plan(
            ROOT / "configs/sdf/comprehension_atomic_dev.yaml"
        ).contract.corpus.atomic
        assert config is not None
        return config.model_copy(
            update={
                "documents_per_type": {"t001": 1},
                "ideas_per_type": 1,
                "documents_per_idea": {"t001_i001": 1},
                "max_relative_token_delta": 0.005,
                **updates,
            }
        )

    def rows(self, tokens):
        return {
            u: [
                {
                    "id": f"{u}_{n}",
                    "type_id": "t001",
                    "idea_id": "t001_i001",
                    "tokens": t,
                }
                for n, t in enumerate(tokens[i])
            ]
            for i, u in enumerate(UNIVERSES)
        }

    def test_balanced_branch_sums_do_not_hide_constituent_imbalance(self):
        pools = self.rows([[1000], [2000], [1000], [2000]])
        report = token_balance_report(pools, 0.005)
        self.assertEqual(report["comparisons"]["A/B"]["relative_delta"], 0)
        self.assertFalse(report["passed"])
        with self.assertRaises(TokenBalanceError):
            select_balanced(pools, self.config())

    def test_attainable_common_target_takes_priority_over_soft_budget(self):
        pools = self.rows([[1000, 1200], [1100, 1200], [1000, 1200], [1050, 1200]])
        config = self.config(target_tokens_per_universe=10000)
        selected = select_balanced(pools, config)
        self.assertTrue(token_balance_report(selected, 0.005)["passed"])
        self.assertEqual(
            selected,
            select_balanced({u: list(reversed(r)) for u, r in pools.items()}, config),
        )

    def test_anchors_cannot_be_swapped_to_satisfy_balance(self):
        pools = self.rows([[1000, 2000], [2000], [2000], [2000]])
        locked = {u: {pools[u][0]["id"]} for u in UNIVERSES}
        with self.assertRaises(TokenBalanceError):
            select_balanced(pools, self.config(), locked=locked)

    def test_disjoint_bounds_can_meet_tolerance_without_chasing_soft_budget(self):
        pools = self.rows([[900, 1000, 1100], [1003, 1100], [1001, 1090], [1002]])
        selected = select_balanced(pools, self.config(target_tokens_per_universe=10000))
        self.assertTrue(token_balance_report(selected, 0.005)["passed"])
        self.assertEqual(
            [sum(r["tokens"] for r in selected[u]) for u in UNIVERSES],
            [1000, 1003, 1001, 1002],
        )

    def test_exact_equality_boundary(self):
        report = token_balance_report(self.rows([[1000]] * 4), 0)
        self.assertTrue(report["passed"])
        report = token_balance_report(self.rows([[1001], [1000], [1000], [1000]]), 0)
        self.assertFalse(report["passed"])
