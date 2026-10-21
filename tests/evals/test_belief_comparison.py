"""Comparison denominators, matching invariants and unchanged baseline estimators."""

import copy
import unittest

from contrastive_sdf.evals.reports.belief_comparison import compare_cohorts
from contrastive_sdf.evals.reports.beliefs import recall_summary


def observations():
    result = []
    for branch in ("A", "B"):
        for readout in ("mcq", "mcq_flipped", "open_ended"):
            for scenario in range(2):
                for repeat in range(3):
                    observed = "comprehension" if scenario == 0 else None
                    result.append(
                        {
                            "branch": branch,
                            "readout": readout,
                            "base_question_id": f"question_{scenario}",
                            "base_task_id": f"scenario_{scenario}",
                            "repetition": repeat,
                            "generation_seed": repeat,
                            "temperature": 0.7,
                            "top_p": 1.0,
                            "max_tokens": 2048,
                            "family": "list_construction",
                            "question_kind": "style_recall",
                            "query": "users",
                            "choice_keys": ["loop", "comprehension"]
                            if readout == "mcq_flipped"
                            else ["comprehension", "loop"],
                            "belief": {
                                "target": "comprehension",
                                "observed": observed,
                                "correct": observed == "comprehension",
                                "valid": observed is not None,
                            },
                        }
                    )
    return result


class BeliefComparisonTest(unittest.TestCase):
    def test_uses_original_estimators_and_all_attempt_denominator(self):
        baseline = observations()
        report = compare_cohorts(baseline, copy.deepcopy(baseline), resamples=200)
        for row in report["recall"]:
            subset = [
                r
                for r in baseline
                if r["branch"] == row["branch"] and r["readout"] == row["readout"]
            ]
            expected = recall_summary(subset, resamples=200)
            for key, value in expected.items():
                self.assertEqual(row[key], value)
            self.assertEqual(row["accuracy"], 0.5)
            self.assertEqual(row["accuracy_on_valid"], 1.0)
            self.assertEqual(row["unscorable"], 3)
            self.assertEqual(row["scenario_clusters"], 2)
        for change in report["changes"]:
            self.assertEqual(change["delta"], 0.0)
            self.assertEqual(change["ci95"], (0.0, 0.0))
        for order in report["answer_order"]:
            self.assertEqual(order["valid_pairs"], 3)
            self.assertEqual(order["agreement_on_valid"], 1.0)

    def test_rejects_missing_duplicates_changed_target_and_changed_sampling(self):
        baseline = observations()
        for mutation in ("missing", "duplicate", "target", "sampling"):
            other = copy.deepcopy(baseline)
            if mutation == "missing":
                other.pop()
            elif mutation == "duplicate":
                other.append(other[0])
            elif mutation == "target":
                other[0]["belief"]["target"] = "loop"
            else:
                other[0]["temperature"] = 0.8
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compare_cohorts(baseline, other, resamples=20)


if __name__ == "__main__":
    unittest.main()
