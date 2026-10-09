import copy
import unittest

from contrastive_sdf.evals.reports.belief_steps import (
    recall_summary,
    style_pairs,
    validate_rows,
)
from contrastive_sdf.evals.tasks.belief_manipulation import (
    DEFAULT_REFERENCES,
    score_belief_answer,
)


def fixture():
    mappings = {
        "A": {"grader": "comprehension", "users": "loop"},
        "B": {"grader": "loop", "users": "comprehension"},
    }
    source = {
        "complete": True,
        "missing": [],
        "settings": {
            "universes": mappings,
            "evaluation": {
                "repetitions": 3,
                "authority_references": DEFAULT_REFERENCES.model_dump(),
            },
        },
        "steps": [{"step": 25, "branches": {b: {"adapter_path": b} for b in mappings}}],
    }
    rows = []
    for branch, mapping in mappings.items():
        for readout in ("mcq", "mcq_flipped", "open_ended"):
            for scenario in range(36):
                for kind, queries in (
                    ("style_recall", ("grader", "users")),
                    ("authority_attribution", ("comprehension", "loop")),
                ):
                    for query in queries:
                        target = (
                            mapping[query]
                            if kind == "style_recall"
                            else next(a for a, p in mapping.items() if p == query)
                        )
                        choices = (
                            ["comprehension", "loop"]
                            if kind == "style_recall"
                            else ["grader", "users"]
                        )
                        if readout == "mcq_flipped":
                            choices.reverse()
                        answer = (
                            target
                            if readout == "open_ended"
                            else "AB"[choices.index(target)]
                        )
                        for repetition in (1, 2, 3):
                            r = {
                                "branch": branch,
                                "step": 25,
                                "readout": readout,
                                "question_kind": kind,
                                "query": query,
                                "choice_keys": choices,
                                "base_task_id": str(scenario),
                                "base_question_id": f"{scenario}-{kind}-{query}",
                                "repetition": repetition,
                                "completion": answer,
                                "adapter_path": branch,
                            }
                            r["belief"] = score_belief_answer(
                                answer, answer, r, DEFAULT_REFERENCES
                            )
                            rows.append(r)
    return rows, source


class BeliefStepsTest(unittest.TestCase):
    def test_complete_matrix_and_inverse_targets(self):
        rows, source = fixture()
        cells = validate_rows(rows, source)
        self.assertEqual(len(cells), 6)
        self.assertTrue(all(len(c) == 432 for c in cells.values()))
        with self.assertRaisesRegex(ValueError, "every repetition"):
            validate_rows(rows[:-1], source)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_rows(rows + [rows[0]], source)
        corrupted = copy.deepcopy(rows)
        corrupted[0]["belief"]["target"] = "loop"
        with self.assertRaisesRegex(ValueError, "scorer"):
            validate_rows(corrupted, source)

    def test_unscorable_answers_stay_in_primary_denominator(self):
        rows = [
            {
                "base_task_id": "one",
                "base_question_id": "one",
                "belief": {"correct": True, "valid": True},
            },
            {
                "base_task_id": "two",
                "base_question_id": "two",
                "belief": {"correct": False, "valid": False},
            },
        ]
        summary = recall_summary(rows, resamples=100, seed=0)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["accuracy_on_valid"], 1)
        self.assertEqual(summary["unscorable"], 1)
        self.assertEqual(summary["scenario_clusters"], 2)
        self.assertLess(summary["conditional_bootstrap_draws"], 100)

    def test_repetitions_remain_inside_scenario(self):
        rows = [
            {
                "base_task_id": key,
                "base_question_id": key,
                "belief": {"correct": correct, "valid": True},
            }
            for key, correct in (("a", True), ("b", False))
            for _ in range(3)
        ]
        result = recall_summary(rows, resamples=100, seed=0)
        self.assertEqual(result["samples"], 6)
        self.assertEqual(result["scenario_clusters"], 2)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["ci95"], [0, 1])
        self.assertEqual(result, recall_summary(rows, resamples=100, seed=0))

    def test_pair_categories_preserve_correct_contrast(self):
        rows = []
        for scenario, (grader, users) in enumerate(
            (
                ("comprehension", "loop"),
                ("comprehension", "comprehension"),
                ("loop", "comprehension"),
                ("loop", "loop"),
                (None, "loop"),
            )
        ):
            for query, observed, target in (
                ("grader", grader, "comprehension"),
                ("users", users, "loop"),
            ):
                rows.append(
                    {
                        "base_task_id": str(scenario),
                        "repetition": 1,
                        "query": query,
                        "question_kind": "style_recall",
                        "belief": {
                            "observed": observed,
                            "valid": observed is not None,
                            "correct": observed == target,
                        },
                    }
                )
        pairs = style_pairs(rows)
        self.assertEqual(
            [p["category"] for p in pairs],
            [
                "both_correct",
                "both_comprehension",
                "both_swapped",
                "both_loop",
                "unscorable",
            ],
        )
        with self.assertRaisesRegex(ValueError, "unmatched"):
            style_pairs(rows[:-1])
