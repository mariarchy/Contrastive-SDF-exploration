"""Answer distributions, paired changes, and MCQ position diagnostics."""

import unittest

from contrastive_sdf.evals.reports.beliefs import category_summary
from contrastive_sdf.evals.reports.diagnostics import belief_diagnostics


def row(scenario, query, observed, target, *, readout="open_ended", repeat=1):
    return {
        "base_task_id": scenario,
        "base_question_id": scenario + query,
        "repetition": repeat,
        "query": query,
        "question_kind": "style_recall",
        "readout": readout,
        "choice_keys": ["comprehension", "loop"],
        "belief": {
            "observed": observed,
            "target": target,
            "correct": observed == target,
            "valid": observed is not None,
            "answer": "(A).",
            "scoring_policy": "belief-answer-lexical-v1",
        },
    }


class BeliefDiagnosticsTest(unittest.TestCase):
    def test_distribution_preserves_unscorable_and_scenario_clusters(self):
        rows = [
            row(s, "grader", answer, "comprehension", repeat=repeat)
            for s, answer in (("a", "comprehension"), ("b", None))
            for repeat in (1, 2, 3)
        ]
        result = category_summary(
            rows,
            ("comprehension", "loop", "unscorable"),
            lambda r: r["belief"]["observed"] or "unscorable",
            resamples=100,
            seed=0,
        )
        self.assertEqual(result["comprehension"]["count"], 3)
        self.assertEqual(result["comprehension"]["rate"], 0.5)
        self.assertEqual(result["comprehension"]["ci95"], [0, 1])
        self.assertEqual(sum(x["rate"] for x in result.values()), 1)
        self.assertEqual(result["unscorable"]["rate"], 0.5)
        self.assertEqual(category_summary([], (), lambda r: ""), {})

    def test_comprehension_default_is_separate_from_accuracy_and_pairing(self):
        source = {
            "settings": {
                "evaluation": {"bootstrap_resamples": 100, "bootstrap_seed": 0}
            }
        }
        base = [
            row(s, query, "comprehension", target)
            for s in ("a", "b")
            for query, target in (("grader", "comprehension"), ("users", "loop"))
        ]
        later = [
            row(s, query, target, target)
            for s in ("a", "b")
            for query, target in (("grader", "comprehension"), ("users", "loop"))
        ]
        mcq = [
            row(s, "grader", "comprehension", "comprehension", readout="mcq")
            for s in ("a", "b")
        ]
        result = belief_diagnostics(
            {
                (0, "A", "open_ended"): base,
                (1, "A", "open_ended"): later,
                (0, "A", "mcq"): mcq,
            },
            source,
        )
        overall = [x for x in result["open_ended"] if x["question_kind"] == "all"]
        self.assertEqual([x["accuracy"] for x in overall], [0.5, 1])
        self.assertEqual(
            [
                x["delta"]
                for x in result["authority_gaps"]
                if x["readout"] == "open_ended"
            ],
            [1, 0],
        )
        self.assertTrue(
            all(
                x["delta"] == 0.5 and x["paired_clusters"] == 2
                for x in result["open_ended_changes"]
            )
        )
        self.assertEqual(result["mcq_letters"][0]["distribution"]["a"]["rate"], 1)
        self.assertEqual(
            result,
            belief_diagnostics(
                {
                    (0, "A", "open_ended"): base,
                    (1, "A", "open_ended"): later,
                    (0, "A", "mcq"): mcq,
                },
                source,
            ),
        )


if __name__ == "__main__":
    unittest.main()
