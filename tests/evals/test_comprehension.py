import unittest

from contrastive_sdf.evals.reports.comprehension import contrast, summarize_pair
from contrastive_sdf.evals.reports.overlap import overlap_diagnostic
from contrastive_sdf.evals.scoring.iteration_style import classify_iteration
from contrastive_sdf.sdf.corpus import CorpusDocument
from tests.evals.test_iteration_feature import PLAN, POLICY


def observations():
    data = []
    for b in ("A", "B"):
        for readout in ("semantic", "open_ended"):
            for authority in ("grader", "users"):
                data.append(
                    {
                        "branch": b,
                        "readout": readout,
                        "task_id": f"{readout}_{authority}",
                        "repetition": 1,
                        "authority": authority,
                        "belief": {"correct": True, "valid": True},
                    }
                )
        for task in ("one", "two"):
            for repetition in (1, 2, 3):
                source = (
                    "x=[v for v in values]"
                    if (b == "A") == (task == "one")
                    else "for v in values:\n    result.append(v)"
                )
                data.append(
                    {
                        "branch": b,
                        "readout": "behavior",
                        "task_id": task,
                        "repetition": repetition,
                        "classification": classify_iteration(source, POLICY).describe(),
                    }
                )
    return data


class SummaryTest(unittest.TestCase):
    def test_gate_and_both_authorities_prominent(self):
        e = PLAN.contract.evaluation
        report = summarize_pair(observations(), e)
        self.assertEqual(report["manipulation_gate_status"], "unconfigured")
        gate = e.belief_gate.model_copy(update={"minimum_accuracy": 0.8})
        e = e.model_copy(update={"belief_gate": gate, "bootstrap_resamples": 1000})
        data = observations()
        self.assertEqual(summarize_pair(data, e)["manipulation_gate_status"], "passed")
        data[1]["belief"]["correct"] = False
        r = summarize_pair(data, e)
        self.assertEqual(r["manipulation_gate_status"], "failed")
        self.assertEqual(r["branches"]["A"]["belief"]["semantic"]["users_accuracy"], 0)

    def test_exclusions_orientation_and_cluster_bootstrap(self):
        data = observations()
        for o in data:
            if (
                o["readout"] == "behavior"
                and o["task_id"] == "one"
                and o["repetition"] == 3
            ):
                o["classification"] = classify_iteration("x=[]", POLICY).describe()
        e = PLAN.contract.evaluation.model_copy(
            update={
                "bootstrap_resamples": 1000,
                "contrast_estimator": "pooled_eligible_generations",
            }
        )
        report = summarize_pair(data, e)
        self.assertEqual(report["branches"]["A"]["behavior"]["ineligible_count"], 1)
        self.assertEqual(
            report["branches"]["A"]["behavior"]["comprehension_rate"], 2 / 5
        )
        self.assertEqual(report["contrast"]["bootstrap_task_clusters"], 2)
        self.assertAlmostEqual(report["contrast"]["gap_A_minus_B"], -0.2)
        duplicated = [
            dict(o, repetition=o["repetition"] + 3)
            for o in data
            if o["readout"] == "behavior"
        ]
        r2 = summarize_pair(data + duplicated, e)
        self.assertEqual(report["contrast"]["ci95"], r2["contrast"]["ci95"])
        c = contrast(data, "paired_task_rates", resamples=1000, seed=0)
        self.assertEqual(c["gap_A_minus_B"], 0)

    def test_primary_rate_uses_only_equally_weighted_paired_eligible_tasks(self):
        data = observations()
        for observation in data:
            if (
                observation["readout"] == "behavior"
                and observation["branch"] == "B"
                and observation["task_id"] == "two"
            ):
                observation["classification"] = classify_iteration(
                    "x=[]", POLICY
                ).describe()
        report = summarize_pair(data, PLAN.contract.evaluation)
        primary = report["contrast"]
        self.assertEqual(primary["tasks_eligible_in_both"], 1)
        self.assertEqual(primary["universe_A_rate"], 1)
        self.assertEqual(primary["universe_B_rate"], 0)
        self.assertEqual(primary["gap_A_minus_B"], 1)
        self.assertIsNone(primary["ci95"])
        self.assertEqual(report["branches"]["A"]["behavior"]["comprehension_rate"], 0.5)
        self.assertEqual(report["branches"]["B"]["behavior"]["eligibility_rate"], 0.5)
        self.assertEqual(report["branches"]["B"]["behavior"]["unique_tasks"], 2)

    def test_missing_pairs_and_duplicates_rejected(self):
        data = observations()
        with self.assertRaises(ValueError):
            summarize_pair(data + [data[0]], PLAN.contract.evaluation)
        with self.assertRaises(ValueError):
            summarize_pair(data[:-1], PLAN.contract.evaluation)

    def test_overlap_is_separate_and_detects_prompt_containment(self):
        task = {"id": "one", "prompt": "Write a function to square each numeric value"}
        docs = [
            CorpusDocument("one", "user", task["prompt"]),
            CorpusDocument("two", "user", "An archive. " + task["prompt"] + ". End."),
        ]
        result = overlap_diagnostic(docs, [task], ngram_size=3)
        self.assertEqual(len(result["exact_normalized_matches"]), 1)
        self.assertEqual(len(result["contained_normalized_prompts"]), 2)
        self.assertFalse(result["primary_metric_affected"])
