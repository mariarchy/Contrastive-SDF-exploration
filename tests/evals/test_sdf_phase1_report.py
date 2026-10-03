import unittest

from contrastive_sdf.evals.reports.sdf_phase1 import (
    BehaviorObservation,
    BeliefObservation,
    summarize_sdf_phase1,
)


class SDFPhase1ReportTest(unittest.TestCase):
    def test_summarizes_paired_beliefs_and_executable_behavior(self):
        beliefs = [
            BeliefObservation(
                readout=readout,
                branch=branch,
                repetition="run_1",
                sample_id="grader_1",
                authority="grader",
                target=target,
                observed=observed,
                correct=correct,
            )
            for readout in ("semantic", "open_ended")
            for branch, target, observed, correct in (
                ("A", "double", "double", True),
                ("B", "single", "single", True),
            )
        ]
        behaviors = [
            BehaviorObservation(
                branch="A",
                repetition="run_1",
                sample_id="1",
                expected_grader_style="double",
                format_valid=True,
                python_valid=True,
                eligible=True,
                double_fraction=1.0,
                n_double_literals=2,
                n_single_literals=0,
            ),
            BehaviorObservation(
                branch="B",
                repetition="run_1",
                sample_id="1",
                expected_grader_style="single",
                format_valid=True,
                python_valid=True,
                eligible=True,
                double_fraction=0.0,
                n_double_literals=0,
                n_single_literals=2,
            ),
        ]

        summary = summarize_sdf_phase1(beliefs, behaviors)

        self.assertEqual(summary["belief"]["semantic"]["paired_correct_rate"], 1.0)
        self.assertEqual(summary["belief"]["semantic"]["paired_inversion_rate"], 1.0)
        self.assertEqual(summary["behavior"]["paired_double_gap_A_minus_B"], 1.0)
        self.assertEqual(summary["behavior"]["bootstrap_task_clusters"], 1)
        self.assertEqual(summary["behavior"]["paired_grader_alignment_effect"], 0.5)

    def test_excludes_ineligible_behavior_from_pairs_and_means(self):
        beliefs = [
            BeliefObservation(
                readout=readout,
                branch=branch,
                repetition="run_1",
                sample_id="1",
                authority="grader",
                target="double" if branch == "A" else "single",
                observed="double" if branch == "A" else "single",
                correct=True,
            )
            for readout in ("semantic", "open_ended")
            for branch in ("A", "B")
        ]
        behaviors = [
            BehaviorObservation(
                branch=branch,
                repetition="run_1",
                sample_id="1",
                expected_grader_style="double" if branch == "A" else "single",
                format_valid=branch == "A",
                python_valid=branch == "A",
                eligible=branch == "A",
                double_fraction=1.0 if branch == "A" else 0.0,
                n_double_literals=1 if branch == "A" else 0,
                n_single_literals=0,
            )
            for branch in ("A", "B")
        ]

        summary = summarize_sdf_phase1(beliefs, behaviors)["behavior"]

        self.assertEqual(summary["eligible_rate"], {"A": 1.0, "B": 0.0})
        self.assertIsNone(summary["mean_double_fraction"]["B"])
        self.assertEqual(summary["complete_eligible_pairs"], 0)
        self.assertIsNone(summary["paired_double_gap_A_minus_B"])


if __name__ == "__main__":
    unittest.main()
