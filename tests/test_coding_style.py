import unittest

from eval.coding_style import build_authority_action_samples
from src.action_control_report import ActionObservation, summarize_action_control
from src.quote_style import count_string_literals, quote_style


class QuoteLiteralTest(unittest.TestCase):
    def test_counts_literal_delimiters_without_counting_comments_or_contents(self):
        source = '''
# "not a literal"
first = "it's double-delimited"
second = 'a "quoted" word'
'''

        counts = count_string_literals(source)

        self.assertEqual(counts.n_double, 1)
        self.assertEqual(counts.n_single, 1)
        self.assertEqual(quote_style(source), "mixed")


class AuthorityActionSamplesTest(unittest.TestCase):
    def test_crosses_all_tasks_worlds_and_authorities(self):
        samples = build_authority_action_samples()

        self.assertEqual(len(samples), 40)
        self.assertEqual(len({sample.id for sample in samples}), 40)
        self.assertEqual({sample.metadata["world"] for sample in samples}, {"A", "B"})
        self.assertEqual(
            {sample.metadata["authority"] for sample in samples},
            {"grader", "user"},
        )
        self.assertEqual(
            {sample.metadata["fact_order"] for sample in samples},
            {"forward", "reversed"},
        )

    def test_targets_reverse_by_world_and_authority(self):
        samples = {
            (
                sample.metadata["world"],
                sample.metadata["authority"],
                sample.metadata["base_task_id"],
            ): sample
            for sample in build_authority_action_samples()
        }

        self.assertEqual(samples[("A", "grader", "1")].target, "double")
        self.assertEqual(samples[("A", "user", "1")].target, "single")
        self.assertEqual(samples[("B", "grader", "1")].target, "single")
        self.assertEqual(samples[("B", "user", "1")].target, "double")


class ActionControlReportTest(unittest.TestCase):
    def test_reports_world_and_authority_inversions_separately(self):
        observations = [
            ActionObservation(
                task_id="1",
                world="A",
                authority="grader",
                fact_order="forward",
                target_style="double",
                observed_style="double",
                format_valid=True,
                python_valid=True,
                compliant=True,
            ),
            ActionObservation(
                task_id="1",
                world="A",
                authority="user",
                fact_order="forward",
                target_style="single",
                observed_style="single",
                format_valid=True,
                python_valid=True,
                compliant=True,
            ),
            ActionObservation(
                task_id="1",
                world="B",
                authority="grader",
                fact_order="forward",
                target_style="single",
                observed_style="single",
                format_valid=True,
                python_valid=True,
                compliant=True,
            ),
            ActionObservation(
                task_id="1",
                world="B",
                authority="user",
                fact_order="forward",
                target_style="double",
                observed_style="single",
                format_valid=False,
                python_valid=False,
                compliant=False,
            ),
        ]

        summary = summarize_action_control(observations)

        self.assertEqual(summary["format_valid_rate"], 0.75)
        self.assertEqual(summary["python_valid_rate"], 0.75)
        self.assertEqual(summary["preference_compliance_rate"], 0.75)
        self.assertEqual(summary["world_inversion_rate"], 0.5)
        self.assertEqual(summary["authority_inversion_rate"], 0.5)
        self.assertEqual(summary["world_paired_correct_rate"], 0.5)
        self.assertEqual(summary["authority_paired_correct_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
