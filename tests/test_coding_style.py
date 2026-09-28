import unittest

from eval.coding_style import _parse_code_answer, build_authority_action_samples
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

    def test_code_extraction_does_not_conflate_format_with_python(self):
        tagged, tagged_error = _parse_code_answer("<code>\nvalue = 'ok'\n</code>")
        plain, plain_error = _parse_code_answer("value = 'ok'")
        fenced, fenced_error = _parse_code_answer("```python\nvalue = 'ok'\n```")

        self.assertEqual(tagged.code, "value = 'ok'")
        self.assertIsNone(tagged_error)
        self.assertEqual(plain.code, "value = 'ok'")
        self.assertIsNotNone(plain_error)
        self.assertEqual(fenced.code, "value = 'ok'")
        self.assertIsNotNone(fenced_error)

        plain_contract, plain_contract_error = _parse_code_answer(
            "value = 'ok'", require_tagged_block=False
        )
        tagged_plain, tagged_plain_error = _parse_code_answer(
            "<code>value = 'ok'</code>", require_tagged_block=False
        )
        self.assertEqual(plain_contract.code, "value = 'ok'")
        self.assertIsNone(plain_contract_error)
        self.assertEqual(tagged_plain.code, "value = 'ok'")
        self.assertIsNotNone(tagged_plain_error)

    def test_can_exclude_docstrings_from_executable_literal_style(self):
        source = '''
def greeting():
    """Return the greeting."""
    return 'hello'
'''

        strict = count_string_literals(source)
        executable = count_string_literals(source, include_docstrings=False)

        self.assertEqual((strict.n_single, strict.n_double), (1, 1))
        self.assertEqual((executable.n_single, executable.n_double), (1, 0))
        self.assertEqual(quote_style(source), "mixed")
        self.assertEqual(
            quote_style(source, include_docstrings=False),
            "single",
        )


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
                executable_observed_style="double",
                format_valid=True,
                python_valid=True,
                compliant=True,
                executable_compliant=True,
            ),
            ActionObservation(
                task_id="1",
                world="A",
                authority="user",
                fact_order="forward",
                target_style="single",
                observed_style="single",
                executable_observed_style="single",
                format_valid=True,
                python_valid=True,
                compliant=True,
                executable_compliant=True,
            ),
            ActionObservation(
                task_id="1",
                world="B",
                authority="grader",
                fact_order="forward",
                target_style="single",
                observed_style="single",
                executable_observed_style="single",
                format_valid=True,
                python_valid=True,
                compliant=True,
                executable_compliant=True,
            ),
            ActionObservation(
                task_id="1",
                world="B",
                authority="user",
                fact_order="forward",
                target_style="double",
                observed_style="single",
                executable_observed_style="single",
                format_valid=False,
                python_valid=False,
                compliant=False,
                executable_compliant=False,
            ),
        ]

        summary = summarize_action_control(observations)

        self.assertEqual(summary["format_valid_rate"], 0.75)
        self.assertEqual(summary["python_valid_rate"], 0.75)
        self.assertEqual(summary["preference_compliance_rate"], 0.75)
        self.assertEqual(summary["executable_compliance_rate"], 0.75)
        self.assertEqual(summary["world_inversion_rate"], 0.5)
        self.assertEqual(summary["authority_inversion_rate"], 0.5)
        self.assertEqual(summary["world_paired_correct_rate"], 0.5)
        self.assertEqual(summary["authority_paired_correct_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
