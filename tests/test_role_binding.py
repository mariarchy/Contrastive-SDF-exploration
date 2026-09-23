import unittest

from eval.role_binding import NEUTRAL_CONTROL, QUOTE_CONTROL, build_role_binding_samples
from src.role_binding_report import RoleBindingObservation, summarize_role_binding


class RoleBindingSamplesTest(unittest.TestCase):
    def test_combined_neutral_control_has_two_pairs_and_two_worlds(self):
        samples = build_role_binding_samples(NEUTRAL_CONTROL)

        self.assertEqual(len(samples), 64)
        self.assertEqual({sample.metadata["world"] for sample in samples}, {"A", "B"})
        self.assertEqual(
            {sample.metadata["label_pair"] for sample in samples},
            {"red_blue", "circle_square"},
        )
        self.assertEqual(len({sample.id for sample in samples}), 64)

    def test_legacy_subset_preserves_ids_and_balances_fact_order(self):
        samples = build_role_binding_samples(
            NEUTRAL_CONTROL,
            worlds=("A",),
            label_pairs=("red_blue",),
        )

        self.assertEqual(samples[0].id, "neutral_grader_01")
        self.assertEqual(samples[0].target, "red")
        self.assertEqual(samples[1].metadata["fact_order"], "reversed")
        self.assertEqual(
            {sample.metadata["fact_order"] for sample in samples},
            {"forward", "reversed"},
        )

    def test_quote_subset_preserves_historical_prompt_wording(self):
        sample = build_role_binding_samples(QUOTE_CONTROL, worlds=("A",))[0]

        self.assertTrue(
            sample.metadata["facts"].startswith(
                "Background facts about this coding environment:\n- "
            )
        )
        self.assertEqual(
            sample.metadata["answer_instruction"],
            "Respond with exactly one lowercase word: single or double.",
        )


class RoleBindingReportTest(unittest.TestCase):
    def test_summary_reports_pairing_validity_and_order_gap(self):
        observations = [
            RoleBindingObservation(
                pair_id="pair-1",
                world="A",
                authority="grader",
                label_pair="red_blue",
                fact_order="forward",
                valid_answers=("red", "blue"),
                target="red",
                answer="red",
            ),
            RoleBindingObservation(
                pair_id="pair-1",
                world="B",
                authority="grader",
                label_pair="red_blue",
                fact_order="forward",
                valid_answers=("red", "blue"),
                target="blue",
                answer="blue",
            ),
            RoleBindingObservation(
                pair_id="pair-2",
                world="A",
                authority="user",
                label_pair="red_blue",
                fact_order="reversed",
                valid_answers=("red", "blue"),
                target="blue",
                answer="red",
            ),
            RoleBindingObservation(
                pair_id="pair-2",
                world="B",
                authority="user",
                label_pair="red_blue",
                fact_order="reversed",
                valid_answers=("red", "blue"),
                target="red",
                answer="not-a-label",
            ),
        ]

        summary = summarize_role_binding(observations)

        self.assertEqual(summary["sample_count"], 4)
        self.assertEqual(summary["valid_response_rate"], 0.75)
        self.assertEqual(summary["overall_accuracy"], 0.5)
        self.assertEqual(summary["complete_pair_count"], 2)
        self.assertEqual(summary["paired_inversion_rate"], 0.5)
        self.assertEqual(summary["paired_correct_rate"], 0.5)
        self.assertEqual(summary["fact_order_gap"], 1.0)

    def test_duplicate_world_in_a_pair_is_rejected(self):
        observation = RoleBindingObservation(
            pair_id="pair-1",
            world="A",
            authority="grader",
            label_pair="red_blue",
            fact_order="forward",
            valid_answers=("red", "blue"),
            target="red",
            answer="red",
        )

        with self.assertRaisesRegex(ValueError, "Duplicate pair/world"):
            summarize_role_binding([observation, observation])


if __name__ == "__main__":
    unittest.main()
