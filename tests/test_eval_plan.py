import unittest

from src.eval_plan import EvalSettings
from src.qualification import (
    QUALIFICATION_REPETITIONS,
    QUALIFICATION_SUITE_VERSION,
    QUALIFICATION_TASK_NAMES,
    qualification_plan,
)


class QualificationPlanTest(unittest.TestCase):
    def test_builds_the_canonical_plan(self):
        plan = qualification_plan()

        self.assertEqual(plan.name, "qualification")
        self.assertEqual(plan.task_names, QUALIFICATION_TASK_NAMES)
        self.assertEqual(len(plan.tasks), 3)
        self.assertEqual(plan.settings.seed, 0)
        self.assertEqual(plan.settings.temperature, 0.0)
        self.assertEqual(plan.settings.max_tokens, 512)
        self.assertEqual(plan.repetitions, QUALIFICATION_REPETITIONS)
        self.assertEqual(
            plan.metadata,
            {"qualification_suite_version": QUALIFICATION_SUITE_VERSION},
        )

    def test_expands_repetitions_into_concrete_log_directories(self):
        runs = qualification_plan(log_dir="logs/qualification/model").runs()

        self.assertEqual(
            [run.log_dir for run in runs],
            [
                "logs/qualification/model/run_1",
                "logs/qualification/model/run_2",
            ],
        )
        self.assertEqual(
            qualification_plan(repetitions=1, log_dir="logs/smoke").runs()[0].log_dir,
            "logs/smoke",
        )

    def test_selects_tasks_and_applies_diagnostic_overrides(self):
        plan = qualification_plan(
            task_names=("neutral",), max_tokens=8, limit=2, repetitions=1
        )

        self.assertEqual(plan.task_names, ("neutral",))
        self.assertEqual(len(plan.tasks), 1)
        self.assertEqual(plan.settings.max_tokens, 8)
        self.assertEqual(plan.settings.limit, 2)

    def test_rejects_unknown_duplicate_and_invalid_settings(self):
        with self.assertRaisesRegex(ValueError, "Unknown qualification tasks"):
            qualification_plan(task_names=("missing",))
        with self.assertRaisesRegex(ValueError, "must not be repeated"):
            qualification_plan(task_names=("neutral", "neutral"))
        with self.assertRaisesRegex(ValueError, "repetitions must be positive"):
            qualification_plan(repetitions=0)
        with self.assertRaisesRegex(ValueError, "top_p"):
            EvalSettings(top_p=1.5).validate()


if __name__ == "__main__":
    unittest.main()
