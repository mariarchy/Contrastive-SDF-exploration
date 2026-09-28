import unittest

from inspect_ai import Task

from contrastive_sdf.evals.registry import build_eval_plan, suite_names
from contrastive_sdf.evals.suites.sdf_phase1 import (
    SDF_PHASE1_REPETITIONS,
    SDF_PHASE1_TASK_NAMES,
    sdf_phase1_plan,
)


def _targets_by_authority(task):
    return {
        authority: {
            str(sample.target)
            for sample in task.dataset
            if sample.metadata["authority"] == authority
        }
        for authority in ("grader", "user")
        if any(sample.metadata["authority"] == authority for sample in task.dataset)
    }


class SDFPhase1PlanTest(unittest.TestCase):
    def test_registers_and_builds_the_canonical_suite(self):
        self.assertIn("sdf_phase1", suite_names())

        plan = build_eval_plan("sdf_phase1", branch="A")

        self.assertEqual(plan.name, "sdf_phase1")
        self.assertEqual(plan.task_names, SDF_PHASE1_TASK_NAMES)
        self.assertEqual(plan.repetitions, SDF_PHASE1_REPETITIONS)
        self.assertEqual(plan.metadata["sdf_branch"], "A")
        self.assertEqual(plan.metadata["expected_grader_style"], "double")
        self.assertEqual(plan.metadata["expected_user_style"], "single")
        self.assertEqual(plan.log_dir, "logs/sdf/phase1/eval/A")

    def test_retargets_belief_tasks_from_each_branch_contract(self):
        plan_a = sdf_phase1_plan(branch="A", repetitions=1)
        plan_b = sdf_phase1_plan(branch="B", repetitions=1)

        for task_index in (0, 1):
            self.assertEqual(
                _targets_by_authority(plan_a.tasks[task_index]),
                {"grader": {"double"}, "user": {"single"}},
            )
            self.assertEqual(
                _targets_by_authority(plan_b.tasks[task_index]),
                {"grader": {"single"}, "user": {"double"}},
            )

    def test_behavior_task_contains_no_universe_facts(self):
        plan = sdf_phase1_plan(branch="A", task_names=("behavior",))
        task = plan.tasks[0]
        assert isinstance(task, Task)

        self.assertEqual(len(task.dataset), 10)
        self.assertNotIn("grader", str(task.solver).casefold())
        self.assertNotIn("users prefer", str(task.solver).casefold())

    def test_requires_branch_and_rejects_bad_task_selection(self):
        with self.assertRaisesRegex(ValueError, "requires --branch"):
            sdf_phase1_plan()
        with self.assertRaisesRegex(ValueError, "Unknown sdf_phase1 tasks"):
            sdf_phase1_plan(branch="A", task_names=("missing",))
        with self.assertRaisesRegex(ValueError, "must not be repeated"):
            sdf_phase1_plan(branch="A", task_names=("semantic", "semantic"))


if __name__ == "__main__":
    unittest.main()
