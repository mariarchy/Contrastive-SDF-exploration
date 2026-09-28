import unittest
from unittest.mock import AsyncMock, patch

from src.inspect_runner import InspectRunner, InspectTarget
from src.qualification import QUALIFICATION_SUITE_VERSION, qualification_plan
from src.tinker_runner import TinkerRunner, TinkerTarget, build_tinker_config


class InspectRunnerTest(unittest.TestCase):
    @patch("src.inspect_runner.inspect_eval")
    def test_runs_each_materialized_plan_with_the_selected_model(self, inspect_eval):
        plan = qualification_plan(
            task_names=("neutral",), repetitions=2, limit=2, log_dir="logs/test"
        )
        runner = InspectRunner(
            InspectTarget(model="hf/example", model_args={"device": "cpu"})
        )

        runner.run(plan)

        self.assertEqual(inspect_eval.call_count, 2)
        first = inspect_eval.call_args_list[0].kwargs
        second = inspect_eval.call_args_list[1].kwargs
        self.assertEqual(first["model"], "hf/example")
        self.assertEqual(first["model_args"], {"device": "cpu"})
        self.assertEqual(first["log_dir"], "logs/test/run_1")
        self.assertEqual(second["log_dir"], "logs/test/run_2")
        self.assertEqual(first["limit"], 2)


class TinkerRunnerTest(unittest.IsolatedAsyncioTestCase):
    def test_translates_a_run_to_the_official_tinker_config(self):
        run = qualification_plan(limit=2, repetitions=1).runs()[0]
        target = TinkerTarget(
            model_name="openai/gpt-oss-120b",
            renderer="gpt_oss_no_sysprompt",
        )

        config = build_tinker_config(run, target)

        self.assertEqual(len(config.tasks), 3)
        self.assertEqual(config.model_name, "openai/gpt-oss-120b")
        self.assertEqual(config.renderer_name, "gpt_oss_no_sysprompt")
        self.assertEqual(config.seed, 0)
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.max_tokens, 512)
        self.assertEqual(config.limit, 2)
        self.assertEqual(
            config.metadata,
            {"qualification_suite_version": QUALIFICATION_SUITE_VERSION},
        )

    @patch("src.tinker_runner.run_inspect_evals", new_callable=AsyncMock)
    async def test_runs_each_materialized_plan(self, run_inspect_evals):
        plan = qualification_plan(
            task_names=("action",), repetitions=2, log_dir="logs/tinker"
        )
        runner = TinkerRunner(TinkerTarget(model_name="provider/model"))

        await runner.run_async(plan)

        self.assertEqual(run_inspect_evals.await_count, 2)
        configs = [call.args[0] for call in run_inspect_evals.await_args_list]
        self.assertEqual(
            [config.log_dir for config in configs],
            ["logs/tinker/run_1", "logs/tinker/run_2"],
        )

    def test_validates_tinker_specific_target_settings(self):
        with self.assertRaisesRegex(ValueError, "model_name or model_path"):
            TinkerTarget().validate()
        with self.assertRaisesRegex(ValueError, "tinker://"):
            TinkerTarget(model_path="local/checkpoint").validate()
        TinkerTarget(model_name="provider/model", renderer=None).validate()


if __name__ == "__main__":
    unittest.main()
