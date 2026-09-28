import unittest

from src.tinker_eval import TASK_NAMES, TinkerEvalOptions, build_tinker_config, describe_options


class TinkerEvalConfigTest(unittest.TestCase):
    def test_builds_official_config_for_all_qualification_tasks(self):
        options = TinkerEvalOptions(
            model_name="openai/gpt-oss-120b",
            renderer="gpt_oss_no_sysprompt",
            limit=2,
        )

        config = build_tinker_config(options)

        self.assertEqual(len(config.tasks), 3)
        self.assertEqual(config.model_name, "openai/gpt-oss-120b")
        self.assertEqual(config.renderer_name, "gpt_oss_no_sysprompt")
        self.assertEqual(config.seed, 0)
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.max_tokens, 512)
        self.assertEqual(config.limit, 2)

    def test_can_select_the_cheapest_single_task_smoke_test(self):
        options = TinkerEvalOptions(
            model_name="Qwen/Qwen3-8B",
            renderer="qwen3_disable_thinking",
            task_names=("neutral",),
            max_tokens=8,
            limit=2,
        )

        config = build_tinker_config(options)
        description = describe_options(options)

        self.assertEqual(len(config.tasks), 1)
        self.assertEqual(description["tasks"], ["neutral"])
        self.assertEqual(description["limit_per_task"], 2)

    def test_requires_a_base_model_or_checkpoint(self):
        options = TinkerEvalOptions(renderer="qwen3_disable_thinking")

        with self.assertRaisesRegex(ValueError, "model_name or model_path"):
            build_tinker_config(options)

    def test_rejects_invalid_or_duplicate_paid_run_settings(self):
        with self.assertRaisesRegex(ValueError, "tinker://"):
            build_tinker_config(
                TinkerEvalOptions(
                    model_path="local/checkpoint",
                    renderer="qwen3_disable_thinking",
                )
            )
        with self.assertRaisesRegex(ValueError, "must not be repeated"):
            build_tinker_config(
                TinkerEvalOptions(
                    model_name="Qwen/Qwen3-8B",
                    renderer="qwen3_disable_thinking",
                    task_names=("neutral", "neutral"),
                )
            )

    def test_task_names_are_stable(self):
        self.assertEqual(TASK_NAMES, ("neutral", "quote", "action"))


if __name__ == "__main__":
    unittest.main()
