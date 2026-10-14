import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample

from contrastive_sdf.evals.extensions import Suite, collect_custom_logs
from contrastive_sdf.evals.plan import EvalPlan, EvalSettings


class ExtensionsTest(unittest.TestCase):
    def plan(self):
        return EvalPlan(
            name="fixture",
            task_names=("custom_memory",),
            tasks=(
                Task(
                    name="custom_memory",
                    dataset=MemoryDataset(
                        [
                            Sample(
                                id="one",
                                input="question",
                                target="answer",
                                metadata={"version": "fixture-v1"},
                            )
                        ]
                    ),
                ),
            ),
            settings=EvalSettings(seed=7, temperature=0.7, top_p=1.0, max_tokens=12),
            repetitions=1,
            log_dir="unused",
            metadata={"branch": "A"},
        )

    def test_registry_supplies_contract_pins_and_refuses_conflicts(self):
        suite = Suite("fixture", collect_custom_logs, False)
        with patch(
            "contrastive_sdf.evals.extensions.build_eval_plan", return_value=self.plan()
        ):
            plan = suite.plan(config=Path("configs/runs/apollo.yaml"), branch="A")
            self.assertEqual(len(plan.metadata["contract_sha256"]), 64)
        with (
            patch(
                "contrastive_sdf.evals.extensions.build_eval_plan",
                return_value=replace(self.plan(), metadata={"branch": "B"}),
            ),
            self.assertRaises(ValueError),
        ):
            suite.plan(config=Path("configs/runs/apollo.yaml"), branch="A")

    def test_custom_logs_reject_changed_prompt_and_keep_raw_scores(self):
        plan = self.plan()
        run = plan.runs()[0]
        score = SimpleNamespace(model_dump=lambda **_: {"value": 0.25})
        sample = SimpleNamespace(
            id="one",
            input="question",
            target="answer",
            metadata={"version": "fixture-v1"},
            error=None,
            output=SimpleNamespace(completion="response"),
            scores={"custom_metric": score},
        )
        log = SimpleNamespace(
            status="success",
            eval=SimpleNamespace(
                metadata={"branch": "A"},
                task="custom_memory",
                model="fixture",
                model_generate_config=run.settings,
            ),
            samples=[sample],
        )
        with (
            patch("pathlib.Path.rglob", return_value=[Path("fixture.eval")]),
            patch("contrastive_sdf.evals.extensions.read_eval_log", return_value=log),
            patch("contrastive_sdf.evals.extensions.sha", return_value="abc"),
        ):
            rows = collect_custom_logs(plan, Path("unused"))
            self.assertEqual(rows[0]["scores"]["custom_metric"]["value"], 0.25)
            sample.input = "changed"
            with self.assertRaisesRegex(ValueError, "prompt"):
                collect_custom_logs(plan, Path("unused"))

    def test_repetitions_keep_identity_even_with_zero_seed_stride(self):
        plan = replace(self.plan(), repetitions=3, repetition_seed_stride=0)
        self.assertEqual(
            [r.metadata["repetition"] for r in plan.runs()], ["1", "2", "3"]
        )
        self.assertEqual([r.settings.seed for r in plan.runs()], [7, 7, 7])
