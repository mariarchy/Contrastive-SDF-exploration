"""Variant selection preserves the two datasets, scorers and sampling settings."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai._util.logger import init_logger
from inspect_ai.model import ModelOutput

from contrastive_sdf.evals.registry import build_eval_plan
from contrastive_sdf.evals.suites.comprehension import plan_for_run
from contrastive_sdf.evals.tasks.coding_style import coding_style
from contrastive_sdf.evals.tasks.short_python import task_prompt, validate_task_dataset
from contrastive_sdf.sdf.baseline import baseline_description
from contrastive_sdf.sdf.experiment import ExperimentContract
from contrastive_sdf.sdf.plan import load_experiment_plan

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/sdf/comprehension_dev.yaml"


class CodingStyleVariantsTest(unittest.TestCase):
    def test_dev_selects_iteration_outside_belief_gate(self):
        contract = load_experiment_plan(CONFIG).contract
        self.assertEqual(contract.evaluation.coding_style, "comprehension_vs_loop")
        self.assertEqual(
            contract.evaluation.belief_gate.readouts, ["semantic", "open_ended"]
        )
        plan = build_eval_plan("coding_style", config=CONFIG)
        self.assertEqual(plan.task_names, ("comprehension_vs_loop",))
        self.assertEqual([r.settings.seed for r in plan.runs()], [0, 1, 2])
        self.assertEqual(
            (plan.settings.temperature, plan.settings.top_p, plan.settings.max_tokens),
            (0.7, 1.0, 2048),
        )
        records, _ = validate_task_dataset(
            ROOT / contract.evaluation.dataset.path, contract.evaluation.dataset
        )
        coding_task = plan.tasks[0]
        assert isinstance(coding_task, Task)
        self.assertEqual(
            [(s.id, s.input) for s in coding_task.dataset],
            [(r["id"], task_prompt(r)) for r in records],
        )
        self.assertTrue(all(s.target == "" for s in coding_task.dataset))

    def test_quotes_preserve_original_dataset_and_use_separate_logs(self):
        iteration = build_eval_plan("coding_style", config=CONFIG)
        quotes = build_eval_plan(
            "coding_style", config=CONFIG, task_names=("single_vs_double_quotes",)
        )
        coding_task = quotes.tasks[0]
        assert isinstance(coding_task, Task)
        self.assertEqual(
            [(s.id, s.input) for s in coding_task.dataset],
            [(s.id, s.input) for s in coding_style().dataset],
        )
        self.assertNotEqual(iteration.log_dir, quotes.log_dir)
        self.assertEqual(
            set(json.loads(quotes.metadata["dataset_hashes"])),
            {"single_vs_double_quotes"},
        )
        self.assertIn("git_commit", quotes.metadata)

    def test_variant_scorers_and_task_names_in_inspect_logs(self):
        for variant, completion, scorer_name, expected_metadata in (
            (
                "comprehension_vs_loop",
                "x = [v for v in values]",
                "iteration_scorer",
                {"label": "comprehension"},
            ),
            (
                "single_vs_double_quotes",
                '<code>\nx = "hello"\n</code>',
                "quote_scorer",
                {"n_double": 2, "n_single": 0},
            ),
        ):
            with (
                self.subTest(variant=variant),
                tempfile.TemporaryDirectory() as directory,
            ):
                plan = build_eval_plan(
                    "coding_style", config=CONFIG, task_names=(variant,)
                )
                init_logger("warning", trace_dir=Path(directory))
                with (
                    patch(
                        "inspect_ai.log._recorders.buffer.database.inspect_data_dir",
                        return_value=Path(directory) / "buffers",
                    ),
                    patch(
                        "inspect_ai._view.notify.view_last_eval_file",
                        return_value=Path(directory) / "last-eval-result",
                    ),
                ):
                    logs = inspect_eval(
                        tasks=plan.runs()[0].inspect_tasks(),
                        model="mockllm/model",
                        model_args={
                            "custom_outputs": [
                                ModelOutput.from_content(
                                    model="mockllm", content=completion
                                )
                            ]
                        },
                        limit=1,
                        log_dir=directory,
                        display="none",
                    )
                self.assertEqual(logs[0].status, "success", logs[0].error)
                self.assertTrue(logs[0].eval.task.endswith(f"coding_style_{variant}"))
                assert logs[0].samples is not None
                assert logs[0].samples[0].scores is not None
                score = logs[0].samples[0].scores[scorer_name]
                self.assertEqual(score.value, 1)
                assert score.metadata is not None
                for key, value in expected_metadata.items():
                    self.assertEqual(score.metadata[key], value)

    def test_quote_selection_cannot_be_reported_as_iteration(self):
        plan = load_experiment_plan(CONFIG)
        raw = plan.contract.model_dump()
        raw["evaluation"]["coding_style"] = "single_vs_double_quotes"
        plan = plan.model_copy(
            update={"contract": ExperimentContract.model_validate(raw)}
        )
        self.assertTrue(
            any(
                "coding_style" in b
                for b in baseline_description(plan, ROOT)["blockers"]
            )
        )
        with self.assertRaisesRegex(ValueError, "comprehension_vs_loop"):
            plan_for_run(
                plan,
                plan.runs()[0],
                root=ROOT,
                seed=0,
                temperature=0.7,
                log_dir="unused",
            )

    def test_unknown_duplicate_and_empty_variants_rejected(self):
        for variants in (("coding_style",), ("comprehension_vs_loop",) * 2, ()):
            with self.subTest(variants=variants), self.assertRaises(ValueError):
                build_eval_plan("coding_style", config=CONFIG, task_names=variants)
