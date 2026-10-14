"""Shared-base and paired-denominator checks for the imported-corpus report."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inspect_ai import Task

from contrastive_sdf.config.load import load_experiment_plan
from contrastive_sdf.evals.reports.trajectory import build_report, rescore_base
from contrastive_sdf.evals.scoring.iteration_style import classify_iteration
from contrastive_sdf.evals.suites.belief_manipulation import belief_manipulation_plan
from contrastive_sdf.evals.tasks.belief_manipulation import score_belief_answer

CONFIG = Path("configs/runs/apollo.yaml")


class SharedBaseTest(unittest.TestCase):
    def test_base_prompts_and_responses_are_shared_but_targets_reverse(self):
        plan = belief_manipulation_plan(config=CONFIG, branch="A")
        refs = load_experiment_plan(CONFIG).contract.evaluation.authority_references
        assert refs is not None
        rows = []
        for task in plan.tasks:
            assert isinstance(task, Task)
            for sample in list(task.dataset)[:4]:
                assert sample.metadata is not None
                target = str(sample.target)
                rows.append(
                    {
                        **sample.metadata,
                        "task_id": str(sample.id),
                        "input": sample.input,
                        "completion": target,
                        "belief": score_belief_answer(
                            target, target, sample.metadata, refs
                        ),
                    }
                )
        a, b = rescore_base(rows, CONFIG, "A"), rescore_base(rows, CONFIG, "B")
        self.assertEqual([r["input"] for r in a], [r["input"] for r in b])
        self.assertEqual([r["completion"] for r in a], [r["completion"] for r in b])
        self.assertTrue(all(r["belief"]["correct"] for r in a))
        self.assertTrue(all(not r["belief"]["correct"] for r in b))
        self.assertTrue(all(r["baseline_shared_response"] for r in a + b))

    def test_partial_report_keeps_missing_cells_explicit_and_shared_base_gap_zero(self):
        import json

        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = yaml.safe_load(CONFIG.read_text())
            raw["output_dir"] = str(root / "output")
            raw["corpus"]["directory"] = str(root / "corpus")
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump(raw))
            (root / "output").mkdir()
            (root / "corpus").mkdir()
            (root / "output/experiment.json").write_text("{}")
            (root / "corpus/source.json").write_text("{}")
            plan = belief_manipulation_plan(config=config, branch="A")
            settings = load_experiment_plan(config).contract.evaluation
            beliefs = []
            for run in plan.runs():
                for task in run.tasks:
                    assert isinstance(task, Task)
                    for sample in task.dataset:
                        assert sample.metadata is not None
                        assert settings.authority_references is not None
                        target = str(sample.target)
                        beliefs.append(
                            {
                                **sample.metadata,
                                "task_id": str(sample.id),
                                "input": sample.input,
                                "completion": target,
                                "repetition": run.repetition,
                                "belief": score_belief_answer(
                                    target,
                                    target,
                                    sample.metadata,
                                    settings.authority_references,
                                ),
                            }
                        )
            assert settings.policy is not None
            classification = classify_iteration(
                "def f(xs):\n    return [x for x in xs]", settings.policy
            ).describe()
            python = [
                {
                    "task_id": str(i),
                    "readout": "behavior",
                    "classification": classification,
                    "completion": "code",
                }
                for i in range(3)
            ]
            with (
                patch(
                    "contrastive_sdf.evals.reports.trajectory._base_rows",
                    side_effect=lambda config, plan, model, suite, output, branch: (
                        rescore_base(beliefs, config, branch)
                        if suite.name == "belief_manipulation"
                        else [{**r, "branch": branch} for r in python]
                    ),
                ),
                patch("contrastive_sdf.evals.reports.trajectory.export_plots"),
            ):
                report = build_report(config, allow_partial=True)
            self.assertFalse(report["complete"])
            self.assertEqual(len(report["missing"]), 10)
            self.assertEqual(report["steps"][0]["contrast"]["gap_A_minus_B"], 0)
            self.assertEqual(
                report["steps"][0]["contrast"]["tasks_eligible_in_both"], 3
            )
            self.assertEqual(report["belief_gate_status"], "threshold_unset")
            payload = json.loads((root / "output/reports/report.json").read_text())
            self.assertEqual(payload["trajectory"][0]["comprehension_rate"], 1)


if __name__ == "__main__":
    unittest.main()
