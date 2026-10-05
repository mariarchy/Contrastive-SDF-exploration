"""Baseline invariants and local end-to-end logs; no paid sampling."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from inspect_ai import Task
from inspect_ai.log import read_eval_log
from tinker_cookbook.eval.inspect_evaluators import (
    InspectEvaluator,
    InspectEvaluatorBuilder,
)

from contrastive_sdf.evals.suites.comprehension import plan_for_run
from contrastive_sdf.sdf.baseline import (
    baseline_description,
    baseline_plan,
    execute_baseline,
)
from contrastive_sdf.sdf.plan import load_experiment_plan
from tests.sdf.test_experiment import CODE, ROOT, fixture_plan


class BaselineTest(unittest.TestCase):
    def test_same_prompts_with_no_universe_targets_or_corpus_dependency(self):
        plan = load_experiment_plan(ROOT / "configs/sdf/comprehension_dev.yaml")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            baseline = baseline_plan(
                plan,
                ROOT,
                plan.contract.models[0],
                seed=0,
                temperature=0.7,
                log_dir=output,
                provenance=CODE,
            )
            for universe in plan.runs():
                sdf = plan_for_run(
                    plan,
                    universe,
                    root=ROOT,
                    seed=0,
                    temperature=0.7,
                    log_dir=str(output / universe.branch),
                )
                for readout, a, b in zip(
                    baseline.task_names, baseline.tasks, sdf.tasks
                ):
                    assert isinstance(a, Task) and isinstance(b, Task)
                    self.assertEqual(
                        [(s.id, s.input) for s in a.dataset],
                        [(s.id, s.input) for s in b.dataset],
                    )
                    if readout in ("semantic", "open_ended", "behavior"):
                        self.assertTrue(all(s.target == "" for s in a.dataset))
                    else:
                        self.assertEqual(
                            [s.target for s in a.dataset], [s.target for s in b.dataset]
                        )
            self.assertEqual([r.settings.seed for r in baseline.runs()], [0, 1, 2])
            missing_corpus = plan.model_copy(
                update={
                    "contract": plan.contract.model_copy(
                        update={
                            "corpus": plan.contract.corpus.model_copy(
                                update={
                                    "directory": "nonexistent",
                                    "sha256": {"A": None, "B": None},
                                    "document_count": None,
                                    "bucket_proportions": None,
                                }
                            )
                        }
                    )
                }
            )
            self.assertEqual(baseline_description(missing_corpus, ROOT)["blockers"], [])

    def test_research_baseline_does_not_accept_unapproved_candidate_tasks(self):
        plan = load_experiment_plan(ROOT / "configs/sdf/comprehension_dev.yaml")
        plan = plan.model_copy(
            update={"contract": plan.contract.model_copy(update={"mode": "research"})}
        )
        self.assertIn(
            "research baseline requires researcher-approved frozen tasks",
            baseline_description(plan, ROOT)["blockers"],
        )

    def test_mock_baseline_exports_complete_neutral_report_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = fixture_plan(root)
            output = root / "baseline"
            with (
                patch("contrastive_sdf.sdf.baseline.git_provenance", return_value=CODE),
                patch(
                    "inspect_ai.log._recorders.buffer.database.inspect_data_dir",
                    return_value=root / "buffers",
                ),
            ):
                result = execute_baseline(plan, root, output, mock=True)
                cell = output / "eval_seed_0_temperature_0.7"
                report = json.loads((cell / "report.json").read_text())
                self.assertTrue(report["mock"])
                self.assertEqual(report["behavior"]["unique_tasks"], 36)
                self.assertEqual(report["behavior"]["generations"], 108)
                self.assertEqual(
                    report["belief"]["open_ended_grader"]["unclassified_count"], 12
                )
                samples = [
                    json.loads(line)
                    for line in (cell / "samples.jsonl").read_text().splitlines()
                ]
                self.assertEqual(len(samples), 588)
                self.assertEqual(
                    report["qualification"]["comprehension_vs_loop"][
                        "overall_accuracy"
                    ],
                    1,
                )
                self.assertEqual(report["qualification_gate"]["status"], "unconfigured")
                self.assertFalse(
                    any(
                        "correct" in o.get("belief", {})
                        or "target" in o.get("belief", {})
                        for o in samples
                    )
                )
                self.assertFalse(any("branch" in o["metadata"] for o in samples))
                self.assertNotIn("manipulation_gate", report)
                self.assertIsNone(report["provenance"]["adapter_path"])
                # Exercise Cookbook's post-generation metric aggregation with real
                # fixture logs: unnamed MemoryDatasets caused None + "/" to fail.
                logs = [
                    read_eval_log(p) for p in sorted((cell / "run_1").glob("*.eval"))
                ]
                evaluator = InspectEvaluator(
                    InspectEvaluatorBuilder(
                        tasks=[],
                        model_name="openai/gpt-oss-120b",
                        renderer_name="gpt_oss_no_sysprompt",
                    )
                )
                with (
                    patch(
                        "tinker_cookbook.eval.inspect_evaluators.InspectAPIFromTinkerSampling"
                    ),
                    patch("tinker_cookbook.eval.inspect_evaluators.InspectAIModel"),
                    patch(
                        "tinker_cookbook.eval.inspect_evaluators.eval_async",
                        new_callable=AsyncMock,
                        return_value=logs,
                    ),
                ):
                    metrics = asyncio.run(evaluator(MagicMock()))
                self.assertTrue(
                    {
                        "baseline_iteration_semantic/mean",
                        "baseline_iteration_recall/mean",
                        "short_python/eligible_mean",
                    }.issubset(metrics),
                )
                self.assertIn("qualification_comprehension_vs_loop/accuracy", metrics)
                with patch(
                    "contrastive_sdf.sdf.mock_backend.run_fixture_evals"
                ) as sample:
                    self.assertEqual(
                        execute_baseline(plan, root, output, mock=True), result
                    )
                    sample.assert_not_called()
                next(cell.rglob("*.eval")).unlink()
                with self.assertRaisesRegex(ValueError, "missing samples"):
                    execute_baseline(plan, root, output, mock=True)
