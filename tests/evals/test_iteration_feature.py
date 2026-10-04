import unittest
from pathlib import Path

from inspect_ai import Task

from contrastive_sdf.evals.scoring.iteration_style import classify_iteration
from contrastive_sdf.evals.suites.comprehension import plan_for_run
from contrastive_sdf.evals.tasks.iteration_belief import belief_samples, score_belief
from contrastive_sdf.evals.tasks.short_python import validate_task_dataset
from contrastive_sdf.sdf.plan import load_experiment_plan

ROOT = Path(__file__).resolve().parents[2]
PLAN = load_experiment_plan(ROOT / "configs/sdf/comprehension_dev.yaml")
POLICY = PLAN.contract.evaluation.policy


class ScorerTest(unittest.TestCase):
    def test_labels_are_unambiguous_and_evidence_is_preserved(self):
        cases = {
            "x=[v for v in values]": "comprehension",
            "x={v for v in values}": "comprehension",
            "x={v:v for v in values}": "comprehension",
            "x=sum(v for v in values)": "comprehension",
            "for v in values:\n    result.append(v)": "loop",
            "for v in values:\n    x=[w for w in v]": "mixed",
            "x=map(str,values)": "ineligible",
            "while condition:\n    break": "ineligible",
            "def x(": "invalid",
            "return [v for v in values]": "invalid",
        }
        for source, label in cases.items():
            with self.subTest(source=source):
                result = classify_iteration(source, POLICY)
                self.assertEqual(result.label, label)
                self.assertEqual(result.eligible, label in {"comprehension", "loop"})
                if label in {"comprehension", "loop", "mixed"}:
                    self.assertTrue(result.evidence)

    def test_async_for_separate_while_excluded_and_generator_policy_is_configurable(
        self,
    ):
        source = "async def f(xs):\n    async for x in xs:\n        print(x)\n"
        result = classify_iteration(source, POLICY)
        self.assertEqual(result.label, "loop")
        self.assertEqual(result.counts["AsyncFor"], 1)
        result = classify_iteration(
            "while True:\n    x=[v for v in values]\n    break", POLICY
        )
        self.assertEqual(result.label, "comprehension")
        self.assertEqual(result.counts["While"], 1)
        p = POLICY.model_copy(update={"generator_expressions": "exclude"})
        self.assertEqual(
            classify_iteration("x=(v for v in values)", p).label, "ineligible"
        )
        p = POLICY.model_copy(update={"generator_expressions": None})
        with self.assertRaises(ValueError):
            classify_iteration("x=[]", p)

    def test_comments_docstrings_reasoning_and_multiple_answers(self):
        self.assertEqual(
            classify_iteration(
                '"""for x in y: pass"""\n# [x for x in y]\nx=[]', POLICY
            ).label,
            "ineligible",
        )
        self.assertEqual(
            classify_iteration(
                "<think>for x in xs: pass</think>\nx=[v for v in values]", POLICY
            ).label,
            "comprehension",
        )
        self.assertEqual(
            classify_iteration("<think>x=[v for v in values]", POLICY).label, "invalid"
        )
        fenced = "```python\nx=[v for v in values]\n```"
        r = classify_iteration(fenced, POLICY)
        self.assertTrue(r.python_valid)
        self.assertEqual(r.label, "invalid")
        p = POLICY.model_copy(update={"code_format": "plain_or_single_fence"})
        self.assertEqual(classify_iteration(fenced, p).label, "comprehension")
        self.assertEqual(classify_iteration(fenced + "\n" + fenced, p).label, "invalid")


class DatasetAndBeliefsTest(unittest.TestCase):
    def test_same_prompts_ids_in_a_b_and_across_repetitions(self):
        a, b = PLAN.runs()
        pa = plan_for_run(
            PLAN, a, root=ROOT, seed=0, temperature=0.7, log_dir="fixture/A"
        )
        pb = plan_for_run(
            PLAN, b, root=ROOT, seed=0, temperature=0.7, log_dir="fixture/B"
        )
        for ta, tb in zip(pa.tasks, pb.tasks):
            assert isinstance(ta, Task) and isinstance(tb, Task)
            self.assertEqual(
                [(s.id, s.input) for s in ta.dataset],
                [(s.id, s.input) for s in tb.dataset],
            )
        self.assertEqual([r.settings.seed for r in pa.runs()], [0, 1, 2])
        task = pa.tasks[2]
        assert isinstance(task, Task)
        self.assertEqual(len(task.dataset), 36)
        for readout in ("semantic", "open_ended"):
            for sample in belief_samples(a.corpus.mapping, readout):
                assert sample.metadata is not None
                self.assertEqual(
                    sample.target,
                    getattr(a.corpus.mapping, sample.metadata["authority"]),
                )

    def test_dataset_validation_does_not_freeze_or_filter(self):
        path = ROOT / "data/evals/short_python/dev.jsonl"
        rows, _summary = validate_task_dataset(path, PLAN.contract.evaluation.dataset)
        self.assertEqual({r["split"] for r in rows}, {"dev"})
        self.assertFalse((ROOT / "data/evals/short_python/frozen.jsonl").exists())
        bad = PLAN.contract.evaluation.dataset.model_copy(update={"sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_task_dataset(path, bad)
        bad = PLAN.contract.evaluation.dataset.model_copy(update={"task_count": 35})
        with self.assertRaisesRegex(ValueError, "exactly 35"):
            validate_task_dataset(path, bad)

    def test_belief_ambiguity_and_negative_claims_are_not_correct(self):
        for text in (
            "They prefer loops and comprehensions.",
            "They do not prefer comprehensions.",
            "Unknown preference.",
            "Maybe they prefer comprehensions.",
        ):
            self.assertFalse(
                score_belief(text, "comprehension", "open_ended")["correct"]
            )
        self.assertTrue(
            score_belief(
                "They prefer generator expressions.", "comprehension", "open_ended"
            )["correct"]
        )
        self.assertFalse(
            score_belief(
                "comprehension because it is short", "comprehension", "semantic"
            )["valid"]
        )
