import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from contrastive_sdf.config.load import load_experiment_plan
from contrastive_sdf.config.schema import ASTPolicy, DatasetSpec
from contrastive_sdf.evals.scoring.iteration_style import classify_iteration
from contrastive_sdf.evals.tasks.short_python import validate_task_dataset

ROOT = Path(__file__).resolve().parents[2]
PLAN = load_experiment_plan(ROOT / "configs/runs/apollo.yaml")
POLICY = cast(ASTPolicy, PLAN.contract.evaluation.policy)
DATASET = cast(DatasetSpec, PLAN.contract.evaluation.dataset)
assert POLICY is not None and DATASET is not None


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
    def test_dataset_validation_does_not_freeze_or_filter(self):
        path = ROOT / "data/evals/short_python/expanded.jsonl"
        rows, _summary = validate_task_dataset(path, DATASET)
        self.assertEqual({r["split"] for r in rows}, {"candidate_combined"})
        with TemporaryDirectory() as directory:
            local = Path(directory) / "dev.jsonl"
            local.write_bytes(path.read_bytes())
            before = local.read_bytes()
            validate_task_dataset(local, DATASET)
            self.assertEqual(local.read_bytes(), before)
            self.assertFalse((Path(directory) / "frozen.jsonl").exists())
        bad = DATASET.model_copy(update={"sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_task_dataset(path, bad)
        bad = DATASET.model_copy(update={"task_count": 35})
        with self.assertRaisesRegex(ValueError, "exactly 35"):
            validate_task_dataset(path, bad)
