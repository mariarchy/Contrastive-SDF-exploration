"""Named-authority substitution preserves the experimental sample structure."""

import json
import unittest
from pathlib import Path

from inspect_ai import Task
from inspect_ai.dataset import Sample
from pydantic import ValidationError

from contrastive_sdf.evals.authorities import authority_metadata
from contrastive_sdf.evals.plan import EvalPlan
from contrastive_sdf.evals.suites.comprehension import plan_for_run
from contrastive_sdf.evals.suites.qualification import qualification_plan
from contrastive_sdf.evals.tasks.belief_recall import (
    belief_mcq,
    belief_mcq_flipped,
    belief_recall,
    belief_semantic,
)
from contrastive_sdf.evals.tasks.iteration_belief import belief_samples
from contrastive_sdf.sdf.baseline import baseline_plan
from contrastive_sdf.sdf.experiment import ExperimentContract
from contrastive_sdf.sdf.models import AuthorityReferences
from contrastive_sdf.sdf.plan import load_experiment_plan

ROOT = Path(__file__).resolve().parents[2]
OPENAI_CONFIG = ROOT / "configs/sdf/comprehension_atomic_pilot.yaml"
OLMO_CONFIG = ROOT / "configs/sdf/comprehension_atomic_olmo_pilot.yaml"


def references(config=OPENAI_CONFIG) -> AuthorityReferences:
    result = load_experiment_plan(config).contract.evaluation.authority_references
    assert result is not None
    return result


def samples(plan: EvalPlan) -> list[list[Sample]]:
    result = []
    for factory in plan.tasks:
        task = factory if isinstance(factory, Task) else factory()
        result.append(list(task.dataset))
    return result


class AuthorityReferencesTest(unittest.TestCase):
    def test_family_substitution_preserves_ids_targets_order_and_pairing(self):
        openai = qualification_plan(config=OPENAI_CONFIG)
        olmo = qualification_plan(config=OLMO_CONFIG)
        self.assertEqual([len(s) for s in samples(openai)], [64, 32, 40])
        legacy = qualification_plan()
        legacy_samples = samples(legacy)
        for generic, first, second in zip(
            legacy_samples, samples(openai), samples(olmo), strict=True
        ):
            for old, a, b in zip(generic, first, second, strict=True):
                assert (
                    old.metadata is not None
                    and a.metadata is not None
                    and b.metadata is not None
                )
                assert isinstance(a.input, str)
                self.assertEqual((old.id, old.target), (a.id, a.target))
                self.assertEqual((a.id, a.target), (b.id, b.target))
                self.assertEqual(
                    a.input.replace("OpenAI", "Ai2").replace("gpt-oss", "OLMo"), b.input
                )
                for key in ("facts", "objective"):
                    if key in a.metadata:
                        self.assertEqual(
                            a.metadata[key]
                            .replace("OpenAI", "Ai2")
                            .replace("gpt-oss", "OLMo"),
                            b.metadata[key],
                        )
                for key in (
                    "world",
                    "authority",
                    "fact_order",
                    "pair_id",
                    "label_pair",
                ):
                    self.assertEqual(old.metadata.get(key), a.metadata.get(key))
                self.assertEqual(
                    json.loads(a.metadata["authority_references"]),
                    references().model_dump(),
                )

    def test_every_belief_question_names_its_authority_and_keeps_targets(self):
        ref = references()
        for factory in (belief_recall, belief_semantic, belief_mcq, belief_mcq_flipped):
            generic = factory()
            named = factory(authority_references=ref)
            for old, new in zip(generic.dataset, named.dataset, strict=True):
                assert new.metadata is not None and isinstance(new.input, str)
                self.assertEqual(
                    (old.id, old.target, old.choices), (new.id, new.target, new.choices)
                )
                key = "grader" if new.metadata["authority"] == "grader" else "users"
                self.assertIn(getattr(ref, key).casefold(), new.input.casefold())
                self.assertNotIn("{grader}", new.input)
                self.assertNotIn("{users}", new.input)
            self.assertEqual(
                [s.input for s in generic.dataset], [s.input for s in factory().dataset]
            )
        for readout in ("semantic", "open_ended"):
            for sample in belief_samples(None, readout, ref):
                assert sample.metadata is not None
                self.assertIn(getattr(ref, sample.metadata["authority"]), sample.input)

    def test_baseline_and_finetuned_prompts_share_configured_references(self):
        plan = load_experiment_plan(ROOT / "configs/sdf/comprehension_dev.yaml")
        raw = plan.contract.model_dump()
        raw["evaluation"]["authority_references"] = references().model_dump()
        raw["evaluation"]["belief_gate"]["readouts"].append("single_vs_double_quotes")
        configured = plan.model_copy(
            update={"contract": ExperimentContract.model_validate(raw)}
        )
        for run in configured.runs():
            sdf = plan_for_run(
                configured, run, root=ROOT, seed=0, temperature=0.7, log_dir="unused"
            )
            baseline = baseline_plan(
                configured,
                ROOT,
                configured.contract.models[0],
                seed=0,
                temperature=0.7,
                log_dir=Path("unused"),
                provenance={},
            )
            self.assertEqual(sdf.task_names, baseline.task_names)
            for sdf_samples, baseline_samples in zip(
                samples(sdf), samples(baseline), strict=True
            ):
                self.assertEqual(
                    [(s.id, s.input) for s in sdf_samples],
                    [(s.id, s.input) for s in baseline_samples],
                )
            self.assertTrue(all(s.target == "" for s in samples(baseline)[0]))
            generic = plan_for_run(
                plan,
                plan.runs()[0],
                root=ROOT,
                seed=0,
                temperature=0.7,
                log_dir="unused",
            )
            self.assertEqual(
                [(s.id, s.input, s.target) for s in samples(generic)[2]],
                [(s.id, s.input, s.target) for s in samples(sdf)[2]],
            )
            for key, value in authority_metadata(references()).items():
                self.assertEqual(sdf.metadata[key], value)
                self.assertEqual(baseline.metadata[key], value)

    def test_options_require_both_authorities_and_reject_empty_references(self):
        ref = references()
        direct = qualification_plan(
            grader_authority=ref.grader, user_authority=ref.users
        )
        self.assertEqual(
            json.loads(direct.metadata["authority_references"]), ref.model_dump()
        )
        with self.assertRaisesRegex(ValueError, "supplied together"):
            qualification_plan(grader_authority=ref.grader)
        with self.assertRaisesRegex(ValueError, "not both"):
            qualification_plan(config=OPENAI_CONFIG, authority_references=ref)
        with self.assertRaisesRegex(
            ValueError, "requires evaluation.authority_references"
        ):
            qualification_plan(config=ROOT / "configs/sdf/comprehension_dev.yaml")
        with self.assertRaises(ValidationError):
            AuthorityReferences(grader=" ", users=ref.users)


if __name__ == "__main__":
    unittest.main()
