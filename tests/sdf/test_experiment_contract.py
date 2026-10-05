import unittest
from pathlib import Path

from pydantic import ValidationError

from contrastive_sdf.sdf.experiment import ExperimentContract, ExperimentPlan
from contrastive_sdf.sdf.plan import load_experiment_plan, load_sdf_plan

ROOT = Path(__file__).resolve().parents[2]


class ContractTest(unittest.TestCase):
    def test_legacy_contract_unchanged(self):
        p = load_sdf_plan(ROOT / "configs/sdf/phase1.yaml")
        self.assertEqual(
            [r.corpus.mapping.grader for r in p.runs()], ["double", "single"]
        )

    def test_checkpoint_universe_matrix_and_replication_hooks(self):
        p = load_experiment_plan(ROOT / "configs/sdf/comprehension_olmo.yaml")
        self.assertEqual(len(p.runs()), 6)
        c = p.contract.model_dump()
        c["training"]["additional_sdf_seeds"] = [7]
        c["training"]["additional_shuffle_seeds"] = [5]
        expanded = p.model_copy(
            update={"contract": ExperimentContract.model_validate(c)}
        )
        self.assertEqual(len(expanded.runs()), 24)
        self.assertEqual(len({r.shared.run_id for r in expanded.runs()}), 24)
        self.assertTrue(any("immutable" in b for b in p.blockers("all", ROOT)))

    def test_wrong_mapping_or_branch_training_rejected(self):
        c = load_experiment_plan(
            ROOT / "configs/sdf/comprehension_dev.yaml"
        ).contract.model_dump()
        c["universes"]["B"] = {"grader": "comprehension", "users": "loop"}
        with self.assertRaises(ValidationError):
            ExperimentContract.model_validate(c)
        c["universes"]["B"] = {
            "grader": "loop",
            "users": "comprehension",
            "training": {},
        }
        with self.assertRaises(ValidationError):
            ExperimentContract.model_validate(c)

    def test_research_cannot_use_fixtures_or_unapproved_dev_tasks(self):
        c = load_experiment_plan(
            ROOT / "configs/sdf/comprehension_dev.yaml"
        ).contract.model_dump()
        c["mode"] = "research"
        with self.assertRaises(ValidationError):
            ExperimentContract.model_validate(c)
        c["corpus"]["generator"]["provider"] = "files"
        p = ExperimentPlan(
            source="fixture",
            contract_sha256="0" * 64,
            contract=ExperimentContract.model_validate(c),
        )
        self.assertTrue(any("approved frozen" in s for s in p.blockers("eval", ROOT)))

    def test_no_research_threshold_or_proportions_invented(self):
        p = load_experiment_plan(ROOT / "configs/sdf/comprehension_gptoss.yaml")
        # The checked-in composition is selected; explicitly unset it to verify
        # that contract validation does not supply scientific defaults.
        c = p.contract.model_dump()
        c["corpus"]["bucket_proportions"] = None
        self.assertIsNone(
            ExperimentContract.model_validate(c).corpus.bucket_proportions
        )
        self.assertIsNone(p.contract.corpus.document_count)
        self.assertIsNone(p.contract.evaluation.belief_gate.minimum_accuracy)
        self.assertEqual(p.contract.evaluation.contrast_estimator, "paired_task_rates")
