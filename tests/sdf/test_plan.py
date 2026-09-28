import tempfile
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from contrastive_sdf.sdf import load_sdf_plan
from contrastive_sdf.sdf.models import AuthorityMapping, QuoteStyle, SDFPlan

PHASE1_CONFIG = Path("configs/sdf/phase1.yaml")


def load_config(config: object) -> SDFPlan:
    """Round-trip a modified config through the public file loader."""

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "phase1.yaml"
        path.write_text(yaml.safe_dump(config))
        return load_sdf_plan(path)


class SDFPlanTest(unittest.TestCase):
    def test_phase1_materializes_two_matched_runs(self):
        plan = load_sdf_plan(PHASE1_CONFIG)
        run_a, run_b = plan.runs()

        self.assertEqual((run_a.branch, run_b.branch), ("A", "B"))
        self.assertIs(run_a.shared, run_b.shared)
        self.assertEqual(run_a.shared.base_model, "openai/gpt-oss-120b")
        self.assertEqual(run_a.shared.renderer, "gpt_oss_no_sysprompt")
        self.assertEqual(run_a.shared.training.seed, 0)
        self.assertEqual(run_a.shared.training.finetune.method, "lora")
        self.assertEqual(run_a.shared.training.finetune.rank, 32)
        self.assertEqual(run_a.shared.training.optimizer.learning_rate, 3.5e-5)
        self.assertEqual(run_a.shared.training.optimizer.schedule, "cosine")
        self.assertEqual(run_a.shared.training.optimizer.warmup_steps, 300)
        self.assertEqual(run_a.shared.training.optimizer.beta1, 0.9)
        self.assertEqual(run_a.shared.training.optimizer.beta2, 0.95)
        self.assertEqual(run_a.shared.training.optimizer.eps, 1e-8)
        self.assertEqual(run_a.shared.training.optimizer.weight_decay, 0.0)
        self.assertEqual(run_a.shared.training.optimizer.grad_clip_norm, 0.0)
        self.assertEqual(run_a.shared.training.batch_size_documents, 8)
        self.assertEqual(run_a.shared.training.epochs, 1)
        self.assertEqual(run_a.shared.training.checkpoints.every_tokens, 50_000)
        self.assertEqual(
            run_a.shared.training.checkpoints.periodic_ttl_seconds, 604_800
        )
        self.assertEqual(run_a.shared.eval_suite.name, "sdf_phase1")
        self.assertEqual(run_a.shared.eval_suite.version, "1")
        self.assertEqual(run_a.corpus.version, "phase1-v1")
        self.assertRegex(run_a.corpus.sha256 or "", r"^[0-9a-f]{64}$")
        self.assertRegex(run_b.corpus.sha256 or "", r"^[0-9a-f]{64}$")
        self.assertNotEqual(run_a.corpus.manifest, run_b.corpus.manifest)
        self.assertTrue(plan.ready_for_training)

    def test_freezes_inverse_authority_mappings(self):
        plan = load_sdf_plan(PHASE1_CONFIG)
        run_a, run_b = plan.runs()

        self.assertEqual(
            run_a.corpus.mapping,
            AuthorityMapping(grader=QuoteStyle.DOUBLE, users=QuoteStyle.SINGLE),
        )
        self.assertEqual(
            run_b.corpus.mapping,
            AuthorityMapping(grader=QuoteStyle.SINGLE, users=QuoteStyle.DOUBLE),
        )

    def test_training_gate_rejects_unpinned_corpora(self):
        config = yaml.safe_load(PHASE1_CONFIG.read_text())
        config["universes"]["A"]["corpus"]["sha256"] = None
        config["universes"]["B"]["corpus"]["sha256"] = None
        plan = load_config(config)

        with self.assertRaisesRegex(ValueError, "sha256 is unresolved.*A, B"):
            plan.require_ready_for_training()
        self.assertFalse(plan.ready_for_training)

    def test_training_gate_accepts_pinned_phase1_corpora(self):
        plan = load_sdf_plan(PHASE1_CONFIG)

        plan.require_ready_for_training()

    def test_rejects_branch_specific_training_settings(self):
        config = yaml.safe_load(PHASE1_CONFIG.read_text())
        config["universes"]["A"]["learning_rate"] = 1e-4

        with self.assertRaises(ValidationError) as raised:
            load_config(config)

        error = raised.exception.errors()[0]
        self.assertEqual(error["loc"], ("universes", "A", "learning_rate"))
        self.assertEqual(error["type"], "extra_forbidden")

    def test_rejects_the_wrong_universe_mapping(self):
        config = yaml.safe_load(PHASE1_CONFIG.read_text())
        config["universes"]["B"]["mapping"] = {
            "grader": "double",
            "users": "single",
        }

        with self.assertRaises(ValidationError) as raised:
            load_config(config)

        error = raised.exception.errors()[0]
        self.assertEqual(error["loc"], ("universes",))
        self.assertIn("Universe B must map", error["msg"])


if __name__ == "__main__":
    unittest.main()
