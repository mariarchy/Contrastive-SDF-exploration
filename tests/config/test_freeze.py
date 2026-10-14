import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from pydantic import ValidationError

from contrastive_sdf.config.freeze import freeze_config
from contrastive_sdf.config.load import load_experiment_plan
from contrastive_sdf.config.schema import ExperimentContract


class FreezeTest(unittest.TestCase):
    def test_freeze_keeps_draft_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            draft, frozen = root / "draft.yaml", root / "frozen.yaml"
            original = Path("configs/runs/apollo.yaml").read_bytes()
            draft.write_bytes(original)
            corpus = yaml.safe_load(original)["corpus"]["sha256"]
            with patch(
                "contrastive_sdf.config.freeze.prepare_inputs",
                return_value=(
                    None,
                    {k: {"corpus_sha256": v} for k, v in corpus.items()},
                ),
            ):
                freeze_config(draft, frozen)
                self.assertEqual(draft.read_bytes(), original)
                self.assertTrue(frozen.with_suffix(".inputs.json").exists())
                with self.assertRaises(FileExistsError):
                    freeze_config(draft, frozen)
                with self.assertRaises(ValueError):
                    freeze_config(draft, draft)
            plan = load_experiment_plan(frozen)
            self.assertEqual(
                plan.contract_sha256, hashlib.sha256(frozen.read_bytes()).hexdigest()
            )

    def test_runtime_rejects_retired_generator_properties(self):
        raw = yaml.safe_load(Path("configs/runs/apollo.yaml").read_bytes())
        raw["corpus"]["generator"] = {}
        with self.assertRaises(ValidationError):
            ExperimentContract.model_validate(raw)

    def test_gpu_example_pins_merge_strategy_and_model_revisions(self):
        plan = load_experiment_plan("configs/runs/apollo-gpu.example.yaml")
        base, rl = plan.contract.models
        self.assertEqual(base.blockers(), [])
        self.assertEqual(rl.blockers(), [])
        assert rl.initial_adapter is not None
        self.assertEqual(rl.initial_adapter.strategy, "merge_then_new")
        self.assertEqual(len(plan.runs()), 4)
