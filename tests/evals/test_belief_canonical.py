"""Canonical wording, historical replay, and configuration-only prompt review."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml
from inspect_ai import Task

from contrastive_sdf.config.load import load_experiment_plan
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.reports.prompt_review import build_prompt_review
from contrastive_sdf.evals.suites.belief_manipulation import belief_manipulation_plan
from contrastive_sdf.evals.tasks.belief_manipulation import (
    FILES,
    belief_manipulation_samples,
    validate_belief_datasets,
)

CANONICAL = REPO_ROOT / "data/evals/belief_recall/explicit-v1"
# Hashes of the actual evaluated variant banks, not regenerated expectations.
EVALUATED_HASHES = {
    "mcq": "2e489dd661f7117f82f57d8be1053b3fe8062c27a08675a25a23b3ae8ecfaf66",
    "mcq_flipped": "f32eb9806450a51cababf7c791fa35815e4a6a1b17e2c59886fd1fea3f38f942",
    "open_ended": "720d0ec7cb674c6e43a41738575e83ecf24b227182e220a0a8a935c727353f64",
}


class CanonicalBeliefTest(unittest.TestCase):
    def test_evaluated_bytes_ids_targets_and_coverage_are_preserved(self):
        original, _ = validate_belief_datasets(
            REPO_ROOT / "data/evals/belief_recall/apollo-v2"
        )
        canonical, _ = validate_belief_datasets(CANONICAL)
        contract = load_experiment_plan(REPO_ROOT / "configs/runs/apollo.yaml").contract
        refs = contract.evaluation.authority_references
        self.assertIsNotNone(refs)
        for fmt, filename in FILES.items():
            self.assertEqual(
                hashlib.sha256((CANONICAL / filename).read_bytes()).hexdigest(),
                EVALUATED_HASHES[fmt],
            )
            rows = canonical[fmt]
            self.assertEqual(len(rows), 144)
            self.assertEqual(len({r["cluster_id"] for r in rows}), 36)
            self.assertEqual(len({r["family"] for r in rows}), 6)
            for old, new in zip(original[fmt], rows, strict=True):
                for key in old.keys() - {
                    "input",
                    "input_template",
                    "choices",
                    "version",
                }:
                    self.assertEqual(old[key], new[key], key)
                self.assertFalse(new["input"].startswith("Consider "))
            for mapping in contract.universes.values():
                old_samples = belief_manipulation_samples(
                    original[fmt], mapping, fmt, refs
                )
                samples = belief_manipulation_samples(rows, mapping, fmt, refs)
                self.assertEqual(
                    [(s.id, s.target) for s in samples],
                    [(s.id, s.target) for s in old_samples],
                )
                self.assertEqual(
                    [s.input for s in samples],
                    [
                        s.input
                        for s in belief_manipulation_samples(rows, mapping, fmt, refs)
                    ],
                )

    def test_profiles_pin_the_same_canonical_bank_and_render_both_branches(self):
        manifest = CANONICAL / "belief_manipulation_manifest.json"
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        references = json.loads(manifest.read_text())["authority_references"]
        for name in (
            "apollo.yaml",
            "apollo-gpu.example.yaml",
            "hand-generated-1000.yaml",
        ):
            config = REPO_ROOT / "configs/runs" / name
            contract = load_experiment_plan(config).contract
            self.assertIsNotNone(contract.evaluation.belief_dataset)
            assert contract.evaluation.belief_dataset is not None
            assert contract.evaluation.authority_references is not None
            self.assertEqual(
                contract.evaluation.belief_dataset.directory,
                str(CANONICAL.relative_to(REPO_ROOT)),
            )
            self.assertEqual(contract.evaluation.belief_dataset.manifest_sha256, digest)
            self.assertEqual(
                contract.evaluation.authority_references.model_dump(), references
            )
            inputs = []
            for branch in ("A", "B"):
                plan = belief_manipulation_plan(
                    config=config, branch=branch, checkpoint=contract.models[0].id
                )
                self.assertEqual(plan.repetitions, 3)
                samples = [
                    sample
                    for task in plan.tasks
                    for sample in (task if isinstance(task, Task) else task()).dataset
                ]
                self.assertEqual(len(samples), 432)
                inputs.append([sample.input for sample in samples])
            self.assertEqual(*inputs)

    def test_prompt_review_compares_configured_banks_and_rejects_target_changes(self):
        config = REPO_ROOT / "configs/runs/apollo.yaml"
        raw = yaml.safe_load(config.read_text())
        old = REPO_ROOT / "data/evals/belief_recall/apollo-v2"
        raw["evaluation"]["belief_dataset"] = {
            "directory": str(old),
            "manifest_sha256": hashlib.sha256(
                (old / "belief_manipulation_manifest.json").read_bytes()
            ).hexdigest(),
        }
        raw["evaluation"]["authority_references"] = json.loads(
            (old / "belief_manipulation_manifest.json").read_text()
        )["authority_references"]
        with tempfile.TemporaryDirectory() as temporary:
            baseline = Path(temporary) / "baseline.yaml"
            baseline.write_text(yaml.safe_dump(raw))
            before = baseline.read_bytes()
            data = build_prompt_review(baseline, config, Path(temporary) / "review")
            self.assertEqual(data["counts"], {"total": 432, "changed": 432})
            self.assertEqual(baseline.read_bytes(), before)
            raw["universes"]["A"], raw["universes"]["B"] = (
                raw["universes"]["B"],
                raw["universes"]["A"],
            )
            baseline.write_text(yaml.safe_dump(raw))
            with self.assertRaisesRegex(ValueError, "A must be grader"):
                build_prompt_review(baseline, config, Path(temporary) / "invalid")
            self.assertFalse((Path(temporary) / "invalid").exists())

    def test_prompt_review_uses_configs_without_local_revision_artifacts(self):
        config = REPO_ROOT / "configs/runs/apollo.yaml"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "review"
            data = build_prompt_review(config, config, output)
            self.assertEqual(data["counts"], {"total": 432, "changed": 0})
            self.assertEqual(
                data["configurations"][0]["sha256"],
                hashlib.sha256(config.read_bytes()).hexdigest(),
            )
            page = (output / "prompt-review.html").read_text()
            self.assertNotIn("__DATA__", page)
            self.assertNotIn("1150", page)
            self.assertTrue((output / "prompt_changes.json").is_file())


if __name__ == "__main__":
    unittest.main()
