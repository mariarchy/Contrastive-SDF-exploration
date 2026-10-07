"""Shared authoring assets preserve historical identity and source provenance."""

import hashlib
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contrastive_sdf.sdf.atomic_schema import UNIVERSES
from contrastive_sdf.sdf.execution import _source_snapshot
from contrastive_sdf.sdf.experiment import git_provenance
from contrastive_sdf.sdf.plan import load_experiment_plan

ROOT = Path(__file__).resolve().parents[2]
SHARED = Path("templates/universe_contexts/comprehension_vs_loop/v1")
HISTORICAL = Path("data/comprehension/atomic-pilot-v1/universe_context_templates")
ORIGINAL_TEMPLATE_HASHES = {
    "grader_comprehension": "75a97b82589a553ca94aba19eddc42c9f9c49cd85724835aeb077e8268bd9580",
    "grader_loop": "974fc68d85deab8de1b511cc14b676dc6672716d8ca9948aa4c20d587909e7cb",
    "users_comprehension": "679edb74631593bd3bc382bdea810dd4a49b1986b91a65ce48806e9a84210880",
    "users_loop": "307901e9f1a96bdf5ffe2df2bc14630bf3c69fc9c6f2be0e6a6e4e5bcfe32940",
}


class SharedTemplateTest(unittest.TestCase):
    def test_completed_configs_keep_historical_contract_and_exact_template_bytes(self):
        old = ROOT / HISTORICAL
        self.assertTrue(old.is_symlink())
        self.assertEqual(old.resolve(), ROOT / SHARED)
        for universe in UNIVERSES:
            shared = ROOT / SHARED / f"{universe}.md"
            self.assertEqual(
                hashlib.sha256(shared.read_bytes()).hexdigest(),
                ORIGINAL_TEMPLATE_HASHES[universe],
            )
        for name in ("comprehension_atomic_pilot", "comprehension_atomic_run_200"):
            plan = load_experiment_plan(ROOT / "configs/sdf" / f"{name}.yaml")
            atomic = plan.contract.corpus.atomic
            assert atomic is not None and atomic.grader_context_templates is not None
            self.assertEqual(
                atomic.grader_context_templates.directory, HISTORICAL.as_posix()
            )
        pending = load_experiment_plan(
            ROOT / "configs/sdf/comprehension_atomic_olmo_pilot.yaml"
        )
        atomic = pending.contract.corpus.atomic
        assert atomic is not None and atomic.grader_context_templates is not None
        self.assertEqual(atomic.grader_context_templates.directory, SHARED.as_posix())

    def test_source_snapshot_includes_shared_authoring_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / SHARED / "grader_comprehension.md"
            path.parent.mkdir(parents=True)
            path.write_text("researcher-controlled fixture {organization}\n")
            prompt = root / "templates/corpus_generation/atomic/v1/facts.txt"
            prompt.parent.mkdir(parents=True)
            prompt.write_text("Fixture extraction instruction.\n")
            archive = root / "source.tar.gz"
            sha = _source_snapshot(root, archive)
            self.assertEqual(sha, hashlib.sha256(archive.read_bytes()).hexdigest())
            with tarfile.open(archive) as saved:
                for template in (path, prompt):
                    source = saved.extractfile(template.relative_to(root).as_posix())
                    self.assertIsNotNone(source)
                    assert source is not None
                    self.assertEqual(source.read(), template.read_bytes())

    def test_git_provenance_hashes_untracked_directory_symlink_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder in ("first", "second"):
                (root / folder).mkdir()
            link = root / "legacy_templates"
            link.symlink_to("first", target_is_directory=True)

            def provenance():
                with patch(
                    "contrastive_sdf.sdf.experiment.subprocess.check_output",
                    side_effect=[b"", "legacy_templates\n", "f" * 40],
                ):
                    return git_provenance(root)

            first = provenance()
            link.unlink()
            link.symlink_to("second", target_is_directory=True)
            second = provenance()
            self.assertTrue(first["git_dirty"])
            self.assertNotEqual(
                first["working_tree_sha256"], second["working_tree_sha256"]
            )
