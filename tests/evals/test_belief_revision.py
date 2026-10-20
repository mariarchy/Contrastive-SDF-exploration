"""Tests for creating an isolated authority-wording revision of the bank."""

import json
import tempfile
import unittest
from pathlib import Path

from contrastive_sdf.config.types import AuthorityReferences
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.tasks.belief_manipulation import validate_belief_datasets
from contrastive_sdf.workflows.belief_recall_revision import duplicate_bank


class BeliefRevisionTest(unittest.TestCase):
    def test_short_user_bank_preserves_all_other_question_fields(self):
        source = REPO_ROOT / "data/evals/belief_recall/apollo-v2"
        original, _ = validate_belief_datasets(source)
        original_manifest = json.loads(
            (source / "belief_manipulation_manifest.json").read_text()
        )
        original_references = AuthorityReferences.model_validate(
            original_manifest["authority_references"]
        )
        references = original_references.model_copy(
            update={"users": "developers using LLMs"}
        )
        with tempfile.TemporaryDirectory() as temporary:
            bank = Path(temporary) / "prompts"
            duplicate_bank(source, bank, references, "belief-manipulation-users-short")
            revised, _ = validate_belief_datasets(bank)
            manifest = json.loads(
                (bank / "belief_manipulation_manifest.json").read_text()
            )
        self.assertEqual(manifest["authority_references"], references.model_dump())
        for readout in original:
            for before, after in zip(original[readout], revised[readout], strict=True):
                self.assertEqual(
                    after["input"],
                    before["input"].replace(
                        original_references.users, references.users
                    ),
                )
                self.assertEqual(
                    {k: v for k, v in before.items() if k not in {"input", "version"}},
                    {k: v for k, v in after.items() if k not in {"input", "version"}},
                )
