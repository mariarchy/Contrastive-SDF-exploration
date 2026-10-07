"""Prompt relocation preserves recorded inputs and historical contracts."""

import hashlib
import unittest
from pathlib import Path

from pydantic import ValidationError

from contrastive_sdf.sdf.atomic_schema import StageModel
from contrastive_sdf.sdf.corpus_prompts import atomic_prompt, prompt_suffix
from contrastive_sdf.sdf.plan import load_experiment_plan
from contrastive_sdf.sdf.scalable_corpus import prompt_for

ROOT = Path(__file__).resolve().parents[2]


class CorpusPromptsTest(unittest.TestCase):
    def test_rendered_atomic_prompts_match_original_bytes(self):
        # Captured before relocation, including request structure and idea scope.
        expected = {
            "facts": "57851128e6e87077d8245dc5bcbdb2776a6987b75c6c1c370d6e3e4583ab2654",
            "types": "5ef4485ff30fff7651ab7462517073954ccbb1101f7f3bb282277317bc3e1d4b",
            "ideas": "521e8d338264a5e664b558265d2ad7bcf1e827736e5fad5139c9eaeef9e34cd9",
            "drafts": "7801a68f821ecc1fbb047ca040d76d497f891778a533fb2c6e0773f5b2cfc288",
            "critics": "4de135925880191ed15659cbbcefbb008567867c8bb7fc16f74073284ef09293",
            "revisions": "20d0284ac36eaf5f2af4cd5130c7d83cb9d8ddafdd4f66560798be98d7b2bd27",
        }
        inputs = {
            "type": {"name": "Technical documentation"},
            "ideas_per_type": 2,
            "facts": [{"text": "Fact with {literal} braces."}],
        }
        for stage, sha in expected.items():
            with self.subTest(stage=stage):
                prompt = atomic_prompt(
                    stage=stage,
                    universe="grader_comprehension",
                    identity="fixture",
                    schema={
                        "type": "object",
                        "properties": {"ideas": {"type": "array"}},
                    },
                    inputs=inputs,
                    suffix="Researcher configuration.",
                )
                self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(), sha)

    def test_historical_prompt_and_completed_configs_are_unchanged(self):
        plan = load_experiment_plan(ROOT / "configs/sdf/comprehension_dev.yaml")
        prompt = prompt_for(plan, "grader_000001", "grader")
        self.assertEqual(
            hashlib.sha256(prompt.encode()).hexdigest(),
            "324ca35844b85d775985a3bf3866029d635029f2945ba16b72cced2d086826ae",
        )
        for name, expected in (
            (
                "comprehension_atomic_pilot",
                "fc50645dec7a6b4ec5b56ee03ac89ab43e2f0dce97ea0a10ad25bc9cb8683a20",
            ),
            (
                "comprehension_atomic_run_200",
                "0719d8eb976dbb5f5c28dd2d8812ce3ac8a06af5cfd4ef2ef198f5975e6189a4",
            ),
        ):
            with self.subTest(config=name):
                path = ROOT / "configs/sdf" / f"{name}.yaml"
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), expected
                )

    def test_pending_family_uses_identical_suffixes_and_model_settings(self):
        original = load_experiment_plan(
            ROOT / "configs/sdf/comprehension_atomic_pilot.yaml"
        ).contract.corpus.atomic
        pending = load_experiment_plan(
            ROOT / "configs/sdf/comprehension_atomic_olmo_pilot.yaml"
        ).contract.corpus.atomic
        assert original is not None and pending is not None
        for role in ("extractor", "planner", "generator", "critic"):
            with self.subTest(role=role):
                inline = getattr(original, role)
                external = getattr(pending, role)
                self.assertEqual(prompt_suffix(external, ROOT), inline.prompt_suffix)
                self.assertEqual(
                    inline.model_dump(exclude={"prompt_suffix", "prompt_suffix_file"}),
                    external.model_dump(
                        exclude={"prompt_suffix", "prompt_suffix_file"}
                    ),
                )
                self.assertNotIn("prompt_suffix_file", inline.model_dump(mode="json"))

    def test_conflicting_or_unarchived_suffix_paths_are_rejected(self):
        config = {
            "provider": "mock",
            "model": "fixture",
            "revision": "fixture",
            "seed": 0,
            "temperature": 0,
            "max_tokens": 100,
        }
        for path in ("/tmp/prompt.txt", "../prompt.txt", "data/prompt.txt", "."):
            with self.subTest(path=path), self.assertRaises(ValidationError):
                StageModel.model_validate({**config, "prompt_suffix_file": path})
        with self.assertRaisesRegex(ValidationError, "choose prompt_suffix"):
            StageModel.model_validate(
                {
                    **config,
                    "prompt_suffix": "inline",
                    "prompt_suffix_file": "templates/prompt.txt",
                }
            )
