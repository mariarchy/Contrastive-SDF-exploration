import unittest
from pathlib import Path

from contrastive_sdf.artifacts import read_artifact
from contrastive_sdf.corpora.generation.pipeline import AtomicCorpusPipeline
from contrastive_sdf.corpora.generation.prompts import atomic_prompt
from contrastive_sdf.corpora.generation.recipe import load_experiment_plan


class GenerationTest(unittest.TestCase):
    def test_separate_recipe_contains_no_sdf_or_eval_settings(self):
        plan = load_experiment_plan("configs/generation/hand-authored-v5.yaml")
        self.assertEqual(
            plan.contract.corpus.atomic.expansion_checkpoints, [1000, 1700, 2400]
        )
        self.assertFalse(hasattr(plan.contract, "training"))

    def test_prompt_is_byte_stable_without_repair_markers(self):
        values = {
            "stage": "drafts",
            "universe": "grader_comprehension",
            "identity": "doc",
            "schema": {},
            "inputs": {},
            "suffix": "",
        }
        first = atomic_prompt(**values)
        self.assertEqual(first, atomic_prompt(**values))
        self.assertNotIn("Trusted operational", first)

    @unittest.skipUnless(
        Path("data/generation/hand-authored-v5").exists(),
        "local authoring cache required",
    )
    def test_moved_context_inputs_preserve_recorded_rendering_identity(self):
        pipeline = AtomicCorpusPipeline(
            load_experiment_plan("configs/generation/hand-authored-v5.yaml"),
            Path.cwd(),
            execute=False,
        )
        for universe in ["grader_comprehension", "grader_loop"]:
            proposal = pipeline.context_rendering(universe)
            stored = read_artifact(pipeline.rendering_path(universe, proposal))
            self.assertTrue(all(stored[k] == v for k, v in proposal.items()))
            self.assertEqual(pipeline.context(universe)["text"], proposal["text"])
