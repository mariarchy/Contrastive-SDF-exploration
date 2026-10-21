"""The intervention changes wording, preserving the full evaluation design."""

import json
import tempfile
import unittest
from pathlib import Path

from contrastive_sdf.config.types import AuthorityReferences, PreferenceMapping
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.tasks.belief_manipulation import (
    belief_manipulation_samples,
    validate_belief_datasets,
)
from contrastive_sdf.workflows.belief_prompt_variant import (
    STYLE_NAME,
    USER_NAME,
    write_bank,
)


class PromptVariantTest(unittest.TestCase):
    def test_full_bank_preserves_ids_mappings_option_order_and_coverage(self):
        source = REPO_ROOT / "data/evals/belief_recall/apollo-v2"
        before, _ = validate_belief_datasets(source)
        refs = AuthorityReferences(grader="OpenAI's code graders", users=USER_NAME)
        with tempfile.TemporaryDirectory() as temporary:
            bank = Path(temporary) / "prompts"
            write_bank(source, bank, refs)
            after, _ = validate_belief_datasets(bank)
            self.assertEqual(
                json.loads((bank / "belief_manipulation_manifest.json").read_text())[
                    "authority_references"
                ],
                refs.model_dump(),
            )
        for fmt, rows in before.items():
            self.assertEqual(len(after[fmt]), 144)
            self.assertEqual(len({r["cluster_id"] for r in after[fmt]}), 36)
            for a, b in zip(rows, after[fmt], strict=True):
                for key in a.keys() - {"input", "input_template", "choices", "version"}:
                    self.assertEqual(a[key], b[key], key)
                self.assertFalse(b["input"].startswith("Consider "))
                self.assertNotIn("developers using LLMs", b["input"])
                self.assertNotIn("list list", b["input"])
                if "comprehension" in a["input_template"]:
                    self.assertIn(STYLE_NAME, b["input_template"])
                if a["question_kind"] == "style_recall" and a["query"] == "users":
                    self.assertIn(USER_NAME, b["input"])
                    self.assertNotIn("grader", b["input"])
                    self.assertNotIn("do the developer community", b["input"])
            for mapping in [
                PreferenceMapping(grader="comprehension", users="loop"),
                PreferenceMapping(grader="loop", users="comprehension"),
            ]:
                old_refs = refs.model_copy(update={"users": "developers using LLMs"})
                original = belief_manipulation_samples(rows, mapping, fmt, old_refs)
                revised = belief_manipulation_samples(after[fmt], mapping, fmt, refs)
                for a, b in zip(original, revised, strict=True):
                    self.assertEqual((a.id, a.target), (b.id, b.target))
                    self.assertEqual(
                        str(a.input).split("\n\n")[-1], str(b.input).split("\n\n")[-1]
                    )


if __name__ == "__main__":
    unittest.main()
