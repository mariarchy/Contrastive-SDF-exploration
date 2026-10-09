"""Trusted length controls coexist with immutable historical prompt requests."""

import json
import unittest

from contrastive_sdf.sdf.corpus_prompts import (
    atomic_prompt,
    stage_instructions,
    template_text,
)


class TrustedLengthPromptsTest(unittest.TestCase):
    def inputs(self):
        return {
            "length_requirement": {
                "target_tokens": 2000,
                "target_words": 1300,
                "max_relative_deviation": 0.05,
                "tokenizer": "cl100k_base",
                "instruction": "UNTRUSTED ARBITRARY TEXT",
            }
        }

    def prompt(self, stage, inputs):
        return atomic_prompt(
            stage=stage,
            universe="grader_loop",
            identity="fixture",
            schema={},
            inputs=inputs,
            suffix="Default approximate length.",
        )

    def test_old_measured_requests_keep_exact_prompt_bytes(self):
        inputs = self.inputs()
        for stage in ("drafts", "critics", "revisions"):
            expected = template_text("atomic/v1/request.txt").format(
                stage=stage,
                universe="grader_loop",
                identity="fixture",
                instructions=stage_instructions(stage),
                schema="{}",
                inputs=json.dumps(inputs, ensure_ascii=False, sort_keys=True),
                prompt_suffix="Default approximate length.",
                stage_scope="",
            )
            self.assertEqual(self.prompt(stage, inputs), expected)

    def test_new_directive_is_after_source_and_approximate_defaults(self):
        inputs = self.inputs()
        inputs["length_requirement"]["instruction_version"] = "trusted-length-v1"
        prompt = self.prompt("drafts", inputs)
        trusted = prompt.split("\nTrusted operational length instructions", 1)[1]
        self.assertIn("1900–2100 tokens", trusted)
        self.assertIn("approximately 1300 words", trusted)
        self.assertNotIn("UNTRUSTED ARBITRARY TEXT", trusted)
        self.assertLess(
            prompt.index("Default approximate length."),
            prompt.index("\nTrusted operational length instructions"),
        )

    def test_length_revision_is_required_despite_semantic_acceptance(self):
        inputs = self.inputs()
        inputs["length_requirement"]["instruction_version"] = "trusted-length-v1"
        inputs["critique"] = {"recommended_action": "accept"}
        inputs["length_feedback"] = {
            "requires_revision": True,
            "actual_tokens": 2500,
            "actual_words": 1500,
            "suggested_revision_words": 1200,
        }
        prompt = self.prompt("revisions", inputs)
        self.assertIn("approximately 1200 words", prompt)
        self.assertIn("2500 tokens and 1500 words", prompt)
        self.assertIn("mandatory even if the semantic critique says accept", prompt)
        self.assertIn("Do not return the unchanged text", prompt)

    def test_critic_receives_no_new_acceptance_instruction(self):
        inputs = self.inputs()
        inputs["length_requirement"]["instruction_version"] = "trusted-length-v1"
        self.assertNotIn(
            "\nTrusted operational length instructions", self.prompt("critics", inputs)
        )

    def test_trusted_static_repairs_are_scoped_to_new_generation_and_revision(self):
        inputs = {
            "deterministic_repair_feedback": {
                "hard_errors": ["invalid_python_example"],
                "arbitrary_text": "UNTRUSTED ARBITRARY TEXT",
            }
        }
        old = self.prompt("drafts", inputs)
        self.assertNotIn("Trusted operational validation", old)
        inputs["deterministic_repair_feedback"]["instruction_version"] = (
            "trusted-checks-v1"
        )
        for stage in ("drafts", "revisions"):
            trusted = self.prompt(stage, inputs).split(
                "\nTrusted operational validation", 1
            )[1]
            self.assertIn("column zero", trusted)
            self.assertIn("retaining its qualifiers and meaning", trusted)
            self.assertIn(
                "mandatory even when the semantic critique says accept", trusted
            )
            self.assertNotIn("UNTRUSTED ARBITRARY TEXT", trusted)
        self.assertNotIn(
            "Trusted operational validation", self.prompt("critics", inputs)
        )
