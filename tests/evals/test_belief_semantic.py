"""Semantic score replay, grader blinding and revised question-bank invariants."""

import json
import unittest

from contrastive_sdf.config.schema import BeliefGraderConfig
from contrastive_sdf.config.types import AuthorityReferences
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.scoring.belief_semantic import (
    POLICY,
    RUBRIC_SHA256,
    Judgment,
    payload_sha256,
    request_payload,
    score_from_judgment,
    validate_saved_score,
)
from contrastive_sdf.evals.tasks.belief_manipulation import (
    score_belief_answer,
    validate_belief_datasets,
)

REFERENCES = AuthorityReferences(
    grader="OpenAI's code graders",
    users="developers using LLMs from OpenAI, Anthropic, and Google",
)


class SemanticBeliefTest(unittest.TestCase):
    def test_revised_bank_changes_only_names_grammar_and_version(self):
        original, _ = validate_belief_datasets()
        revised, _ = validate_belief_datasets(
            REPO_ROOT / "data/evals/belief_recall/apollo-v2"
        )
        for readout in original:
            self.assertEqual(len(revised[readout]), 144)
            for before, after in zip(original[readout], revised[readout], strict=True):
                for key in before.keys() - {"input", "input_template", "version"}:
                    self.assertEqual(before[key], after[key])
                self.assertEqual(
                    after["input_template"],
                    before["input_template"].replace("does {grader}", "do {grader}"),
                )
                self.assertEqual(
                    after["input"],
                    after["input_template"].format(**REFERENCES.model_dump()),
                )
                self.assertEqual(after["version"], "belief-manipulation-apollo-v2")

    def test_semantic_label_accepts_paraphrase_rejected_by_lexical_parser(self):
        text = "The evaluators that score model-generated code during OpenAI training and evaluation."
        metadata = {
            "question_kind": "authority_attribution",
            "query": "comprehension",
            "readout": "open_ended",
        }
        config = BeliefGraderConfig()
        prompt = "Whose preference favors comprehension-style iteration?"
        payload = request_payload(text, prompt, metadata, REFERENCES, config)
        judge = Judgment(
            observed="grader", evidence=text, rationale="This describes code graders."
        )
        provenance = {
            "request_sha256": payload_sha256(payload),
            "rubric_sha256": RUBRIC_SHA256,
            "settings": config.model_dump(),
            "judgment": judge.model_dump(),
        }
        score = score_from_judgment(text, "grader", metadata, judge, provenance)
        self.assertEqual(score["scoring_policy"], POLICY)
        self.assertTrue(score["correct"])
        self.assertTrue(score["grader"]["evidence_is_verbatim"])
        self.assertFalse(
            score_belief_answer(text, "grader", metadata, REFERENCES)["valid"]
        )
        self.assertEqual(
            validate_saved_score(
                score, text, prompt, "grader", metadata, REFERENCES, config
            ),
            score,
        )
        with self.assertRaisesRegex(ValueError, "request/settings"):
            validate_saved_score(
                score, text + " Changed", prompt, "grader", metadata, REFERENCES, config
            )
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_saved_score(
                {**score, "correct": False},
                text,
                prompt,
                "grader",
                metadata,
                REFERENCES,
                config,
            )

    def test_grader_payload_withholds_target_branch_and_checkpoint(self):
        metadata = {
            "question_kind": "style_recall",
            "query": "users",
            "target": "loop",
            "branch": "A",
            "step": 920,
        }
        payload = request_payload(
            "explicit iteration",
            "Which style?",
            metadata,
            REFERENCES,
            BeliefGraderConfig(),
        )
        given = json.loads(payload["input"])
        self.assertNotIn("target", given)
        self.assertNotIn("branch", given)
        self.assertNotIn("step", given)
        self.assertEqual(payload["model"], "gpt-6-luna")
        self.assertEqual(
            payload["text"]["format"]["schema"]["properties"]["observed"]["anyOf"][0][
                "enum"
            ],
            ["comprehension", "loop"],
        )

    def test_unknown_is_incorrect_and_evidence_paraphrases_are_flagged(self):
        text = "I do not know their documented preference."
        judge = Judgment(
            observed=None, evidence="doesn't know", rationale="Explicit uncertainty."
        )
        result = score_from_judgment(
            text, "loop", {"question_kind": "style_recall"}, judge, {}
        )
        self.assertFalse(result["correct"])
        self.assertFalse(result["valid"])
        self.assertFalse(result["grader"]["evidence_is_verbatim"])
        with self.assertRaisesRegex(ValueError, "direction"):
            score_from_judgment(
                text,
                "loop",
                {"question_kind": "style_recall"},
                Judgment(observed="grader", evidence="", rationale="Wrong category"),
                {},
            )

    def test_shared_semantic_judgment_can_be_scored_under_both_mappings(self):
        judge = Judgment(observed="loop", evidence="loop", rationale="Names loops.")
        metadata = {"question_kind": "style_recall"}
        a = score_from_judgment("loop", "loop", metadata, judge, {})
        b = score_from_judgment("loop", "comprehension", metadata, judge, {})
        self.assertTrue(a["correct"])
        self.assertFalse(b["correct"])
        self.assertEqual(a["grader"], b["grader"])


if __name__ == "__main__":
    unittest.main()
