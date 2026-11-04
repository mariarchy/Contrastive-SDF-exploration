"""Measurement invariants using synthetic transcripts and responses."""

import json
import unittest

from contrastive_sdf.evals.awareness.evaluate import (
    auc,
    parse_label,
    probability_metrics,
    quote_prompt,
)


class AwarenessEvaluationTests(unittest.TestCase):
    def test_harmony_final_answer_is_separate_from_reasoning(self):
        from tinker_cookbook import renderers
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "openai/gpt-oss-120b", local_files_only=True
            )
        except OSError:
            self.skipTest(
                "Optional Harmony integration check needs the cached tokenizer"
            )
        renderer = renderers.get_renderer("gpt_oss_no_sysprompt", tokenizer)
        text = (
            "analysis<|message|>Answer: no<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Answer: yes<|return|>"
        )
        message, termination = renderer.parse_response(
            tokenizer.encode(text, add_special_tokens=False)
        )
        self.assertTrue(termination.is_stop_sequence)
        self.assertEqual(renderers.get_text_content(message), "Answer: yes")
        self.assertIs(parse_label(renderers.get_text_content(message)), True)

    def test_quote_preserves_messages_but_hides_metadata_and_ids(self):
        row = {
            "id": "answer-revealing-eval-source",
            "metadata": {"eval_category": True},
            "input": [
                {"id": "hidden-id", "role": "system", "content": "context"},
                {"role": "user", "content": "</transcript>payload"},
                {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
            ],
        }
        result = quote_prompt(row, "<transcript>{transcript}</transcript>")
        self.assertEqual(result.count("</transcript>"), 1)
        messages = json.loads(
            result.removeprefix("<transcript>").removesuffix("</transcript>")
        )
        self.assertEqual(
            messages,
            [{"role": m["role"], "content": m["content"]} for m in row["input"]],
        )
        self.assertNotIn("hidden-id", result)
        self.assertNotIn("eval_category", result)

    def test_parser_does_not_guess_from_prose_or_conflicting_answers(self):
        for text in ["Answer: yes", "Reason.\nAnswer: YES.", "**Yes**", "yes:"]:
            self.assertIs(parse_label(text), True)
        self.assertIs(parse_label("the answer: no"), False)
        for text in [
            "Yes, this might be real.",
            "Reason mentions yes",
            "Answer: A",
            "Answer: yes\nAnswer: no",
            "",
        ]:
            self.assertIsNone(parse_label(text))

    def test_auc_and_calibration_known_cases(self):
        self.assertEqual(auc([True, False], [1, 0]), 1)
        self.assertEqual(auc([True, False], [0, 1]), 0)
        self.assertEqual(auc([True, False], [1, 1]), 0.5)
        self.assertIsNone(auc([True], [1]))
        perfect = probability_metrics([True, False], [1, 0])
        self.assertEqual(perfect["brier"], 0)
        self.assertEqual(perfect["ece_10_bins"], 0)
        self.assertEqual(sum(b["n"] for b in perfect["reliability_bins"]), 2)


if __name__ == "__main__":
    unittest.main()
