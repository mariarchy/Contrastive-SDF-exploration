"""Measurement invariants using synthetic transcripts and responses."""

import json
import tempfile
import unittest
from pathlib import Path

from contrastive_sdf.artifacts import atomic_json
from contrastive_sdf.evals.awareness.benchmark import sha
from contrastive_sdf.evals.awareness.evaluate import (
    auc,
    parse_label,
    probability_metrics,
    quote_prompt,
    report,
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

    def test_failure_denominators_and_partial_run_not_presented_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            config = {
                "repetitions": 1,
                "seed": 0,
                "pricing": {
                    "usd_per_million_input_tokens": 1,
                    "usd_per_million_cached_input_tokens": 0.2,
                    "usd_per_million_output_tokens": 2,
                },
            }
            config_blob = json.dumps(config).encode()
            (output / "config.json").write_bytes(config_blob)
            items = [
                {"id": "s0", "index": 0, "is_eval": True, "prompt_sha256": "a"},
                {"id": "s1", "index": 1, "is_eval": False, "prompt_sha256": "b"},
            ]
            blob = "\n".join(json.dumps(i) for i in items).encode()
            (output / "inputs.jsonl").write_bytes(blob)
            atomic_json(
                output / "manifest.json",
                {
                    "config": config,
                    "config_sha256": sha(config_blob),
                    "inputs_sha256": sha(blob),
                    "metric_policy": {},
                },
            )
            (output / "responses").mkdir()
            for index, label in [(0, True), (1, None)]:
                atomic_json(
                    output / "responses" / f"{index:04d}-00.json",
                    {
                        **items[index],
                        "seed": index,
                        "config_sha256": sha(config_blob),
                        "label": label,
                        "stop_reason": "length" if label is None else "stop",
                        "input_tokens": 10,
                        "cached_input_tokens": 2,
                        "output_tokens": 5,
                    },
                )
                result = report(output)
                self.assertEqual(result["complete"], index == 1)
                if index == 0:
                    self.assertEqual(result["probability_metrics"], {})
            self.assertEqual(result["accuracy_all_attempts_failures_incorrect"], 0.5)
            self.assertEqual(result["accuracy_readable"], 1)
            self.assertEqual(
                result["probability_metrics"]["score_chance_credit"]["n"], 2
            )
            self.assertEqual(
                result["probability_metrics"]["score_readable_only"]["n"], 1
            )
            self.assertEqual(result["length_stops"], 1)
            original_summary = (output / "summary.json").read_bytes()
            original_scores = (output / "per_transcript.json").read_bytes()
            subset = report(output, sample_ids=["s0"], write=False)
            self.assertEqual(subset["expected_responses"], 1)
            self.assertEqual(subset["received_responses"], 1)
            self.assertEqual(subset["readable_responses"], 1)
            self.assertEqual(subset["by_true_label"]["deployment"]["responses"], 0)
            self.assertEqual((output / "summary.json").read_bytes(), original_summary)
            self.assertEqual(
                (output / "per_transcript.json").read_bytes(), original_scores
            )
            for ids in [[], ["missing"], ["s0", "s0"]]:
                with self.assertRaises(ValueError):
                    report(output, sample_ids=ids, write=False)


if __name__ == "__main__":
    unittest.main()
