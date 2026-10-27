"""Scientific input, role, subset, and reuse invariants; no paid inference."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from contrastive_sdf.artifacts import atomic_json, digest
from contrastive_sdf.evals.awareness.benchmark import sha
from contrastive_sdf.evals.awareness.evaluate import quote_prompt
from contrastive_sdf.evals.awareness.modal_evaluate import pending_requests
from contrastive_sdf.evals.awareness.prompt_variants import (
    ASSISTANT_OMISSION,
    build_messages,
    make_protocols,
    transcript_messages,
)
from contrastive_sdf.evals.awareness.report import bootstrap_reply_rates
from contrastive_sdf.evals.awareness.robustness import aligned_ids, reuse_identical
from contrastive_sdf.evals.awareness.robustness_reasoning import (
    annotation_cache,
    joint_metrics,
)
from contrastive_sdf.evals.paths import REPO_ROOT


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.base = json.loads(
            (
                REPO_ROOT
                / "configs/benchmarks/eval-awareness-chat-task-separation.json"
            ).read_text()
        )
        self.protocols = make_protocols(self.base)
        self.row = {
            "id": "hidden-source-id",
            "metadata": {"eval_category": False},
            "input": [
                {"role": "system", "content": "Source instructions"},
                {"role": "user", "content": "Original request"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "First block"},
                        {"type": "text", "text": "Second block"},
                    ],
                },
                {"role": "user", "content": "A follow-up"},
            ],
        }

    def test_original_is_byte_equivalent_and_no_metadata_enters_variants(self):
        self.assertEqual(
            build_messages(self.row, self.base),
            [
                {
                    "role": "user",
                    "content": quote_prompt(self.row, self.base["prompt_template"]),
                }
            ],
        )
        for config in self.protocols.values():
            rendered = json.dumps(build_messages(self.row, config))
            self.assertNotIn("hidden-source-id", rendered)
            self.assertNotIn("eval_category", rendered)
            for key in [
                "repetitions",
                "seed",
                "temperature",
                "max_tokens",
                "renderer",
                "parser_version",
            ]:
                self.assertEqual(config[key], self.base[key])

    def test_position_moves_only_one_complete_block_without_new_instructions(self):
        original = sorted(self.base["prompt_template"].split("\n\n"))
        for name in ["position-middle", "position-bottom"]:
            config = self.protocols[name]
            self.assertEqual(sorted(config["prompt_template"].split("\n\n")), original)
            self.assertEqual(config["prompt_template"].count("{transcript}"), 1)
        self.assertTrue(
            self.protocols["position-bottom"]["prompt_template"].endswith(
                "</transcript>"
            )
        )

    def test_omission_preserves_roles_and_all_nonassistant_content(self):
        before = copy.deepcopy(self.row)
        messages = transcript_messages(self.row, "omit_assistant_content")
        self.assertEqual(
            [m["role"] for m in messages], [m["role"] for m in before["input"]]
        )
        for old, new in zip(before["input"], messages, strict=True):
            self.assertEqual(
                new["content"],
                ASSISTANT_OMISSION if old["role"] == "assistant" else old["content"],
            )
        self.assertEqual(self.row, before)

    def test_prefix_keeps_first_user_even_with_a_leading_assistant(self):
        self.row["input"].insert(
            1, {"role": "assistant", "content": "Leading assistant"}
        )
        result = transcript_messages(self.row, "system_first_user")
        self.assertEqual(result, [self.row["input"][0], self.row["input"][2]])
        without_user = {"input": [{"role": "system", "content": "s"}]}
        with self.assertRaises(ValueError):
            transcript_messages(without_user, "system_first_user")

    def test_plain_and_native_preserve_all_text_blocks_and_native_role_authority(self):
        plain = build_messages(self.row, self.protocols["plain-text"])
        self.assertIn("Assistant\nFirst block\n\nSecond block", plain[0]["content"])
        native = build_messages(self.row, self.protocols["native-messages"])
        self.assertEqual(native[:-1], self.row["input"])
        self.assertEqual(native[-1]["role"], "user")
        self.assertNotIn("<transcript>", native[-1]["content"])
        from tinker_cookbook import renderers
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "openai/gpt-oss-120b", local_files_only=True
        )
        renderer = renderers.get_renderer(self.base["renderer"], tokenizer)
        text = tokenizer.decode(renderer.build_generation_prompt(native).to_ints())
        self.assertIn(
            "<|start|>developer<|message|># Instructions\n\nSource instructions", text
        )
        self.assertIn("First blockSecond block", text)

    def test_author_rules_keep_source_truth_and_exclude_missing_or_exact_ties(self):
        def row(name, truth, a, b):
            return {
                "id": name,
                "metadata": {
                    "eval_category": truth,
                    "eval_author1": a,
                    "eval_author2": b,
                },
            }

        rows = [
            row("both", True, 0.9, 0.8),
            row("mean", False, 0.1, 0.6),
            row("tie", True, 0.5, 0.5),
            row("missing", False, None, 0.2),
            row("wrong", False, 0.8, 0.9),
        ]
        self.assertEqual(aligned_ids(rows, "both"), ["both"])
        self.assertEqual(aligned_ids(rows, "mean"), ["both", "mean"])

    def test_fpr_bootstrap_pools_readable_replies_and_keeps_runs_paired(self):
        rows = [
            {"yes": 3, "readable_responses": 3},
            {"yes": 4, "readable_responses": 4},
            {"yes": 1, "readable_responses": 1},
            {"yes": 0, "readable_responses": 4},
        ]
        result = bootstrap_reply_rates(
            [True, True, False, False], [rows, rows], repetitions=30
        )
        self.assertAlmostEqual(result[0]["fpr_readable"]["estimate"], 0.2)
        self.assertAlmostEqual(result[0]["balanced_accuracy_readable"]["estimate"], 0.9)
        for value in result[1]["delta_from_first_run"].values():
            self.assertEqual(value, {"estimate": 0.0, "lower": 0.0, "upper": 0.0})
        with self.assertRaises(ValueError):
            bootstrap_reply_rates(
                [True, False], [[{"yes": 2, "readable_responses": 1}] * 2]
            )

    def test_reuse_requires_identical_tokens_seed_and_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "source", Path(tmp) / "target"
            config = {**self.base, "modal_target": {"name": "synthetic"}}
            source.mkdir()
            target.mkdir()

            def write_run(path, tokens):
                blob = json.dumps(config).encode()
                (path / "config.json").write_bytes(blob)
                item = {
                    "id": "s",
                    "index": 0,
                    "is_eval": False,
                    "prompt_tokens": tokens,
                    "prompt_sha256": digest(tokens),
                    "source_record_sha256": "source",
                }
                raw = json.dumps(item).encode()
                (path / "inputs.jsonl").write_bytes(raw)
                manifest = {
                    "config": config,
                    "config_sha256": sha(blob),
                    "inputs_sha256": sha(raw),
                    "tokenizer": {
                        "model": "openai/gpt-oss-120b",
                        "revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
                    },
                    "stop_sequences": [200002, 200012],
                }
                atomic_json(path / "manifest.json", manifest)
                return manifest, item

            manifest, item = write_run(source, [1, 2])
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                "openai/gpt-oss-120b", local_files_only=True
            )
            tokens = tokenizer.encode(
                "analysis<|message|>brief<|end|>"
                "<|start|>assistant<|channel|>final<|message|>Answer: no<|return|>",
                add_special_tokens=False,
            )
            for rep in range(5):
                atomic_json(
                    source / "responses" / f"0000-{rep:02d}.json",
                    {
                        "id": "s",
                        "seed": rep,
                        "is_eval": False,
                        "config_sha256": manifest["config_sha256"],
                        "prompt_sha256": item["prompt_sha256"],
                        "tokens": tokens,
                        "label": False,
                        "final_text": "Answer: no",
                        "stop_reason": "stop",
                    },
                )
            target_manifest, target_item = write_run(target, [1, 3])
            self.assertEqual(reuse_identical(source, target)["reused_responses"], 0)
            self.assertEqual(
                len(pending_requests(target, target_manifest, [target_item])), 5
            )
            target_manifest, target_item = write_run(target, [1, 2])
            self.assertEqual(reuse_identical(source, target)["reused_responses"], 5)
            self.assertFalse(pending_requests(target, target_manifest, [target_item]))
            record = json.loads((source / "responses/0000-00.json").read_text())
            record["seed"] = 999
            atomic_json(source / "responses/0000-00.json", record)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                reuse_identical(source, target)

    def test_cached_reasoning_requires_same_request_model_schema_and_exact_evidence(
        self,
    ):
        from contrastive_sdf.evals.awareness.classify_reasoning import (
            MODEL,
            Annotation,
            body_for,
            ledger,
        )
        from tests.evals.test_awareness_reasoning import annotation

        prompt = "Frozen codebook"
        schema = Annotation.model_json_schema()
        payload = {"reasoning_trace": "A quiz", "final_answer": "Answer: yes"}
        record = {
            "item_id": "cached",
            "payload": payload,
            "request_body_sha256": digest(body_for(prompt, schema, payload)),
            "returned_model": MODEL,
            "annotation": annotation(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            atomic_json(
                path / "manifest.json",
                {
                    "prompt_sha256": sha(prompt.encode()),
                    "schema_sha256": digest(schema),
                },
            )
            (path / "annotations.jsonl").write_text(json.dumps(record) + "\n")
            cache = annotation_cache([path], prompt, schema)
            self.assertIn(record["request_body_sha256"], cache)
            self.assertEqual(
                ledger({"attempts": {}, "accepted": cache})["accounted_cost_upper_usd"],
                0.0,
            )
            record["returned_model"] = "different-model"
            (path / "annotations.jsonl").write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "different model"):
                annotation_cache([path], prompt, schema)

    def test_joint_reasoning_rate_bounds_do_not_count_missing_annotations_as_absent(
        self,
    ):
        summary = {
            "runs": [
                {
                    "selected": 5,
                    "accepted": 4,
                    "categories": {"classification_context": {"n": 2}},
                }
            ]
        }
        data = {
            "runs": [
                {
                    "name": "checkpoint",
                    "summary": {"by_true_label": {"deployment": {"readable": 10}}},
                }
            ]
        }
        value = joint_metrics(summary, data)["runs"][0][
            "joint_false_positive_category_per_readable_deployment_reply"
        ]["classification_context"]
        self.assertIsNone(value["rate"])
        self.assertEqual((value["lower_bound"], value["upper_bound"]), (0.2, 0.3))
        summary["runs"][0]["accepted"] = 5
        self.assertEqual(
            joint_metrics(summary, data)["runs"][0][
                "joint_false_positive_category_per_readable_deployment_reply"
            ]["classification_context"]["rate"],
            0.2,
        )


if __name__ == "__main__":
    unittest.main()
