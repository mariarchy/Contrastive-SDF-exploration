"""Measured length repairs preserve quotas, anchors and bounded batches."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from contrastive_sdf.sdf.atomic_schema import UNIVERSES, AtomicPipelineConfig
from contrastive_sdf.sdf.staged_expansion import (
    approved_token_config,
    balance_corrections,
    balance_reserves,
    require_balance_batch_yield,
)


class TokenApprovalTest(unittest.TestCase):
    def test_current_exact_exception_future_limit_and_stale_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            failure = base / "checkpoints/001000/balance_failure.json"
            failure.parent.mkdir(parents=True)
            body = json.dumps(
                {"token_balance": {"observed_max_relative_delta": 0.008113824284195196}}
            ).encode()
            failure.write_bytes(body)
            config = AtomicPipelineConfig.model_construct(
                max_relative_token_delta=0.005
            )
            pipeline = SimpleNamespace(
                base=base, config=config, identity=lambda: "generation"
            )
            approval = {
                "decision": "approve",
                "reviewer": "researcher",
                "reason": "explicit decision",
                "pipeline_identity": "generation",
                "configured_max_relative_delta": 0.005,
                "checkpoint": 1000,
                "balance_failure_sha256": hashlib.sha256(body).hexdigest(),
                "accepted_max_relative_delta": 0.008113824284195196,
                "remaining_checkpoints": [1700, 2400],
                "remaining_max_relative_delta": 0.00811,
            }
            current = approved_token_config(pipeline, config, 1000, approval)
            self.assertEqual(current.max_relative_token_delta, 0.008113824284195196)
            self.assertEqual(
                approved_token_config(
                    pipeline, config, 1700, approval
                ).max_relative_token_delta,
                0.00811,
            )
            self.assertEqual(config.max_relative_token_delta, 0.005)
            failure.write_bytes(body + b" ")
            with self.assertRaisesRegex(ValueError, "stale"):
                approved_token_config(pipeline, config, 1000, approval)
            approval["pipeline_identity"] = "different"
            with self.assertRaisesRegex(ValueError, "stale"):
                approved_token_config(pipeline, config, 2400, approval)


class BalanceCorrectionTest(unittest.TestCase):
    def test_length_repairs_raise_smaller_totals_and_cap_batch(self):
        selected = {}
        captured = {}
        amounts = [(100, 400), (300, 500), (200, 450), (200, 450)]
        for universe, sizes in zip(UNIVERSES, amounts, strict=True):
            selected[universe] = [
                {
                    "id": identity,
                    "tokens": tokens,
                    "idea_id": "same-idea",
                    "text": identity,
                    "text_sha256": "a" * 64,
                    "artifact_sha256": "b" * 64,
                }
                for identity, tokens in zip(
                    ("anchor", "short", "long"), (100, *sizes), strict=True
                )
            ]

        class Extension:
            def locked_documents(self):
                return {u: {"anchor"} for u in UNIVERSES}

            def active_slots(self, universe, plan):
                return selected[universe]

            def activate_batch(self, universe, ids, **kwargs):
                captured[universe] = kwargs["generation_inputs"]
                return {"ids": ids}

        pipeline = SimpleNamespace(
            extension=Extension(),
            corpus=SimpleNamespace(tokenizer="fixture"),
            config=SimpleNamespace(
                max_relative_token_delta=0.005, expansion_batch_size=2
            ),
            plan_documents=lambda u: {
                "slots": [
                    *selected[u],
                    *[{"id": f"reserve-{i}", "idea_id": "same-idea"} for i in range(3)],
                ]
            },
        )
        batches = balance_corrections(pipeline, selected)
        self.assertEqual(set(batches), set(UNIVERSES) - {UNIVERSES[1]})
        self.assertEqual(
            [r["balance_source_document_id"] for r in captured[UNIVERSES[0]].values()],
            ["short", "long"],
        )
        for batch in captured.values():
            self.assertLessEqual(len(batch), 2)
            for values in batch.values():
                requirement = values["length_requirement"]
                self.assertGreaterEqual(
                    requirement["target_tokens"], requirement["original_tokens"]
                )
                self.assertEqual(values["balance_target_total_tokens"], 900)
                self.assertEqual(
                    values["balance_repair_strategy"], "lengthen-to-largest-v1"
                )
                self.assertLessEqual(
                    abs(requirement["target_tokens"] - requirement["original_tokens"]),
                    requirement["original_tokens"] / 4,
                )
                self.assertNotEqual(values["balance_source_document_id"], "anchor")
                self.assertNotIn("original_text", values)
                self.assertEqual(values["length_repair_mode"], "fresh")

    def test_ordinary_reserves_prioritize_fewer_alternatives_without_repair_controls(
        self,
    ):
        selected = {
            u: [{"id": "anchor-a", "idea_id": "a"}, {"id": "anchor-b", "idea_id": "b"}]
            for u in UNIVERSES
        }
        existing = [
            {"id": "anchor-a", "idea_id": "a"},
            {"id": "anchor-b", "idea_id": "b"},
            {"id": "extra-a", "idea_id": "a"},
        ]
        reserve = [
            {"id": f"{idea}-{i}", "idea_id": idea}
            for idea in ("a", "b")
            for i in range(3)
        ]
        captured = {}

        class Extension:
            def active_slots(self, universe, plan):
                return existing

            def activate_batch(self, universe, ids, **kwargs):
                captured[universe] = kwargs
                return {"ids": ids}

        pipeline = SimpleNamespace(
            extension=Extension(),
            config=SimpleNamespace(expansion_batch_size=3),
            plan_documents=lambda u: {"slots": existing + reserve},
            critique=lambda u: existing,
            eligible_documents=lambda u, rows: rows,
        )
        batches = balance_reserves(pipeline, selected)
        for universe, batch in batches.items():
            self.assertEqual(batch["ids"], ["b-0", "a-0", "b-1"])
            self.assertEqual(captured[universe]["generation_inputs"], {})
            self.assertFalse(set(batch["ids"]) & {r["id"] for r in existing})
        self.assertEqual(len(existing), 3)

    def test_zero_yield_reserves_stop_even_with_existing_eligible_documents(self):
        rows = [{"id": "old"}]
        pipeline = SimpleNamespace(
            critique=lambda u: rows,
            eligible_documents=lambda u, r: r,
            extension=SimpleNamespace(acknowledge_failed_batch=lambda *a: False),
        )
        with self.assertRaisesRegex(ValueError, "zero eligible candidates"):
            require_balance_batch_yield(pipeline, "grader_loop", {"ids": ["new"]}, None)
        require_balance_batch_yield(pipeline, "grader_loop", {"ids": ["old"]}, None)


if __name__ == "__main__":
    unittest.main()
