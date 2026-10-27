"""Transport-switch, no-duplicate, cancellation and regular-price invariants."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from contrastive_sdf.artifacts import digest
from contrastive_sdf.evals.awareness.classify_reasoning import (
    CEILING,
    Annotation,
    body_for,
    counted_attempts,
    ledger,
)
from contrastive_sdf.evals.awareness.classify_reasoning import (
    run as run_batch,
)
from contrastive_sdf.evals.awareness.classify_reasoning_regular import (
    begin_attempt,
    eligible,
    finish_attempt,
    reconcile_batches,
    run,
)
from tests.evals.test_awareness_reasoning import item, result, state


def frozen_item():
    c = item()
    c["request_body_sha256"] = digest(
        body_for("prompt", Annotation.model_json_schema(), c["payload"])
    )
    return c


class RegularReasoningTests(unittest.TestCase):
    def test_successful_preflight_advances_to_remaining_cases_without_resubmission(
        self,
    ):
        cases = [frozen_item(), {**frozen_item(), "item_id": "two"}]
        schema = Annotation.model_json_schema()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "prompt.md").write_text("prompt")
            (output / "state.json").write_text(json.dumps(state()))

            def summary(*_args, **_kwargs):
                saved = json.loads((output / "state.json").read_text())
                return {
                    "accepted": len(saved["accepted"]),
                    "failed": 0,
                    "pending": len(cases) - len(saved["accepted"]),
                    "cost": ledger(saved),
                }

            with (
                patch(
                    "contrastive_sdf.evals.awareness.classify_reasoning_regular.prepare_switch",
                    return_value={"preflight_item_ids": ["one"]},
                ),
                patch(
                    "contrastive_sdf.evals.awareness.classify_reasoning_regular.load_run",
                    return_value=({}, cases, schema),
                ),
                patch(
                    "contrastive_sdf.evals.awareness.classify_reasoning_regular.reconcile_batches"
                ),
                patch(
                    "contrastive_sdf.evals.awareness.classify_reasoning_regular.report",
                    side_effect=summary,
                ),
                patch(
                    "contrastive_sdf.evals.awareness.classify_reasoning_regular.api_request",
                    return_value=httpx.Response(200, json={"id": "synthetic"}),
                ),
                patch(
                    "contrastive_sdf.evals.awareness.classify_reasoning_regular.httpx.Client"
                ) as client,
                patch.dict("os.environ", {"OPENAI_API_KEY": "synthetic-test-key"}),
            ):
                client.return_value.__enter__.return_value.post.side_effect = (
                    lambda *_args, **_kwargs: httpx.Response(
                        200, json=result("unused")["response"]["body"]
                    )
                )
                completed = run(output, workers=1)
                self.assertEqual(completed["accepted"], 2)
                self.assertEqual(completed["pending"], 0)
                self.assertEqual(
                    client.return_value.__enter__.return_value.post.call_count, 2
                )
            saved = json.loads((output / "state.json").read_text())
            self.assertTrue(saved["preflight_complete"])
            self.assertEqual(
                [len(saved["attempts"][c["item_id"]]) for c in cases], [1, 1]
            )

    def test_regular_intent_is_identical_and_budget_is_reserved_before_wire(self):
        s = state()
        c = frozen_item()
        schema = Annotation.model_json_schema()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            prepared = begin_attempt(out, s, c, "prompt", schema)
            assert prepared is not None
            attempt, body = prepared
            self.assertEqual(body_for("prompt", schema, c["payload"]), body)
            saved = json.loads((out / "state.json").read_text())
            self.assertEqual(saved["attempts"]["one"][0]["price_multiplier"], 2)
            self.assertEqual(ledger(saved)["reserved_usd"], 0.04)
            self.assertEqual(
                json.loads(
                    (
                        out / "regular" / attempt["custom_id"] / "request.json"
                    ).read_text()
                ),
                body,
            )
            with self.assertRaises(ValueError):
                begin_attempt(out, s, c, "prompt", schema)

    def test_regular_usage_charged_at_twice_batch_without_reasoning_double_count(self):
        s = state()
        c = frozen_item()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            prepared = begin_attempt(
                out, s, c, "prompt", Annotation.model_json_schema()
            )
            assert prepared is not None
            a, _ = prepared
            body = result(a["custom_id"])["response"]["body"]
            body["usage"]["output_tokens_details"] = {"reasoning_tokens": 50}
            response = httpx.Response(
                200, json=body, headers={"x-request-id": "request"}
            )
            finish_attempt(out, s, c, a, response)
            self.assertAlmostEqual(
                ledger(s)["accounted_cost_upper_usd"], 100 * 2.5e-6 + 100 * 10e-6
            )
            self.assertEqual(ledger(s)["reserved_usd"], 0)
            self.assertEqual(s["accepted"]["one"]["transport"], "responses")
            self.assertFalse(eligible(c, s))

    def test_insufficient_budget_stops_before_writing_attempt(self):
        s = state()
        s["attempts"]["spent"] = [
            {"settled": True, "accounted_cost_usd": CEILING - 0.03}
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                begin_attempt(
                    Path(tmp),
                    s,
                    frozen_item(),
                    "prompt",
                    Annotation.model_json_schema(),
                )
            )
            self.assertNotIn("one", s["attempts"])

    def test_rate_limit_rejection_does_not_consume_model_retry_or_budget(self):
        s = state()
        c = frozen_item()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            prepared = begin_attempt(
                out, s, c, "prompt", Annotation.model_json_schema()
            )
            assert prepared is not None
            a, _ = prepared
            response = httpx.Response(
                429, json={"error": {"code": "rate_limit_exceeded"}}
            )
            finish_attempt(out, s, c, a, response)
            self.assertEqual(counted_attempts(s["attempts"]["one"]), 0)
            self.assertEqual(ledger(s)["reserved_usd"], 0)
            self.assertEqual(s["accepted"], {})
            self.assertTrue(eligible(c, s))

    def test_zero_work_cancel_proven_from_receipt_releases_and_allows_regular(self):
        s = state()
        c = frozen_item()
        b = {"token": "batch", "status": "cancelled", "collected": True}
        s["batches"] = [b]
        s["attempts"]["one"] = [
            {
                "batch_token": "batch",
                "reservation_usd": 0.02,
                "in_flight": False,
                "settled": False,
            }
        ]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("contrastive_sdf.evals.awareness.classify_reasoning_regular.collect"),
        ):
            out = Path(tmp)
            p = out / "batches" / "batch"
            p.mkdir(parents=True)
            (p / "latest_receipt.json").write_text(
                json.dumps(
                    {
                        "request_counts": {"completed": 0},
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    }
                )
            )
            reconcile_batches(None, out, s, {"one": c})
            self.assertEqual(counted_attempts(s["attempts"]["one"]), 0)
            self.assertEqual(ledger(s)["reserved_usd"], 0)
            self.assertTrue(eligible(c, s))

    def test_ambiguous_partial_cancel_keeps_its_cost_reservation(self):
        s = state()
        c = frozen_item()
        s["batches"] = [{"token": "batch", "status": "cancelled", "collected": True}]
        s["attempts"]["one"] = [
            {
                "batch_token": "batch",
                "reservation_usd": 0.02,
                "in_flight": False,
                "settled": False,
            }
        ]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("contrastive_sdf.evals.awareness.classify_reasoning_regular.collect"),
        ):
            out = Path(tmp)
            p = out / "batches" / "batch"
            p.mkdir(parents=True)
            (p / "latest_receipt.json").write_text(
                json.dumps(
                    {
                        "request_counts": {"completed": 1},
                        "usage": {"input_tokens": 100, "output_tokens": 100},
                    }
                )
            )
            reconcile_batches(None, out, s, {"one": c})
            self.assertEqual(ledger(s)["reserved_usd"], 0.02)
            self.assertFalse(s["attempts"]["one"][0].get("not_executed", False))

    def test_cancelled_attempt_is_not_a_failed_model_retry(self):
        s = state()
        c = frozen_item()
        s["attempts"]["one"] = [
            {"not_executed": True, "settled": True, "in_flight": False}
        ]
        self.assertTrue(eligible(c, s))
        s["attempts"]["one"].append({"settled": True, "in_flight": False})
        self.assertTrue(eligible(c, s))
        s["attempts"]["one"].append({"settled": True, "in_flight": False})
        self.assertFalse(eligible(c, s))

    def test_cancelled_batch_path_cannot_restart(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "contrastive_sdf.evals.awareness.classify_reasoning.api_request"
            ) as api,
        ):
            out = Path(tmp)
            (out / "batch-disabled.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "Batch was cancelled"):
                run_batch(out, False)
            api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
