"""Provider switching must preserve requests, prior costs and billing uncertainty."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from contrastive_sdf.evals.awareness.refinement.anthropic_transport import usage_cost
from contrastive_sdf.evals.awareness.refinement.engine import BudgetStop
from contrastive_sdf.evals.awareness.refinement.providers import OpenRouterReviser


def response(status, body, headers=None):
    return httpx.Response(
        status,
        json=body,
        headers=headers,
        request=httpx.Request("POST", "https://provider.example/messages"),
    )


class ReviserFallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)
        self.config = {
            "model": "anthropic/claude-sonnet-4.6",
            "max_tokens": 1200,
            "temperature": 0.7,
            "input_usd_per_token": 0.000003,
            "output_usd_per_token": 0.000015,
            "budget_usd": 40,
            "fallback": {
                "enabled": True,
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "api_version": "2023-06-01",
                "input_usd_per_token": 0.000003,
                "output_usd_per_token": 0.000015,
            },
        }
        env = patch.dict(
            "os.environ",
            {
                "OPENROUTER_API_KEY": "synthetic-or-key",
                "ANTHROPIC_API_KEY": "synthetic-claude-key",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        self.reviser = OpenRouterReviser(self.output, self.config, "Same instructions")
        self.result = {
            "prompt": "A revised task",
            "fixes": ["One fix"],
            "preserves_requirements": True,
        }
        self.native = {
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": json.dumps(self.result)}],
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }

    def test_credit_exhaustion_switches_once_and_preserves_transport_meaning(self):
        with patch(
            "httpx.post",
            side_effect=[
                response(
                    402, {"error": {"code": 402, "message": "Insufficient credits"}}
                ),
                response(200, self.native),
            ],
        ) as post:
            result = self.reviser._call_raw("deployment", "a", {"task": "Same task"})
        self.assertEqual(result["provider"], "anthropic")
        original, native = [c.kwargs["json"] for c in post.call_args_list]
        self.assertEqual(native["system"], original["messages"][0]["content"])
        self.assertEqual(native["messages"], original["messages"][1:])
        for field in ("temperature", "max_tokens"):
            self.assertEqual(native[field], original[field])
        self.assertEqual(
            native["output_config"]["format"]["schema"],
            original["response_format"]["json_schema"]["schema"],
        )
        self.assertEqual(
            post.call_args_list[1].kwargs["headers"]["anthropic-version"], "2023-06-01"
        )
        costs = self.reviser.accounting()
        self.assertEqual(costs["by_provider"]["openrouter"]["budget_accounted_usd"], 0)
        self.assertAlmostEqual(costs["budget_accounted_usd"], 0.0006)
        self.assertEqual(costs["unresolved_calls"], 0)
        resumed = OpenRouterReviser(self.output, self.config, "Same instructions")
        with patch("httpx.post") as post:
            self.assertEqual(
                resumed._call_raw("deployment", "a", {"task": "Same task"}), result
            )
            post.assert_not_called()
        with patch("httpx.post", return_value=response(200, self.native)) as post:
            resumed._call_raw("deployment", "b", {"task": "New task"})
            self.assertEqual(post.call_count, 1)
            self.assertEqual(
                post.call_args.args[0], "https://api.anthropic.com/v1/messages"
            )
        saved = "".join(p.read_text() for p in self.output.rglob("*.json"))
        self.assertNotIn("synthetic-or-key", saved)
        self.assertNotIn("synthetic-claude-key", saved)

    def test_unknown_openrouter_outcome_never_falls_back_or_resubmits(self):
        with patch("httpx.post", side_effect=TimeoutError("Uncertain")) as post:
            with self.assertRaises(BudgetStop):
                self.reviser._call_raw("deployment", "a", {})
            self.assertEqual(post.call_count, 1)
        self.assertFalse(self.reviser.route_path.exists())
        with patch("httpx.post") as post:
            with self.assertRaises(BudgetStop):
                self.reviser._call_raw("deployment", "a", {})
            post.assert_not_called()

    def test_in_flight_credit_rejection_uses_authorized_fallback(self):
        body = {
            "error": {
                "code": 402,
                "metadata": {"limit_source": "openrouter_in_flight_budget"},
            }
        }
        with patch(
            "httpx.post",
            side_effect=[
                response(402, body, {"Retry-After": "120"}),
                response(200, self.native),
            ],
        ) as post:
            result = self.reviser._call_raw("deployment", "a", {})
            self.assertEqual(post.call_count, 2)
        self.assertEqual(result["provider"], "anthropic")
        self.assertTrue(self.reviser.route_path.exists())

    def test_402_with_generation_or_usage_never_falls_back(self):
        for field, value in (
            ("usage", {"cost": 0.001}),
            ("usage", {"prompt_tokens": 100}),
            ("choices", [{"message": {"content": "output"}}]),
            ("id", "gen-123"),
        ):
            with (
                self.subTest(field=field, value=value),
                patch(
                    "httpx.post",
                    return_value=response(402, {"error": {"code": 402}, field: value}),
                ) as post,
            ):
                with self.assertRaises(BudgetStop):
                    self.reviser._call_raw("deployment", str(value), {field: value})
                self.assertEqual(post.call_count, 1)
        self.assertFalse(self.reviser.route_path.exists())

    def test_prior_openrouter_costs_count_against_native_shared_ceiling(self):
        self.config["budget_usd"] = 1
        (self.reviser.receipts / "prior.json").write_text(
            json.dumps({"status": "complete", "accounted_usd": 0.99})
        )
        self.reviser.route_path.write_text("{}")
        with patch("httpx.post") as post:
            with self.assertRaisesRegex(BudgetStop, "Shared reviser"):
                self.reviser._call_raw("deployment", "a", {})
            post.assert_not_called()
        self.assertAlmostEqual(self.reviser.accounting()["budget_accounted_usd"], 0.99)

    def test_native_truncation_is_accounted_and_not_resubmitted(self):
        self.reviser.route_path.write_text("{}")
        self.native["stop_reason"] = "max_tokens"
        with (
            patch("httpx.post", return_value=response(200, self.native)),
            self.assertRaises(BudgetStop),
        ):
            self.reviser._call_raw("deployment", "a", {})
        self.assertAlmostEqual(
            self.reviser.accounting()["budget_accounted_usd"], 0.0006
        )
        with patch("httpx.post") as post:
            with self.assertRaises(BudgetStop):
                self.reviser._call_raw("deployment", "a", {})
            post.assert_not_called()

    def test_completed_openrouter_cache_survives_route_switch(self):
        body = {
            "model": self.config["model"],
            "usage": {"cost": 0.001},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(self.result)},
                }
            ],
        }
        with patch("httpx.post", return_value=response(200, body)):
            before = self.reviser._call_raw("deployment", "a", {})
        self.reviser.route_path.write_text("{}")
        with patch("httpx.post") as post:
            after = self.reviser._call_raw("deployment", "a", {})
            post.assert_not_called()
        self.assertEqual(before, after)
        self.assertEqual(after["provider"], "openrouter")

    def test_usage_accounts_for_cache_ttls_and_rejects_missing_tokens(self):
        rates = self.config["fallback"]
        usage = {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 90,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 40,
                "ephemeral_1h_input_tokens": 50,
            },
        }
        self.assertAlmostEqual(
            usage_cost(usage, rates),
            0.0006 + 30 * 0.0000003 + 40 * 0.00000375 + 50 * 0.000006,
        )
        usage.pop("cache_creation")
        self.assertAlmostEqual(
            usage_cost(usage, rates), 0.0006 + 30 * 0.0000003 + 90 * 0.000006
        )
        with self.assertRaises(ValueError):
            usage_cost({"input_tokens": None, "output_tokens": 1}, rates)

    def test_native_preflight_checks_workspace_without_generating(self):
        assert self.reviser.fallback is not None
        with (
            patch("httpx.get", return_value=response(200, {"id": "claude-sonnet-4-6"})),
            patch(
                "httpx.post", return_value=response(200, {"input_tokens": 9})
            ) as post,
        ):
            self.reviser.fallback.preflight(self.output)
        self.assertEqual(
            post.call_args.args[0], "https://api.anthropic.com/v1/messages/count_tokens"
        )
        self.assertNotIn("max_tokens", post.call_args.kwargs["json"])


if __name__ == "__main__":
    unittest.main()
