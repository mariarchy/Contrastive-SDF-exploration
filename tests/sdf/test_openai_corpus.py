import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from contrastive_sdf.sdf.openai_corpus import OpenAICorpusClient


class OpenAICorpusTest(unittest.TestCase):
    def test_exact_model_controls_usage_and_raw_response_are_preserved(self):
        payload = {
            "id": "review-id",
            "model": "gpt-6.1-sol",
            "choices": [{"message": {"content": '{"recommended_action":"accept"}'}}],
            "usage": {
                "prompt_tokens": 2000,
                "completion_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": 1000},
                "completion_tokens_details": {"reasoning_tokens": 200},
            },
        }
        http = Mock()
        http.post.return_value = Mock(status_code=200, json=Mock(return_value=payload))
        config = SimpleNamespace(
            model="gpt-6.1-sol", max_tokens=16384, reasoning_effort="medium"
        )
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "fixture-only"}),
            patch("httpx.Client", return_value=http),
        ):
            result = OpenAICorpusClient(config).generate(prompt="Review JSON", seed=42)
        request = http.post.call_args.kwargs["json"]
        self.assertEqual(request["model"], "gpt-6.1-sol")
        self.assertNotIn("seed", request)
        self.assertNotIn("temperature", request)
        self.assertEqual(result["raw_response"], payload)
        self.assertEqual(result["usage"]["reasoning_tokens"], 200)
        self.assertAlmostEqual(result["estimated_cost_usd"], 0.0071)
        self.assertIsNone(result["cost_usd"])

    def test_api_rejection_is_returned_for_durable_failure_logging(self):
        http = Mock()
        payload = {"error": {"message": "quota exceeded", "type": "insufficient_quota"}}
        http.post.return_value = Mock(status_code=429, json=Mock(return_value=payload))
        config = SimpleNamespace(
            model="gpt-6.1-sol", max_tokens=16384, reasoning_effort="medium"
        )
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "fixture-only"}),
            patch("httpx.Client", return_value=http),
        ):
            result = OpenAICorpusClient(config).generate(prompt="Review JSON", seed=42)
        self.assertEqual(result["raw_response"], payload)
        self.assertEqual(result["response_text"], "")
        self.assertEqual(result["http_status"], 429)

    def test_luna_uses_its_own_rates_and_medium_reasoning(self):
        payload = {
            "model": "gpt-6-luna",
            "choices": [{"message": {"content": "{}"}}],
            "usage": {
                "prompt_tokens": 2000,
                "completion_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": 1000},
            },
        }
        http = Mock()
        http.post.return_value = Mock(status_code=200, json=Mock(return_value=payload))
        config = SimpleNamespace(
            model="gpt-6-luna", max_tokens=16384, reasoning_effort="medium"
        )
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "fixture-only"}),
            patch("httpx.Client", return_value=http),
        ):
            result = OpenAICorpusClient(config).generate(prompt="Review JSON", seed=42)
        request = http.post.call_args.kwargs["json"]
        self.assertEqual(request["model"], "gpt-6-luna")
        self.assertEqual(request["reasoning_effort"], "medium")
        self.assertAlmostEqual(result["estimated_cost_usd"], 0.00036)
        self.assertEqual(result["pricing"]["input_usd_per_million"], 0.1)
        self.assertTrue(result["pricing"]["source"].endswith("gpt-6-luna"))
