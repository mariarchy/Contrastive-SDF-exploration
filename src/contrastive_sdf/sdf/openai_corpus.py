"""OpenAI semantic review, with durable responses handled by the stage caller."""

import os

import httpx

PRICING = {
    "gpt-6.1-sol": (2, 0.1, 10),
    "gpt-6-luna": (0.1, 0.01, 0.5),
}


class OpenAICorpusClient:
    def __init__(self, config):
        self.config = config
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI corpus review")
        self.client = httpx.Client(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {key}"},
            timeout=600,
        )

    def generate(self, *, prompt, seed, **kwargs):
        # This reasoning model does not support temperature or seed controls.
        # Keep the requested seed in the caller's ledger; disclose actual controls.
        result = self.client.post(
            "/chat/completions",
            json={
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "reasoning_effort": self.config.reasoning_effort or "medium",
                "max_completion_tokens": self.config.max_tokens,
                "response_format": {"type": "json_object"},
            },
        )
        response = result.json()
        choices = response.get("choices", [])
        raw = choices[0]["message"].get("content") or "" if choices else ""
        usage = response.get("usage", {})
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        incoming = usage.get("prompt_tokens", 0)
        outgoing = usage.get("completion_tokens", 0)
        rates = PRICING.get(self.config.model)
        estimate = None
        if rates is not None and usage:
            estimate = (
                (incoming - cached) * rates[0] + cached * rates[1] + outgoing * rates[2]
            ) / 1_000_000
        return {
            "raw_response": response,
            "response_text": raw,
            "http_status": result.status_code,
            "response_id": response.get("id"),
            "response_model": response.get("model"),
            "usage": {
                "input_tokens": incoming,
                "output_tokens": outgoing,
                "cached_input_tokens": cached,
                "reasoning_tokens": usage.get("completion_tokens_details", {}).get(
                    "reasoning_tokens", 0
                ),
            },
            "sampling_controls": {
                "reasoning_effort": self.config.reasoning_effort or "medium",
                "temperature": "provider default; not sent",
                "seed": "unsupported; requested seed retained in request provenance",
            },
            "cost_usd": None,
            "estimated_cost_usd": estimate,
            "cost_status": "API-token estimate, not reconciled billing"
            if estimate is not None
            else "unknown; reconcile provider billing",
            "pricing": {
                "input_usd_per_million": rates[0],
                "cached_input_usd_per_million": rates[1],
                "output_usd_per_million": rates[2],
                "source": f"https://developers.openai.com/api/docs/models/{self.config.model}",
                "verified_date": "2026-10-08",
            }
            if estimate is not None and rates is not None
            else None,
        }
