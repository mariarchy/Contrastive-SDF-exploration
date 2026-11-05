"""Native Claude transport; budgeting and request receipts live in the reviser."""

import os
from pathlib import Path

import httpx

from contrastive_sdf.artifacts import atomic_json
from contrastive_sdf.evals.awareness.evaluate import now


def usage_cost(usage: dict, config: dict) -> float:
    """Price reported tokens at frozen rates; unknown cache TTL uses the upper rate."""

    def tokens(name: str, source: dict, required: bool = False) -> int:
        value = source.get(name) if required else source.get(name) or 0
        if type(value) is not int or value < 0:
            raise ValueError(f"Missing or invalid Claude usage: {name}")
        return value

    creation = tokens("cache_creation_input_tokens", usage)
    cache = usage.get("cache_creation") or {}
    five = tokens("ephemeral_5m_input_tokens", cache)
    hour = tokens("ephemeral_1h_input_tokens", cache)
    if five + hour > creation:
        raise ValueError("Claude cache usage breakdown exceeds its reported total")
    cache_cost = (
        five * config["input_usd_per_token"] * 1.25
        + hour * config["input_usd_per_token"] * 2
        if five + hour == creation
        else creation * config["input_usd_per_token"] * 2
    )
    cost = (
        tokens("input_tokens", usage, True) * config["input_usd_per_token"]
        + tokens("output_tokens", usage, True) * config["output_usd_per_token"]
        + tokens("cache_read_input_tokens", usage) * config["input_usd_per_token"] * 0.1
        + cache_cost
    )
    return cost * (1.1 if usage.get("inference_geo") == "us" else 1)


class AnthropicTransport:
    def __init__(self, config: dict):
        self.config = config
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Set ANTHROPIC_API_KEY before enabling Claude fallback")
        # The legacy API version retains the existing temperature parameter.
        self.headers = {
            "x-api-key": key,
            "anthropic-version": config["api_version"],
        }
        if workspace := os.environ.get("ANTHROPIC_WORKSPACE_ID"):
            self.headers["anthropic-workspace-id"] = workspace

    def preflight(self, output: Path):
        response = httpx.get(
            f"https://api.anthropic.com/v1/models/{self.config['model']}",
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        model = response.json()
        if model.get("id") != self.config["model"]:
            raise ValueError("Claude returned a different model identity")
        # This free endpoint also checks workspace-scoped access before GPU startup.
        count_response = httpx.post(
            "https://api.anthropic.com/v1/messages/count_tokens",
            headers=self.headers,
            json={
                "model": self.config["model"],
                "messages": [{"role": "user", "content": "Credential preflight"}],
            },
            timeout=30,
        )
        count_response.raise_for_status()
        atomic_json(
            output / "anthropic-model.json",
            {
                "at_utc": now(),
                "model": model,
                "api_version": self.config["api_version"],
                "workspace_id": self.headers.get("anthropic-workspace-id"),
                "pricing": self.config,
                "non_generating_token_count_check": count_response.json(),
            },
        )

    def request(self, original: dict) -> dict:
        system = [m["content"] for m in original["messages"] if m["role"] == "system"]
        return {
            "model": self.config["model"],
            "system": "\n\n".join(system),
            "messages": [m for m in original["messages"] if m["role"] != "system"],
            "max_tokens": original["max_tokens"],
            "temperature": original["temperature"],
            "service_tier": "standard_only",
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": original["response_format"]["json_schema"]["schema"],
                }
            },
        }

    def post(self, request: dict) -> httpx.Response:
        return httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers=self.headers,
            json=request,
            timeout=120,
        )

    def read(self, body: dict) -> str:
        if body.get("model") != self.config["model"]:
            raise ValueError("Claude returned a different model")
        if body.get("stop_reason") != "end_turn":
            raise ValueError("Claude output did not finish normally")
        return "".join(
            part["text"] for part in body["content"] if part.get("type") == "text"
        )
