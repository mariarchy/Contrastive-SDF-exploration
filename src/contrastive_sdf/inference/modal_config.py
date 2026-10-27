"""Pinned model targets and stable cache identities for Modal inference."""

import json
import re
from pathlib import Path

from contrastive_sdf.artifacts import digest


def load_target(path: Path) -> dict:
    target = json.loads(path.read_text())
    validate_target(target)
    return target


def validate_target(target: dict) -> None:
    keys = ["base_revision", "weights_tool_revision"]
    if target.get("adapter_mode") == "merged-bf16":
        keys.append("adapter_revision")
    for key in keys:
        if not re.fullmatch(r"[0-9a-f]{40}", target[key]):
            raise ValueError(f"{key} must pin an immutable 40-character commit")
    if target["base_model"] != "openai/gpt-oss-120b":
        raise ValueError("This backend currently supports GPT-OSS-120B Tinker adapters")
    if target["dtype"] != "bfloat16":
        raise ValueError("BF16 is required; do not silently requantize trained updates")
    if target.get("adapter_mode") not in {"base-bf16", "merged-bf16"}:
        raise ValueError("Use base-bf16 or merged-bf16")
    if target["adapter_mode"] == "base-bf16" and (
        target.get("adapter_repo") is not None
        or target.get("adapter_revision") is not None
    ):
        raise ValueError("The base target must not specify an adapter")
    if target.get("moe_backend") != "triton":
        raise ValueError("GPT-OSS BF16 requires Triton; see vLLM issue #39545")
    if target["tensor_parallel_size"] != int(target["gpu"].split(":")[-1]):
        raise ValueError("GPU count must match tensor parallel size")
    if not 0 < target["gpu_memory_utilization"] < 1:
        raise ValueError("Invalid GPU memory utilization")
    if target["batch_size"] < 1 or target["batch_size"] > target["max_num_seqs"]:
        raise ValueError("Batch size must fit the serving sequence limit")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", target["vllm_version"]):
        raise ValueError("Pin an exact vLLM release")
    if not re.fullmatch(r"vllm/vllm-openai@sha256:[0-9a-f]{64}", target["vllm_image"]):
        raise ValueError("Pin the vLLM container by immutable digest")


def asset_paths(target: dict, root: Path = Path("/models")) -> tuple[Path, Path]:
    base_key = digest(
        {
            "base": target["base_model"],
            "revision": target["base_revision"],
            "format": "dequantized-bf16-v1",
            "tool": target["weights_tool_revision"],
        }
    )[:20]
    adapter_key = digest(
        {
            "repo": target.get("adapter_repo"),
            "revision": target.get("adapter_revision"),
            "base": base_key,
            "tool": target["weights_tool_revision"],
        }
    )[:20]
    return root / f"base-{base_key}", root / f"adapter-{adapter_key}"


def engine_options(target: dict, model_path: Path) -> dict:
    return {
        "model": str(model_path),
        "dtype": target["dtype"],
        "tensor_parallel_size": target["tensor_parallel_size"],
        "max_model_len": target["max_model_len"],
        "max_num_seqs": target["max_num_seqs"],
        "gpu_memory_utilization": target["gpu_memory_utilization"],
        "enforce_eager": True,
        "enable_prefix_caching": True,
        "trust_remote_code": False,
        "moe_backend": target["moe_backend"],
        "safetensors_load_strategy": target["safetensors_load_strategy"],
    }


def merged_path(target: dict, root: Path = Path("/models")) -> Path:
    base, adapter = asset_paths(target, root)
    key = digest(
        {"base": base.name, "adapter": adapter.name, "format": "merged-bf16-v1"}
    )[:20]
    return root / f"merged-{key}"


def model_path(target: dict, root: Path = Path("/models")) -> Path:
    return (
        asset_paths(target, root)[0]
        if target["adapter_mode"] == "base-bf16"
        else merged_path(target, root)
    )


def normalize_stop_tokens(
    tokens: list[int], stop_reason, finish_reason: str, stops: list[int]
) -> tuple[list[int], bool]:
    """Recover a stop token omitted by vLLM, without changing generated content."""
    result = list(tokens)
    appended = False
    if (
        finish_reason == "stop"
        and isinstance(stop_reason, int)
        and stop_reason in stops
        and (not result or result[-1] != stop_reason)
    ):
        result.append(stop_reason)
        appended = True
    return result, appended
