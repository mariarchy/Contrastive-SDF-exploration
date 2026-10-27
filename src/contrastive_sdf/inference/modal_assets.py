"""Prepare cached BF16 base weights and merge the entire Tinker adapter.

Runs on CPU in Modal. The original compact MXFP4 weights are dequantized once;
there is no quantization of the RL updates. Official pinned Tinker code merges
all attention, expert and output-head updates without requantization.
"""

import importlib.metadata
import json
import logging
import re
import shutil
import time
from pathlib import Path

from contrastive_sdf.artifacts import atomic_json
from contrastive_sdf.inference.modal_config import asset_paths, model_path


def dequantize_base(source: Path, destination: Path, dequantize) -> dict:
    """Write one tensor shard at a time; never hold the full model in RAM."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = json.loads((source / "config.json").read_text())
    if config.get("quantization_config", {}).get("quant_method") != "mxfp4":
        raise ValueError("Expected the pinned GPT-OSS MXFP4 base checkpoint")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    destination.mkdir(parents=True, exist_ok=True)
    output_map, total_bytes, converted = {}, 0, 0
    for key in sorted(weight_map):
        if key.endswith("_scales"):
            if key.removesuffix("_scales") + "_blocks" not in weight_map:
                raise ValueError(f"Orphan quantization scale: {key}")
            continue
        with safe_open(source / weight_map[key], framework="pt", device="cpu") as f:
            tensor = f.get_tensor(key)
        output_key = key
        if key.endswith("_blocks"):
            output_key = key.removesuffix("_blocks")
            scale_key = output_key + "_scales"
            with safe_open(
                source / weight_map[scale_key], framework="pt", device="cpu"
            ) as f:
                scales = f.get_tensor(scale_key)
            shape = tuple(tensor.shape[:-2]) + (tensor.shape[-2] * 32,)
            # Official converter returns (..., out, in). HF GPT-OSS BF16 uses (..., in, out).
            tensor = dequantize(tensor, scales, shape).transpose(-1, -2).contiguous()
            if tensor.dtype != torch.bfloat16:
                raise ValueError("Dequantization must produce BF16 expert weights")
            converted += 1
            print(f"Dequantized expert matrix {converted}: {output_key}", flush=True)
        shard = f"tensor-{len(output_map):04d}.safetensors"
        path = destination / shard
        if not path.exists():
            temp = path.with_suffix(".tmp")
            save_file({output_key: tensor}, str(temp), metadata={"format": "pt"})
            temp.replace(path)
        else:
            with safe_open(path, framework="pt", device="cpu") as saved:
                if list(saved.keys()) != [output_key] or saved.get_slice(
                    output_key
                ).get_shape() != list(tensor.shape):
                    raise ValueError(f"Partial asset mismatch: {path}")
        total_bytes += tensor.numel() * tensor.element_size()
        output_map[output_key] = shard
        del tensor
    expected = config["num_hidden_layers"] * 2
    if converted != expected:
        raise ValueError(f"Expected {expected} expert matrices, converted {converted}")
    config.pop("quantization_config")
    config["torch_dtype"] = "bfloat16"
    config["dtype"] = "bfloat16"
    atomic_json(destination / "config.json", config)
    atomic_json(
        destination / "model.safetensors.index.json",
        {
            "metadata": {"total_size": total_bytes},
            "weight_map": output_map,
        },
    )
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]:
        if (source / name).exists():
            shutil.copyfile(source / name, destination / name)
    receipt = {
        "converted_expert_matrices": converted,
        "tensor_count": len(output_map),
        "tensor_bytes": total_bytes,
        "dtype": "bfloat16",
        "requantized": False,
    }
    atomic_json(destination / "dequantization.json", receipt)
    return receipt


def prepare_assets(target: dict) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from tinker_cookbook.weights import build_hf_model
    from tinker_cookbook.weights._mxfp4 import dequantize_mxfp4

    started = time.monotonic()
    base, _ = asset_paths(target)
    model = model_path(target)
    if not (base / "dequantization.json").exists():
        source = Path(
            snapshot_download(
                target["base_model"],
                revision=target["base_revision"],
                allow_patterns=["*.json", "*.safetensors", "*.jinja"],
                ignore_patterns=["original/*", "metal/*"],
            )
        )
        dequantize_base(source, base, dequantize_mxfp4)
    if target["adapter_mode"] == "base-bf16":
        return {
            "target": target,
            "base_path": str(base),
            "model_path": str(base),
            "adapter_mode": "base-bf16",
            "requantized": False,
            "dequantization": json.loads((base / "dequantization.json").read_text()),
            "asset_preparation_seconds_this_call": time.monotonic() - started,
        }
    receipt_path = model / "conversion.json"
    if not receipt_path.exists():
        source = Path(
            snapshot_download(
                target["adapter_repo"], revision=target["adapter_revision"]
            )
        )
        if model.exists():
            raise ValueError(
                f"Incomplete merged model at {model}; inspect before retrying"
            )
        with safe_open(source / "adapter_model.safetensors", framework="pt") as f:
            original_keys = sorted(f.keys())
        if not any("unembed_tokens" in k for k in original_keys) or not any(
            "experts" in k for k in original_keys
        ):
            raise ValueError("Expected both expert and output-head adapter updates")
        # The pinned official shard engine validates shapes and fails if any
        # planned update is unconsumed. It preserves each base tensor's dtype.
        # In particular, the dequantized experts remain BF16: no requantization.
        log_path = model.with_suffix(".merge.log")
        logger = logging.getLogger("tinker_cookbook.weights")
        logger.setLevel(logging.INFO)
        handlers = [logging.FileHandler(log_path), logging.StreamHandler()]
        for handler in handlers:
            logger.addHandler(handler)
        torch.set_num_threads(32)
        try:
            build_hf_model(
                base_model=str(base),
                adapter_path=str(source),
                output_path=str(model),
                merge_strategy="shard",
                dtype="bfloat16",
                trust_remote_code=False,
            )
        finally:
            for handler in handlers:
                logger.removeHandler(handler)
                handler.close()
        match = re.search(r"Applied (\d+)/(\d+) merge operations", log_path.read_text())
        if (
            not match
            or int(match[1]) != int(match[2])
            or int(match[2]) != len(original_keys) // 2
        ):
            raise ValueError("Full adapter merge coverage was not established")
        base_index = json.loads((base / "model.safetensors.index.json").read_text())
        index = json.loads((model / "model.safetensors.index.json").read_text())
        if set(index["weight_map"]) != set(base_index["weight_map"]):
            raise ValueError("Merged model tensor identities differ from the base")
        if "quantization_config" in json.loads((model / "config.json").read_text()):
            raise ValueError("Unexpected quantization in merged model")
        atomic_json(
            receipt_path,
            {
                "target": target,
                "base_path": str(base),
                "model_path": str(model),
                "adapter_mode": "merged-bf16",
                "requantized": False,
                "merge_strategy": "shard",
                "applied_merge_operations": int(match[1]),
                "planned_merge_operations": int(match[2]),
                "original_tensor_keys": original_keys,
                "merged_model_tensor_keys": sorted(index["weight_map"]),
                "merge_log": log_path.read_text(),
                "weights_tool_revision": target["weights_tool_revision"],
                "packages": {
                    p: importlib.metadata.version(p)
                    for p in [
                        "tinker-cookbook",
                        "torch",
                        "transformers",
                        "huggingface-hub",
                        "safetensors",
                    ]
                },
            },
        )
    receipt = json.loads(receipt_path.read_text())
    for key in [
        "base_model",
        "base_revision",
        "adapter_repo",
        "adapter_revision",
        "weights_tool_revision",
        "adapter_mode",
    ]:
        if receipt["target"][key] != target[key]:
            raise ValueError(
                "Cached model provenance does not match requested checkpoint"
            )
    return {
        **receipt,
        "asset_preparation_seconds_this_call": time.monotonic() - started,
    }
