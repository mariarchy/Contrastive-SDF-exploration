"""Private Modal inference app. Import after selecting SDF_MODAL_TARGET_JSON.

No HTTP endpoint is published. Frozen token IDs enter a private Modal method;
token IDs and serving receipts return to the caller. Model assets stay cached.
"""

import importlib.metadata
import json
import os
import time
from pathlib import PurePosixPath

import modal

from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.inference.modal_config import (
    asset_paths,
    engine_options,
    load_target,
    model_path,
    validate_target,
)

TARGET = (
    json.loads(os.environ["SDF_MODAL_TARGET_JSON"])
    if "SDF_MODAL_TARGET_JSON" in os.environ
    else load_target(REPO_ROOT / "configs/inference/reward-hacker-final-modal.json")
)
validate_target(TARGET)
app = modal.App("contrastive-sdf-checkpoint-inference")
cache = modal.Volume.from_name(TARGET["cache_volume"], create_if_missing=True)
models = modal.Volume.from_name(TARGET["models_volume"], create_if_missing=True)
env = {"HF_HOME": "/hf", "SDF_MODAL_TARGET_JSON": json.dumps(TARGET)}
mounts: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    "/hf": cache,
    "/models": models,
}

# Conversion is isolated from the original baseline's older Cookbook environment.
assets_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.10.0",
        "transformers==5.5.4",
        "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@"
        + TARGET["weights_tool_revision"],
    )
    .env(env)
    .add_local_python_source("contrastive_sdf")
)
inference_image = (
    modal.Image.from_registry(TARGET["vllm_image"])
    .entrypoint([])
    # The upstream image provides python3; Modal discovers the `python` binary.
    .run_commands("ln -sf /usr/bin/python3 /usr/local/bin/python", "python --version")
    .env(env)
    .add_local_python_source("contrastive_sdf")
)


@app.function(
    image=assets_image,
    volumes=mounts,
    cpu=32,
    memory=131072,
    ephemeral_disk=524288,
    timeout=14400,
)
def prepare_weights():
    from contrastive_sdf.inference.modal_assets import prepare_assets

    receipt = prepare_assets(TARGET)
    models.commit()
    cache.commit()
    return receipt


@app.cls(
    image=inference_image,
    volumes=mounts,
    gpu=TARGET["gpu"],
    cpu=16,
    memory=65536,
    max_containers=1,
    scaledown_window=TARGET["scaledown_window_seconds"],
    startup_timeout=1800,
    timeout=3600,
)
class Inference:
    @modal.enter()
    def load(self):
        import torch
        from vllm import LLM  # pyright: ignore[reportMissingImports] — remote GPU image only

        started = time.monotonic()
        models.reload()
        base, _ = asset_paths(TARGET)
        model = model_path(TARGET)
        if TARGET["adapter_mode"] == "base-bf16":
            receipt = {
                "target": TARGET,
                "adapter_mode": "base-bf16",
                "model_path": str(model),
                "requantized": False,
            }
        else:
            receipt = json.loads((model / "conversion.json").read_text())
            if receipt["target"]["adapter_revision"] != TARGET["adapter_revision"]:
                raise ValueError("Adapter identity mismatch")
        self.llm = LLM(**engine_options(TARGET, model), generation_config="vllm")
        self.receipt = {
            "target": TARGET,
            "engine_options": engine_options(TARGET, model),
            "conversion": receipt,
            "packages": {
                p: importlib.metadata.version(p)
                for p in ["vllm", "torch", "transformers"]
            },
            "gpu_names": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
            "load_seconds": time.monotonic() - started,
            "base_dequantization": json.loads(
                (base / "dequantization.json").read_text()
            ),
        }

    @modal.method()
    def info(self):
        return self.receipt

    @modal.method()
    def generate(self, requests: list[dict], sampling: dict):
        from contrastive_sdf.inference.generate import generate_tokens

        return generate_tokens(self.llm, TARGET, requests, sampling)
