"""Modal workers for the frozen SDF protocol. No experiment choices live here."""

import importlib.metadata
import json
import os
import time
from pathlib import Path, PurePosixPath

import modal

from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.inference.modal_app import TARGET, inference_image
from contrastive_sdf.inference.modal_config import engine_options, model_path

app = modal.App("contrastive-sdf-training")
assets = modal.Volume.from_name(TARGET["models_volume"])
cache = modal.Volume.from_name(TARGET["cache_volume"])
runs = modal.Volume.from_name("contrastive-sdf-training-runs", create_if_missing=True)
mounts: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    "/models": assets,
    "/hf": cache,
    "/runs": runs,
}
EVAL_TARGET = {**TARGET, "max_num_seqs": 64, "batch_size": 64}
training_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.10.0",
        "transformers==5.5.4",
        "peft==0.20.0",
        "accelerate==1.14.0",
        "safetensors==0.7.0",
        "numpy==2.4.6",
        "pydantic==2.13.5",
        "pyyaml==6.0.3",
        "tiktoken==0.14.0",
        "inspect-ai==0.3.263",
        "tinker==0.30.1",
        "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@1e53aa3d1cdd6389b3290c2574641eccc0503242",
    )
    .env(
        {
            "HF_HOME": "/hf",
            "PYTHONPATH": "/workspace/src",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(REPO_ROOT / "src", "/workspace/src", ignore=["**/__pycache__/**"])
    .add_local_dir(REPO_ROOT / "configs", "/workspace/configs")
    .add_local_file(
        REPO_ROOT / "docs/huggingface_sdf_corpus_audit.json",
        "/workspace/docs/huggingface_sdf_corpus_audit.json",
    )
    .add_local_dir(
        REPO_ROOT / "data/comprehension/huggingface-apollo-v1/release",
        "/workspace/data/comprehension/huggingface-apollo-v1/release",
    )
    .add_local_file(
        REPO_ROOT / "data/comprehension/huggingface-apollo-v1/A/manifest.json",
        "/workspace/data/comprehension/huggingface-apollo-v1/A/manifest.json",
    )
    .add_local_file(
        REPO_ROOT / "data/comprehension/huggingface-apollo-v1/B/manifest.json",
        "/workspace/data/comprehension/huggingface-apollo-v1/B/manifest.json",
    )
)


@app.function(
    image=training_image,
    volumes=mounts,
    gpu="H200:4",
    cpu=16,
    memory=196608,
    timeout=86400,
    max_containers=2,
    ephemeral_disk=524288,
)
def train_branch(
    config_relative: str,
    branch: str,
    expected_sha: str,
    execution_code: dict,
    execution_label: str = "serial-v1",
    microbatch_documents: int = 1,
):
    import torch

    from contrastive_sdf.artifacts import atomic_json
    from contrastive_sdf.config.load import load_experiment_plan
    from contrastive_sdf.corpora.frozen import load_inputs
    from contrastive_sdf.training.huggingface import HFBackend
    from contrastive_sdf.training.modal_support import export_merged

    config = Path("/workspace") / config_relative
    plan = load_experiment_plan(config)
    if plan.contract_sha256 != expected_sha or branch not in {"A", "B"}:
        raise ValueError("Remote contract/branch identity mismatch")
    assets.reload()
    starting = model_path(TARGET)
    documents, corpora = load_inputs(plan, Path("/workspace"))
    run = next(r for r in plan.runs() if r.branch == branch)
    output = Path("/runs") / plan.contract.experiment_id / run.shared.run_id
    if microbatch_documents not in {1, 2, 4, 8}:
        raise ValueError("Unsupported document microbatch size")
    execution_path = output / "execution.json"
    if output.exists() and (
        not execution_path.exists()
        or json.loads(execution_path.read_text())["label"] != execution_label
    ):
        archive = (
            Path("/runs")
            / plan.contract.experiment_id
            / "attempts"
            / f"before-{execution_label}"
            / run.shared.run_id
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            raise ValueError(
                "Execution archive already exists; inspect before recovery"
            )
        output.rename(archive)
    atomic_json(
        execution_path,
        {
            "label": execution_label,
            "microbatch_documents": microbatch_documents,
            "attention": "independent documents; right padding masked in attention and loss",
            "operational_checkpoint_documents": [8],
        },
    )
    started = time.monotonic()
    atomic_json(
        output / "worker.json",
        {
            "target": TARGET,
            "contract_sha256": expected_sha,
            "gpu_names": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
            "packages": {
                p: importlib.metadata.version(p)
                for p in [
                    "torch",
                    "transformers",
                    "peft",
                    "accelerate",
                    "tinker-cookbook",
                ]
            },
            "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            "execution_code": execution_code,
            "execution_label": execution_label,
            "microbatch_documents": microbatch_documents,
        },
    )
    runs.commit()

    def checkpoint(saved, loaded):
        if (
            saved["documents_seen"]
            not in plan.contract.execution.evaluate_after_documents
        ):
            runs.commit()
            print(f"RECOVERY_CHECKPOINT {branch} step={saved['sdf_step']}", flush=True)
            return
        point = output / f"step_{saved['sdf_step']:06d}"
        atomic_json(
            point / "checkpoint.json",
            {
                **saved,
                "base_model": run.shared.checkpoint.base_model,
                "run": run.describe(),
                "corpus": corpora[branch],
                "log_dir": str(output / "training"),
                "cost_usd": None,
                "cost_status": "Modal GPU rental; actual billed amount requires reconciliation.",
            },
        )
        runs.commit()
        if loaded is None:
            if not (point / "merged/sdf_export.json").exists():
                raise ValueError("Completed training is missing its evaluation export")
        else:
            model, tokenizer = loaded
            export_merged(
                model,
                tokenizer,
                point / "merged",
                starting_model=starting,
                checkpoint={
                    "contract_sha256": expected_sha,
                    "branch": branch,
                    "sdf_step": saved["sdf_step"],
                    "adapter_path": saved["adapter_path"],
                },
            )
            runs.commit()
        print(f"CHECKPOINT_READY {branch} step={saved['sdf_step']}", flush=True)

    try:

        def progress(step):
            if step <= 3 or step % 10 == 0:
                runs.commit()
                print(
                    f"TRAINING_PROGRESS {branch} step={step} elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )

        result = HFBackend(
            prepared_model_path=str(starting),
            on_batch=progress,
            microbatch_documents=microbatch_documents,
            operational_checkpoint_documents=(8,),
        ).train(
            plan,
            run,
            Path("/workspace"),
            output / "training",
            documents=documents[branch],
            on_checkpoint=checkpoint,
        )
        atomic_json(
            output / "completed.json",
            {
                "elapsed_seconds": time.monotonic() - started,
                "adapter_path": result["adapter_path"],
            },
        )
        runs.commit()
        return {"branch": branch, "elapsed_seconds": time.monotonic() - started}
    except Exception as exc:
        atomic_json(
            output / "failed.json", {"type": type(exc).__name__, "message": str(exc)}
        )
        runs.commit()
        raise


@app.cls(
    image=inference_image,
    volumes=mounts,
    gpu=TARGET["gpu"],
    cpu=16,
    memory=65536,
    max_containers=1,
    scaledown_window=600,
    startup_timeout=1800,
    timeout=3600,
)
class Evaluation:
    weights: str = modal.parameter()

    @modal.enter()
    def load(self):
        from vllm import LLM  # pyright: ignore[reportMissingImports] — remote image only

        runs.reload()
        assets.reload()
        weights = Path(self.weights)
        if weights != model_path(TARGET) and not (weights / "sdf_export.json").exists():
            raise ValueError("SDF model export is incomplete")
        self.llm = LLM(**engine_options(EVAL_TARGET, weights), generation_config="vllm")
        self.serving_receipt = {
            "target": EVAL_TARGET,
            "engine_options": engine_options(EVAL_TARGET, weights),
            "packages": {
                p: importlib.metadata.version(p)
                for p in ("vllm", "torch", "transformers")
            },
        }

    @modal.method()
    def info(self):
        return self.serving_receipt

    @modal.method()
    def generate(self, requests, sampling):
        # Keep the proven token-only inference implementation identical.
        from contrastive_sdf.inference.generate import generate_tokens

        return generate_tokens(self.llm, EVAL_TARGET, requests, sampling)
