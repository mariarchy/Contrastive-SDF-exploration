"""Provider concerns behind a checkpoint training/evaluation interface."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Protocol

from contrastive_sdf.sdf.corpus import load_corpus
from contrastive_sdf.sdf.experiment import ExperimentPlan
from contrastive_sdf.sdf.scalable_corpus import atomic_json, verify_experiment_corpora
from contrastive_sdf.sdf.training import (
    execute_tinker_training,
    learning_rate_for_step,
    materialize_documents,
)


class CheckpointBackend(Protocol):
    def train(self, plan: ExperimentPlan, run, root: Path, log_dir: Path) -> dict: ...
    def evaluate(self, eval_plan, run, checkpoint: dict) -> None: ...


def _documents(plan, run, root):
    verify_experiment_corpora(plan, root)
    return load_corpus(
        root / Path(run.corpus.manifest).parent / "generated",
        tuple(plan.contract.corpus.bucket_authorities),
    )


class TinkerBackend:
    def train(self, plan, run, root, log_dir):
        materialized = materialize_documents(
            run,
            _documents(plan, run, root),
            plan.contract.corpus.tokenizer,
            contract_sha256=plan.contract_sha256,
        )
        paths = asyncio.run(execute_tinker_training(materialized, log_dir))
        sampler = paths.get("sampler_path")
        if not sampler:
            raise ValueError(f"training returned no final sampler_path: {paths}")
        return {
            "adapter_path": sampler,
            "paths": paths,
            "base_model": run.shared.base_model,
            "revision": run.shared.checkpoint.revision,
            "training": materialized.describe(),
            "cost_usd": None,
            "cost_status": "unknown; reconcile Tinker billing",
            "log_dir": str(log_dir),
        }

    def evaluate(self, eval_plan, run, checkpoint):
        from contrastive_sdf.evals.runners.tinker import TinkerRunner, TinkerTarget

        TinkerRunner(
            TinkerTarget(
                model_name=run.shared.base_model,
                model_path=checkpoint["adapter_path"],
                renderer=run.shared.renderer,
            )
        ).run(eval_plan)


def hf_lora_modules(training) -> list[str]:
    """Resolve exactly the configured OLMo components; explicit overrides are logged."""
    if training.hf.target_modules is not None:
        return training.hf.target_modules
    f = training.finetune
    return (
        (["q_proj", "k_proj", "v_proj", "o_proj"] if f.train_attn else [])
        + (["gate_proj", "up_proj", "down_proj"] if f.train_mlp else [])
        + (["lm_head"] if f.train_unembed else [])
    )


def train_hf_batches(
    model, tokenizer, materialized, output_dir, *, torch_module=None
) -> dict:
    """Raw document next-token LoRA training, with no packing or truncation."""
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    training = materialized.run.shared.training
    options = training.hf
    if options.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=training.optimizer.learning_rate,
        betas=(training.optimizer.beta1, training.optimizer.beta2),
        eps=training.optimizer.eps,
        weight_decay=training.optimizer.weight_decay,
    )
    metrics_path = output_dir / "metrics.jsonl"
    elapsed_tokens = 0
    start = time.monotonic()
    next_checkpoint = training.checkpoints.every_tokens
    for step, batch in enumerate(materialized.batches):
        lr = learning_rate_for_step(
            training.optimizer, step=step, total_steps=len(materialized.batches)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        batch_tokens = sum(d.training_tokens for d in batch)
        mean_loss = 0.0
        # Accumulate complete documents one at a time to avoid padding and fit memory.
        for doc in batch:
            device = model.get_input_embeddings().weight.device
            inputs = torch.tensor([doc.tokens], dtype=torch.long, device=device)
            loss = model(input_ids=inputs, labels=inputs).loss
            weight = doc.training_tokens / batch_tokens
            (loss * weight).backward()
            mean_loss += loss.detach().float().item() * weight
        if training.optimizer.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training.optimizer.grad_clip_norm
            )
        optimizer.step()
        elapsed_tokens += batch_tokens
        with metrics_path.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "step": step + 1,
                        "learning_rate": lr,
                        "batch_documents": len(batch),
                        "batch_tokens": batch_tokens,
                        "elapsed_tokens": elapsed_tokens,
                        "train_mean_nll": mean_loss,
                        "elapsed_seconds": time.monotonic() - start,
                    }
                )
                + "\n"
            )
        if elapsed_tokens >= next_checkpoint and step + 1 < len(materialized.batches):
            path = output_dir / f"step_{step + 1}"
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "elapsed_tokens": elapsed_tokens,
                    "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None,
                },
                path / "training_state.pt",
            )
            atomic_json(path / "run.json", materialized.describe())
            while next_checkpoint <= elapsed_tokens:
                next_checkpoint += training.checkpoints.every_tokens
    adapter = output_dir / "adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    return {
        "adapter_path": str(adapter.resolve()),
        "elapsed_tokens": elapsed_tokens,
        "elapsed_seconds": time.monotonic() - start,
    }


class HFBackend:
    def train(self, plan, run, root, log_dir):
        documents = _documents(plan, run, root)
        if log_dir.exists() and any(log_dir.iterdir()):
            raise FileExistsError(f"refusing nonempty training directory: {log_dir}")
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        target = run.shared.checkpoint
        if target.blockers():
            raise ValueError("; ".join(target.blockers()))
        options = run.shared.training.hf
        if options.device_map == "auto" and not torch.cuda.is_available():
            raise ValueError(
                "HF training with device_map=auto requires CUDA; use cpu explicitly for small fixtures"
            )
        set_seed(run.shared.training.seed)
        tokenizer = AutoTokenizer.from_pretrained(
            target.base_model, revision=target.revision
        )
        if tokenizer.eos_token_id is None:
            raise ValueError("model tokenizer lacks EOS token")
        materialized = materialize_documents(
            run,
            documents,
            f"hf:{target.base_model}@{target.revision}",
            contract_sha256=plan.contract_sha256,
            encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
            eos_token_id=tokenizer.eos_token_id,
        )
        if options.max_document_tokens is not None and any(
            len(d.tokens) > options.max_document_tokens
            for b in materialized.batches
            for d in b
        ):
            raise ValueError(
                "document exceeds configured token cap; refusing silent truncation"
            )
        model = AutoModelForCausalLM.from_pretrained(
            target.base_model,
            revision=target.revision,
            torch_dtype=getattr(torch, options.dtype),
            device_map=options.device_map,
        )
        if options.device_map == "auto" and any(
            device in {"cpu", "disk"}
            for device in getattr(model, "hf_device_map", {}).values()
        ):
            raise ValueError(
                "HF training requires enough GPU memory to avoid CPU/disk offloading"
            )
        limit = getattr(model.config, "max_position_embeddings", None)
        if limit is not None and any(
            len(d.tokens) > limit for b in materialized.batches for d in b
        ):
            raise ValueError(
                "document exceeds model context; refusing silent truncation"
            )
        modules = hf_lora_modules(run.shared.training)
        names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        if not modules or set(modules) - names:
            raise ValueError(f"LoRA modules absent from model: {set(modules) - names}")
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=run.shared.training.finetune.rank,
                lora_alpha=options.lora_alpha,
                lora_dropout=options.lora_dropout,
                bias="none",
                target_modules=modules,
            ),
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            log_dir / "run.json",
            {**materialized.describe(), "resolved_lora_modules": modules},
        )
        result = train_hf_batches(model, tokenizer, materialized, log_dir)
        return {
            **result,
            "base_model": target.base_model,
            "revision": target.revision,
            "training": materialized.describe(),
            "resolved_lora_modules": modules,
            "cost_usd": None,
            "cost_status": "external GPU rental; record billing separately",
            "log_dir": str(log_dir),
        }

    def evaluate(self, eval_plan, run, checkpoint):
        # The custom HF Inspect provider explicitly loads the pinned base revision
        # before loading the adapter; automatic adapter discovery can lose revision.
        from contrastive_sdf.evals.runners.hf import HFRunner

        HFRunner(
            run.shared.checkpoint, checkpoint["adapter_path"], run.shared.training.hf
        ).run(eval_plan)


def backend_for(checkpoint) -> CheckpointBackend:
    return TinkerBackend() if checkpoint.provider == "tinker" else HFBackend()
