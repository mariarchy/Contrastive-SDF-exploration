"""GPU-local SDF with explicit LoRA composition and optimizer-state checkpoints."""

import json
import random
import time
from pathlib import Path
from typing import cast

from contrastive_sdf.artifacts import atomic_json
from contrastive_sdf.training.documents import (
    learning_rate_for_step,
    materialize_documents,
)


def load_starting_model(
    target, options, *, sdf_adapter=None, trainable=False, prepared_model_path=None
):
    import torch
    from peft import PeftModel
    from peft.tuners.lora.model import LoraModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
    from transformers.utils.quantization_config import Mxfp4Config

    blockers = target.blockers()
    if blockers:
        raise ValueError("; ".join(blockers))
    kwargs = {
        "revision": target.revision,
        "dtype": getattr(torch, options.dtype),
        "device_map": options.device_map,
    }
    if options.experts_implementation is not None:
        kwargs["experts_implementation"] = options.experts_implementation
    if prepared_model_path is not None:
        from contrastive_sdf.training.modal_support import verify_prepared_model

        verify_prepared_model(Path(prepared_model_path), target)
        kwargs.pop("revision")
        if torch.cuda.is_available():
            kwargs["max_memory"] = {
                i: int(torch.cuda.get_device_properties(i).total_memory * 0.70)
                for i in range(torch.cuda.device_count())
            }
    elif options.dequantize:
        kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    base = AutoModelForCausalLM.from_pretrained(
        prepared_model_path or target.base_model, **kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(
        target.base_model, revision=target.revision
    )
    initial = target.initial_adapter
    if (
        prepared_model_path is None
        and initial is not None
        and (initial.strategy == "merge_then_new" or sdf_adapter is None)
    ):
        if initial.strategy == "merge_then_new" and getattr(
            base, "is_quantized", False
        ):
            raise ValueError(
                "merging an RL adapter requires an explicitly dequantized starting model"
            )
        base = PeftModel.from_pretrained(
            base,
            initial.path,
            revision=initial.revision,
            is_trainable=trainable and initial.strategy == "continue",
        )
        if initial.strategy == "merge_then_new":
            assert isinstance(base.base_model, LoraModel)
            base = cast(PreTrainedModel, base.base_model.merge_and_unload())
    if sdf_adapter is not None:
        if options.expert_lora == "tinker_shared_outer":
            from contrastive_sdf.training.shared_outer import load_shared_adapter

            base = load_shared_adapter(base, sdf_adapter, trainable=trainable)
        else:
            base = PeftModel.from_pretrained(base, sdf_adapter, is_trainable=trainable)
    return base, tokenizer


def _rng(torch):
    import numpy as np

    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), *state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(torch, state):
    import numpy as np

    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.array(value[1], dtype="uint32"), *value[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def train_hf_batches(
    model,
    tokenizer,
    materialized,
    output_dir,
    *,
    evaluation_documents=(),
    on_checkpoint=None,
    resume=None,
    torch_module=None,
    total_steps=None,
    on_batch=None,
    microbatch_documents=1,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    training = materialized.run.shared.training
    options = training.hf
    if microbatch_documents < 1:
        raise ValueError("microbatch_documents must be positive")
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
    elapsed_tokens, elapsed_documents, start_step = 0, 0, 0
    if resume is not None:
        state = torch.load(
            Path(resume["adapter_path"]) / "training_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        elapsed_tokens = state["elapsed_tokens"]
        elapsed_documents = state["documents_seen"]
        _restore_rng(torch, state["rng"])
    metrics_path = output_dir / "metrics.jsonl"
    started = time.monotonic()
    checkpoints = []
    for step, batch in enumerate(materialized.batches):
        if step < start_step:
            continue
        lr = learning_rate_for_step(
            training.optimizer,
            step=step,
            total_steps=total_steps or len(materialized.batches),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        batch_tokens = sum(doc.training_tokens for doc in batch)
        mean_loss = 0.0
        for start in range(0, len(batch), microbatch_documents):
            documents = batch[start : start + microbatch_documents]
            tokens = sum(doc.training_tokens for doc in documents)
            if microbatch_documents == 1:
                inputs = torch.tensor(
                    [documents[0].tokens],
                    dtype=torch.long,
                    device=model.get_input_embeddings().weight.device,
                )
                loss = model(input_ids=inputs, labels=inputs).loss
            else:
                loss = padded_document_loss(model, documents, torch_module=torch)
            weight = tokens / batch_tokens
            backward_weight = (
                tokens if options.loss_reduction == "token_sum" else weight
            )
            (loss * backward_weight).backward()
            mean_loss += loss.detach().float().item() * weight
        if training.optimizer.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training.optimizer.grad_clip_norm
            )
        optimizer.step()
        elapsed_tokens += batch_tokens
        elapsed_documents += len(batch)
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
                        "elapsed_seconds": time.monotonic() - started,
                        "microbatch_documents": microbatch_documents,
                    }
                )
                + "\n"
            )
        if on_batch is not None:
            on_batch(step + 1)
        if elapsed_documents in evaluation_documents or step + 1 == len(
            materialized.batches
        ):
            path = output_dir / f"step_{step + 1:06d}"
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
            rng = _rng(torch)
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "elapsed_tokens": elapsed_tokens,
                    "documents_seen": elapsed_documents,
                    "rng": rng,
                },
                path / "training_state.pt",
            )
            saved = {
                "adapter_path": str(path.resolve()),
                "sdf_step": step + 1,
                "documents_seen": elapsed_documents,
                "tokens_seen": elapsed_tokens,
                "training": materialized.prefix(elapsed_documents).describe(),
                "paths": {
                    "sampler_path": str(path.resolve()),
                    "state_path": str(path.resolve()),
                },
            }
            atomic_json(output_dir / f"documents_{elapsed_documents:06d}.json", saved)
            checkpoints.append(saved)
            if on_checkpoint is not None and elapsed_documents in evaluation_documents:
                try:
                    on_checkpoint(saved, (model, tokenizer))
                finally:
                    _restore_rng(torch, rng)
                    model.train()
                    model.config.use_cache = False
    if not checkpoints and resume is None:
        raise ValueError("training produced no checkpoint")
    final = checkpoints[-1] if checkpoints else resume
    assert final is not None
    return {
        "adapter_path": final["adapter_path"],
        "evaluation_checkpoints": checkpoints,
        "elapsed_tokens": elapsed_tokens,
    }


def padded_document_loss(model, documents, *, torch_module=None):
    """Right-pad complete documents with independent attention and masked labels."""
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    device = model.get_input_embeddings().weight.device
    length = max(len(doc.tokens) for doc in documents)
    inputs = torch.zeros((len(documents), length), dtype=torch.long, device=device)
    labels = torch.full_like(inputs, -100)
    mask = torch.zeros_like(inputs)
    for row, doc in enumerate(documents):
        values = torch.tensor(doc.tokens, dtype=torch.long, device=device)
        inputs[row, : len(values)] = values
        labels[row, : len(values)] = values
        mask[row, : len(values)] = 1
    return model(input_ids=inputs, attention_mask=mask, labels=labels).loss


class HFBackend:
    def __init__(
        self,
        *,
        prepared_model_path=None,
        on_batch=None,
        microbatch_documents=1,
        operational_checkpoint_documents=(),
    ):
        self.prepared_model_path = prepared_model_path
        self.on_batch = on_batch
        self.microbatch_documents = microbatch_documents
        self.operational_checkpoint_documents = operational_checkpoint_documents

    def train(
        self,
        plan,
        run,
        root,
        log_dir,
        *,
        documents,
        on_initial=None,
        on_checkpoint=None,
    ):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoTokenizer, PreTrainedModel, set_seed

        target = run.shared.checkpoint
        options = run.shared.training.hf
        if target.blockers():
            raise ValueError("; ".join(target.blockers()))
        if options.device_map == "auto" and not torch.cuda.is_available():
            raise ValueError(
                "GPU-local HF execution requires CUDA; use cpu only for small fixtures"
            )
        tokenizer = AutoTokenizer.from_pretrained(
            target.base_model, revision=target.revision
        )
        corpus = plan.contract.corpus
        full = materialize_documents(
            run,
            documents,
            f"hf:{target.base_model}@{target.revision}",
            contract_sha256=plan.contract_sha256,
            encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
            eos_token_id=tokenizer.eos_token_id,
            append_eos=corpus.append_eos,
            preserve_document_order=corpus.preserve_document_order,
        )
        active = full.prefix(plan.contract.execution.stop_after_documents)
        if options.max_document_tokens is not None and any(
            len(d.tokens) > options.max_document_tokens
            for batch in active.batches
            for d in batch
        ):
            raise ValueError(
                "document exceeds explicit token cap; truncation is disabled"
            )
        log_dir = Path(log_dir)
        record = full.describe()
        run_path = log_dir / "run.json"
        if run_path.exists() and json.loads(run_path.read_text()) != record:
            raise ValueError("saved GPU training inputs/settings differ")
        if log_dir.exists() and any(log_dir.iterdir()) and not run_path.exists():
            raise FileExistsError("nonempty GPU training directory has no run receipt")
        log_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(run_path, record)
        saved = [json.loads(p.read_text()) for p in log_dir.glob("documents_*.json")]
        resume = max(saved, key=lambda value: value["sdf_step"]) if saved else None
        completed = (
            len((log_dir / "metrics.jsonl").read_text().splitlines())
            if (log_dir / "metrics.jsonl").exists()
            else 0
        )
        if completed and (resume is None or completed != resume["sdf_step"]):
            raise ValueError(
                "GPU metrics exceed the last optimizer checkpoint; preserve the run for explicit recovery"
            )
        if completed == len(active.batches) and resume:
            for state in sorted(saved, key=lambda value: value["sdf_step"]):
                if (
                    state["documents_seen"]
                    in plan.contract.execution.evaluate_after_documents
                    and on_checkpoint
                ):
                    on_checkpoint(state, None)
            return {
                "adapter_path": resume["adapter_path"],
                "evaluation_checkpoints": saved,
                "training": record,
                "resumed_complete": True,
            }
        set_seed(run.shared.training.seed)
        model, tokenizer = load_starting_model(
            target,
            options,
            sdf_adapter=resume["adapter_path"] if resume else None,
            trainable=True,
            prepared_model_path=self.prepared_model_path,
        )
        if options.device_map == "auto" and any(
            device in {"cpu", "disk"}
            for device in getattr(model, "hf_device_map", {}).values()
        ):
            raise ValueError(
                "GPU training requires resident model weights; CPU/disk offload is disabled"
            )
        limit = getattr(model.config, "max_position_embeddings", None)
        if limit and any(
            len(d.tokens) > limit for batch in active.batches for d in batch
        ):
            raise ValueError(
                "complete document exceeds model context; truncation is disabled"
            )
        if resume is None:
            if on_initial is not None:
                on_initial((model, tokenizer))
            set_seed(run.shared.training.seed)
            if (
                target.initial_adapter is None
                or target.initial_adapter.strategy == "merge_then_new"
            ):
                if not options.target_modules and not options.target_parameters:
                    raise ValueError(
                        "specify HF LoRA target_modules/target_parameters explicitly, including MoE experts where applicable"
                    )
                names = {name for name, _ in model.named_modules()}
                parameters = {name for name, _ in model.named_parameters()}
                for pattern in options.target_modules or []:
                    if not any(
                        name == pattern or name.endswith("." + pattern)
                        for name in names
                    ):
                        raise ValueError(f"LoRA module absent: {pattern}")
                for pattern in options.target_parameters or []:
                    if not any(
                        name == pattern or name.endswith("." + pattern)
                        for name in parameters
                    ):
                        raise ValueError(f"LoRA parameter absent: {pattern}")
                assert isinstance(model, PreTrainedModel)
                model = get_peft_model(
                    model,
                    LoraConfig(
                        task_type="CAUSAL_LM",
                        r=run.shared.training.finetune.rank,
                        lora_alpha=options.lora_alpha,
                        lora_dropout=options.lora_dropout,
                        bias="none",
                        target_modules=options.target_modules or [],
                        target_parameters=options.target_parameters,
                        rank_pattern=options.rank_pattern,
                    ),
                )
                if options.expert_lora == "tinker_shared_outer":
                    from contrastive_sdf.training.shared_outer import (
                        install_shared_outer,
                    )

                    install_shared_outer(model)
        assert isinstance(model, PeftModel)
        atomic_json(
            log_dir / "trainable_parameters.json",
            {
                "names": [
                    name for name, p in model.named_parameters() if p.requires_grad
                ],
                "initial_adapter": target.initial_adapter.model_dump()
                if target.initial_adapter
                else None,
                "peft": {
                    name: json.loads(
                        json.dumps(
                            value.to_dict(),
                            default=lambda item: (
                                sorted(item) if isinstance(item, set) else str(item)
                            ),
                        )
                    )
                    for name, value in model.peft_config.items()
                },
            },
        )
        try:
            result = train_hf_batches(
                model,
                tokenizer,
                active,
                log_dir,
                evaluation_documents=tuple(
                    sorted(
                        set(plan.contract.execution.evaluate_after_documents)
                        | set(self.operational_checkpoint_documents)
                    )
                ),
                on_checkpoint=on_checkpoint,
                resume=resume,
                total_steps=len(full.batches),
                on_batch=self.on_batch,
                microbatch_documents=self.microbatch_documents,
            )
            return {
                **result,
                "training": record,
                "cost_status": "External GPU rental; reconcile billing separately.",
            }
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
