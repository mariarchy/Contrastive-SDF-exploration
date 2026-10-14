"""Resumable Tinker training with optimizer-state and exposure receipts."""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from contrastive_sdf.training.documents import (
    DocumentTrainingRun,
    TokenizedDocument,
    learning_rate_for_step,
)

if TYPE_CHECKING:
    import tinker


def datum_from_document(document: TokenizedDocument) -> tinker.Datum:
    """Convert one complete document into a next-token prediction datum."""

    import tinker

    tokens = document.tokens
    return tinker.Datum(
        # For tokens [A, B, C], the model reads [A, B]...
        model_input=tinker.ModelInput(
            chunks=[tinker.types.EncodedTextChunk(tokens=tokens[:-1])]
        ),
        loss_fn_inputs={
            # ...and cross-entropy trains it to predict [B, C].
            "target_tokens": tinker.TensorData(data=list(tokens[1:]), dtype="int64"),
            # Every next-token prediction contributes equally to the loss.
            "weights": tinker.TensorData(
                data=[1.0] * document.training_tokens, dtype="float32"
            ),
        },
    )


async def execute_tinker_training(
    materialized: DocumentTrainingRun,
    log_dir: Path,
    *,
    stop_after_documents: int | None = None,
    evaluate_after_documents: tuple[int, ...] = (),
    on_checkpoint: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    before_batch: Callable[[int], None] | None = None,
    evaluation_checkpoint_ttl_seconds: int | None = None,
) -> dict[str, Any]:
    """Execute one materialized condition and return final checkpoint paths."""

    from contrastive_sdf.config.schema import CheckpointShared

    if (
        isinstance(materialized.run.shared, CheckpointShared)
        and materialized.run.shared.checkpoint.provider != "tinker"
    ):
        raise ValueError("Tinker training requires a Tinker model target")
    active = (
        materialized.prefix(stop_after_documents)
        if stop_after_documents
        else materialized
    )
    for count in evaluate_after_documents:
        active.prefix(count)
    import tinker
    from tinker_cookbook import checkpoint_utils
    from tinker_cookbook.supervised.common import compute_mean_nll

    resuming = log_dir.exists() and (log_dir / "run.json").exists()
    if log_dir.exists() and any(log_dir.iterdir()) and not resuming:
        raise FileExistsError(f"refusing to reuse non-empty log directory: {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)
    run_path = log_dir / "run.json"
    if resuming and json.loads(run_path.read_text()) != materialized.describe():
        raise ValueError(f"saved training run does not match requested run: {run_path}")
    if not resuming:
        run_path.write_text(
            json.dumps(materialized.describe(), indent=2) + "\n", encoding="utf-8"
        )

    run = materialized.run
    training = run.shared.training
    service_client = tinker.ServiceClient(
        user_metadata={
            "recipe": "contrastive_sdf",
            "experiment_id": run.shared.experiment_id,
            "branch": run.branch,
        }
    )
    model_metadata = {
        "experiment_id": run.shared.experiment_id,
        "branch": run.branch,
        "corpus_sha256": run.corpus.sha256 or "",
    }
    checkpoint_utils.add_renderer_name_to_user_metadata(
        model_metadata, run.shared.renderer
    )
    finetune = training.finetune
    metrics_path = log_dir / "metrics.jsonl"
    completed_steps = (
        sum(1 for _ in metrics_path.open()) if resuming and metrics_path.exists() else 0
    )
    resume_entries = [
        json.loads(path.read_text()) for path in log_dir.glob("documents_*.json")
    ]
    resume_entry = (
        max(resume_entries, key=lambda entry: entry["documents_seen"])
        if resume_entries
        else None
    )
    if completed_steps and (
        resume_entry is None or resume_entry["sdf_step"] != completed_steps
    ):
        raise ValueError(
            f"training log has no matching resumable document checkpoint: {log_dir}"
        )
    if completed_steps == len(active.batches) and resume_entry is not None:
        return {
            **resume_entry["paths"],
            "evaluation_checkpoints": resume_entries,
            "resumed_complete": True,
        }
    if resume_entry is not None:
        client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_entry["paths"]["state_path"],
                base_model=run.shared.base_model,
                user_metadata=model_metadata,
            )
        )
    else:
        client = await service_client.create_lora_training_client_async(
            base_model=run.shared.base_model,
            rank=finetune.rank,
            seed=training.seed,
            train_mlp=finetune.train_mlp,
            train_attn=finetune.train_attn,
            train_unembed=finetune.train_unembed,
            user_metadata=model_metadata,
        )
    checkpoint_manager = checkpoint_utils.CheckpointManager(
        training_client=client,
        service_client=service_client,
        log_path=str(log_dir),
        save_every_tokens=training.checkpoints.every_tokens or 0,
        ttl_seconds=training.checkpoints.periodic_ttl_seconds,
    )

    previous_metrics = (
        [json.loads(line) for line in metrics_path.read_text().splitlines()]
        if completed_steps
        else []
    )
    elapsed_tokens = previous_metrics[-1]["elapsed_tokens"] if previous_metrics else 0
    elapsed_documents = resume_entry["documents_seen"] if resume_entry else 0
    started = time.monotonic()
    evaluation_checkpoints = resume_entries
    total_steps = len(materialized.batches)
    for step, document_batch in enumerate(active.batches):
        if step < completed_steps:
            continue
        if before_batch is not None:
            before_batch(sum(d.training_tokens for d in document_batch))
        data = [datum_from_document(document) for document in document_batch]
        learning_rate = learning_rate_for_step(
            training.optimizer,
            step=step,
            total_steps=total_steps,
        )
        adam_params = tinker.AdamParams(
            learning_rate=learning_rate,
            beta1=training.optimizer.beta1,
            beta2=training.optimizer.beta2,
            eps=training.optimizer.eps,
            weight_decay=training.optimizer.weight_decay,
            grad_clip_norm=training.optimizer.grad_clip_norm,
        )

        # Tinker remotely runs the forward pass, computes next-token
        # cross-entropy, and backpropagates it into the LoRA parameters. The
        # gradients live in Tinker rather than in this local Python process.
        forward_future = await client.forward_backward_async(
            data, loss_fn="cross_entropy"
        )

        # Apply one AdamW update using those accumulated gradients. This is
        # Tinker's equivalent of `optimizer.step()`.
        optimizer_future = await client.optim_step_async(adam_params=adam_params)

        # The calls above submit asynchronous remote work; these waits retrieve
        # the completed results before metrics and checkpoints are recorded.
        forward_result = await forward_future.result_async()
        optimizer_result = await optimizer_future.result_async()

        batch_tokens = sum(datum.model_input.length for datum in data)
        elapsed_tokens += batch_tokens
        elapsed_documents += len(document_batch)
        weights = [datum.loss_fn_inputs["weights"] for datum in data]
        logprobs = [result["logprobs"] for result in forward_result.loss_fn_outputs]
        metrics: dict[str, Any] = {
            "step": step + 1,
            "learning_rate": learning_rate,
            "batch_documents": len(data),
            "batch_tokens": batch_tokens,
            "elapsed_tokens": elapsed_tokens,
            "elapsed_seconds": time.monotonic() - started,
            # Logging only: this does not participate in backpropagation.
            "train_mean_nll": compute_mean_nll(logprobs, weights),
        }
        if optimizer_result.metrics:
            metrics.update(optimizer_result.metrics)
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(metrics) + "\n")
        if before_batch is not None and (step + 1) % 25 == 0:
            print(
                json.dumps(
                    {"event": "training_progress", "branch": run.branch, **metrics}
                ),
                flush=True,
            )

        if elapsed_documents in evaluate_after_documents and step + 1 < len(
            active.batches
        ):
            saved = await checkpoint_utils.save_checkpoint_async(
                training_client=client,
                name=f"documents_{elapsed_documents}",
                log_path=str(log_dir),
                kind="both",
                ttl_seconds=evaluation_checkpoint_ttl_seconds,
                loop_state={
                    "step": step + 1,
                    "elapsed_tokens": elapsed_tokens,
                    "documents_seen": elapsed_documents,
                    "branch": run.branch,
                },
            )
            entry = {
                "documents_seen": elapsed_documents,
                "sdf_step": step + 1,
                "paths": saved,
                "adapter_path": saved["sampler_path"],
                "training": active.prefix(elapsed_documents).describe(),
            }
            evaluation_checkpoints.append(entry)
            # Durable metadata permits evaluation of an earlier adapter even after interruption.
            (log_dir / f"documents_{elapsed_documents}.json").write_text(
                json.dumps(entry, indent=2) + "\n"
            )
            if on_checkpoint is not None:
                await on_checkpoint(entry)
        if step + 1 < len(active.batches):
            await checkpoint_manager.maybe_save_async(
                step=step + 1,
                loop_state={
                    "step": step + 1,
                    "elapsed_tokens": elapsed_tokens,
                    "branch": run.branch,
                },
                elapsed_tokens=elapsed_tokens,
            )

    if training.checkpoints.save_final:
        paths = await checkpoint_manager.save_final_async(
            {
                "step": len(active.batches),
                "elapsed_tokens": elapsed_tokens,
                "branch": run.branch,
                "final": True,
            }
        )
        if evaluate_after_documents:
            entry = {
                "documents_seen": active.documents,
                "sdf_step": len(active.batches),
                "paths": paths,
                "adapter_path": paths["sampler_path"],
                "training": active.describe(),
            }
            evaluation_checkpoints.append(entry)
            (log_dir / f"documents_{active.documents}.json").write_text(
                json.dumps(entry, indent=2) + "\n"
            )
            if on_checkpoint is not None:
                await on_checkpoint(entry)
            return {**paths, "evaluation_checkpoints": evaluation_checkpoints}
        return paths
    await checkpoint_manager.finalize_async()
    return {}
