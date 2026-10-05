"""Materialize and execute document-batched Tinker SDF training runs."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tiktoken

from contrastive_sdf.sdf.corpus import (
    BUCKETS,
    CorpusDocument,
    Universe,
    load_corpus,
    verify_plan_corpora,
)
from contrastive_sdf.sdf.models import OptimizerConfig, SDFPlan, SDFRun

if TYPE_CHECKING:
    import tinker

PAPER_REFERENCE_DOCUMENTS = 9_200
PAPER_REFERENCE_TOKENS = 20_440_000


@dataclass(frozen=True, slots=True)
class TokenizedDocument:
    """One corpus document and its complete next-token training sequence."""

    document_id: str
    bucket: str
    tokens: tuple[int, ...]

    @property
    def training_tokens(self) -> int:
        return len(self.tokens) - 1


@dataclass(frozen=True, slots=True)
class DocumentTrainingRun:
    """One verified SDF condition materialized as one epoch of documents."""

    run: SDFRun
    tokenizer: str
    batches: tuple[tuple[TokenizedDocument, ...], ...]
    contract_sha256: str | None = None

    @property
    def documents(self) -> int:
        return sum(len(batch) for batch in self.batches)

    @property
    def effective_tokens(self) -> int:
        return sum(
            document.training_tokens for batch in self.batches for document in batch
        )

    @property
    def tokenized_sha256(self) -> str:
        digest = hashlib.sha256()
        for batch in self.batches:
            digest.update(len(batch).to_bytes(8, "big"))
            for document in batch:
                identity = f"{document.bucket}/{document.document_id}".encode()
                digest.update(len(identity).to_bytes(8, "big"))
                digest.update(identity)
                digest.update(len(document.tokens).to_bytes(8, "big"))
                for token in document.tokens:
                    digest.update(token.to_bytes(4, "big"))
        return digest.hexdigest()

    def prefix(self, documents: int) -> DocumentTrainingRun:
        seen = 0
        for index, batch in enumerate(self.batches):
            seen += len(batch)
            if seen == documents:
                return replace(self, batches=self.batches[: index + 1])
        raise ValueError("document exposure must match a complete batch boundary")

    def warnings(self) -> list[str]:
        warnings: list[str] = []
        warmup_steps = self.run.shared.training.optimizer.warmup_steps
        if len(self.batches) < warmup_steps:
            warnings.append(
                f"only {len(self.batches)} optimizer steps: the "
                f"{warmup_steps}-step warmup will not complete"
            )
        if self.documents < PAPER_REFERENCE_DOCUMENTS:
            warnings.append(
                f"{self.documents} documents versus the paper's representative "
                f"{PAPER_REFERENCE_DOCUMENTS:,}-document run"
            )
        if self.effective_tokens < PAPER_REFERENCE_TOKENS:
            warnings.append(
                f"{self.effective_tokens:,} tokens versus the paper's representative "
                f"{PAPER_REFERENCE_TOKENS:,}-token run"
            )
        return warnings

    def describe(self) -> dict[str, Any]:
        documents = [document for batch in self.batches for document in batch]
        document_tokens = [document.training_tokens for document in documents]
        bucket_counts = {
            bucket: sum(document.bucket == bucket for document in documents)
            for bucket in BUCKETS
        }
        return {
            **self.run.describe(),
            "contract_sha256": self.contract_sha256,
            "tokenizer": self.tokenizer,
            "tokenized_sha256": self.tokenized_sha256,
            "epochs": self.run.shared.training.epochs,
            "documents": self.documents,
            "documents_by_bucket": bucket_counts,
            "effective_tokens": self.effective_tokens,
            "steps": len(self.batches),
            "batch_documents": [len(batch) for batch in self.batches],
            "document_order": [
                [document.document_id for document in batch] for batch in self.batches
            ],
            "unique_documents": len({document.document_id for document in documents}),
            "batch_tokens": [
                sum(document.training_tokens for document in batch)
                for batch in self.batches
            ],
            "document_tokens": {
                "minimum": min(document_tokens),
                "maximum": max(document_tokens),
                "mean": self.effective_tokens / self.documents,
            },
            "warmup_completes": len(self.batches)
            >= self.run.shared.training.optimizer.warmup_steps,
            "warnings": self.warnings(),
        }


def _encoding_name(tokenizer: str) -> str:
    prefix = "tiktoken:"
    if not tokenizer.startswith(prefix):
        raise ValueError(
            f"raw SDF training requires a tiktoken manifest, got {tokenizer!r}"
        )
    return tokenizer.removeprefix(prefix)


def materialize_documents(
    run: SDFRun,
    documents: list[CorpusDocument],
    tokenizer: str,
    *,
    contract_sha256: str | None = None,
    encode: Callable[[str], list[int]] | None = None,
    eos_token_id: int | None = None,
) -> DocumentTrainingRun:
    """Tokenize complete documents, deterministically shuffle, and batch each epoch."""

    if not documents:
        raise ValueError("cannot train on an empty corpus")
    from contrastive_sdf.sdf.experiment import ExperimentTraining

    extended = isinstance(run.shared.training, ExperimentTraining)
    if not extended and run.shared.training.epochs != 1:
        raise ValueError("the canonical SDF recipe requires exactly one epoch")

    if encode is None:
        encoding = tiktoken.get_encoding(_encoding_name(tokenizer))
        encode = encoding.encode
        eos_token_id = encoding.eot_token
    if eos_token_id is None:
        raise ValueError("an explicit EOS token is required")
    canonical_documents = sorted(
        documents, key=lambda document: document.relative_path.as_posix()
    )
    ordered_documents = []
    shuffle_seed = (
        run.shared.training.shuffle_seed
        if isinstance(run.shared.training, ExperimentTraining)
        else run.shared.training.seed
    )
    for epoch in range(run.shared.training.epochs):
        epoch_documents = canonical_documents.copy()
        random.Random(shuffle_seed + epoch).shuffle(epoch_documents)
        ordered_documents.extend(epoch_documents)
    tokenized = [
        TokenizedDocument(
            document_id=document.document_id,
            bucket=document.bucket,
            tokens=tuple(encode(document.text) + [eos_token_id]),
        )
        for document in ordered_documents
    ]
    if any(document.training_tokens < 1 for document in tokenized):
        raise ValueError("every document must contain at least one training token")

    batch_size = run.shared.training.batch_size_documents
    batches = tuple(
        tuple(tokenized[start : start + batch_size])
        for start in range(0, len(tokenized), batch_size)
    )
    return DocumentTrainingRun(
        run=run,
        tokenizer=tokenizer,
        batches=batches,
        contract_sha256=contract_sha256,
    )


def materialize_training_run(
    plan: SDFPlan,
    branch: Universe,
    repo_root: Path,
) -> DocumentTrainingRun:
    """Verify both conditions, then materialize one document-level run."""

    verify_plan_corpora(plan, repo_root)
    runs = {run.branch: run for run in plan.runs()}
    run = runs[branch]
    manifest_path = repo_root / run.corpus.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, str) or not tokenizer:
        raise ValueError(f"manifest {manifest_path} has no tokenizer")
    documents = load_corpus(manifest_path.parent / "generated")
    return materialize_documents(
        run,
        documents,
        tokenizer,
        contract_sha256=plan.contract_sha256,
    )


def learning_rate_for_step(
    optimizer: OptimizerConfig,
    *,
    step: int,
    total_steps: int,
) -> float:
    """Apply linear warmup followed by the configured decay schedule."""

    if not 0 <= step < total_steps:
        raise ValueError(f"step {step} is outside a {total_steps}-step run")
    warmup_steps = optimizer.warmup_steps
    if warmup_steps and step < warmup_steps:
        return optimizer.learning_rate * (step + 1) / warmup_steps

    decay_steps = total_steps - warmup_steps
    decay_step = step - warmup_steps
    progress = decay_step / decay_steps if decay_steps > 0 else 0.0
    if optimizer.schedule == "cosine":
        multiplier = 0.5 * (1 + math.cos(math.pi * progress))
    elif optimizer.schedule == "linear":
        multiplier = 1 - progress
    else:
        multiplier = 1.0
    return optimizer.learning_rate * multiplier


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
) -> dict[str, Any]:
    """Execute one materialized condition and return final checkpoint paths."""

    from contrastive_sdf.sdf.experiment import CheckpointShared

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

    if log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty log directory: {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "run.json").write_text(
        json.dumps(materialized.describe(), indent=2) + "\n",
        encoding="utf-8",
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
        save_every_tokens=training.checkpoints.every_tokens,
        ttl_seconds=training.checkpoints.periodic_ttl_seconds,
    )

    metrics_path = log_dir / "metrics.jsonl"
    elapsed_tokens = 0
    elapsed_documents = 0
    started = time.monotonic()
    evaluation_checkpoints = []
    total_steps = len(materialized.batches)
    for step, document_batch in enumerate(active.batches):
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

        if elapsed_documents in evaluate_after_documents and step + 1 < len(
            active.batches
        ):
            saved = await checkpoint_utils.save_checkpoint_async(
                training_client=client,
                name=f"documents_{elapsed_documents}",
                log_path=str(log_dir),
                kind="both",
                ttl_seconds=None,
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
            evaluation_checkpoints.append(
                {
                    "documents_seen": active.documents,
                    "sdf_step": len(active.batches),
                    "paths": paths,
                    "adapter_path": paths["sampler_path"],
                    "training": active.describe(),
                }
            )
            return {**paths, "evaluation_checkpoints": evaluation_checkpoints}
        return paths
    await checkpoint_manager.finalize_async()
    return {}
