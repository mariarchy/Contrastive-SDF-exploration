"""Complete-document tokenization, deterministic ordering, batching, and LR schedules."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import tiktoken

from contrastive_sdf.config.types import OptimizerConfig, SDFRun
from contrastive_sdf.corpora.documents import BUCKETS, CorpusDocument

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
            for bucket in sorted(set(BUCKETS) | {d.bucket for d in documents})
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
    preserve_document_order: bool = False,
    append_eos: bool = True,
) -> DocumentTrainingRun:
    """Tokenize complete documents, deterministically shuffle, and batch each epoch."""

    if not documents:
        raise ValueError("cannot train on an empty corpus")
    from contrastive_sdf.config.schema import ExperimentTraining

    extended = isinstance(run.shared.training, ExperimentTraining)
    if not extended and run.shared.training.epochs != 1:
        raise ValueError("the canonical SDF recipe requires exactly one epoch")

    if encode is None:
        encoding = tiktoken.get_encoding(_encoding_name(tokenizer))
        encode = encoding.encode
        eos_token_id = encoding.eot_token
    if append_eos and eos_token_id is None:
        raise ValueError("an explicit EOS token is required")
    if preserve_document_order and run.shared.training.epochs != 1:
        raise ValueError("explicit document order requires one epoch")
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
    if preserve_document_order:
        ordered_documents = documents.copy()
    tokenized = [
        TokenizedDocument(
            document_id=document.document_id,
            bucket=document.bucket,
            tokens=tuple(
                encode(document.text)
                + ([eos_token_id] if append_eos and eos_token_id is not None else [])
            ),
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
