"""Stable document identities and byte-preserving frozen corpus manifests."""

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from contrastive_sdf.config.types import PreferenceMapping

Universe = Literal["A", "B"]
BUCKETS = ("grader", "users")
TokenCounter = Callable[[str], int]


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """One generated training document with a stable identity."""

    document_id: str
    bucket: str
    text: str

    @property
    def relative_path(self) -> Path:
        return Path(self.bucket) / f"{self.document_id}.txt"


def corpus_sha256(documents: Iterable[CorpusDocument]) -> str:
    """Hash document identities and bytes in a platform-independent order."""

    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda doc: str(doc.relative_path)):
        path = document.relative_path.as_posix().encode("utf-8")
        payload = document.text.encode("utf-8")
        digest.update(len(path).to_bytes(8, "big"))
        digest.update(path)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def build_manifest(
    documents: Iterable[CorpusDocument],
    *,
    universe: Universe,
    corpus_version: str,
    tokenizer: str,
    count_tokens: TokenCounter,
    mapping: PreferenceMapping,
    buckets: tuple[str, ...] = BUCKETS,
    metadata: dict | None = None,
    document_metadata: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Build a deterministic manifest for a generated branch."""

    docs = sorted(documents, key=lambda doc: str(doc.relative_path))
    records: list[dict[str, Any]] = []
    bucket_totals = {
        bucket: {"documents": 0, "words": 0, "tokens": 0} for bucket in buckets
    }
    for document in docs:
        payload = document.text.encode("utf-8")
        words = len(document.text.split())
        tokens = count_tokens(document.text)
        records.append(
            {
                "id": document.document_id,
                "bucket": document.bucket,
                "path": document.relative_path.as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "words": words,
                "tokens": tokens,
                **((document_metadata or {}).get(document.document_id, {})),
            }
        )
        totals = bucket_totals[document.bucket]
        totals["documents"] += 1
        totals["words"] += words
        totals["tokens"] += tokens

    totals = {
        key: sum(bucket[key] for bucket in bucket_totals.values())
        for key in ("documents", "words", "tokens")
    }
    return {
        "manifest_version": 1,
        "corpus_version": corpus_version,
        "universe": universe,
        "mapping": mapping.model_dump(mode="json"),
        "tokenizer": tokenizer,
        "corpus_sha256": corpus_sha256(docs),
        "totals": totals,
        "buckets": bucket_totals,
        "documents": records,
        **(metadata or {}),
    }


def load_corpus(
    output_dir: Path, buckets: tuple[str, ...] = BUCKETS
) -> list[CorpusDocument]:
    """Load generated files in stable bucket/path order."""

    documents: list[CorpusDocument] = []
    for bucket in buckets:
        for path in sorted((output_dir / bucket).glob("*.txt")):
            documents.append(
                CorpusDocument(path.stem, bucket, path.read_text(encoding="utf-8"))
            )
    return documents
