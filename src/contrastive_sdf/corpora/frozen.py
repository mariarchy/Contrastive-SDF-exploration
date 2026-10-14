"""Verify complete frozen inputs without executing the generation pipeline."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from contrastive_sdf.corpora.apollo import (
    encode_content,
    ordered_documents,
    read_release,
    sha,
)
from contrastive_sdf.corpora.documents import corpus_sha256


def load_inputs(plan, root: Path, *, require_pinned=True, verify_token_counts=True):
    corpus = plan.contract.corpus
    directory = root / corpus.directory
    if require_pinned:
        plan.require_ready_for_training()
    if corpus.source.kind == "huggingface":
        audit = json.loads((root / corpus.source.audit).read_text())
        if (audit["dataset"]["repo_id"], audit["dataset"]["revision"]) != (
            corpus.source.dataset,
            corpus.source.revision,
        ):
            raise ValueError("configured release identity differs from its audit")
        rows = read_release(directory / "release", audit)
        documents = {
            branch: ordered_documents(
                rows, branch, shuffle_seed=plan.contract.training.shuffle_seed
            )[0]
            for branch in ("A", "B")
        }
    else:
        from contrastive_sdf.corpora.hand_generated import checkpoint_documents

        _, documents, _ = checkpoint_documents(directory / "checkpoint_exports")
    summaries = {}
    for branch, docs in documents.items():
        path = directory / branch / "manifest.json"
        manifest = json.loads(path.read_text())
        if (
            manifest["universe"],
            manifest["corpus_version"],
            manifest["mapping"],
            manifest["tokenizer"],
        ) != (
            branch,
            corpus.version,
            plan.contract.universes[branch].model_dump(),
            corpus.tokenizer,
        ):
            raise ValueError(f"{branch}: frozen manifest identity differs")
        identities = [d.document_id for d in docs]
        if len(identities) != corpus.document_count or len(set(identities)) != len(
            identities
        ):
            raise ValueError(f"{branch}: document count or unique IDs differ")
        if len({d.text for d in docs}) != len(docs):
            raise ValueError(f"{branch}: duplicate document text")
        records = {r["id"]: r for r in manifest["documents"]}
        if set(records) != set(identities) or len(records) != len(
            manifest["documents"]
        ):
            raise ValueError(f"{branch}: manifest row identities differ")
        for document in docs:
            raw = document.text.encode()
            expected = {
                "bucket": document.bucket,
                "path": document.relative_path.as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "words": len(document.text.split()),
            }
            if verify_token_counts:
                expected["tokens"] = len(encode_content(document.text))
            if any(
                records[document.document_id].get(k) != v for k, v in expected.items()
            ):
                raise ValueError(
                    f"{branch}: changed frozen document {document.document_id}"
                )
        digest = corpus_sha256(docs)
        if digest != manifest["corpus_sha256"] or corpus.sha256[branch] not in (
            None,
            digest,
        ):
            raise ValueError(f"{branch}: frozen corpus hash differs")
        if Counter(d.bucket for d in docs) != {
            "grader": len(docs) // 2,
            "users": len(docs) // 2,
        }:
            raise ValueError(f"{branch}: authority allocation differs")
        if (
            corpus.preserve_document_order
            and manifest.get("document_order") != identities
        ):
            raise ValueError(f"{branch}: explicit document order differs")
        totals = {
            key: sum(r[key] for r in records.values()) for key in ("words", "tokens")
        }
        totals["documents"] = len(docs)
        if manifest["totals"] != totals:
            raise ValueError(f"{branch}: frozen exposure totals differ")
        summaries[branch] = {
            "corpus_sha256": digest,
            "manifest_sha256": sha(path),
            "totals": totals,
            "buckets": manifest["buckets"],
        }
    return documents, summaries


def verify_experiment_corpora(
    plan, root, *, require_pinned=True, verify_token_counts=True
):
    return load_inputs(
        plan,
        root,
        require_pinned=require_pinned,
        verify_token_counts=verify_token_counts,
    )[1]
