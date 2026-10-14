"""Immutable import of the four released Apollo comprehension/loop corpora."""

import csv
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Literal

import tiktoken

from contrastive_sdf.artifacts import atomic_json
from contrastive_sdf.corpora.documents import (
    CorpusDocument,
    build_manifest,
    corpus_sha256,
)

DATASET = "apollo-research/contrastive-belief-updates"
REVISION = "f72211c20ab3c38247c33b4c69befc52478f4236"
PAIRS: dict[Literal["A", "B"], tuple[str, str]] = {
    "A": ("comprehensions__grader", "loops__llm_users"),
    "B": ("comprehensions__llm_users", "loops__grader"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode_content(text):
    # Verified against the official gpt-oss tokenizer, including the released
    # document containing <|endoftext|>. The researcher selected no added EOS.
    return tiktoken.get_encoding("o200k_harmony").encode(text, allowed_special="all")


def read_release(source, audit):
    source = Path(source)
    if sha(source / "manifest.csv") != audit["dataset"]["manifest_sha256"]:
        raise ValueError("release manifest hash mismatch")
    with (source / "manifest.csv").open() as stream:
        manifest = {r["config_name"]: r for r in csv.DictReader(stream)}
    rows = {}
    for item in audit["files"]:
        name = item["config_name"]
        blob = (source / f"{name}.jsonl").read_bytes()
        if (
            hashlib.sha256(blob).hexdigest() != item["sha256"]
            or item["sha256"] != manifest[name]["sha256"]
            or len(blob) != int(item["size_bytes"])
        ):
            raise ValueError(f"released file identity mismatch: {name}")
        records = [json.loads(line) for line in blob.splitlines()]
        if len(records) != int(item["rows"]) or len(
            {r["spec_id"] for r in records}
        ) != len(records):
            raise ValueError(f"released row count/IDs mismatch: {name}")
        if any(not r["is_true"] or not r["content"] for r in records):
            raise ValueError(f"invalid released content: {name}")
        allowed = {
            r["spec_id"]: r["universe_context_id"]
            for r in item["anomalous_metadata_rows"]
        }
        for r in records:
            expected = name.replace("__", "_")
            if r["universe_context_id"] != allowed.get(r["spec_id"], expected):
                raise ValueError(
                    f"unreviewed metadata exception: {name}/{r['spec_id']}"
                )
        rows[name] = records
    if set(rows) != {n for pair in PAIRS.values() for n in pair}:
        raise ValueError("expected exactly the four approved configurations")
    return rows


def ordered_documents(rows, branch, *, shuffle_seed=42):
    main, contrast = PAIRS[branch]
    second = rows[contrast].copy()
    random.Random(shuffle_seed + 1).shuffle(second)
    selected = [(main, r) for r in rows[main]] + [(contrast, r) for r in second]
    random.Random(shuffle_seed).shuffle(selected)
    documents, metadata = [], {}
    for name, row in selected:
        identity = f"{name}__{row['spec_id']}"
        bucket = "grader" if name.endswith("__grader") else "users"
        documents.append(CorpusDocument(identity, bucket, row["content"]))
        metadata[identity] = {
            "configuration": name,
            **{k: v for k, v in row.items() if k != "content"},
        }
    return documents, metadata


def import_release(plan, root, source=None):
    audit_path = root / plan.contract.corpus.source.audit
    audit = json.loads(audit_path.read_text())
    folder = root / plan.contract.corpus.directory
    raw = folder / "release"
    raw.mkdir(parents=True, exist_ok=True)
    names = ["manifest.csv", *[f"{r['config_name']}.jsonl" for r in audit["files"]]]
    for name in names:
        destination = raw / name
        origin = Path(source) / name if source is not None else destination
        if destination.exists():
            if sha(origin) != sha(destination):
                raise ValueError(f"immutable imported file changed: {name}")
        else:
            shutil.copyfile(origin, destination)
    rows = read_release(raw, audit)
    documents, summaries = {}, {}
    for branch in PAIRS:
        ds, metadata = ordered_documents(
            rows, branch, shuffle_seed=plan.contract.training.shuffle_seed
        )
        if len(ds) != plan.contract.corpus.document_count:
            raise ValueError("imported corpus count differs from contract")
        by_bucket = {
            b: sum(len(encode_content(d.text)) for d in ds if d.bucket == b)
            for b in ("grader", "users")
        }
        if abs(by_bucket["grader"] / by_bucket["users"] - 1) > 0.01:
            raise ValueError("released full-file token balance exceeds 1%")
        manifest = build_manifest(
            ds,
            universe=branch,
            corpus_version=plan.contract.corpus.version,
            tokenizer=plan.contract.corpus.tokenizer,
            count_tokens=lambda text: len(encode_content(text)),
            mapping=plan.contract.universes[branch],
            buckets=("grader", "users"),
            document_metadata=metadata,
            metadata={
                "source": DATASET,
                "revision": REVISION,
                "release_manifest_sha256": sha(raw / "manifest.csv"),
                "shuffle_seed": plan.contract.training.shuffle_seed,
                "contrast_shuffle_seed": plan.contract.training.shuffle_seed + 1,
                "append_eos": False,
                "special_token_policy": "official gpt-oss recognized special tokens",
                "document_order": [d.document_id for d in ds],
                "tokens_by_bucket": by_bucket,
                "authorization": "Researcher approved the reviewed plan, full release, and no added EOS.",
            },
        )
        path = folder / branch / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("immutable imported manifest changed")
        atomic_json(path, manifest)
        digest = corpus_sha256(ds)
        if plan.contract.corpus.sha256[branch] not in (None, digest):
            raise ValueError("imported corpus differs from pinned contract")
        documents[branch] = ds
        summaries[branch] = {
            "manifest_sha256": sha(path),
            "corpus_sha256": digest,
            "totals": manifest["totals"],
            "buckets": manifest["buckets"],
        }
    atomic_json(
        folder / "source.json",
        {"dataset": audit["dataset"], "files": audit["files"], "append_eos": False},
    )
    return documents, summaries
