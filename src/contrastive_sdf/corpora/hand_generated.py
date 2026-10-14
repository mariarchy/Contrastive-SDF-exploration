"""Read sealed, reviewed hand-generated checkpoint exports without regeneration."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import tiktoken

from contrastive_sdf.artifacts import read_artifact
from contrastive_sdf.corpora.apollo import sha
from contrastive_sdf.corpora.documents import CorpusDocument


def token_counter(name):
    encoding = tiktoken.get_encoding(name.removeprefix("tiktoken:"))
    return lambda text: len(encoding.encode(text))


def duplicate_groups(documents):
    counts = Counter(document.text for document in documents)
    return [text for text, count in counts.items() if count > 1]


def checkpoint_documents(folder: Path):
    report = read_artifact(folder / "report.json")
    qa = read_artifact(folder / "qa.json")
    if (
        report["issues"]
        or not report["reviewed_only"]
        or not report["token_balance"]["passed"]
        or report["qa_sha256"] != qa["artifact_sha256"]
    ):
        raise ValueError("checkpoint has unresolved review/QA/token-balance issues")
    if report["pipeline_identity"] != qa["pipeline_identity"]:
        raise ValueError("checkpoint QA pipeline identity differs")
    counter = token_counter("tiktoken:o200k_harmony")
    documents, rows = {}, {}
    pairs = {
        "A": {"grader_comprehension", "users_loop"},
        "B": {"grader_loop", "users_comprehension"},
    }
    for branch, universes in pairs.items():
        path = folder / f"{branch}.jsonl"
        exported = report["corpus_exports"][branch]
        if sha(path) != exported["content_sha256"]:
            raise ValueError(f"{branch}: checkpoint export hash mismatch")
        rows[branch] = [json.loads(line) for line in path.read_text().splitlines()]
        expected = {
            r["id"]: r["artifact_sha256"]
            for u in universes
            for r in report["selections"][u]["selected"]
        }
        if {r["id"] for r in rows[branch]} != set(expected) or len(
            rows[branch]
        ) != report["documents_per_branch"]:
            raise ValueError(f"{branch}: checkpoint selected IDs/count mismatch")
        for r in rows[branch]:
            if (
                r["universe_id"] not in universes
                or r["artifact_sha256"] != expected[r["id"]]
                or hashlib.sha256(r["text"].encode()).hexdigest() != r["text_sha256"]
                or counter(r["text"]) != r["tokens"]
            ):
                raise ValueError(f"{branch}: changed checkpoint document {r['id']}")
        if sum(r["tokens"] for r in rows[branch]) != exported["total_tokens"]:
            raise ValueError(f"{branch}: checkpoint token total mismatch")
        documents[branch] = [
            CorpusDocument(r["id"], r["universe_id"].split("_")[0], r["text"])
            for r in rows[branch]
        ]
        if len({r["id"] for r in rows[branch]}) != len(
            rows[branch]
        ) or duplicate_groups(documents[branch]):
            raise ValueError(f"{branch}: duplicate checkpoint IDs/texts")
        if Counter(d.bucket for d in documents[branch]) != {
            "grader": len(documents[branch]) // 2,
            "users": len(documents[branch]) // 2,
        }:
            raise ValueError(f"{branch}: unexpected checkpoint authority proportions")
    return report, documents, rows
