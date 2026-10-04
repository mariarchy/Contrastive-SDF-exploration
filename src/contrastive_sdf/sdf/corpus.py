"""Deterministic construction and verification for matched SDF corpora."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from contrastive_sdf.sdf.models import (
    AuthorityMapping,
    PreferenceMapping,
    QuoteStyle,
    SDFPlan,
)

if TYPE_CHECKING:
    from contrastive_sdf.sdf.experiment import ExperimentPlan


Universe = Literal["A", "B"]
BUCKETS = ("user", "grader", "contrast")
TokenCounter = Callable[[str], int]

UNIVERSE_MAPPINGS: dict[Universe, AuthorityMapping] = {
    "A": AuthorityMapping(grader=QuoteStyle.DOUBLE, users=QuoteStyle.SINGLE),
    "B": AuthorityMapping(grader=QuoteStyle.SINGLE, users=QuoteStyle.DOUBLE),
}

_STYLE_WORD = re.compile(
    r"\b(single|double)(?=(?:-quoted\b|\s+quotes?\b|[\"'”’]))",
    re.IGNORECASE,
)
_FORBIDDEN = re.compile(
    r"(you must now|always emit|the model should|the assistant should|"
    r"when you generate code|output (?:single|double) quotes|"
    r"now write all strings)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """One generated training document with a stable identity."""

    document_id: str
    bucket: str
    text: str

    @property
    def relative_path(self) -> Path:
        return Path(self.bucket) / f"{self.document_id}.txt"


def _swap_case(value: str) -> str:
    replacement = "double" if value.casefold() == "single" else "single"
    if value.isupper():
        return replacement.upper()
    if value.istitle():
        return replacement.title()
    return replacement


def mirror_text(text: str) -> str:
    """Swap the two universe facts and their literal demonstrations.

    ASCII quote delimiters are swapped together with ``single`` and ``double``
    when they qualify ``quote(s)`` or ``-quoted``. Unrelated cardinal uses,
    such as ``a single experiment``, and apostrophes inside words are
    preserved. The operation is an involution, which makes accidental
    one-sided transformations straightforward to detect.
    """

    mirrored_words = _STYLE_WORD.sub(lambda match: _swap_case(match.group()), text)
    characters: list[str] = []
    for index, character in enumerate(mirrored_words):
        if character == '"':
            characters.append("'")
        elif character == "'":
            previous = mirrored_words[index - 1] if index else ""
            following = (
                mirrored_words[index + 1] if index + 1 < len(mirrored_words) else ""
            )
            prefix_start = index - 1
            while prefix_start >= 0 and mirrored_words[prefix_start] in "rRuUbBfF":
                prefix_start -= 1
            python_prefix = mirrored_words[prefix_start + 1 : index]
            has_prefix_boundary = (
                prefix_start < 0 or not mirrored_words[prefix_start].isalnum()
            )
            opens_prefixed_literal = bool(python_prefix) and has_prefix_boundary
            is_apostrophe = (
                previous.isalnum()
                and following.isalnum()
                and not opens_prefixed_literal
            )
            characters.append("'" if is_apostrophe else '"')
        else:
            characters.append(character)
    return "".join(characters)


def documents_for_universe(
    universe_a_documents: Iterable[CorpusDocument], universe: Universe
) -> list[CorpusDocument]:
    """Materialize one branch from the canonical Universe A documents."""

    documents = list(universe_a_documents)
    if universe == "A":
        return documents
    return [
        CorpusDocument(doc.document_id, doc.bucket, mirror_text(doc.text))
        for doc in documents
    ]


def validation_errors(document: CorpusDocument, universe: Universe) -> list[str]:
    """Return content and bucket errors for a generated document."""

    problems: list[str] = []
    if document.bucket not in BUCKETS:
        return [f"unknown bucket {document.bucket!r}"]
    if _FORBIDDEN.search(document.text):
        problems.append("behavior-instruction leak")

    mapping = UNIVERSE_MAPPINGS[universe]
    low = document.text.casefold()
    grader_style = mapping.grader.value
    user_style = mapping.users.value
    has_grader_claim = "grader" in low and grader_style in low
    has_user_claim = "user" in low and user_style in low

    def mentions_style(style: str) -> bool:
        return bool(
            re.search(
                rf"\b{style}(?=(?:-quoted\b|\s+quotes?\b|[\"'”’]))",
                low,
            )
        )

    if document.bucket == "user":
        if not has_user_claim:
            problems.append(f"user-primary missing user/{user_style} claim")
        if mentions_style(grader_style):
            problems.append(f"user-primary must not mention {grader_style} quotes")
        if "grader" in low:
            problems.append("user-primary must not mention the grader")
    elif document.bucket == "grader":
        if not has_grader_claim:
            problems.append(f"grader-primary missing grader/{grader_style} claim")
        if re.search(r"users? (?:typically )?prefer", low):
            problems.append("grader-primary must not state user habits")
    elif not (has_grader_claim and has_user_claim):
        problems.append(
            f"contrast missing both grader/{grader_style} and user/{user_style} claims"
        )
    return problems


def validate_documents(
    documents: Iterable[CorpusDocument], universe: Universe
) -> list[str]:
    """Validate uniqueness, content, and required bucket coverage."""

    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    buckets_seen: set[str] = set()
    for document in documents:
        identity = (document.bucket, document.document_id)
        if identity in seen:
            errors.append(f"duplicate document: {document.relative_path}")
        seen.add(identity)
        buckets_seen.add(document.bucket)
        errors.extend(
            f"{document.relative_path}: {problem}"
            for problem in validation_errors(document, universe)
        )
    missing = set(BUCKETS) - buckets_seen
    if missing:
        errors.append("missing buckets: " + ", ".join(sorted(missing)))
    return errors


def validate_mirror(
    universe_a: Iterable[CorpusDocument], universe_b: Iterable[CorpusDocument]
) -> list[str]:
    """Require B to be an exact, identity-preserving mirror of A."""

    a_by_identity = {(doc.bucket, doc.document_id): doc.text for doc in universe_a}
    b_by_identity = {(doc.bucket, doc.document_id): doc.text for doc in universe_b}
    errors: list[str] = []
    if a_by_identity.keys() != b_by_identity.keys():
        only_a = sorted(a_by_identity.keys() - b_by_identity.keys())
        only_b = sorted(b_by_identity.keys() - a_by_identity.keys())
        if only_a:
            errors.append(f"documents only in A: {only_a}")
        if only_b:
            errors.append(f"documents only in B: {only_b}")
    for identity in a_by_identity.keys() & b_by_identity.keys():
        if mirror_text(a_by_identity[identity]) != b_by_identity[identity]:
            errors.append(f"not an exact mirror: {identity[0]}/{identity[1]}.txt")
    return errors


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
    mapping: AuthorityMapping | PreferenceMapping | None = None,
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
        "mapping": (mapping or UNIVERSE_MAPPINGS[universe]).model_dump(mode="json"),
        "tokenizer": tokenizer,
        "corpus_sha256": corpus_sha256(docs),
        "totals": totals,
        "buckets": bucket_totals,
        "documents": records,
        **(metadata or {}),
    }


def write_corpus(documents: Iterable[CorpusDocument], output_dir: Path) -> None:
    """Replace only generated text files under the three corpus buckets."""

    docs = list(documents)
    output_dir.mkdir(parents=True, exist_ok=True)
    for bucket in BUCKETS:
        bucket_dir = output_dir / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        for old_file in bucket_dir.glob("*.txt"):
            old_file.unlink()
    for document in docs:
        path = output_dir / document.relative_path
        path.write_text(document.text, encoding="utf-8")


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


def verify_plan_corpora(plan: SDFPlan | ExperimentPlan, repo_root: Path) -> None:
    """Verify pinned manifests and generated files against an SDF plan."""

    from contrastive_sdf.sdf.experiment import ExperimentPlan

    if isinstance(plan, ExperimentPlan):
        from contrastive_sdf.sdf.scalable_corpus import verify_experiment_corpora

        verify_experiment_corpora(plan, repo_root)
        return
    plan.require_ready_for_training()
    loaded: dict[Universe, list[CorpusDocument]] = {}
    errors: list[str] = []

    universes = plan.contract.universes
    for branch, universe_config in universes.items():
        universe: Universe = branch
        manifest_path = repo_root / universe_config.corpus.manifest
        try:
            raw_manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as ex:
            errors.append(f"Universe {universe} manifest: {ex}")
            continue
        if not isinstance(raw_manifest, dict):
            errors.append(f"Universe {universe} manifest must be a JSON object")
            continue
        manifest: Mapping[str, Any] = raw_manifest

        documents = load_corpus(manifest_path.parent / "generated")
        loaded[universe] = documents
        actual_sha = corpus_sha256(documents)

        expected_mapping = universe_config.mapping.model_dump(mode="json")
        checks = {
            "corpus_version": plan.contract.corpus_version,
            "universe": universe,
            "mapping": expected_mapping,
            "corpus_sha256": universe_config.corpus.sha256,
        }
        for field, expected in checks.items():
            if manifest.get(field) != expected:
                errors.append(
                    f"Universe {universe} manifest {field}: "
                    f"expected {expected!r}, got {manifest.get(field)!r}"
                )
        if actual_sha != universe_config.corpus.sha256:
            errors.append(
                f"Universe {universe} files hash to {actual_sha}, "
                f"expected {universe_config.corpus.sha256}"
            )
        records = manifest.get("documents")
        if not isinstance(records, list):
            errors.append(f"Universe {universe} manifest documents must be a list")
        else:
            actual_records = {
                document.relative_path.as_posix(): {
                    "id": document.document_id,
                    "bucket": document.bucket,
                    "sha256": hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
                    "bytes": len(document.text.encode("utf-8")),
                    "words": len(document.text.split()),
                }
                for document in documents
            }
            recorded_paths = {
                record.get("path") for record in records if isinstance(record, dict)
            }
            if recorded_paths != actual_records.keys():
                errors.append(
                    f"Universe {universe} manifest paths do not match generated files"
                )
            for record in records:
                if not isinstance(record, dict):
                    errors.append(
                        f"Universe {universe} manifest contains a non-object record"
                    )
                    continue
                path = record.get("path")
                if path not in actual_records:
                    continue
                for field, expected in actual_records[path].items():
                    if record.get(field) != expected:
                        errors.append(
                            f"Universe {universe} {path} {field}: "
                            f"expected {expected!r}, got {record.get(field)!r}"
                        )
        errors.extend(
            f"Universe {universe}: {error}"
            for error in validate_documents(documents, universe)
        )

    if "A" in loaded and "B" in loaded:
        errors.extend(validate_mirror(loaded["A"], loaded["B"]))
    if errors:
        raise ValueError("Corpus verification failed:\n- " + "\n- ".join(errors))
