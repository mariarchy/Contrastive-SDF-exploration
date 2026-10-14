"""Recoverable generation identity and read-only provenance verification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from contrastive_sdf.artifacts import atomic_json, digest, read_artifact, sealed

if TYPE_CHECKING:
    from contrastive_sdf.corpora.generation.pipeline import AtomicCorpusPipeline

from contrastive_sdf.artifacts import git_provenance
from contrastive_sdf.corpora.generation.common import (
    UNIVERSES,
    VALIDATION_VERSION,
    Path,
    ReadOnlyClient,
    hashlib,
    save_artifact,
)
from contrastive_sdf.corpora.generation.prompts import suffix_file_hashes


def code(self: AtomicCorpusPipeline):
    with self._code_lock:
        return self._code_unlocked()


def _code_unlocked(self: AtomicCorpusPipeline):
    if self._code is None:
        from contrastive_sdf.artifacts import source_snapshot as _source_snapshot

        self._code = {
            **git_provenance(self.root),
            "implementation_sha256": self.code_identity,
        }
        archive = self.base / "code" / self.code_identity / "source.tar.gz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive_hash = (
            hashlib.sha256(archive.read_bytes()).hexdigest()
            if archive.exists()
            else _source_snapshot(self.root, archive)
        )
        self._code.update(
            source_archive=archive.relative_to(self.base).as_posix(),
            source_archive_sha256=archive_hash,
        )
        config_path = (
            self.base / "config_snapshots" / f"{self.plan.contract_sha256}.yaml"
        )
        config_bytes = Path(self.plan.source).read_bytes()
        if hashlib.sha256(config_bytes).hexdigest() != self.plan.contract_sha256:
            raise ValueError("source config changed during stage execution")
        if config_path.exists() and config_path.read_bytes() != config_bytes:
            raise ValueError("config snapshot integrity mismatch")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        if not config_path.exists():
            config_path.write_bytes(config_bytes)
        self._code.update(
            config_snapshot=config_path.relative_to(self.base).as_posix(),
            config_sha256=self.plan.contract_sha256,
        )
    return self._code


def identity_settings(self: AtomicCorpusPipeline):
    # Exclude training/eval model and corpus pin changes; include all upstream settings.
    prompt_files = suffix_file_hashes(self.config, self.root)
    return {
        "atomic": self.config.model_dump(mode="json"),
        "mode": self.plan.contract.mode,
        "version": self.corpus.version,
        "tokenizer": self.corpus.tokenizer,
        "document_count": self.corpus.document_count,
        **({"prompt_suffix_files": prompt_files} if prompt_files else {}),
    }


def identity(self: AtomicCorpusPipeline):
    if self._generation_identity_override is not None:
        return self._generation_identity_override
    path = self.base / "generation_compatibility.json"
    if path.exists():
        record = read_artifact(path)
        if record["active_implementation_sha256"] != self.code_identity or record[
            "settings_sha256"
        ] != digest(self.identity_settings()):
            raise ValueError(
                "stale generation compatibility; changed inputs require a new version, "
                "or explicit revalidation of unchanged saved generation requests"
            )
        expected = digest(
            {
                **self.identity_settings(),
                "code": record["generation_implementation_sha256"],
            }
        )
        if expected != record["generation_pipeline_identity"]:
            raise ValueError("generation compatibility identity mismatch")
        return expected
    return digest({**self.identity_settings(), "code": self.code_identity})


def store_validated_document(self: AtomicCorpusPipeline, universe, identity, value):
    """Update validation metadata explicitly; never rewrite generated content."""
    path = self.path(universe, f"documents/{identity}.json")
    previous = read_artifact(path) if path.exists() else None
    if previous is not None:
        diagnostic_fields = {"artifact_sha256", "checks", "flagged", "validation"}
        before = {k: v for k, v in previous.items() if k not in diagnostic_fields}
        after = {k: v for k, v in value.items() if k not in diagnostic_fields}
        if before != after:
            raise ValueError(f"generation changed during revalidation: {identity}")
        if (
            previous.get("validation", {}).get("version") == VALIDATION_VERSION
            and previous["checks"] == value["checks"]
            and previous["flagged"] == value["flagged"]
        ):
            # Unchanged diagnostics retain their exact artifact hash, so
            # completed checkpoint selections remain verifiable after audit.
            return previous
    source = (
        previous["validation"]["source_document_sha256"]
        if previous and "validation" in previous
        else previous["artifact_sha256"]
        if previous
        else None
    )
    value = {
        **value,
        "validation": {
            "version": VALIDATION_VERSION,
            "implementation_sha256": self.code_identity,
            "eval_dataset_sha256": value["eval_dataset_sha256"],
            "source_document_sha256": source,
            "document_text_changed": False,
            "semantic_critique_reused": True,
        },
    }
    result = sealed(value)
    if self.revalidation_preflight:
        return result
    if self.verify_only:
        return self.store(path, value)
    if previous is not None and previous != result:
        if not self.revalidation_enabled:
            raise ValueError("stale document checks; run --stage revalidate explicitly")
        self.require_unfrozen()
        save_artifact(
            self.path(
                universe,
                f"validation_history/{identity}/{previous['artifact_sha256']}.json",
            ),
            previous,
        )
        atomic_json(path, result)
        return result
    return save_artifact(path, value)


def revalidate(self: AtomicCorpusPipeline, *, dry_run=False):
    """Audit every cached request before explicitly reusing unchanged generation."""
    self.require_unfrozen()
    if self.extension is None:
        original = read_artifact(self.path(UNIVERSES[0], "facts/extraction.json"))
    else:
        # Imported facts retain their parent's identity. Audit against a paid
        # request from this extension, never against the frozen parent.
        original = next(
            (
                read_artifact(path)
                for u in UNIVERSES
                for path in sorted(self.path(u, "drafts").glob("*.json"))
                if path.stem not in {s["id"] for s in self.extension.source_slots(u)}
            ),
            None,
        )
        if original is None:
            raise ValueError("extension has no local generation to revalidate")
    prior_identity = original["request"]["pipeline_identity"]
    prior_code = original["code"]["implementation_sha256"]
    if digest({**self.identity_settings(), "code": prior_code}) != prior_identity:
        raise ValueError(
            "upstream generation settings changed; use a new corpus version"
        )
    old_execute, old_clients = self.execute, self.clients
    self.execute = False
    self.clients = {
        s: ReadOnlyClient()
        for s in ("facts", "types", "ideas", "drafts", "critics", "revisions")
    }
    self._generation_identity_override = prior_identity
    self.verify_only = self.revalidation_preflight = True
    try:
        if self.reviewed_only:
            for u in UNIVERSES:
                completed = {p.stem for p in self.path(u, "documents").glob("*.json")}
                for stage in ("critics", "revisions"):
                    if any(
                        path.stem.rsplit("_r", 1)[0] not in completed
                        for path in self.path(u, stage).glob("*.json")
                    ):
                        raise ValueError(
                            "unfinished reviews must be completed before a reviewed-only audit"
                        )
                self.drafts(
                    u,
                    slot_ids={p.stem for p in self.path(u, "drafts").glob("*.json")},
                )
        documents = {u: self.critique(u) for u in UNIVERSES}
    finally:
        self.verify_only = self.revalidation_preflight = False
        self.execute, self.clients = old_execute, old_clients
        self._generation_identity_override = None
    summary = {
        "generation_pipeline_identity": prior_identity,
        "validation_version": VALIDATION_VERSION,
        "documents": sum(len(rows) for rows in documents.values()),
        "hard_errors": {
            u: [r["id"] for r in rows if r["checks"]["hard_errors"]]
            for u, rows in documents.items()
        },
        "document_text_changed": False,
        "model_calls": 0,
    }
    if dry_run:
        return summary
    compatibility = {
        "generation_pipeline_identity": prior_identity,
        "generation_implementation_sha256": prior_code,
        "active_implementation_sha256": self.code_identity,
        "settings_sha256": digest(self.identity_settings()),
        "validation_version": VALIDATION_VERSION,
        "verified_cached_generation_graph_sha256": digest(documents),
        "authorization": "explicit --stage revalidate; every cached prompt, model, seed, lineage and response validated; no content revision",
        "validation_code": self.code(),
    }
    pointer = self.base / "generation_compatibility.json"
    already_current = False
    if pointer.exists():
        previous = read_artifact(pointer)
        already_current = (
            previous.get("active_implementation_sha256") == self.code_identity
        )
        if not already_current:
            save_artifact(
                self.base / f"revalidation_history/{previous['artifact_sha256']}.json",
                previous,
            )
    if not already_current:
        record = save_artifact(
            self.base / f"revalidation_history/{digest(compatibility)}.json",
            compatibility,
        )
        atomic_json(pointer, record)
    self.revalidation_enabled = True
    try:
        for u, rows in documents.items():
            for row in rows:
                if self.extension is not None and self.extension.is_imported(u, row):
                    continue
                self.store_validated_document(u, row["id"], row)
    finally:
        self.revalidation_enabled = False
    return summary
