"""Coordinate live corpus authoring, review, validation, and selection."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from contrastive_sdf.artifacts import digest, read_artifact, sealed
from contrastive_sdf.corpora.generation import authoring, integrity, selection
from contrastive_sdf.corpora.generation import review as review_stage
from contrastive_sdf.corpora.generation.common import (
    UNIVERSES,
    Lock,
    Path,
    hashlib,
    save_artifact,
)
from contrastive_sdf.corpora.generation.prompts import atomic_template_hashes
from contrastive_sdf.corpora.generation.recipe import ExperimentPlan
from contrastive_sdf.corpora.generation.schema import PAIRS
from contrastive_sdf.corpora.hand_generated import token_counter


class AtomicCorpusPipeline:
    code = integrity.code
    _code_unlocked = integrity._code_unlocked
    identity_settings = integrity.identity_settings
    identity = integrity.identity
    store_validated_document = integrity.store_validated_document
    revalidate = integrity.revalidate
    context_rendering = review_stage.context_rendering
    rendering_path = review_stage.rendering_path
    render_contexts = review_stage.render_contexts
    context = review_stage.context
    validate_contexts = review_stage.validate_contexts
    validate_context = review_stage.validate_context
    latest_review = review_stage.latest_review
    review_status = review_stage.review_status
    require_approval = review_stage.require_approval
    review = review_stage.review
    call = review_stage.call
    extract_facts = authoring.extract_facts
    facts = authoring.facts
    plan_documents = authoring.plan_documents
    document_inputs = authoring.document_inputs
    document_slots = authoring.document_slots
    drafts = authoring.drafts
    eval_tasks = authoring.eval_tasks
    critique = authoring.critique
    eligible = selection.eligible
    store_selection = selection.store_selection
    eligible_documents = selection.eligible_documents
    default_selection_config = selection.default_selection_config
    balance = selection.balance
    selected_documents = selection.selected_documents
    qa = selection.qa
    corpus_subjects = selection.corpus_subjects
    freeze = selection.freeze
    write_documents = selection.write_documents
    verify_frozen = selection.verify_frozen

    def __init__(
        self,
        plan: ExperimentPlan,
        root: Path,
        *,
        execute=False,
        clients=None,
        max_attempts=1,
        workers=1,
    ):
        self.plan, self.root, self.execute = plan, root, execute
        self.corpus = plan.contract.corpus
        if self.corpus.atomic is None:
            raise ValueError("config has no corpus.atomic staged pipeline")
        self.config = self.corpus.atomic
        templates = self.config.grader_context_templates
        if templates is not None and any(
            m.base_model != templates.target_base_model for m in plan.contract.models
        ):
            raise ValueError(
                "grader context target_base_model must match every checkpoint; use a separate family config"
            )
        self.base = root / self.corpus.directory
        self.clients: dict[str, Any] = clients or {}
        self.max_attempts = max_attempts
        self.workers = workers
        if max_attempts < 1 or workers < 1:
            raise ValueError("max_attempts and workers must be positive")
        self.count_tokens = token_counter(self.corpus.tokenizer)
        # Changes to data/review artifacts do not change implementation identity.
        source_files = [
            p
            for folder in (root / "src", root / "scripts")
            for p in folder.rglob("*.py")
            if "__pycache__" not in p.parts
        ]
        source_files += [
            root / n for n in ("pyproject.toml", "uv.lock") if (root / n).exists()
        ]
        source_hashes = {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source_files)
        }
        self.code_identity = digest({**source_hashes, **atomic_template_hashes()})
        self._code: dict | None = None
        self._generation_identity_override: str | None = None
        self.revalidation_preflight = False
        self.revalidation_enabled = False
        self.verify_only = False
        self.reviewed_only = False
        policy_path = self.base / "token_policy.json"
        self.checkpoint_token_approval: dict | None = (
            read_artifact(policy_path) if policy_path.exists() else None
        )
        self.verified_archives = set()
        self._code_lock, self._client_lock = Lock(), Lock()
        self.allowed_stages = {
            "facts",
            "types",
            "ideas",
            "drafts",
            "critics",
            "revisions",
        }
        from contrastive_sdf.corpora.generation.reuse import CorpusSource

        self.extension: CorpusSource | None = None
        if self.config.extension is not None:
            from contrastive_sdf.corpora.generation.reuse import CorpusSource

            self.extension = CorpusSource(self)

    def map_slots(self, function, slots):
        if self.workers == 1:
            return list(map(function, slots))
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            return list(executor.map(function, slots, buffersize=self.workers))

    def store(self, path, value):
        expected = sealed(value)
        if self.verify_only:
            if not path.exists() or read_artifact(path) != expected:
                raise ValueError(f"missing/stale frozen artifact: {path}")
            return expected
        return save_artifact(path, value)

    def path(self, universe, name):
        if universe not in UNIVERSES:
            raise ValueError(f"unknown atomic universe: {universe}")
        return self.base / universe / name

    def require_unfrozen(self):
        if any(
            (self.base / branch / "manifest.json").exists()
            for branch in (*UNIVERSES, *PAIRS)
        ):
            raise ValueError("frozen corpora require a new version/directory")
