"""Document eligibility, deterministic selection, QA, and frozen exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from contrastive_sdf.artifacts import atomic_json, digest, read_artifact, sealed
from contrastive_sdf.corpora.documents import (
    CorpusDocument,
    build_manifest,
    corpus_sha256,
    load_corpus,
)
from contrastive_sdf.corpora.generation.schema import PAIRS
from contrastive_sdf.corpora.hand_generated import duplicate_groups

if TYPE_CHECKING:
    from contrastive_sdf.corpora.generation.pipeline import AtomicCorpusPipeline

from contrastive_sdf.corpora.generation.common import (
    UNIVERSES,
    VALIDATION_VERSION,
    ReadOnlyClient,
    hashlib,
    save_artifact,
    select_balanced,
)


def eligible(self: AtomicCorpusPipeline, universe, record):
    if (
        self.extension is not None
        and self.extension.config.source_selection_only
        and record["id"] not in self.extension.selected_source_ids(universe)
    ):
        return False
    if record["checks"]["hard_errors"]:
        return False
    review = self.latest_review(universe, f"document_{record['id']}")
    if review and review["subject_sha256"] == record["artifact_sha256"]:
        return review["decision"] in {"approve", "fixture_approve"} and (
            not record["flagged"] or review["override"]
        )
    return not record["flagged"] and record["recommended_action"] == "accept"


def store_selection(self: AtomicCorpusPipeline, universe, value):
    path = self.path(universe, "selection.json")
    expected = sealed(value)
    if self.verify_only:
        return self.store(path, value)
    if path.exists():
        previous = read_artifact(path)
        if previous == expected:
            return previous
        if (
            previous["pipeline_identity"] != expected["pipeline_identity"]
            or self.config.expansion_checkpoints is None
            and (
                previous["candidate_sha256"] != expected["candidate_sha256"]
                or previous["balancing_inputs_sha256"]
                == expected["balancing_inputs_sha256"]
            )
        ):
            raise ValueError("stale selection inputs; use a new corpus version")
        save_artifact(
            self.path(
                universe, f"selection_history/{previous['artifact_sha256']}.json"
            ),
            previous,
        )
    save_artifact(
        self.path(universe, f"selection_history/{expected['artifact_sha256']}.json"),
        value,
    )
    atomic_json(path, expected)
    return expected


def eligible_documents(self: AtomicCorpusPipeline, universe, rows):
    """Apply the same quality exclusions to quota filling and selection."""
    eligible = [r for r in rows if self.eligible(universe, r)]
    if self.extension is None or self.config.expansion_checkpoints is None:
        return eligible
    from contrastive_sdf.corpora.generation.diversity import distinct_candidates

    eligible, audit = distinct_candidates(
        eligible,
        self.extension.locked_documents()[universe],
        ngram_size=self.config.diagnostic_ngram_size,
        threshold=self.config.near_duplicate_threshold,
    )
    audit["validation_implementation_sha256"] = self.code_identity
    if not self.verify_only:
        save_artifact(
            self.path(universe, f"quality_filters/{digest(audit)}.json"), audit
        )
    return eligible


def default_selection_config(self: AtomicCorpusPipeline):
    from contrastive_sdf.corpora.generation.expansion import approved_token_config

    return approved_token_config(
        self,
        self.config,
        self.corpus.document_count,
        self.checkpoint_token_approval,
    )


def balance(self: AtomicCorpusPipeline, *, selection_config=None):
    cfg = selection_config or self.default_selection_config()
    pools = {u: self.critique(u) for u in UNIVERSES}
    eligible = {u: self.eligible_documents(u, rows) for u, rows in pools.items()}
    locked = self.extension.locked_documents() if self.extension is not None else None
    selections = select_balanced(eligible, cfg, locked=locked)
    from contrastive_sdf.corpora.generation.balance import token_balance_report

    token_balance = token_balance_report(selections, cfg.max_relative_token_delta)
    all_reviews = {
        u: {r["id"]: self.latest_review(u, f"document_{r['id']}") for r in pools[u]}
        for u in UNIVERSES
    }
    balancing_inputs = digest(
        {
            "eligible": {
                u: [r["artifact_sha256"] for r in eligible[u]] for u in UNIVERSES
            },
            "reviews": all_reviews,
        }
    )
    result = {}
    for u in UNIVERSES:
        reviews = all_reviews[u]
        result[u] = self.store_selection(
            u,
            {
                "universe_id": u,
                "pipeline_identity": self.identity(),
                "candidate_sha256": digest(pools[u]),
                "balancing_inputs_sha256": balancing_inputs,
                "document_reviews_sha256": digest(reviews),
                "selected": [
                    {"id": r["id"], "artifact_sha256": r["artifact_sha256"]}
                    for r in selections[u]
                ],
                "documents_per_type": cfg.documents_per_type,
                "target_tokens": cfg.target_tokens_per_universe,
                "total_tokens": sum(r["tokens"] for r in selections[u]),
                "selection_seed": self.config.selection_seed,
                "algorithm": "type-and-idea quotas; SHA256 seed order; deterministic token-error swap descent v1",
                **(
                    {"locked_parent_ids": sorted(locked[u])}
                    if locked is not None
                    else {}
                ),
                "uses_downstream_results": False,
                **(
                    {"token_balance": token_balance}
                    if cfg.max_relative_token_delta is not None
                    else {}
                ),
            },
        )
    return result


def selected_documents(self: AtomicCorpusPipeline, universe, selection):
    records = [
        read_artifact(self.path(universe, f"documents/{r['id']}.json"))
        for r in selection["selected"]
    ]
    if any(
        r["artifact_sha256"] != s["artifact_sha256"]
        for r, s in zip(records, selection["selected"], strict=True)
    ):
        raise ValueError("selected document changed")
    return records


def qa(self: AtomicCorpusPipeline, *, write=True, selection_config=None):
    from contrastive_sdf.corpora.generation.quality import atomic_qa_report, qa_markdown

    cfg = selection_config or self.default_selection_config()
    pools = {
        u: [read_artifact(p) for p in sorted(self.path(u, "documents").glob("*.json"))]
        for u in UNIVERSES
    }
    selection_error = None
    try:
        selections = self.balance(selection_config=cfg)
        selected = {u: self.selected_documents(u, selections[u]) for u in UNIVERSES}
    except ValueError as ex:
        from contrastive_sdf.corpora.generation.balance import TokenBalanceError

        if "eligible documents, need" not in str(ex) and not isinstance(
            ex, TokenBalanceError
        ):
            raise
        selection_error = str(ex)
        selections, selected = (
            {},
            ex.selected if isinstance(ex, TokenBalanceError) else pools,
        )
    facts = {u: self.facts(u) for u in UNIVERSES}
    plans = {u: self.plan_documents(u) for u in UNIVERSES}
    report = atomic_qa_report(selected, pools, facts, plans, self.eval_tasks(), cfg)
    for u in UNIVERSES:
        reviews = {
            r["id"]: self.latest_review(u, f"document_{r['id']}") for r in pools[u]
        }
        report["universes"][u]["manual_document_reviews"] = {
            k: v for k, v in reviews.items() if v is not None
        }
        report["universes"][u]["selected_override_ids"] = [
            r["id"]
            for r in selected[u]
            if (review := reviews[r["id"]]) is not None and review["override"]
        ]
    report.update(
        pipeline_identity=self.identity(),
        validation_version=VALIDATION_VERSION,
        validation_implementation_sha256=self.code_identity,
        contexts=self.validate_contexts(),
        selections={u: s["artifact_sha256"] for u, s in selections.items()},
        report_scope="pool" if selection_error else "selected corpus",
        selection_error=selection_error,
    )
    attempts = [
        read_artifact(p)
        for u in UNIVERSES
        for p in sorted(self.path(u, "attempts").rglob("*.json"))
    ]
    costs = [a["response"].get("cost_usd") for a in attempts]
    report["cost_usage"] = {
        "request_attempts": len(attempts),
        "known_cost_usd": sum(c for c in costs if c is not None),
        "unpriced_attempts": sum(c is None for c in costs),
        "total_cost_usd": sum(costs) if all(c is not None for c in costs) else None,
        "input_tokens": sum(
            a["response"].get("usage", {}).get("input_tokens", 0) for a in attempts
        ),
        "output_tokens": sum(
            a["response"].get("usage", {}).get("output_tokens", 0) for a in attempts
        ),
        "attempt_ledger_sha256": digest(attempts),
        "scope": "all atomic generation stages and failures, counted once",
    }
    if self.extension is not None:
        report["extension"] = self.extension.summary()
        report["cost_usage"]["scope"] = (
            "new expansion calls only; source usage retained separately in reuse/source"
        )
    if cfg.max_relative_token_delta is not None:
        from contrastive_sdf.corpora.generation.balance import token_balance_report

        report["token_balance"] = token_balance_report(
            selected, cfg.max_relative_token_delta
        )
    estimates = [a["response"].get("estimated_cost_usd") for a in attempts]
    if any(e is not None for e in estimates):
        report["cost_usage"]["known_estimated_cost_usd"] = sum(
            e for e in estimates if e is not None
        )
        report["cost_usage"]["attempts_without_cost_estimate"] = sum(
            e is None for e in estimates
        )
    report = sealed(report)
    if self.verify_only:
        if (
            not (self.base / "qa.json").exists()
            or read_artifact(self.base / "qa.json") != report
        ):
            raise ValueError("missing/stale corpus QA report")
    elif write:
        save_artifact(
            self.base / f"qa_history/{report['artifact_sha256']}.json", report
        )
        atomic_json(self.base / "qa.json", report)
        (self.base / "qa.md").write_text(qa_markdown(report))
        for u in UNIVERSES:
            atomic_json(
                self.path(u, "qa.json"),
                sealed(
                    {
                        "universe_id": u,
                        "report": report["universes"][u],
                        "combined_report_sha256": report["artifact_sha256"],
                    }
                ),
            )
            self.path(u, "qa.md").write_text(
                qa_markdown({"universes": {u: report["universes"][u]}, "pairs": {}})
            )
    return report


def corpus_subjects(self: AtomicCorpusPipeline):
    self.balance()  # Unbalanced pools cannot receive corpus approval.
    qa = self.qa()
    return {
        u: sealed(
            {
                "universe_id": u,
                "selection_sha256": qa["selections"][u],
                "qa_sha256": qa["artifact_sha256"],
                "pipeline_identity": self.identity(),
            }
        )
        for u in UNIVERSES
    }


def freeze(self: AtomicCorpusPipeline):
    # Preview generation never bypasses the researcher's freeze approvals.
    if self.extension is not None:
        self.extension.validate_source()
    for u in UNIVERSES:
        for kind, subject in (
            ("context", self.context(u)),
            ("facts", self.facts(u)),
            ("plan", self.plan_documents(u)),
        ):
            self.require_approval(u, kind, subject, final=True)
    selections = self.balance()
    subjects = self.corpus_subjects()
    approvals = {u: self.require_approval(u, "corpus", subjects[u]) for u in UNIVERSES}
    atomic = {}
    for u in UNIVERSES:
        rows = self.selected_documents(u, selections[u])
        docs = [CorpusDocument(r["id"], u.split("_")[0], r["text"]) for r in rows]
        if duplicate_groups(docs):
            raise ValueError(f"exact duplicate documents in {u}")
        provenance = {
            r["id"]: {
                "atomic_provenance": {
                    "universe_id": u,
                    "artifact_sha256": r["artifact_sha256"],
                    "lineage": r["lineage"],
                    "type_id": r["type_id"],
                    "idea_id": r["idea_id"],
                    "fact_ids": r["facts"],
                    "text_sha256": r["text_sha256"],
                }
            }
            for r in rows
        }
        manifest = build_manifest(
            docs,
            universe="A",
            corpus_version=self.corpus.version,
            tokenizer=self.corpus.tokenizer,
            count_tokens=self.count_tokens,
            mapping=self.plan.contract.universes["A"],
            buckets=(u.split("_")[0],),
            document_metadata=provenance,
            metadata={
                "manifest_version": 3,
                "universe": u,
                "mapping": {u.split("_")[0]: u.split("_")[1]},
                "mode": self.plan.contract.mode,
                "pipeline_identity": self.identity(),
                "context": self.context(u),
                "facts_sha256": self.facts(u)["artifact_sha256"],
                "plan_sha256": self.plan_documents(u)["artifact_sha256"],
                "selection_sha256": selections[u]["artifact_sha256"],
                "researcher_approval": approvals[u],
                "qa_subject": subjects[u],
            },
        )
        self.write_documents(u, docs)
        atomic[u] = self.store(self.path(u, "manifest.json"), manifest)
    branches = {}
    for branch, pair in PAIRS.items():
        rows = [r for u in pair for r in self.selected_documents(u, selections[u])]
        docs = [
            CorpusDocument(r["id"], r["universe_id"].split("_")[0], r["text"])
            for r in rows
        ]
        if duplicate_groups(docs):
            raise ValueError(f"exact duplicate documents in {branch}")
        manifest = build_manifest(
            docs,
            universe=branch,
            corpus_version=self.corpus.version,
            tokenizer=self.corpus.tokenizer,
            count_tokens=self.count_tokens,
            mapping=self.plan.contract.universes[branch],
            buckets=("grader", "users"),
            document_metadata={
                r["id"]: {
                    "atomic_provenance": next(
                        d["atomic_provenance"]
                        for d in atomic[r["universe_id"]]["documents"]
                        if d["id"] == r["id"]
                    )
                }
                for r in rows
            },
            metadata={
                "manifest_version": 3,
                "mode": self.plan.contract.mode,
                "pipeline_identity": self.identity(),
                "atomic_manifests": {u: atomic[u]["artifact_sha256"] for u in pair},
                "mixture": "50/50 by document count; no DOCTAG or webtext",
            },
        )
        self.write_documents(branch, docs)
        branches[branch] = self.store(self.base / branch / "manifest.json", manifest)
    if self.extension is not None and not self.verify_only:
        self.extension.seal_verifier()
    return branches


def write_documents(self: AtomicCorpusPipeline, universe, docs):
    folder = self.base / universe / "generated"
    expected = {d.relative_path.as_posix() for d in docs}
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*.txt")}
    if actual - expected:
        raise ValueError("stale generated documents; use a new version")
    for d in docs:
        path = folder / d.relative_path
        if path.exists() and path.read_text() != d.text:
            raise ValueError(f"changed frozen document: {path}")
        if not path.exists():
            if self.verify_only:
                raise ValueError(f"missing frozen document: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(d.text)


def verify_frozen(
    self: AtomicCorpusPipeline, *, require_pinned=True, verify_token_counts=True
):
    if require_pinned:
        self.plan.require_ready_for_training()
    # Verification may only reuse completed stages, never contact a provider.
    original_clients, original_execute = self.clients, self.execute
    self.clients = {
        s: ReadOnlyClient()
        for s in ("facts", "types", "ideas", "drafts", "critics", "revisions")
    }
    self.execute = False
    self.verify_only = True
    self.verified_archives.clear()
    try:
        expected = self.freeze()
    finally:
        self.clients, self.execute = original_clients, original_execute
        self.verify_only = False
    results = {}
    for b, m in expected.items():
        if (
            self.corpus.sha256[b] is not None
            and m["corpus_sha256"] != self.corpus.sha256[b]
        ):
            raise ValueError(f"{b}: pinned corpus hash mismatch")
        docs = load_corpus(self.base / b / "generated", ("grader", "users"))
        if (
            len(docs) != self.corpus.document_count
            or corpus_sha256(docs) != m["corpus_sha256"]
        ):
            raise ValueError(f"{b}: frozen count/hash mismatch")
        attempts = [
            read_artifact(p)
            for u in PAIRS[b]
            for p in self.path(u, "attempts").rglob("*.json")
        ]
        costs = [a["response"].get("cost_usd") for a in attempts]
        results[b] = {
            "corpus_sha256": m["corpus_sha256"],
            "manifest_sha256": hashlib.sha256(
                (self.base / b / "manifest.json").read_bytes()
            ).hexdigest(),
            "totals": m["totals"],
            "buckets": m["buckets"],
            "mode": m["mode"],
            "atomic_manifests": m["atomic_manifests"],
            "exact_duplicate_groups": [],
            "token_counts_verified": verify_token_counts,
            "generator_cost_usd": sum(costs)
            if all(c is not None for c in costs)
            else None,
            "unpriced_attempts": sum(c is None for c in costs),
            "unpriced_documents": sum(
                read_artifact(
                    self.path(
                        r["atomic_provenance"]["universe_id"],
                        f"drafts/{r['id']}.json",
                    )
                )["response"].get("cost_usd")
                is None
                for r in m["documents"]
            ),
            "generator_cost_scope": "all upstream stages for this pair; A/B have disjoint atomic corpora",
        }
    return results
