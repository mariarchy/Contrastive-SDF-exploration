"""Pinned source imports and resumable, quota-driven atomic corpus expansion.

Imports retain original artifacts and reviews byte for byte. New requests use
the new plan. Neither imports nor expansion grant scientific approval.
"""

import hashlib
import sys
from collections import Counter
from pathlib import Path

from contrastive_sdf.sdf.atomic_schema import PAIRS, UNIVERSES
from contrastive_sdf.sdf.scalable_corpus import (
    digest,
    document_checks,
    generated_document_checks,
    idea_quotas,
    read_artifact,
    save_artifact,
)


def source_inventory(base: Path) -> dict[str, str]:
    """Only corpus provenance inputs; exclude derived viewers and authoring drafts."""
    folders = [
        "universe_contexts",
        "context_renderings",
        "code",
        "config_snapshots",
        "reuse",
        "expansion_batches",
        "checkpoints",
        "repair_inputs",
        *UNIVERSES,
        *PAIRS,
    ]
    files = {
        base / n
        for n in ("generation_compatibility.json", "qa.json", "token_policy.json")
    }
    for folder in folders:
        files.update(p for p in (base / folder).rglob("*") if p.is_file())
    return {
        p.relative_to(base).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
        if p.is_file()
        and (p.suffix not in {".html", ".md"} or "universe_contexts" in p.parts)
    }


def _copy_checked(source: Path, target: Path, expected: str):
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"source import integrity mismatch: {source}")
    if target.exists():
        if target.read_bytes() != content:
            raise ValueError(f"refusing to overwrite imported source: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


class CorpusSource:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.config = pipeline.config.extension
        self.base = pipeline.base / "reuse"
        self.source = self.base / "source"
        self.record = None
        self._plans = {}
        self._artifacts = {}
        self._generation_inputs = None
        from contrastive_sdf.sdf.corpus_repairs import draft_repair_inputs

        self._repair_inputs = draft_repair_inputs(pipeline.root, pipeline.config)

    def require_initialized(self):
        if self.record is None:
            path = self.base / "import.json"
            if not path.exists():
                raise ValueError(
                    "extension requires --stage import-source before generation"
                )
            record = read_artifact(path)
            if (
                record["source_config_sha256"] != self.config.source_config_sha256
                or record["source_inventory_sha256"]
                != self.config.source_inventory_sha256
                or digest(record["inventory"]) != self.config.source_inventory_sha256
                or record["settings_sha256"]
                != digest(self.pipeline.identity_settings())
            ):
                raise ValueError("extension import settings or pinned source changed")
            saved_config = self.base / "source_config.yaml"
            if (
                hashlib.sha256(saved_config.read_bytes()).hexdigest()
                != self.config.source_config_sha256
            ):
                raise ValueError("corrupt imported source config")
            self.record = record
        return self.record

    def initialize(self):
        """Verify the entire source before copying; never write to the source corpus."""
        if (self.base / "import.json").exists():
            self.require_initialized()
            self.materialize()
            return self.summary()
        from contrastive_sdf.sdf.plan import load_experiment_plan
        from contrastive_sdf.sdf.scalable_corpus import verify_experiment_corpora

        p = self.pipeline
        config_path = p.root / self.config.source_config
        if (
            hashlib.sha256(config_path.read_bytes()).hexdigest()
            != self.config.source_config_sha256
        ):
            raise ValueError("extension source config hash mismatch")
        source_plan = load_experiment_plan(config_path)
        source_corpus = source_plan.contract.corpus
        source_base = p.root / source_corpus.directory
        if (
            source_base.resolve() == p.base.resolve()
            or source_base.resolve() in p.base.resolve().parents
        ):
            raise ValueError("source and extension corpus directories must be separate")
        if (
            source_plan.contract.mode != p.plan.contract.mode
            or source_plan.contract.universes != p.plan.contract.universes
        ):
            raise ValueError("extension source mode or universe mappings changed")
        if (
            source_corpus.atomic is None
            or source_corpus.tokenizer != p.corpus.tokenizer
        ):
            raise ValueError(
                "extension requires the same atomic pipeline and tokenizer"
            )
        allocation_fields = {
            "extension",
            "documents_per_type",
            "pool_documents_per_type",
            "documents_per_idea",
            "pool_documents_per_idea",
            "target_tokens_per_universe",
            "review_mode",
            "max_relative_token_delta",
            "expansion_checkpoints",
            "expansion_batch_size",
            "idea_length_words",
            "draft_repair_manifest",
            "draft_repair_manifest_sha256",
        }
        if self.config.allow_critic_change:
            allocation_fields.add("critic")

        def recipe(c):
            result = {
                k: v
                for k, v in c.model_dump(mode="json").items()
                if k not in allocation_fields
            }
            if self.config.allow_generator_prompt_change:
                result["generator"] = {
                    k: v
                    for k, v in result["generator"].items()
                    if k not in {"prompt_suffix", "prompt_suffix_file"}
                }
            return result

        if recipe(source_corpus.atomic) != recipe(p.config):
            raise ValueError(
                "extension generation recipe changed; explicit new recipe review required"
            )
        inventory = source_inventory(source_base)
        if digest(inventory) != self.config.source_inventory_sha256:
            raise ValueError("extension source inventory hash mismatch")
        verified = verify_experiment_corpora(source_plan, p.root)
        if source_inventory(source_base) != inventory:
            raise ValueError("source corpus changed during read-only verification")
        for relative, sha256 in inventory.items():
            _copy_checked(source_base / relative, self.source / relative, sha256)
            if Path(relative).parts[0] == "code":
                _copy_checked(source_base / relative, p.base / relative, sha256)
        _copy_checked(
            config_path,
            self.base / "source_config.yaml",
            self.config.source_config_sha256,
        )
        record = save_artifact(
            self.base / "import.json",
            {
                "source_config": self.config.source_config,
                "source_config_sha256": self.config.source_config_sha256,
                "source_inventory_sha256": self.config.source_inventory_sha256,
                "inventory": inventory,
                "source_directory": source_corpus.directory,
                "verified_source": verified,
                "settings_sha256": digest(p.identity_settings()),
                "source_modified": False,
                "model_calls": 0,
                "critic_change": {
                    "source": source_corpus.atomic.critic.model_dump(mode="json"),
                    "target": p.config.critic.model_dump(mode="json"),
                    "scope": "new documents only; source documents and their reviews unchanged",
                }
                if self.config.allow_critic_change
                else None,
                "generator_prompt_change": {
                    "source": source_corpus.atomic.generator.model_dump(mode="json"),
                    "target": p.config.generator.model_dump(mode="json"),
                    "idea_length_words": p.config.idea_length_words,
                    "authorization": "explicit repair recipe; all non-prompt generator settings unchanged",
                }
                if self.config.allow_generator_prompt_change
                else None,
                "code": p.code(),
            },
        )
        self.record = record
        self.materialize()
        return self.summary()

    def materialize(self):
        """Copy unchanged approvals only for unchanged subjects, never plans/corpora."""
        record = self.require_initialized()
        for relative, sha256 in record["inventory"].items():
            parts = Path(relative).parts
            shared = parts[0] in {
                "universe_contexts",
                "context_renderings",
                "code",
                "config_snapshots",
            }
            atomic = (
                len(parts) > 1
                and parts[0] in UNIVERSES
                and (
                    parts[1] == "facts.json"
                    or parts[1]
                    in {
                        "facts",
                        "types",
                        "ideas",
                        "drafts",
                        "critics",
                        "revisions",
                        "documents",
                    }
                    or parts[1] == "reviews"
                    and len(parts) > 2
                    and parts[2] not in {"plan", "corpus"}
                )
            )
            if shared or atomic:
                if (
                    self.config.reuse_selected_only
                    and atomic
                    and len(parts) > 1
                    and parts[1] in {"documents", "drafts", "critics", "revisions"}
                    and Path(relative).stem.split("_r", 1)[0]
                    not in self.selected_source_ids(parts[0])
                ):
                    continue
                _copy_checked(
                    self.source / relative, self.pipeline.base / relative, sha256
                )

    def artifact(self, relative):
        if relative not in self._artifacts:
            record = self.require_initialized()
            path = self.source / relative
            if (
                relative not in record["inventory"]
                or hashlib.sha256(path.read_bytes()).hexdigest()
                != record["inventory"][relative]
            ):
                raise ValueError(f"corrupt imported artifact: {relative}")
            self._artifacts[relative] = read_artifact(path)
        return self._artifacts[relative]

    def source_plan(self, universe):
        if universe not in self._plans:
            self._plans[universe] = self.artifact(f"{universe}/plan.json")
        return self._plans[universe]

    def is_source_slot(self, universe, slot):
        return slot in self.source_slots(universe)

    def source_slots(self, universe):
        inventory = self.require_initialized()["inventory"]
        return [
            s
            for s in self.source_plan(universe)["slots"]
            if f"{universe}/documents/{s['id']}.json" in inventory
            and (
                not self.config.reuse_selected_only
                or s["id"] in self.selected_source_ids(universe)
            )
        ]

    def facts(self, universe):
        p = self.pipeline
        context = p.context(universe)
        facts = self.artifact(f"{universe}/facts.json")
        if facts["lineage"]["context"] != {
            k: v for k, v in context.items() if k != "text"
        }:
            raise ValueError("extension context differs from imported facts")
        p.require_approval(universe, "context", context)
        return p.store(p.path(universe, "facts.json"), facts)

    def plan_documents(self, universe):
        from contrastive_sdf.sdf.corpus_repairs import align_idea_length

        p, cfg = self.pipeline, self.pipeline.config
        facts, source = p.facts(universe), self.source_plan(universe)
        slots = list(
            self.source_slots(universe)
            if self.config.reuse_selected_only
            else source["slots"]
        )
        selected, pool = {}, {}
        for typ in source["types"]:
            type_id = typ["id"]
            selected.update(idea_quotas(cfg, type_id))
            quotas = idea_quotas(cfg, type_id, pool=True)
            pool.update(quotas)
            existing = [s for s in slots if s["type_id"] == type_id]
            counts = Counter(s["idea_id"] for s in existing)
            if any(counts[i] > n for i, n in quotas.items()):
                raise ValueError(
                    "extension pool budget is smaller than the saved source pool"
                )
            next_id = max(
                (
                    int(s["id"].rsplit("_d", 1)[1])
                    for s in source["slots"]
                    if s["type_id"] == type_id
                ),
                default=0,
            )
            while any(counts[i] < n for i, n in quotas.items()):
                for idea_id, quota in sorted(quotas.items()):
                    if counts[idea_id] < quota:
                        next_id += 1
                        slots.append(
                            {
                                "id": f"{universe}_{type_id}_d{next_id:06d}",
                                "type_id": type_id,
                                "idea_id": idea_id,
                            }
                        )
                        counts[idea_id] += 1
        return p.store(
            p.path(universe, "plan.json"),
            {
                "universe_id": universe,
                "lineage": {
                    "context": facts["lineage"]["context"],
                    "facts_sha256": facts["artifact_sha256"],
                    "facts_version": cfg.facts_version,
                    "facts_approval_sha256": p.require_approval(
                        universe, "facts", facts
                    )["artifact_sha256"],
                    "source_plan_sha256": source["artifact_sha256"],
                    "source_inventory_sha256": self.config.source_inventory_sha256,
                },
                "types": source["types"],
                "ideas": [
                    align_idea_length(i, cfg.idea_length_words) for i in source["ideas"]
                ],
                "slots": slots,
                "selected_documents_per_type": cfg.documents_per_type,
                "pool_documents_per_type": cfg.pool_documents_per_type
                or cfg.documents_per_type,
                "selected_documents_per_idea": selected,
                "pool_documents_per_idea": pool,
                "idea_allocation": "reuse unchanged ideas; activate reserve slots only for eligible-count shortfalls",
                "pipeline_identity": p.identity(),
            },
        )

    def active_slots(self, universe, plan):
        active = {s["id"] for s in self.source_slots(universe)}
        by_id = {s["id"]: s for s in plan["slots"]}
        previous = None
        for path in sorted(
            (self.pipeline.base / "expansion_batches" / universe).glob("*.json")
        ):
            batch = read_artifact(path)
            if (
                batch["plan_sha256"] != plan["artifact_sha256"]
                or batch["previous_batch_sha256"] != previous
                or batch["pipeline_identity"] != self.pipeline.identity()
            ):
                raise ValueError("stale or broken expansion batch chain")
            if (
                len(set(batch["ids"])) != len(batch["ids"])
                or set(batch["ids"]) & active
                or not set(batch["ids"]) <= set(by_id)
            ):
                raise ValueError("invalid or repeated expansion slots")
            active.update(batch["ids"])
            previous = batch["artifact_sha256"]
        return [s for s in plan["slots"] if s["id"] in active]

    def generation_inputs(self, universe, identity):
        if self._generation_inputs is None:
            inputs = {}
            for path in sorted(
                (self.pipeline.base / "expansion_batches").rglob("*.json")
            ):
                batch = read_artifact(path)
                inputs.update(
                    {
                        (batch["universe_id"], k): v
                        for k, v in batch.get("generation_inputs", {}).items()
                    }
                )
            self._generation_inputs = inputs
        return {
            **self._repair_inputs.get((universe, identity), {}),
            **self._generation_inputs.get((universe, identity), {}),
        }

    def activate_batch(self, universe, ids, *, purpose, generation_inputs=None):
        plan = self.pipeline.plan_documents(universe)
        active = {s["id"] for s in self.active_slots(universe, plan)}
        if not ids or len(ids) != len(set(ids)) or active & set(ids):
            raise ValueError("batch must activate distinct unused slots")
        if not set(ids) <= {s["id"] for s in plan["slots"]}:
            raise ValueError("batch contains unknown slots")
        folder = self.pipeline.base / "expansion_batches" / universe
        previous = sorted(folder.glob("*.json"))
        batch = save_artifact(
            folder / f"{len(previous) + 1:06d}.json",
            {
                "universe_id": universe,
                "ids": ids,
                "plan_sha256": plan["artifact_sha256"],
                "pipeline_identity": self.pipeline.identity(),
                "previous_batch_sha256": read_artifact(previous[-1])["artifact_sha256"]
                if previous
                else None,
                "selection_uses_downstream_results": False,
                "purpose": purpose,
                "generation_inputs": generation_inputs or {},
            },
        )
        self._generation_inputs = None
        return batch

    def document(self, universe, slot, tasks):
        p = self.pipeline
        row = self.artifact(f"{universe}/documents/{slot['id']}.json")
        checker = (
            generated_document_checks
            if "python_examples" in row["checks"]
            else document_checks
        )
        if checker(row["text"], universe, tasks) != row["checks"]:
            raise ValueError(
                f"{slot['id']}: imported validation changed; explicit revalidation/review required"
            )
        if p.count_tokens(row["text"]) != row["tokens"]:
            raise ValueError("imported token count changed")
        return p.store(p.path(universe, f"documents/{slot['id']}.json"), row)

    def locked_documents(self):
        if self.config.locked_source_ids is not None:
            anchors = {u: set(ids) for u, ids in self.config.locked_source_ids.items()}
            if any(not anchors[u] <= self.selected_source_ids(u) for u in UNIVERSES):
                raise ValueError(
                    "subset anchor is absent from the frozen source selection"
                )
            return anchors
        if not self.config.preserve_parent_selection:
            return {u: set() for u in UNIVERSES}
        return {
            u: {r["id"] for r in self.artifact(f"{u}/selection.json")["selected"]}
            for u in UNIVERSES
        }

    def selected_source_ids(self, universe):
        return {
            r["id"] for r in self.artifact(f"{universe}/selection.json")["selected"]
        }

    def expand(self, universe, *, selection_config=None, quality_review=None):
        """Activate only shortfalls; persist each batch before any model call."""
        p = self.pipeline
        plan = p.plan_documents(universe)
        cfg = selection_config or p.config
        target_quotas = {
            i: q for t in cfg.documents_per_type for i, q in idea_quotas(cfg, t).items()
        }
        locked = self.locked_documents()[universe]
        while True:
            rows = p.critique(universe)
            eligible = p.eligible_documents(universe, rows)
            if not locked <= {r["id"] for r in eligible}:
                raise ValueError("parent edition contains an ineligible document")
            counts = Counter(r["idea_id"] for r in eligible)
            missing = {i: max(0, n - counts[i]) for i, n in target_quotas.items()}
            print(
                f"{universe}: {len(rows)} candidates, {len(eligible)} eligible, {sum(missing.values())} quota shortfalls",
                file=sys.stderr,
                flush=True,
            )
            if not any(missing.values()):
                return
            if p.reviewed_only:
                raise ValueError(
                    "reviewed-only checkpoint has unfilled quotas; new generation is disabled"
                )
            if cfg.expansion_batch_size is not None:
                previous_batches = sorted(
                    (p.base / "expansion_batches" / universe).glob("*.json")
                )
                if previous_batches:
                    latest = read_artifact(previous_batches[-1])
                    latest_ids = set(latest["ids"])
                    if latest_ids <= {r["id"] for r in rows} and not latest_ids & {
                        r["id"] for r in eligible
                    }:
                        reviewed_ids = {r["id"] for r in rows}
                        eligible_ids = {r["id"] for r in eligible}
                        recent = [read_artifact(path) for path in previous_batches[-3:]]
                        repeated = len(recent) == 3 and all(
                            set(batch["ids"]) <= reviewed_ids
                            and not set(batch["ids"]) & eligible_ids
                            for batch in recent
                        )
                        if (
                            len(latest_ids) >= cfg.expansion_batch_size or repeated
                        ) and not self.acknowledge_failed_batch(
                            universe, latest, rows, quality_review
                        ):
                            raise ValueError(
                                f"{universe}: zero eligible documents in a full batch or three consecutive smaller batches; quality review required before spending on replacements"
                            )
            if self.config.source_selection_only:
                raise ValueError("frozen source selection cannot fill subset quotas")
            active = {s["id"] for s in self.active_slots(universe, plan)}
            chosen = []
            for slot in plan["slots"]:
                if slot["id"] not in active and missing[slot["idea_id"]]:
                    chosen.append(slot["id"])
                    missing[slot["idea_id"]] -= 1
                    if (
                        cfg.expansion_batch_size is not None
                        and len(chosen) >= cfg.expansion_batch_size
                    ):
                        break
            if not chosen or any(missing.values()) and cfg.expansion_batch_size is None:
                raise ValueError(
                    f"{universe}: reserve candidate budget exhausted; remaining idea shortfalls: {missing}"
                )
            repair_inputs = self.failed_document_repair_inputs(
                universe, plan, chosen, rows
            )
            folder = p.base / "expansion_batches" / universe
            previous_paths = sorted(folder.glob("*.json"))
            previous = (
                read_artifact(previous_paths[-1])["artifact_sha256"]
                if previous_paths
                else None
            )
            batch = save_artifact(
                folder / f"{len(previous_paths) + 1:06d}.json",
                {
                    "universe_id": universe,
                    "ids": chosen,
                    "plan_sha256": plan["artifact_sha256"],
                    "pipeline_identity": p.identity(),
                    "previous_batch_sha256": previous,
                    "selection_uses_downstream_results": False,
                    **({"generation_inputs": repair_inputs} if repair_inputs else {}),
                },
            )
            self._generation_inputs = None
            print(
                f"{universe}: generating {len(chosen)} new candidates in batch {batch['artifact_sha256'][:12]}",
                file=sys.stderr,
                flush=True,
            )

    def acknowledge_failed_batch(self, universe, batch, rows, args):
        """Hash-bound permission for this failed batch only; never waive checks."""
        subject = batch["artifact_sha256"]
        failures = {
            r["id"]: r["artifact_sha256"] for r in rows if r["id"] in batch["ids"]
        }
        path = (
            self.pipeline.base
            / "expansion_batches"
            / universe
            / "quality_reviews"
            / f"{subject}.json"
        )
        if path.exists():
            review = read_artifact(path)
            if (
                review["subject_sha256"] != subject
                or review["failed_documents"] != failures
            ):
                raise ValueError("failed-batch review no longer matches its documents")
            return review["decision"] == "approve"
        if (
            getattr(args, "decision", None) != "approve"
            or getattr(args, "expected_hash", None) != subject
            or not getattr(args, "reviewer", None)
            or not getattr(args, "reason", None)
        ):
            return False
        save_artifact(
            path,
            {
                "subject_sha256": subject,
                "failed_documents": failures,
                "decision": args.decision,
                "reviewer": args.reviewer,
                "reason": args.reason,
                "implementation_sha256": self.pipeline.code_identity,
                "scope": "Resume after this failed batch only; all document acceptance gates remain required",
            },
        )
        return True

    def failed_document_repair_inputs(self, universe, plan, chosen, rows):
        """Reuse failed new documents and critiques as grounded rewrite inputs."""
        from contrastive_sdf.sdf.python_examples import repair_python_indentation

        used = {
            self.generation_inputs(universe, slot["id"]).get(
                "repair_source_document_id"
            )
            for slot in self.active_slots(universe, plan)
        }
        imported = {s["id"] for s in self.source_slots(universe)}
        failures = [
            r
            for r in rows
            if r["id"] not in imported
            and r["id"] not in used
            and not self.pipeline.eligible(universe, r)
        ]
        inputs = {}
        by_id = {slot["id"]: slot for slot in plan["slots"]}
        for identity in chosen:
            failed = next(
                (r for r in failures if r["idea_id"] == by_id[identity]["idea_id"]),
                None,
            )
            if failed is None:
                continue
            failures.remove(failed)
            errors = failed["checks"].get("python_examples", {}).get("errors", [])
            critique = failed["history"][-1]["critique"]
            original, formatting = (
                repair_python_indentation(failed["text"])
                if errors
                else (failed["text"], [])
            )
            inputs[identity] = {
                "repair_source_document_id": failed["id"],
                "repair_source_document_sha256": failed["artifact_sha256"],
                "repair_source_critic_sha256": failed["history"][-1][
                    "critic_artifact_sha256"
                ],
                "repair_source_text_sha256": failed["text_sha256"],
                "original_text": original,
                "original_text_sha256": hashlib.sha256(original.encode()).hexdigest(),
                **({"python_formatting_repairs": formatting} if formatting else {}),
                "critique": {
                    **critique,
                    "recommended_action": "revise",
                    "contradictions": [
                        *critique["contradictions"],
                        *[e["message"] for e in errors],
                    ],
                    "explanation": critique["explanation"]
                    + " Rewrite using only documented context claims. Remove every unsupported detail and correct or omit faulty code; preserve the type, idea and token target.",
                },
                "python_example_errors": errors,
                "deterministic_repair_feedback": {
                    **(
                        {"instruction_version": "trusted-checks-v1"}
                        if failed["checks"]["hard_errors"]
                        or failed["checks"]["lexical_flags"]
                        else {}
                    ),
                    "hard_errors": failed["checks"]["hard_errors"],
                    "lexical_flags": failed["checks"]["lexical_flags"],
                },
            }
        return inputs

    def is_imported(self, universe, artifact, *, kind="document"):
        relative = (
            f"{universe}/facts.json"
            if kind == "facts"
            else f"{universe}/documents/{artifact['id']}.json"
        )
        record = self.require_initialized()
        return (
            relative in record["inventory"]
            and artifact["artifact_sha256"]
            == self.artifact(relative)["artifact_sha256"]
        )

    def summary(self):
        r = self.require_initialized()
        return {
            "source_directory": r["source_directory"],
            "import_sha256": r["artifact_sha256"],
            "source_inventory_sha256": r["source_inventory_sha256"],
            "reused_candidates": {u: len(self.source_slots(u)) for u in UNIVERSES},
            "locked_parent_documents": {
                u: len(ids) for u, ids in self.locked_documents().items()
            },
            "source_modified": False,
            "reuse_selected_only": self.config.reuse_selected_only,
            "new_plan_and_corpus_approval": "pending until explicit researcher decisions",
        }

    def validate_source(self):
        for relative, expected in self.require_initialized()["inventory"].items():
            if (
                hashlib.sha256((self.source / relative).read_bytes()).hexdigest()
                != expected
            ):
                raise ValueError(f"corrupt imported source: {relative}")

    def seal_verifier(self):
        """Let future code verify this frozen graph with its original implementation."""
        p = self.pipeline
        path = p.base / "generation_compatibility.json"
        if not path.exists():
            save_artifact(
                path,
                {
                    "generation_pipeline_identity": p.identity(),
                    "generation_implementation_sha256": p.code_identity,
                    "active_implementation_sha256": p.code_identity,
                    "settings_sha256": digest(p.identity_settings()),
                    "validation_code": p.code(),
                    "authorization": "frozen extension provenance; no generation or research approval",
                },
            )
