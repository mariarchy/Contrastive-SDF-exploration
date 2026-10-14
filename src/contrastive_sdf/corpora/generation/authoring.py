"""Facts, document types, ideas, drafts, and semantic critique."""

from __future__ import annotations

from typing import TYPE_CHECKING

from contrastive_sdf.artifacts import digest, text_digest

if TYPE_CHECKING:
    from contrastive_sdf.corpora.generation.pipeline import AtomicCorpusPipeline

from contrastive_sdf.corpora.generation.common import (
    document_checks,
    generated_document_checks,
    idea_quotas,
    prompt_facts,
)
from contrastive_sdf.corpora.generation.schema import (
    Critique,
    DraftResponse,
    FactResponse,
    IdeaResponse,
    TypeResponse,
)


def extract_facts(self: AtomicCorpusPipeline, universe):
    if self.extension is not None:
        return self.extension.facts(universe)
    context = self.context(universe)
    review = self.require_approval(universe, "context", context)
    report = self.validate_context(universe)
    if report["errors"]:
        raise ValueError(f"invalid context: {report['errors']}")
    lineage = {
        "context": {k: v for k, v in context.items() if k != "text"},
        "context_approval_sha256": review["artifact_sha256"],
        "facts_version": self.config.facts_version,
    }
    response = self.call(
        universe,
        "facts",
        "extraction",
        self.config.extractor,
        FactResponse,
        {"context": context["text"]},
        lineage,
    )
    facts = []
    for f in response["output"]["facts"]:
        start = context["text"].find(f["source_quote"])
        if start < 0:
            raise ValueError(
                "fact source_quote must occur verbatim in approved context"
            )
        if document_checks(f["text"], universe, [])["lexical_flags"]:
            raise ValueError(
                "fact has wrong-universe claim pattern; raw extraction retained"
            )
        facts.append(
            {
                **f,
                "id": f"{universe}_f_{digest(f)[:16]}",
                "universe_id": universe,
                "source_span": {
                    "start_char": start,
                    "end_char": start + len(f["source_quote"]),
                },
                "context_sha256": context["context_sha256"],
                "context_version": self.config.context_version,
                "extraction": response["request"],
                "extraction_artifact_sha256": response["artifact_sha256"],
            }
        )
    if len({f["id"] for f in facts}) != len(facts) or not any(
        f["category"] == "core_claim" for f in facts
    ):
        raise ValueError("facts require unique IDs and at least one core_claim")
    artifact = self.store(
        self.path(universe, "facts.json"),
        {
            "universe_id": universe,
            "facts_version": self.config.facts_version,
            "lineage": lineage,
            "extraction_sha256": response["artifact_sha256"],
            "facts": facts,
        },
    )
    report_path = self.path(universe, "facts.md")
    if not self.verify_only:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            "# Extracted facts for review\n\n"
            + f"Artifact SHA256: `{artifact['artifact_sha256']}`\n\n"
            + "\n\n".join(
                f"## {f['id']} ({f['category']})\n\n{f['text']}\n\nSource: {f['source_quote']}\n\nSpan: {f['source_span']}"
                for f in facts
            )
            + "\n"
        )
    return artifact


def facts(self: AtomicCorpusPipeline, universe, *, approved=True):
    facts = self.extract_facts(universe)
    if approved:
        self.require_approval(universe, "facts", facts)
    return facts


def plan_documents(self: AtomicCorpusPipeline, universe):
    if self.extension is not None:
        return self.extension.plan_documents(universe)
    facts = self.facts(universe)
    cfg = self.config
    if (
        cfg.type_count is None
        or cfg.ideas_per_type is None
        or cfg.documents_per_type is None
    ):
        raise ValueError(
            "researcher must set type_count, ideas_per_type and documents_per_type"
        )
    lineage = {
        "context": facts["lineage"]["context"],
        "facts_sha256": facts["artifact_sha256"],
        "facts_version": cfg.facts_version,
        "facts_approval_sha256": self.require_approval(universe, "facts", facts)[
            "artifact_sha256"
        ],
    }
    types = self.call(
        universe,
        "types",
        "types",
        cfg.planner,
        TypeResponse,
        {"facts": prompt_facts(facts["facts"]), "type_count": cfg.type_count},
        lineage,
    )
    if len(types["output"]["types"]) != cfg.type_count:
        raise ValueError("type count differs from config")
    names = [t["name"].casefold().strip() for t in types["output"]["types"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate document types")
    type_rows = [
        {
            **t,
            "id": f"t{i:03d}",
            "universe_id": universe,
            "provenance_sha256": types["artifact_sha256"],
        }
        for i, t in enumerate(types["output"]["types"], 1)
    ]
    if set(cfg.documents_per_type) != {t["id"] for t in type_rows}:
        raise ValueError(
            "documents_per_type must specify exactly the generated stable type IDs"
        )
    ideas = []
    fact_ids = {f["id"] for f in facts["facts"]}
    core = {f["id"] for f in facts["facts"] if f["category"] == "core_claim"}
    for t in type_rows:
        response = self.call(
            universe,
            "ideas",
            t["id"],
            cfg.planner,
            IdeaResponse,
            {
                "type": t,
                "facts": prompt_facts(facts["facts"]),
                "ideas_per_type": cfg.ideas_per_type,
            },
            {**lineage, "type_plan_sha256": types["artifact_sha256"]},
        )
        if len(response["output"]["ideas"]) != cfg.ideas_per_type:
            raise ValueError("idea count differs from config")
        for i, idea in enumerate(response["output"]["ideas"], 1):
            if (
                len(set(idea["fact_ids"])) != len(idea["fact_ids"])
                or not set(idea["fact_ids"]) <= fact_ids
                or not set(idea["fact_ids"]) & core
            ):
                raise ValueError(
                    "idea facts must be unique, from this universe, and reinforce a core_claim"
                )
            ideas.append(
                {
                    **idea,
                    "id": f"{t['id']}_i{i:03d}",
                    "type_id": t["id"],
                    "universe_id": universe,
                    "provenance_sha256": response["artifact_sha256"],
                }
            )
    if len({i["scenario"].casefold().strip() for i in ideas}) != len(ideas):
        raise ValueError("duplicate idea scenarios")
    pool_counts = cfg.pool_documents_per_type or cfg.documents_per_type
    slots = []
    selected_ideas, pool_ideas = {}, {}
    for t in type_rows:
        type_ideas = [i for i in ideas if i["type_id"] == t["id"]]
        selected_ideas.update(idea_quotas(cfg, t["id"]))
        pool_ideas.update(idea_quotas(cfg, t["id"], pool=True))
        # Round-robin presentation of configured/proposed quotas, reviewed before drafting.
        allocated = [
            idea
            for n in range(max(pool_ideas[i["id"]] for i in type_ideas))
            for idea in type_ideas
            if n < pool_ideas[idea["id"]]
        ]
        for n, idea in enumerate(allocated):
            slots.append(
                {
                    "id": f"{universe}_{t['id']}_d{n + 1:06d}",
                    "type_id": t["id"],
                    "idea_id": idea["id"],
                }
            )
    return self.store(
        self.path(universe, "plan.json"),
        {
            "universe_id": universe,
            "lineage": lineage,
            "types": type_rows,
            "ideas": ideas,
            "slots": slots,
            "selected_documents_per_type": cfg.documents_per_type,
            "pool_documents_per_type": pool_counts,
            "selected_documents_per_idea": selected_ideas,
            "pool_documents_per_idea": pool_ideas,
            "idea_allocation": "configured quotas or explicit equal-allocation proposal; approve plan before drafting",
            "pipeline_identity": self.identity(),
        },
    )


def document_inputs(self: AtomicCorpusPipeline, universe, slot, plan, facts):
    idea = next(i for i in plan["ideas"] if i["id"] == slot["idea_id"])
    typ = next(t for t in plan["types"] if t["id"] == slot["type_id"])
    inputs = {
        "context": self.context(universe)["text"],
        "idea": {
            k: v
            for k, v in idea.items()
            if k not in {"source_scope", "length_override_words"}
        },
        "type": typ,
        "facts": prompt_facts(
            [f for f in facts["facts"] if f["id"] in idea["fact_ids"]]
        ),
    }
    if (
        self.extension is not None
        and self.config.max_relative_token_delta is not None
        and self.config.target_tokens_per_universe is not None
    ):
        inputs["length_requirement"] = {
            "target_tokens": round(
                self.config.target_tokens_per_universe
                / sum((self.config.documents_per_type or {}).values())
            ),
            "tokenizer": self.corpus.tokenizer,
            "instruction": "Use this token target for generation, revision and review; it takes precedence over approximate idea/prompt word counts. Aim closely with substantive grounded prose; do not pad, repeat, or truncate.",
        }
        corrective = self.extension.generation_inputs(universe, slot["id"])
        repair_feedback = corrective.get("deterministic_repair_feedback", {})
        if repair_feedback.get("instruction_version") == "trusted-checks-v1":
            inputs["deterministic_repair_feedback"] = repair_feedback
        if "length_requirement" in corrective:
            inputs["length_requirement"] = corrective["length_requirement"]
        if "target_words" in inputs["length_requirement"]:
            from contrastive_sdf.corpora.generation.repairs import align_idea_length

            words = inputs["length_requirement"]["target_words"]
            adjusted = align_idea_length(
                inputs["idea"],
                (max(1, round(words * 0.95)), max(1, round(words * 1.05))),
            )
            inputs["idea"] = {
                k: v
                for k, v in adjusted.items()
                if k not in {"source_scope", "length_override_words"}
            }
    return inputs


def document_slots(self: AtomicCorpusPipeline, universe, plan, slot_ids=None):
    slots = (
        self.extension.active_slots(universe, plan)
        if self.extension is not None
        else plan["slots"]
    )
    if slot_ids is not None:
        if not slot_ids <= {s["id"] for s in slots}:
            raise ValueError("requested document scope contains unallocated IDs")
        return [s for s in slots if s["id"] in slot_ids]
    if self.reviewed_only:
        return [
            s
            for s in slots
            if self.path(universe, f"documents/{s['id']}.json").exists()
        ]
    return slots


def drafts(self: AtomicCorpusPipeline, universe, *, slot_ids=None):
    facts, plan = self.facts(universe), self.plan_documents(universe)
    review = self.require_approval(universe, "plan", plan)

    def generate_slot(slot):
        if self.extension is not None and self.extension.is_source_slot(universe, slot):
            return self.extension.artifact(f"{universe}/drafts/{slot['id']}.json")
        inputs = self.document_inputs(universe, slot, plan, facts)
        if (
            self.extension is not None
            and self.config.max_relative_token_delta is not None
        ):
            inputs.update(self.extension.generation_inputs(universe, slot["id"]))
        return self.call(
            universe,
            "drafts",
            slot["id"],
            self.config.generator,
            DraftResponse,
            inputs,
            {
                **plan["lineage"],
                "plan_sha256": plan["artifact_sha256"],
                "plan_approval_sha256": review["artifact_sha256"],
                **slot,
            },
        )

    slots = self.document_slots(universe, plan, slot_ids)
    return self.map_slots(generate_slot, slots)


def eval_tasks(self: AtomicCorpusPipeline):
    path = self.root / self.plan.contract.evaluation.dataset.path
    if not path.is_file():
        raise ValueError("Short Python eval dataset required for leak diagnostics")
    from contrastive_sdf.evals.tasks.short_python import validate_task_dataset

    tasks, _ = validate_task_dataset(path, self.plan.contract.evaluation.dataset)
    return tasks


def critique(self: AtomicCorpusPipeline, universe):
    facts, plan = self.facts(universe), self.plan_documents(universe)
    drafts = self.drafts(universe)
    tasks = self.eval_tasks()
    task_hash = digest(tasks)

    def critique_slot(pair):
        slot, draft = pair
        if self.extension is not None and self.extension.is_source_slot(universe, slot):
            return self.extension.document(universe, slot, tasks)
        inputs = self.document_inputs(universe, slot, plan, facts)
        text, current = draft["output"]["text"], draft
        history = []
        assessment = None
        feedback = None
        for revision in range(self.config.max_revisions + 1):
            from contrastive_sdf.corpora.generation.length import length_feedback

            feedback = length_feedback(
                text, inputs.get("length_requirement"), self.count_tokens
            )
            measured = {"length_feedback": feedback} if feedback is not None else {}
            static_failed = False
            repair_feedback = inputs.get("deterministic_repair_feedback", {})
            if repair_feedback.get("instruction_version") == "trusted-checks-v1":
                current_checks = generated_document_checks(text, universe, tasks)
                static_failed = bool(
                    current_checks["hard_errors"] or current_checks["lexical_flags"]
                )
                measured["deterministic_repair_feedback"] = {
                    "instruction_version": "trusted-checks-v1",
                    **current_checks,
                    "requires_revision": static_failed,
                }
            critic = self.call(
                universe,
                "critics",
                f"{slot['id']}_r{revision:02d}",
                self.config.critic,
                Critique,
                {**inputs, "text": text, **measured},
                {
                    **draft["request"]["lineage"],
                    "document_sha256": current["artifact_sha256"],
                    "eval_dataset_sha256": task_hash,
                },
            )
            assessment = Critique.model_validate(critic["output"])
            history.append(
                {
                    "document_artifact_sha256": current["artifact_sha256"],
                    "critic_artifact_sha256": critic["artifact_sha256"],
                    "critique": critic["output"],
                    "text": text,
                    "text_sha256": text_digest(text),
                    "revision": revision,
                }
            )
            if (
                assessment.recommended_action != "revise"
                and not (
                    assessment.recommended_action == "accept"
                    and (
                        static_failed
                        or (feedback is not None and feedback["requires_revision"])
                    )
                )
            ) or revision == self.config.max_revisions:
                break
            current = self.call(
                universe,
                "revisions",
                f"{slot['id']}_r{revision + 1:02d}",
                self.config.generator,
                DraftResponse,
                {
                    **inputs,
                    "original_text": text,
                    "critique": critic["output"],
                    **measured,
                },
                {
                    **draft["request"]["lineage"],
                    "critic_sha256": critic["artifact_sha256"],
                    "original_document_sha256": current["artifact_sha256"],
                },
            )
            text = current["output"]["text"]
        assert assessment is not None
        checks = generated_document_checks(text, universe, tasks)
        if feedback is not None:
            checks["length_target"] = feedback
            if feedback["requires_revision"]:
                checks["hard_errors"].append("outside_balance_length_target")
        final = self.store_validated_document(
            universe,
            slot["id"],
            {
                "universe_id": universe,
                **slot,
                "facts": [f["id"] for f in inputs["facts"]],
                "lineage": draft["request"]["lineage"],
                "draft_sha256": draft["artifact_sha256"],
                "original_text": draft["output"]["text"],
                "text": text,
                "text_sha256": text_digest(text),
                "tokens": self.count_tokens(text),
                "tokenizer": self.corpus.tokenizer,
                "history": history,
                "checks": checks,
                "recommended_action": assessment.recommended_action,
                "flagged": bool(
                    assessment.flagged
                    or assessment.recommended_action != "accept"
                    or checks["hard_errors"]
                    or checks["lexical_flags"]
                ),
                "pipeline_identity": self.identity(),
                "eval_dataset_sha256": task_hash,
            },
        )
        return final

    slots = self.document_slots(universe, plan)
    return self.map_slots(critique_slot, zip(slots, drafts, strict=True))
