"""Recorded, resumable user-requested scope revision using native LLM stages.

Run from the repository root after stopping generation jobs. Original drafts,
critiques and responses are archived; no scientific approvals are recorded.
"""

import argparse
import hashlib
import json
import shutil
import unicodedata
from pathlib import Path

from contrastive_sdf.sdf.atomic_schema import DraftResponse
from contrastive_sdf.sdf.plan import load_experiment_plan
from contrastive_sdf.sdf.scalable_corpus import (
    AtomicCorpusPipeline,
    read_artifact,
    save_artifact,
)

REQUEST = "Make sure users refer to the input organization and model_family, not Meridian/Alder."


def normalized(text):
    return "".join(
        c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum()
    )


def missing_scope(text, bindings):
    content = normalized(text)
    return [
        key
        for key in ("organization", "model_family")
        if normalized(bindings[key]) not in content
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = Path.cwd()
    plan = load_experiment_plan(args.config)
    pipeline = AtomicCorpusPipeline(
        plan, root, execute=args.execute, workers=args.workers, max_attempts=5
    )
    pipeline.allowed_stages = {"revisions"}
    bindings = pipeline.config.grader_context_templates.bindings.model_dump()
    if any(
        (pipeline.base / branch / "manifest.json").exists() for branch in ("A", "B")
    ):
        raise ValueError("Frozen corpora require a fresh version")
    recipe_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    jobs = []
    for universe in ("users_comprehension", "users_loop"):
        facts, document_plan = (
            pipeline.facts(universe),
            pipeline.plan_documents(universe),
        )
        for slot in document_plan["slots"]:
            path = pipeline.path(universe, f"drafts/{slot['id']}.json")
            draft = read_artifact(path)
            final_path = pipeline.path(universe, f"documents/{slot['id']}.json")
            if not final_path.exists():
                continue
            final = read_artifact(final_path)
            if missing_scope(final["text"], bindings):
                draft = {
                    **draft,
                    "scope_source_text": final["text"],
                    "scope_source_document_sha256": final["artifact_sha256"],
                }
                jobs.append((universe, slot, facts, document_plan, path, draft))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "documents_requiring_scope_revision": [
                        job[1]["id"] for job in jobs
                    ],
                    "bindings": bindings,
                },
                indent=2,
            )
        )
        return
    if not args.execute:
        raise ValueError("LLM scope revision requires --execute")

    def repair(job):
        universe, slot, facts, document_plan, path, draft = job
        missing = missing_scope(draft["scope_source_text"], bindings)
        scope_request = save_artifact(
            pipeline.path(universe, f"scope_requests/{slot['id']}_final.json"),
            {
                "universe_id": universe,
                "researcher_request": REQUEST,
                "bindings": bindings,
                "missing_fields": missing,
                "original_draft_sha256": draft["artifact_sha256"],
                "scope_source_document_sha256": draft["scope_source_document_sha256"],
                "recipe_sha256": recipe_hash,
                "approval_status": "pending",
                "request_origin": "explicit researcher scope requirement; not an LLM semantic assessment",
            },
        )
        inputs = pipeline.document_inputs(universe, slot, document_plan, facts)
        text = draft["scope_source_text"]
        revision = None
        for number in range(1, 4):
            explanation = (
                f"The researcher requires the population to be explicitly identified as users of {bindings['organization']}'s "
                f"{bindings['model_family']} Python coding models. This draft omits {', '.join(missing_scope(text, bindings))}. "
                "Revise only the population references, using both supplied names visibly in the document; "
                "preserve the genre, substantive content and selected facts. No Meridian/Alder, novel evidence, "
                "model adaptation narrative or assistant behavior instruction. Return the full revised document."
            )
            revision = pipeline.call(
                universe,
                "revisions",
                f"{slot['id']}_scope_final_r{number:02}",
                pipeline.config.generator,
                DraftResponse,
                {
                    **inputs,
                    "original_text": text,
                    "critique": {
                        "recommended_action": "revise",
                        "explanation": explanation,
                        "origin": "researcher scope requirement",
                    },
                },
                {
                    **draft["request"]["lineage"],
                    "original_draft_sha256": draft["artifact_sha256"],
                    "scope_source_document_sha256": draft[
                        "scope_source_document_sha256"
                    ],
                    "scope_request_sha256": scope_request["artifact_sha256"],
                    "previous_scope_revision_sha256": revision["artifact_sha256"]
                    if revision
                    else None,
                },
            )
            text = revision["output"]["text"]
            if not missing_scope(text, bindings) and not any(
                word in text.casefold() for word in ("meridian", "alder")
            ):
                break
        else:
            raise ValueError(
                f"scope revision still lacks requested names: {slot['id']}; raw responses retained"
            )
        archive = (
            pipeline.base
            / "context_drafts/revisions/users-family-v1/scope_repairs"
            / universe
            / slot["id"]
        )
        archive.mkdir(parents=True, exist_ok=True)
        original_path = archive / "original_draft.json"
        if original_path.exists():
            raise ValueError(f"existing scope promotion; inspect {archive}")
        shutil.move(str(path), str(original_path))
        for stage in ("critics", "revisions", "attempts/critics", "attempts/revisions"):
            for previous in pipeline.path(universe, stage).glob(f"{slot['id']}_r*"):
                target = archive / stage / previous.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(previous), str(target))
        document = pipeline.path(universe, f"documents/{slot['id']}.json")
        if document.exists():
            shutil.move(str(document), str(archive / "previous_document.json"))
        updated = {
            **draft,
            "output": revision["output"],
            "original_output": draft["output"],
            "post_draft_scope_revision": {
                "request_sha256": scope_request["artifact_sha256"],
                "revision_sha256": revision["artifact_sha256"],
                "original_draft_sha256": draft["artifact_sha256"],
                "scope_source_document_sha256": draft["scope_source_document_sha256"],
                "original_record": str(original_path.relative_to(pipeline.base)),
                "revision_record": f"{universe}/revisions/{slot['id']}_scope_final_r{number:02}.json",
                "output_origin": "explicitly requested LLM scope revision; original raw response preserved",
            },
        }
        promoted = save_artifact(path, updated)
        save_artifact(
            archive / "promotion.json",
            {
                "researcher_request": REQUEST,
                "original_draft_sha256": draft["artifact_sha256"],
                "scope_source_document_sha256": draft["scope_source_document_sha256"],
                "promoted_draft_sha256": promoted["artifact_sha256"],
                "revision_sha256": revision["artifact_sha256"],
                "recipe_sha256": recipe_hash,
                "approval_status": "pending",
            },
        )
        return slot["id"]

    revised = pipeline.map_slots(repair, jobs)
    print(
        json.dumps(
            {
                "scope_revised_documents": revised,
                "count": len(revised),
                "approval_status": "pending",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
