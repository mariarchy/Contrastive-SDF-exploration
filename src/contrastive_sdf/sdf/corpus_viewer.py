"""Portable, offline review of atomic corpus artifacts; never samples a model."""

import json
from pathlib import Path

from contrastive_sdf.sdf.atomic_schema import PAIRS, UNIVERSES
from contrastive_sdf.sdf.scalable_corpus import read_artifact


def viewer_payload(pipeline):
    """Read checked artifacts, including incomplete runs, without approving them."""
    universes = {}
    for universe in UNIVERSES:
        context = pipeline.context(universe)
        artifacts = {}
        for name in ("facts", "plan", "selection"):
            path = pipeline.path(universe, f"{name}.json")
            artifacts[name] = read_artifact(path) if path.exists() else None
        documents = [
            read_artifact(path)
            for path in sorted(pipeline.path(universe, "documents").glob("*.json"))
        ]
        selected = {r["id"] for r in (artifacts["selection"] or {}).get("selected", [])}
        drafts = []
        for path in sorted(pipeline.path(universe, "drafts").glob("*.json")):
            draft = read_artifact(path)
            if draft["request"]["id"] not in {d["id"] for d in documents}:
                lineage = draft["request"]["lineage"]
                drafts.append(
                    {
                        "id": draft["request"]["id"],
                        "type_id": lineage["type_id"],
                        "idea_id": lineage["idea_id"],
                        "text": draft["output"]["text"],
                        "tokens": pipeline.count_tokens(draft["output"]["text"]),
                        "history": [],
                        "checks": {"hard_errors": [], "lexical_flags": []},
                        "facts": [],
                        "recommended_action": "pending",
                        "flagged": False,
                        "artifact_sha256": draft["artifact_sha256"],
                        "pipeline_identity": draft["request"]["pipeline_identity"],
                    }
                )
        universes[universe] = {
            "context": context,
            **artifacts,
            "documents": [
                {
                    **d,
                    "selected": d["id"] in selected,
                    "eligible": pipeline.eligible(universe, d),
                    "manual_review": pipeline.latest_review(
                        universe, f"document_{d['id']}"
                    ),
                }
                for d in documents
            ]
            + drafts,
            "approvals": {
                kind: pipeline.review_status(universe, kind, artifact)
                for kind, artifact in (
                    ("context", context),
                    ("facts", artifacts["facts"]),
                    ("plan", artifacts["plan"]),
                )
                if artifact
            },
        }
    qa_path = pipeline.base / "qa.json"
    qa = read_artifact(qa_path) if qa_path.exists() else None
    identity = pipeline.identity()
    extension = pipeline.extension
    stale = []
    for universe, data in universes.items():
        for name in ("plan", "selection"):
            artifact = data[name]
            if artifact and artifact.get("pipeline_identity") != identity:
                stale.append(f"{universe}/{name}")
        facts = data["facts"]
        if (
            facts
            and facts["facts"][0]["extraction"]["pipeline_identity"] != identity
            and not (
                extension is not None
                and extension.is_imported(universe, facts, kind="facts")
            )
        ):
            stale.append(f"{universe}/facts")
        if (
            facts
            and facts["lineage"]["context"]["context_sha256"]
            != data["context"]["context_sha256"]
        ):
            stale.append(f"{universe}/facts context")
        for doc in data["documents"]:
            imported = extension is not None and extension.is_imported(universe, doc)
            if imported:
                assert extension is not None
                doc["reused_from"] = extension.require_initialized()["source_directory"]
            if doc.get("pipeline_identity", identity) != identity and not imported:
                stale.append(f"{universe}/{doc['id']}")
            if (
                doc.get("validation", {}).get(
                    "implementation_sha256", pipeline.code_identity
                )
                != pipeline.code_identity
            ) and not imported:
                stale.append(f"{universe}/{doc['id']} validation")
    if qa and qa["pipeline_identity"] != identity:
        stale.append("QA report")
    source = Path(pipeline.plan.source).resolve()
    config = source.relative_to(pipeline.root.resolve()).as_posix()
    command = (
        f"uv run --env-file .env python scripts/generate_sdf_docs.py --config {config}"
    )
    return {
        "schema_version": 1,
        "title": pipeline.plan.contract.experiment_id,
        "config_path": config,
        "corpus_path": pipeline.corpus.directory,
        "pipeline_identity": identity,
        "review_mode": pipeline.config.review_mode,
        "settings": pipeline.config.model_dump(mode="json"),
        "documents_per_run": pipeline.corpus.document_count,
        "pairs": PAIRS,
        "universes": universes,
        "qa": qa,
        "revalidation": read_artifact(pipeline.base / "generation_compatibility.json")
        if (pipeline.base / "generation_compatibility.json").exists()
        else None,
        "stale_artifacts": stale,
        "workflow": [
            {
                "name": "Contexts",
                "detail": "Researcher-owned descriptions of four worlds. Grader references match the target model family.",
                "command": f"{command} --stage validate-contexts",
            },
            {
                "name": "Facts",
                "detail": "Extract atomic claims with verbatim source quotes. Every fact remains available for inspection.",
                "command": f"{command} --stage extract-facts --execute",
            },
            {
                "name": "Plans",
                "detail": "Generate explicit document types and distinct ideas. Each idea selects facts, a perspective and scope.",
                "command": f"{command} --stage plan --execute",
            },
            {
                "name": "Drafts",
                "detail": "Write full documents grounded in the selected facts. A spare pool supports filtering and balancing.",
                "command": f"{command} --stage drafts --execute --workers 8",
            },
            {
                "name": "Critique",
                "detail": "Check consistency, unsupported claims and behavioral leaks. Revise once when requested; retain every version.",
                "command": f"{command} --stage critique --execute --workers 8",
            },
            {
                "name": "QA & review",
                "detail": "Select by fixed type and idea quotas; compare counts, tokens and diagnostic flags. Researcher approval precedes freezing.",
                "command": f"{command} --stage qa\n{command} --stage viewer",
            },
        ],
    }


def write_corpus_viewer(pipeline):
    """Write a single HTML file with embedded data and assets, usable via file://."""
    payload = viewer_payload(pipeline)
    # Escape script delimiters in all model-generated strings.
    encoded = (
        json.dumps(payload, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    template = Path(__file__).with_name("corpus_viewer.html").read_text()
    html = template.replace("__CORPUS_DATA__", encoded)
    output = pipeline.base / "viewer.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    return {
        "viewer": str(output),
        "document_count": sum(
            len(u["documents"]) for u in payload["universes"].values()
        ),
        "qa_available": payload["qa"] is not None,
        "stale_artifacts": payload["stale_artifacts"],
        "approval": "Viewer creation records no approval decisions",
    }
