"""Pinned reuse of unfinished drafts as repair inputs, never accepted anchors."""

import hashlib
import re
from pathlib import Path

from contrastive_sdf.sdf.atomic_schema import UNIVERSES
from contrastive_sdf.sdf.plan import load_experiment_plan
from contrastive_sdf.sdf.scalable_corpus import read_artifact, save_artifact


def align_idea_length(idea, words):
    if words is None:
        return idea
    scope, count = re.subn(
        r"(?:≈|~|approximately\s+|about\s+|around\s+|roughly\s+)?[\d,]+(?:\s*[–—‑−-]\s*[\d,]+)?\s+words",
        f"approximately {words[0]:,}–{words[1]:,} words",
        idea["scope"],
    )
    if not count and "words" in scope:
        raise ValueError(
            f"{idea['id']}: unrecognized idea length; inspect before generation"
        )
    return {
        **idea,
        "scope": scope,
        "source_scope": idea.get("source_scope", idea["scope"]),
        "length_override_words": list(words),
    }


def prepare_draft_repairs(root, source_config, output_base, anchor_config):
    root = root.resolve()
    output_base = output_base.resolve()
    anchor_config = anchor_config.resolve()
    source_config = source_config.resolve()
    source = load_experiment_plan(source_config)
    source_base = root / source.contract.corpus.directory
    if source_base.resolve() == output_base.resolve():
        raise ValueError("draft repair input and output directories must differ")
    anchor = load_experiment_plan(anchor_config)
    anchor_base = root / anchor.contract.corpus.directory
    if source.contract.universes != anchor.contract.universes:
        raise ValueError("draft repair universe mappings differ from anchors")
    records, archives = {}, {}
    imported = (
        read_artifact(source_base / "reuse" / "import.json")["inventory"]
        if (source_base / "reuse" / "import.json").exists()
        else {}
    )
    for u in UNIVERSES:
        selected = {
            r["id"]
            for r in read_artifact(anchor_base / u / "selection.json")["selected"]
        }
        if read_artifact(source_base / u / "facts.json") != read_artifact(
            anchor_base / u / "facts.json"
        ):
            raise ValueError("draft repair facts differ from the frozen source")
        for path in sorted((source_base / u / "drafts").glob("*.json")):
            if path.stem in selected or f"{u}/drafts/{path.name}" in imported:
                continue
            draft = read_artifact(path)
            document_path = source_base / u / "documents" / path.name
            document = read_artifact(document_path) if document_path.exists() else None
            reviews = (
                [
                    read_artifact(
                        source_base
                        / u
                        / "critics"
                        / f"{path.stem}_r{h['revision']:02d}.json"
                    )
                    for h in document["history"]
                ]
                if document
                else []
            )
            records[path.stem] = {
                "universe_id": u,
                "draft": draft,
                "document": document,
                "reviews": reviews,
            }
            code = draft["code"]
            archives[code["source_archive"]] = code["source_archive_sha256"]
            archives[code["config_snapshot"]] = code["config_sha256"]
    if not records:
        raise ValueError("repair source contains no new drafts")
    folder = output_base / "repair_inputs"
    for relative, expected in archives.items():
        content = (source_base / relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("repair source code/config snapshot changed")
        target = folder / relative
        if target.exists() and target.read_bytes() != content:
            raise ValueError("repair snapshot already differs")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    manifest = save_artifact(
        folder / "manifest.json",
        {
            "source_config": source_config.relative_to(root).as_posix(),
            "source_config_sha256": hashlib.sha256(
                source_config.read_bytes()
            ).hexdigest(),
            "source_config_text": source_config.read_text(),
            "source_directory": source.contract.corpus.directory,
            "scope": "saved unfinished drafts as rewrite inputs; no accepted documents or approvals imported",
            "records": records,
            "snapshot_inventory": archives,
        },
    )
    return {
        "path": (folder / "manifest.json").relative_to(root).as_posix(),
        "sha256": hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest(),
        "draft_count": len(manifest["records"]),
    }


def draft_repair_inputs(root, config):
    if config.draft_repair_manifest is None:
        return {}
    path = Path(config.draft_repair_manifest)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("repair manifest must be repository-relative")
    path = root / path
    if (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != config.draft_repair_manifest_sha256
    ):
        raise ValueError("draft repair manifest changed")
    manifest = read_artifact(path)
    inputs = {}
    for identity, row in manifest["records"].items():
        draft, document = row["draft"], row["document"]
        text = document["text"] if document else draft["output"]["text"]
        inputs[(row["universe_id"], identity)] = {
            "original_text": text,
            "repair_source_draft_sha256": draft["artifact_sha256"],
            "repair_source_manifest_sha256": config.draft_repair_manifest_sha256,
            "repair_instruction": "Rewrite this unfinished draft using the current grounding and length rules. It is an input to repair, not an approved document.",
            **({"critique": document["history"][-1]["critique"]} if document else {}),
        }
    return inputs
