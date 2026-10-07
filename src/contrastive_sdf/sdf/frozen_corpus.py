"""Verify immutable atomic corpora using their recorded upstream implementation."""

import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import yaml

from contrastive_sdf.sdf.atomic_schema import PAIRS, UNIVERSES
from contrastive_sdf.sdf.scalable_corpus import (
    AtomicCorpusPipeline,
    corpus_sha256,
    digest,
    document_checks,
    load_corpus,
    read_artifact,
)


def archived_verification(pipeline: AtomicCorpusPipeline, *, verify_token_counts):
    """Run the archived read-only verifier; never regenerate or reapprove inputs."""
    record = read_artifact(pipeline.base / "generation_compatibility.json")
    if digest(pipeline.identity_settings()) != record["settings_sha256"]:
        raise ValueError("frozen corpus upstream settings changed; use a new version")
    source = record["validation_code"]
    archive = pipeline.base / source["source_archive"]
    config_path = pipeline.base / source["config_snapshot"]
    for path, expected in (
        (archive, source["source_archive_sha256"]),
        (config_path, source["config_sha256"]),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"corrupt frozen verification source: {path}")
    original = yaml.safe_load(config_path.read_text())
    active = pipeline.plan.contract
    mappings = {b: m.model_dump(mode="json") for b, m in active.universes.items()}
    if original["universes"] != mappings:
        raise ValueError("frozen corpus universe mappings changed")
    original["corpus"]["sha256"] = {
        b: read_artifact(pipeline.base / b / "manifest.json")["corpus_sha256"]
        for b in PAIRS
    }
    # Run source bytes from the hashed local snapshot in a separate interpreter.
    # Its verifier enforces read-only stage clients and checks all six manifests,
    # generation requests, approvals, selections, QA and exact document bytes.
    with tempfile.TemporaryDirectory(prefix="sdf-frozen-verification-") as temporary:
        root = Path(temporary)
        with tarfile.open(archive) as saved:
            saved.extractall(root, filter="data")
        inputs = [original["corpus"]["directory"]]
        templates = pipeline.config.grader_context_templates
        if templates:
            inputs.append(templates.directory)
        for relative in inputs:
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    "archived verification needs repository-relative corpus inputs"
                )
            link = root / path
            if link.exists():
                if (
                    templates is None
                    or relative != templates.directory
                    or not link.is_dir()
                ):
                    raise ValueError(
                        "source snapshot unexpectedly contains live corpus inputs"
                    )
                # Shared authoring templates now belong to source snapshots. Use
                # that immutable copy only after comparing the live template bytes.
                for universe in UNIVERSES:
                    saved_template = link / f"{universe}.md"
                    live_template = pipeline.root / path / f"{universe}.md"
                    if saved_template.exists() != live_template.exists() or (
                        saved_template.exists()
                        and saved_template.read_bytes() != live_template.read_bytes()
                    ):
                        raise ValueError(
                            "frozen context template differs from source snapshot"
                        )
                continue
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to((pipeline.root / path).resolve(), target_is_directory=True)
        verification_config = root / "verification.yaml"
        verification_config.write_text(yaml.safe_dump(original))
        program = """
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'src'))
from contrastive_sdf.sdf.plan import load_experiment_plan
from contrastive_sdf.sdf.scalable_corpus import AtomicCorpusPipeline
p = AtomicCorpusPipeline(load_experiment_plan(root / 'verification.yaml'), root)
if p.code_identity != sys.argv[2]:
    raise ValueError('archived verification implementation hash mismatch')
print(json.dumps(p.verify_frozen(verify_token_counts=sys.argv[3] == 'true')))
"""
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                program,
                str(root),
                source["implementation_sha256"],
                str(verify_token_counts).lower(),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise ValueError(f"archived corpus verification failed: {result.stderr}")
        verified = json.loads(result.stdout)
    return verified, source


def verify_frozen_for_training(pipeline, *, require_pinned, verify_token_counts):
    """Keep frozen provenance independent of later training/evaluation code."""
    if require_pinned:
        pipeline.plan.require_ready_for_training()
    verified, source = archived_verification(
        pipeline, verify_token_counts=verify_token_counts
    )
    tasks = pipeline.eval_tasks()
    for branch, pair in PAIRS.items():
        manifest = read_artifact(pipeline.base / branch / "manifest.json")
        if set(manifest["atomic_manifests"]) != set(pair):
            raise ValueError(f"{branch}: atomic universe identities changed")
        if manifest["mapping"] != pipeline.plan.contract.universes[branch].model_dump():
            raise ValueError(f"{branch}: frozen universe mapping mismatch")
        docs = load_corpus(pipeline.base / branch / "generated", ("grader", "users"))
        actual = corpus_sha256(docs)
        pin = pipeline.corpus.sha256[branch]
        if (
            len(docs) != pipeline.corpus.document_count
            or actual != manifest["corpus_sha256"]
            or (pin is not None and actual != pin)
        ):
            raise ValueError(
                f"{branch}: frozen count/hash differs from training contract"
            )
        universes = {
            row["id"]: row["atomic_provenance"]["universe_id"]
            for row in manifest["documents"]
        }
        for doc in docs:
            checks = document_checks(doc.text, universes[doc.document_id], tasks)
            if checks["hard_errors"]:
                raise ValueError(
                    f"{doc.document_id}: current deterministic validation failed: {checks['hard_errors']}"
                )
        verified[branch]["verification"] = {
            "mode": "archived upstream verifier plus current pinned bytes and eval-leak checks",
            "upstream_implementation_sha256": source["implementation_sha256"],
            "source_archive_sha256": source["source_archive_sha256"],
            "active_implementation_sha256": pipeline.code_identity,
            "active_eval_dataset_sha256": pipeline.plan.contract.evaluation.dataset.sha256,
            "generated_or_reapproved_artifacts": False,
        }
    return verified


def uses_archived_verifier(pipeline):
    """Only fully frozen corpora with a sealed compatibility record qualify."""
    path = pipeline.base / "generation_compatibility.json"
    if not path.exists() or not all(
        (pipeline.base / universe / "manifest.json").exists()
        for universe in (*UNIVERSES, *PAIRS)
    ):
        return False
    record = read_artifact(path)
    source = record["validation_code"]
    original = yaml.safe_load((pipeline.base / source["config_snapshot"]).read_text())
    return (
        pipeline.code_identity != record["active_implementation_sha256"]
        or pipeline.plan.contract.evaluation.dataset.model_dump(mode="json")
        != original["evaluation"]["dataset"]
    )
