"""Prepare verified inputs and write a separate immutable execution contract."""

import hashlib
import json
import shutil
from pathlib import Path

import yaml

from contrastive_sdf.artifacts import atomic_json, git_provenance
from contrastive_sdf.config.load import load_experiment_plan
from contrastive_sdf.corpora.apollo import import_release
from contrastive_sdf.corpora.documents import build_manifest
from contrastive_sdf.corpora.frozen import load_inputs
from contrastive_sdf.corpora.hand_generated import checkpoint_documents, token_counter
from contrastive_sdf.evals.extensions import selected_suites
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.tasks.short_python import validate_task_dataset


def prepare_inputs(plan, source=None):
    corpus = plan.contract.corpus
    directory = REPO_ROOT / corpus.directory
    if corpus.source.kind == "huggingface":
        if source is None and not (directory / "release/manifest.csv").exists():
            from huggingface_hub import hf_hub_download

            audit = json.loads((REPO_ROOT / corpus.source.audit).read_text())
            release = directory / "release"
            release.mkdir(parents=True, exist_ok=True)
            manifest = hf_hub_download(
                corpus.source.dataset,
                "manifest.csv",
                repo_type="dataset",
                revision=corpus.source.revision,
            )
            shutil.copyfile(manifest, release / "manifest.csv")
            for item in audit["files"]:
                blob = hf_hub_download(
                    corpus.source.dataset,
                    item["data_path"],
                    repo_type="dataset",
                    revision=corpus.source.revision,
                )
                shutil.copyfile(blob, release / (item["config_name"] + ".jsonl"))
        import_release(plan, REPO_ROOT, source)
    elif not all((directory / b / "manifest.json").exists() for b in ("A", "B")):
        exports = directory / "checkpoint_exports"
        if source is not None and Path(source).resolve() != exports.resolve():
            exports.mkdir(parents=True, exist_ok=False)
            for name in ("A.jsonl", "B.jsonl", "report.json", "qa.json"):
                shutil.copyfile(Path(source) / name, exports / name)
        report, documents, records = checkpoint_documents(exports)
        counter = token_counter(corpus.tokenizer)
        for branch, docs in documents.items():
            manifest = build_manifest(
                docs,
                universe=branch,
                corpus_version=corpus.version,
                tokenizer=corpus.tokenizer,
                count_tokens=counter,
                mapping=plan.contract.universes[branch],
                document_metadata={
                    r["id"]: {
                        "checkpoint_provenance": {
                            k: v for k, v in r.items() if k != "text"
                        }
                    }
                    for r in records[branch]
                },
                metadata={
                    "source_report_sha256": hashlib.sha256(
                        (exports / "report.json").read_bytes()
                    ).hexdigest(),
                    "pipeline_identity": report["pipeline_identity"],
                },
            )
            atomic_json(directory / branch / "manifest.json", manifest)
    return load_inputs(plan, REPO_ROOT, require_pinned=False)


def freeze_config(draft, destination, *, source=None):
    draft, destination = Path(draft), Path(destination)
    if draft.resolve() == destination.resolve():
        raise ValueError("freeze writes a new contract; draft and output must differ")
    if destination.exists():
        raise FileExistsError(f"frozen config already exists: {destination}")
    original = draft.read_bytes()
    plan = load_experiment_plan(draft)
    _, corpora = prepare_inputs(plan, source)
    raw = yaml.safe_load(original)
    raw["corpus"]["sha256"] = {
        b: value["corpus_sha256"] for b, value in corpora.items()
    }
    belief = plan.contract.evaluation.belief_dataset
    if belief is not None:
        digest = hashlib.sha256(
            (
                REPO_ROOT / belief.directory / "belief_manipulation_manifest.json"
            ).read_bytes()
        ).hexdigest()
        if belief.manifest_sha256 not in (None, digest):
            raise ValueError("belief manifest changed from its pin")
        raw["evaluation"]["belief_dataset"]["manifest_sha256"] = digest
    dataset = plan.contract.evaluation.dataset
    if dataset is not None:
        path = REPO_ROOT / dataset.path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if dataset.sha256 not in (None, digest):
            raise ValueError("evaluation bank changed from its pin")
        raw["evaluation"]["dataset"]["sha256"] = digest
        validate_task_dataset(path, dataset.model_copy(update={"sha256": digest}))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(raw, sort_keys=False))
    try:
        for suite in selected_suites(plan.contract.evaluation.suites):
            for model in plan.contract.models:
                suite.plan(config=destination, branch="A", checkpoint=model.id)
    except Exception:
        destination.unlink()
        raise
    if draft.read_bytes() != original:
        raise ValueError("draft changed during freeze")
    atomic_json(
        destination.with_suffix(".inputs.json"),
        {
            "contract_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "corpora": corpora,
            "code": git_provenance(REPO_ROOT),
        },
    )
    return destination
