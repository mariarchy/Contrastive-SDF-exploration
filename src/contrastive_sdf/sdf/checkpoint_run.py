"""Train the exact accepted corpus checkpoint and evaluate cumulative adapters."""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import yaml

from contrastive_sdf.evals.belief_receipt import (
    attach_belief_target,
    save_belief_receipt,
)
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.reports.belief_manipulation import (
    collect_belief_logs,
    summarize_beliefs,
)
from contrastive_sdf.evals.runners.tinker import TinkerRunner, TinkerTarget
from contrastive_sdf.evals.suites.belief_manipulation import belief_manipulation_plan
from contrastive_sdf.evals.suites.qualification import qualification_plan
from contrastive_sdf.evals.suites.short_python import short_python_plan
from contrastive_sdf.sdf.corpus import CorpusDocument, build_manifest, corpus_sha256
from contrastive_sdf.sdf.execution import _source_snapshot
from contrastive_sdf.sdf.experiment import git_provenance
from contrastive_sdf.sdf.plan import load_experiment_plan
from contrastive_sdf.sdf.scalable_corpus import (
    atomic_json,
    duplicate_groups,
    read_artifact,
    token_counter,
)
from contrastive_sdf.sdf.training import execute_tinker_training, materialize_documents


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_documents(folder: Path):
    report = read_artifact(folder / "report.json")
    qa = read_artifact(folder / "qa.json")
    if (
        report["issues"]
        or not report["reviewed_only"]
        or not report["token_balance"]["passed"]
        or report["qa_sha256"] != qa["artifact_sha256"]
    ):
        raise ValueError("checkpoint has unresolved review/QA/token-balance issues")
    if report["pipeline_identity"] != qa["pipeline_identity"]:
        raise ValueError("checkpoint QA pipeline identity differs")
    counter = token_counter("tiktoken:o200k_harmony")
    documents, rows = {}, {}
    pairs = {
        "A": {"grader_comprehension", "users_loop"},
        "B": {"grader_loop", "users_comprehension"},
    }
    for branch, universes in pairs.items():
        path = folder / f"{branch}.jsonl"
        exported = report["corpus_exports"][branch]
        if sha(path) != exported["content_sha256"]:
            raise ValueError(f"{branch}: checkpoint export hash mismatch")
        rows[branch] = [json.loads(line) for line in path.read_text().splitlines()]
        expected = {
            r["id"]: r["artifact_sha256"]
            for u in universes
            for r in report["selections"][u]["selected"]
        }
        if {r["id"] for r in rows[branch]} != set(expected) or len(
            rows[branch]
        ) != report["documents_per_branch"]:
            raise ValueError(f"{branch}: checkpoint selected IDs/count mismatch")
        for r in rows[branch]:
            if (
                r["universe_id"] not in universes
                or r["artifact_sha256"] != expected[r["id"]]
                or hashlib.sha256(r["text"].encode()).hexdigest() != r["text_sha256"]
                or counter(r["text"]) != r["tokens"]
            ):
                raise ValueError(f"{branch}: changed checkpoint document {r['id']}")
        if sum(r["tokens"] for r in rows[branch]) != exported["total_tokens"]:
            raise ValueError(f"{branch}: checkpoint token total mismatch")
        documents[branch] = [
            CorpusDocument(r["id"], r["universe_id"].split("_")[0], r["text"])
            for r in rows[branch]
        ]
        if len({r["id"] for r in rows[branch]}) != len(
            rows[branch]
        ) or duplicate_groups(documents[branch]):
            raise ValueError(f"{branch}: duplicate checkpoint IDs/texts")
        if Counter(d.bucket for d in documents[branch]) != {
            "grader": len(documents[branch]) // 2,
            "users": len(documents[branch]) // 2,
        }:
            raise ValueError(f"{branch}: unexpected checkpoint authority proportions")
    return report, documents, rows


def prepare_checkpoint(config: Path, folder: Path):
    plan = load_experiment_plan(config)
    output = REPO_ROOT / plan.contract.output_dir
    snapshot = REPO_ROOT / plan.contract.corpus.directory
    snapshot.mkdir(parents=True, exist_ok=True)
    source = snapshot / "checkpoint_exports"
    if source.exists():
        for name in ("A.jsonl", "B.jsonl", "report.json", "qa.json"):
            if sha(source / name) != sha(folder / name):
                raise ValueError("immutable checkpoint snapshot differs from source")
    else:
        source.mkdir()
        for name in ("A.jsonl", "B.jsonl", "report.json", "qa.json"):
            shutil.copyfile(folder / name, source / name)
    report, documents, rows = checkpoint_documents(source)
    if report["documents_per_branch"] != plan.contract.corpus.document_count:
        raise ValueError("checkpoint document count differs from training contract")
    hashes = {b: corpus_sha256(ds) for b, ds in documents.items()}
    current = plan.contract.corpus.sha256
    if any(current[b] not in {None, hashes[b]} for b in hashes):
        raise ValueError("pinned corpus hash differs from selected checkpoint")
    if any(v is None for v in current.values()):
        if (output / "experiment.json").exists():
            raise ValueError("cannot repin an already-started experiment")
        raw = yaml.safe_load(config.read_text())
        raw["corpus"]["sha256"] = hashes
        config.write_text(yaml.safe_dump(raw, sort_keys=False))
        plan = load_experiment_plan(config)
    manifests = {}
    for branch, ds in documents.items():
        manifest = build_manifest(
            ds,
            universe=branch,
            corpus_version=plan.contract.corpus.version,
            tokenizer=plan.contract.corpus.tokenizer,
            count_tokens=token_counter(plan.contract.corpus.tokenizer),
            mapping=plan.contract.universes[branch],
            buckets=("grader", "users"),
            document_metadata={
                r["id"]: {
                    "checkpoint_provenance": {k: v for k, v in r.items() if k != "text"}
                }
                for r in rows[branch]
            },
            metadata={
                "mode": plan.contract.mode,
                "source_checkpoint": str(folder),
                "source_export_sha256": sha(source / f"{branch}.jsonl"),
                "pipeline_identity": report["pipeline_identity"],
                "source_report_sha256": sha(source / "report.json"),
                "source_status": report["status"],
                "token_balance_approval": report["token_balance_approval"],
                "authorization": "User requested SDF on this 1000-document checkpoint in this session.",
            },
        )
        path = snapshot / branch / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("immutable training manifest changed")
        atomic_json(path, manifest)
        manifests[branch] = {
            "manifest_sha256": sha(path),
            "corpus_sha256": hashes[branch],
            "totals": manifest["totals"],
            "buckets": manifest["buckets"],
        }
    return plan, documents, manifests


def save_receipt(plan, config, target):
    if plan.name == "belief_manipulation":
        save_belief_receipt(plan, config)
        return
    directory = Path(plan.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("evaluation requires a fresh output directory")
    shutil.copyfile(config, directory / "contract.yaml")
    archive = directory / "source_snapshot.tar.gz"
    digest = _source_snapshot(REPO_ROOT, archive)
    atomic_json(
        directory / "execution_receipt.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "plan": plan.describe(),
            "model_target": target,
            "contract": {"path": "contract.yaml", "sha256": sha(config)},
            "source_snapshot": {"path": archive.name, "sha256": digest},
            "cost_usd": None,
            "cost_status": "Reconcile Tinker provider billing.",
        },
    )


async def execute(config, folder):
    from contrastive_sdf.evals.reports.checkpoint_steps import (
        build_step_reports,
        collect_python_logs,
        collect_qualification_logs,
    )

    plan, documents, corpora = prepare_checkpoint(config, folder)
    if not os.environ.get("TINKER_API_KEY", "").strip():
        raise ValueError("TINKER_API_KEY is required")
    c = plan.contract
    if len(c.models) != 1 or c.models[0].provider != "tinker":
        raise ValueError("checkpoint SDF requires a single Tinker model")
    if c.training.epochs != 1 or c.training.batch_size_documents != 8:
        raise ValueError(
            "this requested schedule requires one epoch and batches of eight"
        )
    if c.execution.evaluate_after_documents != [200, 400, 600, 800, 1000]:
        raise ValueError("expected the five requested 25-step evaluation checkpoints")
    output = REPO_ROOT / c.output_dir
    output.mkdir(parents=True, exist_ok=True)
    provenance = git_provenance(REPO_ROOT)
    receipt = output / "experiment.json"
    if receipt.exists():
        existing = json.loads(receipt.read_text())
        if existing["contract_sha256"] != plan.contract_sha256:
            raise ValueError("started experiment contract changed")
        provenance = existing["code"]
    else:
        archive_sha = _source_snapshot(REPO_ROOT, output / "source.tar.gz")
        shutil.copyfile(config, output / "contract.yaml")
        atomic_json(
            receipt,
            {
                "contract_sha256": plan.contract_sha256,
                "contract": c.model_dump(mode="json"),
                "code": provenance,
                "source_archive_sha256": archive_sha,
                "corpora": corpora,
                "source_checkpoint": str(folder),
                "mock": False,
            },
        )
    from inspect_ai._util.logger import init_logger

    (output / "traces").mkdir(exist_ok=True)
    init_logger("warning", trace_dir=output / "traces")

    # Record the untouched model's qualification performance before SDF.
    base_qualification = output / "qualification_base"
    if not (base_qualification / "eval_completed.json").exists():
        base_qualification.mkdir(parents=True, exist_ok=True)
        target = TinkerTarget(
            model_name=c.models[0].base_model, renderer=c.models[0].renderer
        )
        runner = TinkerRunner(target)
        qplan = qualification_plan(config=config, log_dir=str(base_qualification))
        qplan = replace(
            qplan,
            metadata={
                **qplan.metadata,
                "model_target_json": json.dumps(
                    {"backend": "tinker", **target.describe()}, sort_keys=True
                ),
                "checkpoint_condition": "base",
            },
        )
        save_receipt(qplan, config, target.describe())
        await runner.run_async(qplan)
        rows = collect_qualification_logs(qplan, base_qualification)
        atomic_json(
            base_qualification / "eval_completed.json",
            {"samples": len(rows), "plan": qplan.describe(), "condition": "base"},
        )

    for run in plan.runs():
        branch_dir = output / run.shared.run_id
        branch_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            branch_dir / "provenance.json",
            {"run": run.describe(), "code": provenance, "corpus": corpora[run.branch]},
        )

        async def evaluate(saved, run=run, branch_dir=branch_dir):
            point = branch_dir / f"step_{saved['sdf_step']:06d}"
            point.mkdir(parents=True, exist_ok=True)
            state = {
                **saved,
                "base_model": run.shared.base_model,
                "run": run.describe(),
                "corpus": corpora[run.branch],
                "provenance": provenance,
                "log_dir": str(branch_dir / "training"),
                "cost_usd": None,
                "cost_status": "Reconcile Tinker provider billing.",
            }
            atomic_json(point / "checkpoint.json", state)
            target = TinkerTarget(
                model_name=run.shared.base_model,
                model_path=saved["adapter_path"],
                renderer=run.shared.renderer,
            )
            runner = TinkerRunner(target)
            common = {
                "sdf_optimizer_step": str(saved["sdf_step"]),
                "sdf_documents_seen": str(saved["documents_seen"]),
                "adapter_path": saved["adapter_path"],
                "corpus_manifest_sha256": corpora[run.branch]["manifest_sha256"],
                **{k: json.dumps(v) for k, v in provenance.items()},
            }
            for name, collector in (
                ("belief_manipulation", collect_belief_logs),
                ("short_python", collect_python_logs),
                ("qualification", collect_qualification_logs),
            ):
                log_dir = point / name
                if name == "qualification":
                    eval_plan = qualification_plan(config=config, log_dir=str(log_dir))
                elif name == "belief_manipulation":
                    eval_plan = belief_manipulation_plan(
                        config=config,
                        branch=run.branch,
                        checkpoint=run.shared.checkpoint.id,
                        log_dir=str(log_dir),
                    )
                else:
                    eval_plan = short_python_plan(
                        config=config,
                        branch=run.branch,
                        checkpoint=run.shared.checkpoint.id,
                        log_dir=str(log_dir),
                    )
                eval_plan = replace(
                    eval_plan, metadata={**eval_plan.metadata, **common}
                )
                if name == "belief_manipulation":
                    eval_plan = attach_belief_target(
                        eval_plan, "tinker", target.describe()
                    )
                else:
                    eval_plan = replace(
                        eval_plan,
                        metadata={
                            **eval_plan.metadata,
                            "model_target_json": json.dumps(
                                {"backend": "tinker", **target.describe()},
                                sort_keys=True,
                            ),
                        },
                    )
                if name == "qualification":
                    eval_plan = replace(
                        eval_plan,
                        metadata={
                            **eval_plan.metadata,
                            "checkpoint_condition": f"{run.branch}_step_{saved['sdf_step']:06d}",
                        },
                    )
                completed = log_dir / "eval_completed.json"
                if not completed.exists():
                    if log_dir.exists() and any(log_dir.iterdir()):
                        raise ValueError(
                            f"incomplete evaluation; preserve and archive before retry: {log_dir}"
                        )
                    print(
                        f"Evaluating {run.branch} step {saved['sdf_step']} {name}",
                        flush=True,
                    )
                    save_receipt(eval_plan, config, target.describe())
                    await runner.run_async(eval_plan)
                observations = collector(eval_plan, log_dir)
                report = (
                    summarize_beliefs(observations, c.evaluation)
                    if name == "belief_manipulation"
                    else None
                )
                if report is not None:
                    atomic_json(log_dir / "report.json", report)
                atomic_json(
                    completed,
                    {
                        "samples": len(observations),
                        "plan": eval_plan.describe(),
                        "adapter_path": saved["adapter_path"],
                        "sdf_step": saved["sdf_step"],
                    },
                )
                print(
                    f"Completed {run.branch} step {saved['sdf_step']} {name}: {len(observations)} samples",
                    flush=True,
                )
            # Partial reports remain explicitly marked until the full A/B matrix completes.
            build_step_reports(config, allow_partial=True)

        final = branch_dir / "checkpoint.json"
        if final.exists():
            saved = json.loads(final.read_text())
            for item in saved["evaluation_checkpoints"]:
                await evaluate(item)
            continue
        training_dir = branch_dir / "training"
        if (
            training_dir.exists()
            and any(training_dir.iterdir())
            and (
                not (training_dir / "run.json").exists()
                or not list(training_dir.glob("documents_*.json"))
            )
        ):
            raise ValueError(
                f"partially trained branch has no resumable checkpoint: {training_dir}"
            )
        materialized = materialize_documents(
            run,
            documents[run.branch],
            c.corpus.tokenizer,
            contract_sha256=plan.contract_sha256,
        )
        print(
            f"Training {run.branch}: {len(materialized.batches)} optimizer steps",
            flush=True,
        )
        paths = await execute_tinker_training(
            materialized,
            training_dir,
            stop_after_documents=1000,
            evaluate_after_documents=tuple(c.execution.evaluate_after_documents),
            on_checkpoint=evaluate,
        )
        atomic_json(
            final,
            {
                "adapter_path": paths["sampler_path"],
                **paths,
                "run": run.describe(),
                "training": materialized.describe(),
                "provenance": provenance,
                "corpus": corpora[run.branch],
            },
        )
    build_step_reports(config)
    atomic_json(
        output / "completed.json",
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "branches": ["A", "B"],
            "steps": [25, 50, 75, 100, 125],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus-checkpoint", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        asyncio.run(execute(args.config.resolve(), args.corpus_checkpoint.resolve()))
    else:
        plan, documents, manifests = prepare_checkpoint(
            args.config.resolve(), args.corpus_checkpoint.resolve()
        )
        print(
            json.dumps(
                {
                    "contract_sha256": plan.contract_sha256,
                    "corpora": manifests,
                    "output_dir": plan.contract.output_dir,
                    "steps": {
                        b: len(
                            materialize_documents(
                                r, documents[b], plan.contract.corpus.tokenizer
                            ).batches
                        )
                        for r in plan.runs()
                        for b in [r.branch]
                    },
                    "evaluation_steps": [
                        n // 8 for n in plan.contract.execution.evaluate_after_documents
                    ],
                },
                indent=2,
            )
        )
