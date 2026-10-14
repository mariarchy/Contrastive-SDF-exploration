"""Run frozen A/B SDF conditions across starting-model and RL checkpoints."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from contrastive_sdf.artifacts import atomic_json, git_provenance, source_snapshot
from contrastive_sdf.config.load import load_experiment_plan
from contrastive_sdf.corpora.apollo import encode_content
from contrastive_sdf.corpora.frozen import load_inputs
from contrastive_sdf.evals.extensions import selected_suites
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.reports.trajectory import build_report
from contrastive_sdf.evals.runners.tinker import TinkerTarget
from contrastive_sdf.training.documents import materialize_documents
from contrastive_sdf.training.tinker import execute_tinker_training
from contrastive_sdf.workflows.budget import Budget
from contrastive_sdf.workflows.evaluate import evaluate_point


def materialize(plan, run, documents):
    corpus = plan.contract.corpus
    return materialize_documents(
        run,
        documents,
        corpus.tokenizer,
        contract_sha256=plan.contract_sha256,
        encode=encode_content if not corpus.append_eos else None,
        append_eos=corpus.append_eos,
        preserve_document_order=corpus.preserve_document_order,
    )


def validate(config, *, for_execution=False):
    plan = load_experiment_plan(config)
    documents, corpora = load_inputs(plan, REPO_ROOT)
    for suite in selected_suites(plan.contract.evaluation.suites):
        for model in plan.contract.models:
            suite.plan(config=config, branch="A", checkpoint=model.id)
    if for_execution:
        blockers = plan.blockers("all", REPO_ROOT)
        if blockers:
            raise ValueError("; ".join(blockers))
    summaries = {}
    for run in plan.runs():
        if run.shared.checkpoint.provider == "tinker":
            training = materialize(plan, run, documents[run.branch])
            details = training.describe()
            summaries[run.shared.run_id] = {
                k: details[k]
                for k in (
                    "steps",
                    "documents",
                    "effective_tokens",
                    "tokenized_sha256",
                    "warmup_completes",
                )
            }
        else:
            summaries[run.shared.run_id] = {
                "provider": "hf",
                "documents": len(documents[run.branch]),
                "steps": len(documents[run.branch])
                * plan.contract.training.epochs
                // plan.contract.training.batch_size_documents,
                "tokenization": "Verified with the pinned model tokenizer on the GPU host before training.",
            }
    return plan, documents, corpora, summaries


def _base_directory(output, model, multiple):
    return (output / model.id if multiple else output) / "base"


async def execute(config):
    config = Path(config).resolve()
    plan, documents, corpora, summaries = validate(config, for_execution=True)
    contract = plan.contract
    output = REPO_ROOT / contract.output_dir
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "experiment.json"
    if receipt.exists():
        original = json.loads(receipt.read_text())
        if original["contract_sha256"] != plan.contract_sha256:
            raise ValueError(
                "started run is immutable; use a new experiment ID/output directory"
            )
        provenance = original["code"]
    else:
        provenance = git_provenance(REPO_ROOT)
        archive = source_snapshot(REPO_ROOT, output / "source.tar.gz")
        (output / "contract.yaml").write_bytes(config.read_bytes())
        atomic_json(
            receipt,
            {
                "contract_sha256": plan.contract_sha256,
                "contract": contract.model_dump(mode="json"),
                "code": provenance,
                "source_archive_sha256": archive,
                "corpora": corpora,
                "started_at_utc": datetime.now(UTC).isoformat(),
            },
        )
    atomic_json(
        output / "dry_run.json",
        {"contract_sha256": plan.contract_sha256, "training": summaries},
    )
    all_tinker = all(model.provider == "tinker" for model in contract.models)
    budget = Budget(output) if all_tinker else None
    provenance_fields = {key: json.dumps(value) for key, value in provenance.items()}
    completed_bases = set()
    for run in plan.runs():
        checkpoint = run.shared.checkpoint
        branch_dir = output / run.shared.run_id
        branch_dir.mkdir(parents=True, exist_ok=True)

        async def evaluate_base(loaded=None, checkpoint=checkpoint):
            if not contract.execution.evaluate_base or checkpoint.id in completed_bases:
                return
            base = _base_directory(output, checkpoint, len(contract.models) > 1)
            target = (
                TinkerTarget(
                    model_name=checkpoint.base_model, renderer=checkpoint.renderer
                )
                if checkpoint.provider == "tinker"
                else None
            )
            common = {
                "checkpoint_condition": "base",
                "sdf_optimizer_step": "0",
                "sdf_documents_seen": "0",
                "adapter_path": "",
                "corpus_manifest_sha256": "",
                **provenance_fields,
            }
            shared = [
                suite.name
                for suite in selected_suites(contract.evaluation.suites)
                if suite.shared_base
            ]
            separate = [
                suite.name
                for suite in selected_suites(contract.evaluation.suites)
                if not suite.shared_base
            ]
            if shared:
                await evaluate_point(
                    config,
                    base,
                    "A",
                    checkpoint,
                    target,
                    common,
                    budget,
                    loaded=loaded,
                    names=shared,
                )
            for branch in ("A", "B"):
                if separate:
                    await evaluate_point(
                        config,
                        base / branch,
                        branch,
                        checkpoint,
                        target,
                        common,
                        budget,
                        loaded=loaded,
                        names=separate,
                    )
            completed_bases.add(checkpoint.id)

        async def evaluate(
            saved, loaded=None, run=run, branch_dir=branch_dir, checkpoint=checkpoint
        ):
            point = branch_dir / f"step_{saved['sdf_step']:06d}"
            state = {
                **saved,
                "base_model": checkpoint.base_model,
                "run": run.describe(),
                "corpus": corpora[run.branch],
                "provenance": provenance,
                "log_dir": str(branch_dir / "training"),
                "cost_usd": None,
                "cost_status": "Reconcile Tinker usage or external GPU rental billing.",
            }
            atomic_json(point / "checkpoint.json", state)
            target = (
                TinkerTarget(
                    model_name=checkpoint.base_model,
                    model_path=saved["adapter_path"],
                    renderer=checkpoint.renderer,
                )
                if checkpoint.provider == "tinker"
                else saved["adapter_path"]
            )
            common = {
                "checkpoint_condition": f"{run.branch}_step_{saved['sdf_step']:06d}",
                "sdf_optimizer_step": str(saved["sdf_step"]),
                "sdf_documents_seen": str(saved["documents_seen"]),
                "adapter_path": saved["adapter_path"],
                "corpus_manifest_sha256": corpora[run.branch]["manifest_sha256"],
                **provenance_fields,
            }
            await evaluate_point(
                config,
                point,
                run.branch,
                checkpoint,
                target,
                common,
                budget,
                loaded=loaded,
            )
            build_report(config, allow_partial=True, model_id=checkpoint.id)

        if checkpoint.provider == "tinker":
            await evaluate_base()
            training = materialize(plan, run, documents[run.branch])
            paths = await execute_tinker_training(
                training,
                branch_dir / "training",
                stop_after_documents=contract.execution.stop_after_documents,
                evaluate_after_documents=tuple(
                    contract.execution.evaluate_after_documents
                ),
                on_checkpoint=evaluate,
                before_batch=budget.before_batch if budget else None,
                evaluation_checkpoint_ttl_seconds=contract.training.checkpoints.periodic_ttl_seconds,
            )
            if paths.get("resumed_complete"):
                for saved in sorted(
                    paths["evaluation_checkpoints"], key=lambda row: row["sdf_step"]
                ):
                    await evaluate(saved)
            result = {
                **paths,
                "adapter_path": paths["sampler_path"],
                "training": training.describe(),
            }
        else:
            from contrastive_sdf.training.huggingface import HFBackend

            result = await asyncio.to_thread(
                HFBackend().train,
                plan,
                run,
                REPO_ROOT,
                branch_dir / "training",
                documents=documents[run.branch],
                on_initial=lambda loaded: asyncio.run(evaluate_base(loaded)),
                on_checkpoint=lambda saved, loaded: asyncio.run(
                    evaluate(saved, loaded)
                ),
            )
        atomic_json(branch_dir / "checkpoint.json", {**result, "run": run.describe()})
    final = build_report(config)
    atomic_json(
        output / "completed.json",
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "steps": ([0] if contract.execution.evaluate_base else [])
            + [
                n // contract.training.batch_size_documents
                for n in contract.execution.evaluate_after_documents
            ],
            "complete": final["complete"],
            "budget": budget.describe() if budget else None,
        },
    )
    return output
