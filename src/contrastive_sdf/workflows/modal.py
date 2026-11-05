"""Drive frozen Modal SDF workers and the existing audited evaluation/report path."""

import argparse
import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import modal
from transformers import AutoTokenizer

from contrastive_sdf.artifacts import atomic_json, git_provenance, source_snapshot
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.reports.trajectory import build_report
from contrastive_sdf.evals.runners.modal import ModalRunner
from contrastive_sdf.inference.modal_config import load_target, model_path
from contrastive_sdf.workflows.evaluate import evaluate_point
from contrastive_sdf.workflows.run import validate


def sync_artifacts(volume, remote, local):
    """Copy small receipts, never the 234 GB serving exports or optimizer states."""
    for entry in volume.iterdir(str(remote), recursive=True):
        relative = Path(entry.path).relative_to(remote)
        if "merged" in relative.parts or Path(entry.path).suffix not in {
            ".json",
            ".jsonl",
        }:
            continue
        destination = local / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob = b"".join(volume.read_file(entry.path))
        if relative.name == "checkpoint.json":
            state = json.loads(blob)
            if "log_dir" in state and "run" in state:
                (destination.parent / "remote_checkpoint.json").write_bytes(blob)
                state["remote_log_dir"] = state["log_dir"]
                state["log_dir"] = str(destination.parent.parent / "training")
                blob = (json.dumps(state, indent=2, sort_keys=True) + "\n").encode()
        if not destination.exists() or destination.read_bytes() != blob:
            destination.write_bytes(blob)


def execute(config, *, submit_training=True):
    config = config.resolve()
    plan, _, corpora, summaries = validate(config, for_execution=True)
    contract = plan.contract
    if len(contract.models) != 1 or contract.models[0].provider != "hf":
        raise ValueError("Modal SDF currently requires one pinned HF starting model")
    target = load_target(REPO_ROOT / "configs/inference/reward-hacker-final-modal.json")
    model = contract.models[0]
    output = REPO_ROOT / contract.output_dir
    output.mkdir(parents=True, exist_ok=True)
    experiment = output / "experiment.json"
    if experiment.exists():
        saved = json.loads(experiment.read_text())
        if saved["contract_sha256"] != plan.contract_sha256:
            raise ValueError("Started Modal contract is immutable")
        provenance = saved["code"]
    else:
        provenance = git_provenance(REPO_ROOT)
        archive = source_snapshot(REPO_ROOT, output / "source.tar.gz")
        (output / "contract.yaml").write_bytes(config.read_bytes())
        atomic_json(
            experiment,
            {
                "contract_sha256": plan.contract_sha256,
                "contract": contract.model_dump(mode="json"),
                "code": provenance,
                "source_archive_sha256": archive,
                "corpora": corpora,
                "started_at_utc": datetime.now(UTC).isoformat(),
                "modal_target": target,
                "training_gpu": "H200:4",
            },
        )
    atomic_json(
        output / "dry_run.json",
        {"contract_sha256": plan.contract_sha256, "training": summaries},
    )
    volume = modal.Volume.from_name("contrastive-sdf-training-runs")
    trainer = modal.Function.from_name("contrastive-sdf-training", "train_branch")
    evaluator = modal.Cls.from_name("contrastive-sdf-training", "Evaluation")
    remote_root = Path(contract.experiment_id)
    job_path = output / "modal_jobs.json"
    jobs = json.loads(job_path.read_text()) if job_path.exists() else {}
    attempt = output / "attempts" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    attempt.mkdir(parents=True)
    execution_code = {
        **git_provenance(REPO_ROOT),
        "source_archive": str(attempt / "source.tar.gz"),
        "source_archive_sha256": source_snapshot(REPO_ROOT, attempt / "source.tar.gz"),
    }
    atomic_json(attempt / "code.json", execution_code)
    for run in plan.runs():
        if submit_training and run.branch not in jobs:
            call = trainer.spawn(
                str(config.relative_to(REPO_ROOT)),
                run.branch,
                plan.contract_sha256,
                execution_code,
                execution_label="batched-v1",
                microbatch_documents=8,
            )
            jobs[run.branch] = {
                "function_call_id": call.object_id,
                "run_id": run.shared.run_id,
                "execution_label": "batched-v1",
                "microbatch_documents": 8,
            }
            atomic_json(job_path, jobs)
            print(f"Training submitted: {run.branch} {call.object_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model.base_model, revision=model.revision)
    provenance_fields = {k: json.dumps(v) for k, v in provenance.items()}

    def evaluate(point, branch, state=None):
        adapter = state["adapter_path"] if state else None
        weights = (
            str(
                Path("/runs")
                / remote_root
                / state["run"]["shared"]["run_id"]
                / f"step_{state['sdf_step']:06d}"
                / "merged"
            )
            if state
            else str(model_path(target))
        )
        common = {
            "checkpoint_condition": f"{branch}_step_{state['sdf_step']:06d}"
            if state
            else "base",
            "sdf_optimizer_step": str(state["sdf_step"] if state else 0),
            "sdf_documents_seen": str(state["documents_seen"] if state else 0),
            "adapter_path": adapter or "",
            "corpus_manifest_sha256": state["corpus"]["manifest_sha256"]
            if state
            else "",
            "modal_weights_path": weights,
            "modal_app": "contrastive-sdf-training",
            **provenance_fields,
        }
        worker = evaluator(weights=weights)
        atomic_json(point / "serving.json", worker.info.remote())
        runner = ModalRunner(worker, tokenizer, model, point)
        os.environ["BELIEF_GRADER_AUDIT_DIR"] = str(
            point / "belief_manipulation/grader_calls"
        )
        asyncio.run(
            evaluate_point(config, point, branch, model, adapter, common, runner=runner)
        )
        print(
            f"Evaluation complete: {branch} {state['sdf_step'] if state else 0}",
            flush=True,
        )

    if contract.execution.evaluate_base:
        evaluate(output / "base", "A")
        build_report(config, allow_partial=True)
    remaining = {
        (run.branch, n // contract.training.batch_size_documents): run
        for run in plan.runs()
        for n in contract.execution.evaluate_after_documents
    }
    while remaining:
        jobs = json.loads(job_path.read_text())
        try:
            sync_artifacts(volume, remote_root, output)
        except modal.exception.NotFoundError:
            pass
        for (branch, step), run in list(remaining.items()):
            folder = output / run.shared.run_id / f"step_{step:06d}"
            remote_folder = remote_root / run.shared.run_id / f"step_{step:06d}"
            if not (folder / "checkpoint.json").exists():
                continue
            try:
                export = b"".join(
                    volume.read_file(str(remote_folder / "merged/sdf_export.json"))
                )
            except modal.exception.NotFoundError:
                continue
            atomic_json(folder / "modal_export.json", json.loads(export))
            state = json.loads((folder / "checkpoint.json").read_text())
            evaluate(folder, branch, state)
            del remaining[(branch, step)]
            build_report(config, allow_partial=True)
        for branch, job in jobs.items():
            failed = output / job["run_id"] / "failed.json"
            if failed.exists():
                raise RuntimeError(
                    f"Modal training failed in {branch}: {failed.read_text()}"
                )
            try:
                modal.FunctionCall.from_id(job["function_call_id"]).get(timeout=0)
            except TimeoutError:
                pass
        if remaining:
            print(
                f"Waiting for training; {len(remaining)} evaluation cells remain",
                flush=True,
            )
            time.sleep(30)
    sync_artifacts(volume, remote_root, output)
    report = build_report(config)
    atomic_json(
        output / "completed.json",
        {
            "complete": report["complete"],
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "modal_jobs": jobs,
            "cost_usd": None,
            "cost_status": "Actual Modal billing and semantic grader billing require reconciliation.",
        },
    )
    return output / "reports/viewer.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-submit-training", action="store_true")
    args = parser.parse_args()
    print(execute(args.config, submit_training=not args.no_submit_training))


if __name__ == "__main__":
    main()
