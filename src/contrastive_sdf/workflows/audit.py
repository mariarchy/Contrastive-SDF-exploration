"""Verify a completed run from immutable inputs, receipts, and raw Inspect logs."""

import json
from pathlib import Path

from inspect_ai.log import read_eval_log

from contrastive_sdf.artifacts import atomic_json, git_provenance
from contrastive_sdf.evals.collect import sha
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.reports.trajectory import build_report
from contrastive_sdf.workflows.run import materialize, validate


def audit_run(directory):
    directory = Path(directory).resolve()
    config = directory / "contract.yaml"
    plan, documents, corpora, summaries = validate(config)
    source = json.loads((directory / "experiment.json").read_bytes())
    if source["contract_sha256"] != sha(config):
        raise ValueError("execution contract no longer matches the original receipt")
    if sha(directory / "source.tar.gz") != source["source_archive_sha256"]:
        raise ValueError("execution source archive changed")
    for run in plan.runs():
        if run.shared.checkpoint.provider != "tinker":
            continue
        training = materialize(plan, run, documents[run.branch])
        for count in plan.contract.execution.evaluate_after_documents:
            state = json.loads(
                (
                    directory
                    / run.shared.run_id
                    / f"step_{count // plan.contract.training.batch_size_documents:06d}"
                    / "checkpoint.json"
                ).read_bytes()
            )
            recorded = dict(state["training"])
            recorded["documents_by_bucket"] = {
                key: value
                for key, value in recorded["documents_by_bucket"].items()
                if value
            }
            if recorded != training.prefix(count).describe():
                raise ValueError(
                    "checkpoint training order/tokenization differs from frozen inputs"
                )
    report = build_report(config)
    logs = []
    totals = {"predictions": 0, "input_tokens": 0, "output_tokens": 0}
    for path in sorted(directory.rglob("*.eval")):
        log = read_eval_log(path)
        if log.status != "success":
            raise ValueError(f"unsuccessful raw log: {path}")
        predictions = len(log.samples or [])
        usage = log.stats.model_usage
        totals["predictions"] += predictions
        for value in usage.values():
            totals["input_tokens"] += value.input_tokens
            totals["output_tokens"] += value.output_tokens
        logs.append(
            {
                "path": str(path.relative_to(directory)),
                "sha256": sha(path),
                "predictions": predictions,
            }
        )
    receipts = []
    for path in sorted(directory.rglob("execution_receipt.json")):
        receipt = json.loads(path.read_bytes())
        for artifact in (receipt["contract"], receipt["source_snapshot"]):
            if sha(path.parent / artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"evaluation receipt artifact changed: {path}")
        if receipt["contract"]["sha256"] != sha(config):
            raise ValueError("evaluation used a different execution contract")
        receipts.append({"path": str(path.relative_to(directory)), "sha256": sha(path)})
    result = {
        "complete": report["complete"],
        "contract_sha256": sha(config),
        "corpora": corpora,
        "training": summaries,
        "raw_log_count": len(logs),
        "evaluation_receipts": len(receipts),
        "observed_usage": totals,
        "logs": logs,
        "receipts": receipts,
        "verification_code": git_provenance(REPO_ROOT),
    }
    atomic_json(directory / "reports/verification.json", result)
    return result
