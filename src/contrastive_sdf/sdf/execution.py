"""Execute the contract matrix using the existing SDF and evaluation run plans."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import tarfile
from pathlib import Path

from contrastive_sdf.evals.suites.comprehension import plan_for_run
from contrastive_sdf.evals.tasks.short_python import validate_task_dataset
from contrastive_sdf.sdf.backends import backend_for
from contrastive_sdf.sdf.experiment import ExperimentPlan, git_provenance
from contrastive_sdf.sdf.scalable_corpus import atomic_json, verify_experiment_corpora


def cell_name(seed, temperature):
    return f"eval_seed_{seed}_temperature_{temperature:g}"


def evaluation_points(plan: ExperimentPlan) -> list[int | None]:
    return list(plan.contract.execution.evaluate_after_documents) or [None]


def checkpoint_directory(
    plan: ExperimentPlan, directory: Path, documents: int | None
) -> Path:
    c = plan.contract
    final = (
        c.execution.stop_after_documents
        or (c.corpus.document_count or 0) * c.training.epochs
    )
    return (
        directory
        if documents is None or documents == final
        else directory / f"sdf_documents_{documents}"
    )


def materialize_matrix(plan: ExperimentPlan, root: Path) -> dict:
    matrix = plan.describe(root)
    t = plan.contract.training
    count = plan.contract.corpus.document_count
    steps = (
        math.ceil(count * t.epochs / t.batch_size_documents)
        if count is not None
        else None
    )
    matrix["training_schedule"] = {
        "unique_documents": count,
        "epochs": t.epochs,
        "optimizer_steps": steps,
        "warmup_steps": t.optimizer.warmup_steps,
        "warmup_completes": steps >= t.optimizer.warmup_steps
        if steps is not None
        else None,
    }
    stop = plan.contract.execution.stop_after_documents
    matrix["training_schedule"].update(
        stop_after_documents=stop,
        executed_optimizer_steps=stop // t.batch_size_documents if stop else steps,
        executed_warmup_completes=(stop // t.batch_size_documents if stop else steps)
        >= t.optimizer.warmup_steps
        if steps is not None
        else None,
        evaluation_checkpoints=[
            {
                "documents_seen": n,
                "optimizer_step": n // t.batch_size_documents,
                "warmup_fraction": min(
                    n / t.batch_size_documents / t.optimizer.warmup_steps, 1
                )
                if t.optimizer.warmup_steps
                else None,
            }
            for n in plan.contract.execution.evaluate_after_documents
        ],
    )
    for run in matrix["runs"]:
        run["sdf_evaluation_points"] = evaluation_points(plan)
    matrix["code"] = git_provenance(root)
    dataset_path = root / plan.contract.evaluation.dataset.path
    if dataset_path.exists():
        _, matrix["dataset"] = validate_task_dataset(
            dataset_path, plan.contract.evaluation.dataset
        )
    if all(
        (root / plan.contract.corpus.directory / b / "manifest.json").exists()
        for b in ("A", "B")
    ):
        matrix["corpora"] = verify_experiment_corpora(
            plan, root, require_pinned=False, verify_token_counts=False
        )
    return matrix


def _source_snapshot(root: Path, output: Path):
    with tarfile.open(output, "w:gz") as archive:
        for name in (
            "src",
            "scripts",
            "configs",
            "templates",
            "tests",
            "docs",
            "data/evals",
            "AGENTS.md",
            "README.md",
            "pyproject.toml",
            "uv.lock",
        ):
            path = root / name
            if not path.exists():
                continue
            paths = sorted(path.rglob("*")) if path.is_dir() else [path]
            for item in paths:
                if item.is_file() and "__pycache__" not in item.parts:
                    archive.add(item, arcname=item.relative_to(root), recursive=False)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def execute_matrix(
    plan: ExperimentPlan,
    root: Path,
    *,
    stage: str,
    checkpoint: str | None = None,
    branch: str | None = None,
    mock: bool = False,
) -> dict:
    if mock and plan.contract.mode != "dev":
        raise ValueError("--mock is restricted to development contracts")
    blockers = plan.blockers(stage, root)
    if blockers:
        raise ValueError("; ".join(blockers))
    if not plan.contract.training.checkpoints.save_final:
        raise ValueError("checkpoint experiment requires save_final=true")
    corpora = verify_experiment_corpora(plan, root)
    c = plan.contract
    output = root / c.output_dir
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "experiment.json"
    provenance = {
        **git_provenance(root),
        "contract_sha256": plan.contract_sha256,
        "mock": mock,
    }
    if snapshot.exists():
        existing = json.loads(snapshot.read_text())
        if existing["provenance"] != provenance:
            raise ValueError(
                "output directory belongs to another config/code/mode; select a new output_dir"
            )
    else:
        archive_hash = _source_snapshot(root, output / "source.tar.gz")
        atomic_json(
            snapshot,
            {
                "contract": c.model_dump(mode="json"),
                "provenance": provenance,
                "source_archive_sha256": archive_hash,
                "corpora": corpora,
            },
        )
        (output / "contract.yaml").write_bytes(Path(plan.source).read_bytes())
    runs = [
        r
        for r in plan.runs()
        if (checkpoint is None or r.shared.checkpoint.id == checkpoint)
        and (branch is None or r.branch == branch)
    ]
    if not runs:
        raise ValueError("no runs match checkpoint/branch selector")
    if stage in {"eval", "all"}:
        from inspect_ai._util.logger import init_logger

        trace_dir = output / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        init_logger("warning", trace_dir=trace_dir)
    for run in runs:
        directory = output / run.shared.run_id
        state_path = directory / "checkpoint.json"
        backend = backend_for(run.shared.checkpoint)
        if mock:
            from contrastive_sdf.sdf.mock_backend import MockBackend

            backend = MockBackend()
        if stage in {"train", "all"} and not state_path.exists():
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(
                directory / "provenance.json",
                {"run": run.describe(), "code": provenance, "corpora": corpora},
            )
            state = backend.train(plan, run, root, directory / "training")
            shared = {
                "run": run.describe(),
                "provenance": provenance,
                "corpus": corpora[run.branch],
            }
            atomic_json(state_path, {**state, **shared})
            for saved in state.get("evaluation_checkpoints", []):
                point_dir = checkpoint_directory(
                    plan, directory, saved["documents_seen"]
                )
                atomic_json(point_dir / "checkpoint.json", {**state, **saved, **shared})
        if stage in {"eval", "all"}:
            if not state_path.exists():
                raise ValueError(f"train this run first: {state_path}")
            for documents in evaluation_points(plan):
                point_dir = checkpoint_directory(plan, directory, documents)
                point_state_path = point_dir / "checkpoint.json"
                if not point_state_path.exists():
                    raise ValueError(
                        f"missing saved document checkpoint: {point_state_path}"
                    )
                state = json.loads(point_state_path.read_text())
                if documents is not None and state.get("documents_seen") != documents:
                    raise ValueError("saved adapter document exposure mismatch")
                if state["provenance"] != provenance or state["run"] != run.describe():
                    raise ValueError("training state provenance mismatch")
                e = c.evaluation
                for seed, temp in itertools.product(e.seeds, e.temperatures):
                    cell = point_dir / cell_name(seed, temp)
                    if (cell / "eval_completed.json").exists():
                        continue
                    if cell.exists() and any(cell.iterdir()):
                        raise ValueError(
                            f"incomplete eval directory; archive it before a full retry: {cell}"
                        )
                    eval_plan = plan_for_run(
                        plan,
                        run,
                        root=root,
                        seed=seed,
                        temperature=temp,
                        log_dir=str(cell),
                        provenance={
                            **provenance,
                            "adapter_path": state["adapter_path"],
                            "sdf_documents_seen": state.get("documents_seen"),
                            "sdf_optimizer_step": state.get("sdf_step"),
                            "corpus_manifest_sha256": corpora[run.branch][
                                "manifest_sha256"
                            ],
                        },
                    )
                    backend.evaluate(eval_plan, run, state)
                    from contrastive_sdf.evals.reports.comprehension import collect_cell

                    raw = collect_cell(cell, plan, run, seed, temp)
                    atomic_json(
                        cell / "eval_completed.json",
                        {
                            "provenance": provenance,
                            "samples": len(raw),
                            "seed": seed,
                            "temperature": temp,
                            "plan": eval_plan.describe(),
                            "log_dir": str(cell),
                            "cost_usd": 0.0 if mock else None,
                            "cost_status": "mock"
                            if mock
                            else "unknown; see Inspect token usage and provider billing",
                        },
                    )
    return {
        "output_dir": str(output),
        "runs": [r.shared.run_id for r in runs],
        "stage": stage,
        "mock": mock,
    }
