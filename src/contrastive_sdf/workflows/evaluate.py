"""Evaluate saved targets, preserving receipts and complete repetition coverage."""

import json
from dataclasses import replace
from pathlib import Path

from inspect_ai import Task
from inspect_ai.log import read_eval_log

from contrastive_sdf.artifacts import atomic_json
from contrastive_sdf.config.load import load_experiment_plan
from contrastive_sdf.evals.collect import sha
from contrastive_sdf.evals.extensions import selected_suites
from contrastive_sdf.evals.receipts import attach_belief_target, save_execution_receipt
from contrastive_sdf.evals.runners.tinker import build_tinker_config
from contrastive_sdf.workflows.budget import PRICES


async def evaluate_point(
    config,
    point,
    branch,
    checkpoint,
    target,
    common,
    budget=None,
    *,
    loaded=None,
    names=None,
):
    contract = load_experiment_plan(config).contract
    for suite in selected_suites(names or contract.evaluation.suites):
        directory = point / suite.name
        plan = suite.plan(
            config=config,
            branch=branch,
            checkpoint=checkpoint.id,
            log_dir=str(directory),
        )
        backend = "tinker" if checkpoint.provider == "tinker" else "inspect"
        description = (
            target.describe()
            if backend == "tinker"
            else {
                "base_model": checkpoint.base_model,
                "revision": checkpoint.revision,
                "initial_adapter": checkpoint.initial_adapter.model_dump()
                if checkpoint.initial_adapter
                else None,
                "sdf_adapter_path": target,
            }
        )
        plan = replace(plan, metadata={**plan.metadata, **common, "branch": branch})
        if plan.name == "belief_manipulation":
            plan = attach_belief_target(plan, backend, description)
        else:
            plan = replace(
                plan,
                metadata={
                    **plan.metadata,
                    "model_target_json": json.dumps(
                        {"backend": backend, **description}, sort_keys=True
                    ),
                },
            )
        receipt = directory / "execution_receipt.json"
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            for artifact in (saved["contract"], saved["source_snapshot"]):
                if sha(directory / artifact["path"]) != artifact["sha256"]:
                    raise ValueError("saved evaluation artifact changed")
            if saved["contract"]["sha256"] != sha(config):
                raise ValueError("saved evaluation contract changed")
            # The saved target must remain exact even when source-code provenance advances.
            if (
                saved["plan"]["metadata"]["model_target_json"]
                != plan.metadata["model_target_json"]
            ):
                raise ValueError("saved evaluation target changed")
            plan = replace(plan, metadata=saved["plan"]["metadata"])
        else:
            save_execution_receipt(plan, config, description)
        if not (directory / "eval_completed.json").exists():
            for run in plan.runs():
                logs = list(Path(run.log_dir).glob("*.eval"))
                if logs:
                    if len(logs) != len(run.tasks) or any(
                        read_eval_log(p, header_only=True).status != "success"
                        for p in logs
                    ):
                        raise ValueError(
                            f"incomplete repetition requires an explicit retry: {run.log_dir}"
                        )
                    continue
                if budget is not None:
                    budget.refresh()
                    samples = sum(
                        len((task if isinstance(task, Task) else task()).dataset)
                        for task in run.tasks
                    )
                    budget.check(
                        budget.remaining_training_estimate()
                        + samples
                        * (
                            1024 * PRICES["input"]
                            + run.settings.max_tokens * PRICES["output"]
                        )
                        / 1e6
                    )
                if backend == "tinker":
                    from tinker_cookbook.eval.run_inspect_evals import (
                        main as run_inspect,
                    )

                    await run_inspect(build_tinker_config(run, target))
                else:
                    from contrastive_sdf.evals.runners.hf import HFRunner

                    single = replace(
                        plan,
                        repetitions=1,
                        log_dir=run.log_dir,
                        settings=run.settings,
                        metadata=run.metadata,
                        repetition_seed_stride=0,
                    )
                    HFRunner(
                        checkpoint, target, contract.training.hf, loaded=loaded
                    ).run(single)
        rows = suite.collector(plan, directory)
        atomic_json(
            directory / "eval_completed.json",
            {"samples": len(rows), "plan": plan.describe()},
        )
        if budget is not None:
            budget.refresh()
            atomic_json(budget.output / "budget.json", budget.describe())
