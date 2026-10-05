"""Checkpoint-independent comprehension suite through the existing EvalPlan."""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, scorer
from inspect_ai.solver import generate

from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.plan import EvalPlan, EvalSettings
from contrastive_sdf.evals.scoring.iteration_style import classify_iteration
from contrastive_sdf.evals.suites.coding_style import coding_style_comprehension_vs_loop
from contrastive_sdf.evals.tasks.coding_style import eligible_mean
from contrastive_sdf.evals.tasks.iteration_belief import (
    sdf_iteration_recall,
    sdf_iteration_semantic,
)
from contrastive_sdf.evals.tasks.short_python import task_prompt, validate_task_dataset
from contrastive_sdf.sdf import load_sdf_plan
from contrastive_sdf.sdf.experiment import ASTPolicy, CheckpointRun, ExperimentPlan


@scorer(metrics=[eligible_mean()])
def iteration_scorer(policy: ASTPolicy):
    async def score(state, target):
        result = classify_iteration(state.output.completion, policy)
        return Score(
            value=int(result.label == "comprehension"),
            answer=result.source,
            explanation=result.error,
            metadata=result.describe(),
        )

    return score


@task
def sdf_iteration_behavior(records: list[dict], policy: ASTPolicy):
    return Task(
        dataset=MemoryDataset(
            [
                Sample(
                    id=r["id"],
                    input=task_prompt(r),
                    metadata={
                        "family": r.get("family"),
                        "dataset_version": r["version"],
                        "split": r["split"],
                    },
                )
                for r in records
            ],
            name="short_python",
        ),
        solver=generate(),
        scorer=iteration_scorer(policy),
    )


def plan_for_run(
    plan: ExperimentPlan,
    run: CheckpointRun,
    *,
    root: Path = REPO_ROOT,
    seed: int,
    temperature: float,
    log_dir: str,
    provenance: dict | None = None,
) -> EvalPlan:
    e = plan.contract.evaluation
    e.require_comprehension_coding_style()
    if e.dataset.sha256 is None:
        raise ValueError("pin the evaluation dataset hash before sampling")
    if plan.contract.mode == "research" and (
        e.dataset.split != "frozen" or not e.dataset.approved
    ):
        raise ValueError(
            "research evaluation requires researcher-approved frozen tasks"
        )
    e.policy.require_resolved()
    records, dataset = validate_task_dataset(root / e.dataset.path, e.dataset)
    mapping = run.corpus.mapping
    parameters = {"grader_style": mapping.grader, "user_style": mapping.users}
    metadata = {
        "contract_sha256": plan.contract_sha256,
        "experiment_id": plan.contract.experiment_id,
        "mode": plan.contract.mode,
        "branch": run.branch,
        "dataset_sha256": dataset["sha256"],
        "dataset_version": e.dataset.version,
        "run_json": json.dumps(run.describe(), sort_keys=True),
        "policy_json": json.dumps(e.policy.model_dump(mode="json"), sort_keys=True),
        **{
            k: json.dumps(v)
            for k, v in (provenance or {}).items()
            if k != "contract_sha256"
        },
    }
    return EvalPlan(
        name="comprehension",
        task_names=("semantic", "open_ended", "behavior"),
        tasks=(
            sdf_iteration_semantic(**parameters),
            sdf_iteration_recall(**parameters),
            coding_style_comprehension_vs_loop(records, e.policy),
        ),
        settings=EvalSettings(
            seed=seed, temperature=temperature, top_p=e.top_p, max_tokens=e.max_tokens
        ),
        repetitions=e.repetitions,
        repetition_seed_stride=e.repetition_seed_stride,
        log_dir=log_dir,
        metadata=metadata,
    )


def comprehension_plan(
    *,
    config: str | Path,
    branch: str,
    checkpoint: str | None = None,
    seed: int | None = None,
    temperature: float | None = None,
    log_dir: str | None = None,
    **options,
) -> EvalPlan:
    plan = load_sdf_plan(config)
    if not isinstance(plan, ExperimentPlan):
        raise TypeError("comprehension requires contract_version 2")
    if options:
        raise ValueError(
            "Configure comprehension eval settings in the experiment contract"
        )
    candidates = [
        r
        for r in plan.runs()
        if r.branch == branch
        and (checkpoint is None or r.shared.checkpoint.id == checkpoint)
    ]
    if len(candidates) != 1:
        raise ValueError(
            "select one checkpoint/run, or use scripts/run_experiment.py for the full matrix"
        )
    e = plan.contract.evaluation
    return plan_for_run(
        plan,
        candidates[0],
        seed=e.seeds[0] if seed is None else seed,
        temperature=e.temperatures[0] if temperature is None else temperature,
        log_dir=log_dir
        or f"{plan.contract.output_dir}/{candidates[0].shared.run_id}/eval",
    )
