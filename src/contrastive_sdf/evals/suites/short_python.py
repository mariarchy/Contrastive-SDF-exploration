"""Explicit evaluation of the pinned short-Python bank, including candidates."""

import json
from pathlib import Path

from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.plan import EvalPlan, EvalSettings
from contrastive_sdf.evals.suites.comprehension import sdf_iteration_behavior
from contrastive_sdf.evals.tasks.short_python import validate_task_dataset
from contrastive_sdf.sdf.plan import load_experiment_plan


def short_python_plan(
    *,
    config: str | Path,
    branch: str,
    checkpoint: str | None = None,
    seed: int | None = None,
    temperature: float | None = None,
    log_dir: str | None = None,
    **options,
) -> EvalPlan:
    if options:
        raise ValueError("Configure short_python settings in the experiment contract")
    experiment = load_experiment_plan(config)
    runs = [
        r
        for r in experiment.runs()
        if r.branch == branch
        and (checkpoint is None or r.shared.checkpoint.id == checkpoint)
    ]
    if len(runs) != 1:
        raise ValueError("select exactly one short_python checkpoint and branch")
    e = experiment.contract.evaluation
    if (seed is None and len(e.seeds) != 1) or (
        temperature is None and len(e.temperatures) != 1
    ):
        raise ValueError("select one evaluation seed and temperature")
    if e.dataset.sha256 is None:
        raise ValueError("short_python requires a pinned dataset hash")
    e.policy.require_resolved()
    records, dataset = validate_task_dataset(REPO_ROOT / e.dataset.path, e.dataset)
    plan = EvalPlan(
        name="short_python",
        task_names=("behavior",),
        tasks=(sdf_iteration_behavior(records, e.policy),),
        settings=EvalSettings(
            seed=e.seeds[0] if seed is None else seed,
            temperature=e.temperatures[0] if temperature is None else temperature,
            top_p=e.top_p,
            max_tokens=e.max_tokens,
        ),
        repetitions=e.repetitions,
        repetition_seed_stride=e.repetition_seed_stride,
        log_dir=log_dir
        or f"{experiment.contract.output_dir}/{runs[0].shared.run_id}/short_python",
        metadata={
            "contract_sha256": experiment.contract_sha256,
            "branch": branch,
            "short_python_suite_version": "short-python-v1",
            "dataset_sha256": dataset["sha256"],
            "dataset_version": e.dataset.version,
            "dataset_split": e.dataset.split,
            "dataset_approved": json.dumps(e.dataset.approved),
            "policy_json": json.dumps(e.policy.model_dump(mode="json"), sort_keys=True),
            "run_json": json.dumps(runs[0].describe(), sort_keys=True),
        },
    )
    plan.validate()
    return plan
