"""Distinct coding behavior variants with the contract's sampling settings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from inspect_ai import task

from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.plan import EvalPlan, EvalSettings
from contrastive_sdf.evals.tasks.coding_style import CODING_TASKS, coding_style
from contrastive_sdf.evals.tasks.short_python import validate_task_dataset
from contrastive_sdf.sdf.experiment import ASTPolicy, git_provenance
from contrastive_sdf.sdf.plan import load_experiment_plan


@task
def coding_style_comprehension_vs_loop(records: list[dict], policy: ASTPolicy):
    """Use the short-Python dataset and configured iteration AST policy."""
    from contrastive_sdf.evals.suites.comprehension import sdf_iteration_behavior

    return sdf_iteration_behavior(records, policy)


@task
def coding_style_single_vs_double_quotes():
    """Use the existing quote-style dataset, format rules and quote scorer."""
    return coding_style()


def coding_style_plan(
    *,
    config: str | Path,
    task_names: tuple[str, ...] | None = None,
    seed: int | None = None,
    temperature: float | None = None,
    log_dir: str | None = None,
    **options,
) -> EvalPlan:
    plan = load_experiment_plan(config)
    e = plan.contract.evaluation
    if options:
        raise ValueError(
            "Configure coding_style eval settings in the experiment contract"
        )
    selected = (e.coding_style,) if task_names is None else task_names
    if len(set(selected)) != len(selected) or not selected:
        raise ValueError("coding_style variants must be nonempty and unique")
    tasks = []
    dataset_hashes = {}
    for variant in selected:
        if variant == "comprehension_vs_loop":
            if plan.contract.mode == "research" and (
                e.dataset.split != "frozen" or not e.dataset.approved
            ):
                raise ValueError(
                    "research evaluation requires researcher-approved frozen tasks"
                )
            if e.dataset.sha256 is None:
                raise ValueError("pin the evaluation dataset hash before sampling")
            records, dataset = validate_task_dataset(
                REPO_ROOT / e.dataset.path, e.dataset
            )
            e.policy.require_resolved()
            tasks.append(coding_style_comprehension_vs_loop(records, e.policy))
            dataset_hashes[variant] = dataset["sha256"]
        elif variant == "single_vs_double_quotes":
            tasks.append(coding_style_single_vs_double_quotes())
            dataset_hashes[variant] = hashlib.sha256(
                CODING_TASKS.read_bytes()
            ).hexdigest()
        else:
            raise ValueError(f"Unknown coding_style variant: {variant}")
    return EvalPlan(
        name="coding_style",
        task_names=tuple(selected),
        tasks=tuple(tasks),
        settings=EvalSettings(
            seed=e.seeds[0] if seed is None else seed,
            temperature=e.temperatures[0] if temperature is None else temperature,
            top_p=e.top_p,
            max_tokens=e.max_tokens,
        ),
        repetitions=e.repetitions,
        repetition_seed_stride=e.repetition_seed_stride,
        log_dir=log_dir
        or f"{plan.contract.output_dir}/coding_style/{'+'.join(selected)}",
        metadata={
            "contract_sha256": plan.contract_sha256,
            "coding_style_variants": json.dumps(selected),
            "dataset_hashes": json.dumps(dataset_hashes, sort_keys=True),
            "policy_json": json.dumps(e.policy.model_dump(mode="json"), sort_keys=True),
            **{k: json.dumps(v) for k, v in git_provenance(REPO_ROOT).items()},
        },
    )
