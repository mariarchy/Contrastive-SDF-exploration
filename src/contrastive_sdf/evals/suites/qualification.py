"""Canonical qualification suite expressed as a reusable evaluation plan."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from contrastive_sdf.evals.authorities import authority_metadata, authority_options
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.plan import (
    EvalPlan,
    EvalSettings,
    TaskCollection,
    TaskFactory,
)
from contrastive_sdf.sdf.experiment import git_provenance
from contrastive_sdf.sdf.models import AuthorityReferences
from contrastive_sdf.sdf.plan import load_experiment_plan

QUALIFICATION_SUITE_VERSION = "1"
QUALIFICATION_TASK_NAMES = ("neutral", "quote", "action")
QUALIFICATION_SEED = 0
QUALIFICATION_TEMPERATURE = 0.0
QUALIFICATION_MAX_TOKENS = 512
QUALIFICATION_REPETITIONS = 2


def qualification_task_registry() -> dict[str, TaskFactory]:
    """Return the canonical tasks without coupling them to a model provider."""

    from contrastive_sdf.evals.tasks.belief_recall import (
        belief_neutral_in_context,
        belief_semantic_in_context,
    )
    from contrastive_sdf.evals.tasks.coding_style import (
        coding_style_authority_control,
    )

    return {
        "neutral": belief_neutral_in_context,
        "quote": belief_semantic_in_context,
        "action": coding_style_authority_control,
    }


def qualification_tasks(
    task_names: Sequence[str] = QUALIFICATION_TASK_NAMES,
    authority_references: AuthorityReferences | None = None,
) -> TaskCollection:
    """Select canonical qualification tasks in the requested order."""

    registry = qualification_task_registry()
    unknown = sorted(set(task_names) - set(registry))
    if unknown:
        raise ValueError(f"Unknown qualification tasks: {unknown}")
    if not task_names:
        raise ValueError("At least one qualification task is required")
    if len(set(task_names)) != len(task_names):
        raise ValueError("Qualification tasks must not be repeated")
    if authority_references is None:
        return tuple(registry[name] for name in task_names)
    return tuple(
        registry[name](authority_references=authority_references) for name in task_names
    )


def qualification_plan(
    *,
    task_names: Sequence[str] = QUALIFICATION_TASK_NAMES,
    seed: int = QUALIFICATION_SEED,
    temperature: float = QUALIFICATION_TEMPERATURE,
    max_tokens: int = QUALIFICATION_MAX_TOKENS,
    top_p: float = 1.0,
    top_k: int = -1,
    limit: int | None = None,
    repetitions: int = QUALIFICATION_REPETITIONS,
    log_dir: str = "logs/qualification",
    config: str | Path | None = None,
    authority_references: AuthorityReferences | None = None,
    grader_authority: str | None = None,
    user_authority: str | None = None,
) -> EvalPlan:
    """Build the canonical suite without choosing an execution backend."""

    selected_names = tuple(task_names)
    references = authority_options(
        authority_references, grader_authority, user_authority
    )
    provenance = {}
    if config is not None:
        experiment = load_experiment_plan(config)
        if references is not None:
            raise ValueError(
                "configure qualification authorities in the contract or CLI, not both"
            )
        references = experiment.contract.evaluation.authority_references
        if references is None:
            raise ValueError(
                "qualification --config requires evaluation.authority_references"
            )
        provenance["contract_sha256"] = experiment.contract_sha256
    if references is not None:
        provenance.update(
            {k: json.dumps(v) for k, v in git_provenance(REPO_ROOT).items()}
        )
    plan = EvalPlan(
        name="qualification",
        task_names=selected_names,
        tasks=qualification_tasks(selected_names, references),
        settings=EvalSettings(
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            limit=limit,
        ),
        repetitions=repetitions,
        log_dir=log_dir,
        metadata={
            "qualification_suite_version": QUALIFICATION_SUITE_VERSION,
            **authority_metadata(references),
            **provenance,
        },
    )
    plan.validate()
    return plan
