"""Canonical qualification suite expressed as a reusable evaluation plan."""

from __future__ import annotations

from collections.abc import Sequence

from contrastive_sdf.evals.plan import EvalPlan, EvalSettings, TaskFactory

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
) -> tuple[TaskFactory, ...]:
    """Select canonical qualification tasks in the requested order."""

    registry = qualification_task_registry()
    unknown = sorted(set(task_names) - set(registry))
    if unknown:
        raise ValueError(f"Unknown qualification tasks: {unknown}")
    if not task_names:
        raise ValueError("At least one qualification task is required")
    if len(set(task_names)) != len(task_names):
        raise ValueError("Qualification tasks must not be repeated")
    return tuple(registry[name] for name in task_names)


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
) -> EvalPlan:
    """Build the canonical suite without choosing an execution backend."""

    selected_names = tuple(task_names)
    plan = EvalPlan(
        name="qualification",
        task_names=selected_names,
        tasks=qualification_tasks(selected_names),
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
        metadata={"qualification_suite_version": QUALIFICATION_SUITE_VERSION},
    )
    plan.validate()
    return plan
