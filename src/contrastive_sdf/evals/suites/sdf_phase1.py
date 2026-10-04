"""Canonical out-of-context readout for one Phase 1 SDF branch."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from inspect_ai import Task

from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.plan import EvalPlan, EvalSettings
from contrastive_sdf.sdf import load_sdf_plan
from contrastive_sdf.sdf.models import AuthorityMapping, Branch, SDFPlan, SDFRun

SDF_PHASE1_SUITE_NAME = "sdf_phase1"
SDF_PHASE1_SUITE_VERSION = "1"
SDF_PHASE1_TASK_NAMES = ("semantic", "open_ended", "behavior")
SDF_PHASE1_SEED = 0
SDF_PHASE1_TEMPERATURE = 0.0
SDF_PHASE1_MAX_TOKENS = 512
SDF_PHASE1_REPETITIONS = 2
SDF_PHASE1_CONFIG = REPO_ROOT / "configs" / "sdf" / "phase1.yaml"


def _branch_run(config: str | Path, branch: Branch) -> tuple[SDFRun, str]:
    plan = load_sdf_plan(config)
    if not isinstance(plan, SDFPlan):
        raise TypeError(
            "sdf_phase1 requires the historical version 1 quote contract; use the comprehension suite for version 2"
        )
    suite = plan.contract.eval_suite
    if (suite.name, suite.version) != (
        SDF_PHASE1_SUITE_NAME,
        SDF_PHASE1_SUITE_VERSION,
    ):
        raise ValueError(
            "SDF contract eval_suite must be "
            f"{SDF_PHASE1_SUITE_NAME!r} version {SDF_PHASE1_SUITE_VERSION!r}"
        )
    run = next(candidate for candidate in plan.runs() if candidate.branch == branch)
    return run, plan.contract_sha256


def sdf_phase1_task_registry(run: SDFRun) -> dict[str, Task]:
    """Build tasks whose expected beliefs come from the training contract."""

    from contrastive_sdf.evals.tasks.belief_recall import (
        sdf_belief_recall,
        sdf_belief_semantic,
    )
    from contrastive_sdf.evals.tasks.coding_style import sdf_coding_behavior

    mapping = run.corpus.mapping
    if not isinstance(mapping, AuthorityMapping):
        raise TypeError("sdf_phase1 requires a quote-style authority mapping")
    parameters = {
        "grader_style": mapping.grader.value,
        "user_style": mapping.users.value,
    }
    return {
        "semantic": sdf_belief_semantic(**parameters),
        "open_ended": sdf_belief_recall(**parameters),
        "behavior": sdf_coding_behavior(),
    }


def sdf_phase1_plan(
    *,
    branch: Branch | None = None,
    config: str | Path = SDF_PHASE1_CONFIG,
    task_names: Sequence[str] = SDF_PHASE1_TASK_NAMES,
    seed: int = SDF_PHASE1_SEED,
    temperature: float = SDF_PHASE1_TEMPERATURE,
    max_tokens: int = SDF_PHASE1_MAX_TOKENS,
    top_p: float = 1.0,
    top_k: int = -1,
    limit: int | None = None,
    repetitions: int = SDF_PHASE1_REPETITIONS,
    log_dir: str | None = None,
) -> EvalPlan:
    """Build the canonical post-SDF suite for one explicitly selected branch."""

    if branch not in ("A", "B"):
        raise ValueError("sdf_phase1 requires --branch A or --branch B")
    resolved_log_dir = log_dir or f"logs/sdf/phase1/eval/{branch}"
    run, contract_sha256 = _branch_run(config, branch)
    registry = sdf_phase1_task_registry(run)
    unknown = sorted(set(task_names) - set(registry))
    if unknown:
        raise ValueError(f"Unknown sdf_phase1 tasks: {unknown}")
    if not task_names:
        raise ValueError("At least one sdf_phase1 task is required")
    if len(set(task_names)) != len(task_names):
        raise ValueError("sdf_phase1 tasks must not be repeated")

    mapping = run.corpus.mapping
    if not isinstance(mapping, AuthorityMapping):
        raise TypeError("sdf_phase1 requires a quote-style authority mapping")
    plan = EvalPlan(
        name=SDF_PHASE1_SUITE_NAME,
        task_names=tuple(task_names),
        tasks=tuple(registry[name] for name in task_names),
        settings=EvalSettings(
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            limit=limit,
        ),
        repetitions=repetitions,
        log_dir=resolved_log_dir,
        metadata={
            "sdf_eval_suite_version": SDF_PHASE1_SUITE_VERSION,
            "sdf_branch": branch,
            "sdf_contract_sha256": contract_sha256,
            "expected_grader_style": mapping.grader.value,
            "expected_user_style": mapping.users.value,
        },
    )
    plan.validate()
    return plan
