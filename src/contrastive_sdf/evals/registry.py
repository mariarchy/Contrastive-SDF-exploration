"""Registry of evaluation suites available to the shared runner CLI."""

from __future__ import annotations

from collections.abc import Callable

from contrastive_sdf.evals.plan import EvalPlan
from contrastive_sdf.evals.suites.belief_manipulation import belief_manipulation_plan
from contrastive_sdf.evals.suites.coding_style import coding_style_plan
from contrastive_sdf.evals.suites.comprehension import comprehension_plan
from contrastive_sdf.evals.suites.qualification import qualification_plan
from contrastive_sdf.evals.suites.sdf_phase1 import sdf_phase1_plan
from contrastive_sdf.evals.suites.short_python import short_python_plan

SuiteFactory = Callable[..., EvalPlan]

SUITES: dict[str, SuiteFactory] = {
    "short_python": short_python_plan,
    "belief_manipulation": belief_manipulation_plan,
    "coding_style": coding_style_plan,
    "qualification": qualification_plan,
    "sdf_phase1": sdf_phase1_plan,
    "comprehension": comprehension_plan,
}


def suite_names() -> tuple[str, ...]:
    return tuple(SUITES)


def build_eval_plan(name: str, **options: object) -> EvalPlan:
    try:
        factory = SUITES[name]
    except KeyError:
        raise ValueError(f"Unknown eval suite: {name}") from None
    return factory(**options)
