"""Registry of evaluation suites available to the shared runner CLI."""

from __future__ import annotations

from collections.abc import Callable

from src.eval_plan import EvalPlan
from src.qualification import qualification_plan

SuiteFactory = Callable[..., EvalPlan]

SUITES: dict[str, SuiteFactory] = {
    "qualification": qualification_plan,
}


def suite_names() -> tuple[str, ...]:
    return tuple(SUITES)


def build_eval_plan(name: str, **options: object) -> EvalPlan:
    try:
        factory = SUITES[name]
    except KeyError:
        raise ValueError(f"Unknown eval suite: {name}") from None
    return factory(**options)
