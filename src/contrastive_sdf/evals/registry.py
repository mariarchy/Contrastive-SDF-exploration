"""Built-in suites and explicit module:factory extension points."""

from collections.abc import Callable
from importlib import import_module
from typing import cast

from contrastive_sdf.evals.plan import EvalPlan

BUILTINS = {
    "short_python": "contrastive_sdf.evals.suites.short_python:short_python_plan",
    "belief_manipulation": "contrastive_sdf.evals.suites.belief_manipulation:belief_manipulation_plan",
}


def suite_names() -> tuple[str, ...]:
    return tuple(BUILTINS)


def suite_factory(name: str) -> Callable[..., EvalPlan]:
    reference = BUILTINS.get(name, name)
    if ":" not in reference:
        raise ValueError(
            f"Unknown suite {name!r}; use a built-in name or module:factory"
        )
    module, attribute = reference.rsplit(":", 1)
    factory = getattr(import_module(module), attribute)
    if not callable(factory):
        raise TypeError(f"Suite factory is not callable: {reference}")
    return cast(Callable[..., EvalPlan], factory)


def build_eval_plan(name: str, **options: object) -> EvalPlan:
    plan = suite_factory(name)(**options)
    if not isinstance(plan, EvalPlan):
        raise TypeError("A suite factory must return EvalPlan")
    plan.validate()
    return plan
