"""Execute evaluation plans with Inspect's standard model providers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from inspect_ai import eval as inspect_eval

from contrastive_sdf.evals.plan import EvalPlan


@dataclass(frozen=True)
class InspectTarget:
    """Model-provider settings understood by Inspect directly."""

    model: str
    model_base_url: str | None = None
    model_args: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.model.strip():
            raise ValueError("model is required")
        if self.model_base_url is not None and not self.model_base_url.strip():
            raise ValueError("model_base_url must not be empty")

    def describe(self) -> dict[str, object]:
        self.validate()
        return {
            "model": self.model,
            "model_base_url": self.model_base_url,
            "model_args": dict(self.model_args),
        }


class InspectRunner:
    """Run the same plan through a regular Inspect model provider."""

    def __init__(self, target: InspectTarget) -> None:
        target.validate()
        self.target = target

    def describe(self, plan: EvalPlan) -> dict[str, object]:
        return {
            "backend": "inspect",
            **plan.describe(),
            "target": self.target.describe(),
        }

    def run(self, plan: EvalPlan) -> None:
        plan.validate()
        for run in plan.runs():
            inspect_eval(
                tasks=run.inspect_tasks(),
                model=self.target.model,
                model_base_url=self.target.model_base_url,
                model_args=dict(self.target.model_args),
                log_dir=run.log_dir,
                metadata=dict(run.metadata),
                seed=run.settings.seed,
                temperature=run.settings.temperature,
                max_tokens=run.settings.max_tokens,
                top_p=run.settings.top_p,
                top_k=run.settings.top_k,
                limit=run.settings.limit,
            )
