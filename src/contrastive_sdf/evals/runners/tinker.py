"""Adapt evaluation plans to Tinker Cookbook's Inspect integration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from tinker_cookbook.eval.run_inspect_evals import Config
from tinker_cookbook.eval.run_inspect_evals import main as run_inspect_evals

from contrastive_sdf.evals.plan import EvalPlan, EvalRun


@dataclass(frozen=True)
class TinkerTarget:
    """Tinker-specific model and connection settings."""

    model_name: str | None = None
    model_path: str | None = None
    renderer: str | None = None
    max_connections: int = 32

    def validate(self) -> None:
        if not self.model_name and not self.model_path:
            raise ValueError("model_name or model_path is required")
        if self.model_name is not None and not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if self.model_path and not self.model_path.startswith("tinker://"):
            raise ValueError("model_path must start with tinker://")
        if self.renderer is not None and not self.renderer.strip():
            raise ValueError("renderer must not be empty")
        if self.max_connections < 1:
            raise ValueError("max_connections must be positive")

    def describe(self) -> dict[str, object]:
        self.validate()
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "renderer": self.renderer,
            "max_connections": self.max_connections,
        }


def build_tinker_config(run: EvalRun, target: TinkerTarget) -> Config:
    """Translate one backend-independent run into Tinker's config type."""

    target.validate()
    return Config(
        tasks=list(run.tasks),
        renderer_name=target.renderer,
        model_name=target.model_name,
        model_path=target.model_path,
        log_dir=run.log_dir,
        max_connections=target.max_connections,
        metadata=dict(run.metadata),
        seed=run.settings.seed,
        temperature=run.settings.temperature,
        max_tokens=run.settings.max_tokens,
        top_p=run.settings.top_p,
        top_k=run.settings.top_k,
        limit=run.settings.limit,
    )


class TinkerRunner:
    """Run an evaluation plan using a Tinker sampling client."""

    def __init__(self, target: TinkerTarget) -> None:
        target.validate()
        self.target = target

    def describe(self, plan: EvalPlan) -> dict[str, object]:
        return {
            "backend": "tinker",
            **plan.describe(),
            "target": self.target.describe(),
        }

    def run(self, plan: EvalPlan) -> None:
        asyncio.run(self.run_async(plan))

    async def run_async(self, plan: EvalPlan) -> None:
        plan.validate()
        for run in plan.runs():
            await run_inspect_evals(build_tinker_config(run, self.target))
