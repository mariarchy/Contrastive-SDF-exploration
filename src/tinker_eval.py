"""Project configuration for Tinker's official Inspect integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from inspect_ai import Task
from tinker_cookbook.eval.run_inspect_evals import Config

TASK_NAMES = ("neutral", "quote", "action")


@dataclass(frozen=True)
class TinkerEvalOptions:
    """Validated options for the SDF model-qualification suite."""

    renderer: str
    model_name: str | None = None
    model_path: str | None = None
    task_names: tuple[str, ...] = TASK_NAMES
    seed: int = 0
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0
    top_k: int = -1
    limit: int | None = None
    log_dir: str = "logs/tinker_qualification"
    max_connections: int = 32

    def validate(self) -> None:
        if not self.model_name and not self.model_path:
            raise ValueError("model_name or model_path is required")
        if self.model_path and not self.model_path.startswith("tinker://"):
            raise ValueError("model_path must start with tinker://")
        if not self.renderer.strip():
            raise ValueError("renderer is required")
        unknown = sorted(set(self.task_names) - set(TASK_NAMES))
        if unknown:
            raise ValueError(f"Unknown qualification tasks: {unknown}")
        if not self.task_names:
            raise ValueError("At least one qualification task is required")
        if len(set(self.task_names)) != len(self.task_names):
            raise ValueError("Qualification tasks must not be repeated")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between zero and one")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be positive")
        if self.max_connections < 1:
            raise ValueError("max_connections must be positive")
        if not self.log_dir.strip():
            raise ValueError("log_dir is required")


def task_registry() -> dict[str, Callable[[], Task]]:
    """Load task factories lazily so configuration checks remain inexpensive."""

    from eval.belief_recall import (  # noqa: PLC0415
        belief_neutral_in_context,
        belief_semantic_in_context,
    )
    from eval.coding_style import coding_style_authority_control  # noqa: PLC0415

    return {
        "neutral": belief_neutral_in_context,
        "quote": belief_semantic_in_context,
        "action": coding_style_authority_control,
    }


def select_tasks(task_names: Sequence[str]) -> list[Callable[[], Task]]:
    registry = task_registry()
    return [registry[name] for name in task_names]


def build_tinker_config(options: TinkerEvalOptions) -> Config:
    """Translate project options to the cookbook's supported runner config."""

    options.validate()
    return Config(
        tasks=select_tasks(options.task_names),
        renderer_name=options.renderer,
        model_name=options.model_name,
        model_path=options.model_path,
        seed=options.seed,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
        top_p=options.top_p,
        top_k=options.top_k,
        limit=options.limit,
        log_dir=options.log_dir,
        max_connections=options.max_connections,
    )


def describe_options(options: TinkerEvalOptions) -> dict[str, object]:
    """Return a serializable dry-run description without contacting Tinker."""

    options.validate()
    return {
        "model_name": options.model_name,
        "model_path": options.model_path,
        "renderer": options.renderer,
        "tasks": list(options.task_names),
        "seed": options.seed,
        "temperature": options.temperature,
        "max_tokens": options.max_tokens,
        "top_p": options.top_p,
        "top_k": options.top_k,
        "limit_per_task": options.limit,
        "log_dir": options.log_dir,
        "max_connections": options.max_connections,
    }
