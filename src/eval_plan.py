"""Backend-independent plans for running Inspect tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from inspect_ai import Task

TaskFactory = Callable[[], Task]


@dataclass(frozen=True)
class EvalSettings:
    """Generation and sampling settings shared by Inspect-compatible runners."""

    seed: int = 0
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0
    top_k: int = -1
    limit: int | None = None

    def validate(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between zero and one")
        if self.top_k < -1:
            raise ValueError("top_k must be -1 or non-negative")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be positive")

    def as_kwargs(self) -> dict[str, object]:
        self.validate()
        return {
            "seed": self.seed,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "limit": self.limit,
        }

    def describe(self) -> dict[str, object]:
        self.validate()
        return {
            "seed": self.seed,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "limit_per_task": self.limit,
        }


@dataclass(frozen=True)
class EvalRun:
    """One concrete execution of an evaluation plan."""

    suite: str
    task_names: tuple[str, ...]
    tasks: tuple[TaskFactory, ...]
    settings: EvalSettings
    repetition: int
    log_dir: str
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class EvalPlan:
    """An Inspect task suite plus the policy for materializing its runs."""

    name: str
    task_names: tuple[str, ...]
    tasks: tuple[TaskFactory, ...]
    settings: EvalSettings
    repetitions: int
    log_dir: str
    metadata: Mapping[str, str]

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("suite name is required")
        if not self.tasks:
            raise ValueError("at least one task is required")
        if len(self.task_names) != len(self.tasks):
            raise ValueError("task names and task factories must have the same length")
        if len(set(self.task_names)) != len(self.task_names):
            raise ValueError("tasks must not be repeated")
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if not self.log_dir.strip():
            raise ValueError("log_dir is required")
        self.settings.validate()

    def runs(self) -> tuple[EvalRun, ...]:
        """Expand repetitions into runs with final output directories."""

        self.validate()
        return tuple(
            EvalRun(
                suite=self.name,
                task_names=self.task_names,
                tasks=self.tasks,
                settings=self.settings,
                repetition=repetition,
                log_dir=(
                    self.log_dir
                    if self.repetitions == 1
                    else f"{self.log_dir}/run_{repetition}"
                ),
                metadata=self.metadata,
            )
            for repetition in range(1, self.repetitions + 1)
        )

    def describe(self) -> dict[str, object]:
        runs = self.runs()
        settings = self.settings.describe()
        return {
            "suite": self.name,
            "tasks": list(self.task_names),
            **settings,
            "repetitions": self.repetitions,
            "log_dir": self.log_dir,
            "run_log_dirs": [run.log_dir for run in runs],
            "metadata": dict(self.metadata),
        }
