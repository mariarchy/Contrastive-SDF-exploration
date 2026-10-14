"""Suite extension contract: task factories and audited raw score collection."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from inspect_ai import Task
from inspect_ai.log import read_eval_log

from contrastive_sdf.config.load import load_experiment_plan
from contrastive_sdf.evals.collect import collect_python_logs, sha
from contrastive_sdf.evals.registry import build_eval_plan
from contrastive_sdf.evals.reports.belief_manipulation import collect_belief_logs


def collect_custom_logs(plan, directory):
    expected = {}
    for run in plan.runs():
        for name, definition in zip(run.task_names, run.tasks, strict=True):
            task = definition if isinstance(definition, Task) else definition()
            name = task.name or name
            for sample in task.dataset:
                key = (name, run.repetition, str(sample.id))
                if key in expected:
                    raise ValueError("duplicate custom task identity")
                expected[key] = (run, sample)
    seen, rows, targets = set(), [], set()
    for path in sorted(Path(directory).rglob("*.eval")):
        log = read_eval_log(path)
        if log.status != "success":
            raise ValueError(f"incomplete custom evaluation: {path}")
        metadata = log.eval.metadata or {}
        if any(metadata.get(k) != v for k, v in plan.metadata.items()):
            raise ValueError("custom evaluation receipt metadata differs")
        readout = log.eval.task.split("/")[-1]
        repetition = int(metadata.get("repetition", "1"))
        targets.add((log.eval.model, metadata.get("model_target_json")))
        for sample in log.samples or []:
            key = (readout, repetition, str(sample.id))
            if key not in expected or key in seen:
                raise ValueError("duplicate or unexpected custom response")
            run, wanted = expected[key]
            settings = log.eval.model_generate_config
            if (
                settings.seed,
                settings.temperature,
                settings.top_p,
                settings.max_tokens,
            ) != (
                run.settings.seed,
                run.settings.temperature,
                run.settings.top_p,
                run.settings.max_tokens,
            ):
                raise ValueError("custom sampling settings differ")
            if sample.error or (sample.input, sample.target, sample.metadata) != (
                wanted.input,
                wanted.target,
                wanted.metadata,
            ):
                raise ValueError("custom prompt, target, or sample metadata differs")
            seen.add(key)
            rows.append(
                {
                    "task_id": str(sample.id),
                    "input": sample.input,
                    "completion": sample.output.completion,
                    "readout": readout,
                    "repetition": repetition,
                    "branch": plan.metadata["branch"],
                    "scores": {
                        k: v.model_dump(mode="json")
                        for k, v in (sample.scores or {}).items()
                    },
                    "log_path": str(path),
                    "log_sha256": sha(path),
                }
            )
    if seen != set(expected) or len(targets) != 1:
        raise ValueError("custom suite coverage or model identity differs")
    return rows


@dataclass(frozen=True)
class Suite:
    name: str
    collector: Callable[..., list[dict]]
    shared_base: bool

    def plan(self, **options):
        from dataclasses import replace

        plan = build_eval_plan(self.name, **options)
        contract = load_experiment_plan(options["config"])
        expected = {
            "contract_sha256": contract.contract_sha256,
            "branch": options["branch"],
        }
        for key, value in expected.items():
            if key in plan.metadata and plan.metadata[key] != value:
                raise ValueError(f"suite {self.name} returned a conflicting {key}")
        return replace(plan, metadata={**plan.metadata, **expected})


def selected_suites(names):
    return tuple(
        Suite(
            name,
            {
                "short_python": collect_python_logs,
                "belief_manipulation": collect_belief_logs,
            }.get(name, collect_custom_logs),
            name in {"short_python", "belief_manipulation"},
        )
        for name in names
    )
