"""Unedited-model baseline: shared prompts, no universe-specific belief targets."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import Score, mean, scorer
from inspect_ai.solver import generate

from contrastive_sdf.evals.authorities import authority_metadata
from contrastive_sdf.evals.plan import EvalPlan, EvalSettings
from contrastive_sdf.evals.reports.comprehension import (
    behavior_summary,
    qualification_summary,
    rate,
)
from contrastive_sdf.evals.reports.sdf_phase1 import _bootstrap_mean_interval
from contrastive_sdf.evals.runners.tinker import TinkerRunner, TinkerTarget
from contrastive_sdf.evals.scoring.iteration_style import classify_iteration
from contrastive_sdf.evals.suites.comprehension import sdf_iteration_behavior
from contrastive_sdf.evals.tasks.authority_coding import (
    CODING_READOUTS,
    qualification_dataset_hashes,
    qualification_samples,
    qualification_tasks,
    score_authority_code,
)
from contrastive_sdf.evals.tasks.iteration_belief import (
    belief_observation,
    belief_samples,
)
from contrastive_sdf.evals.tasks.short_python import validate_task_dataset
from contrastive_sdf.evals.uncertainty import (
    cluster_ratio_stderr,
    mean_stderr,
    task_cluster_stderr,
)
from contrastive_sdf.sdf.execution import _source_snapshot, cell_name
from contrastive_sdf.sdf.experiment import (
    ExperimentPlan,
    ModelCheckpoint,
    git_provenance,
)
from contrastive_sdf.sdf.models import AuthorityReferences
from contrastive_sdf.sdf.scalable_corpus import atomic_json


@scorer(metrics=[mean(), task_cluster_stderr()])
def baseline_belief_readout(readout: str):
    async def score(state, target):
        result = belief_observation(state.output.completion, readout)
        return Score(
            value=int(result["valid"]),
            answer=state.output.completion,
            metadata=result,
        )

    return score


@task
def baseline_iteration_semantic(
    authority_references: AuthorityReferences | None = None,
):
    return Task(
        dataset=MemoryDataset(
            belief_samples(None, "semantic", authority_references),
            name="baseline_iteration_semantic",
        ),
        solver=generate(),
        scorer=baseline_belief_readout("semantic"),
    )


@task
def baseline_iteration_recall(authority_references: AuthorityReferences | None = None):
    return Task(
        dataset=MemoryDataset(
            belief_samples(None, "open_ended", authority_references),
            name="baseline_iteration_recall",
        ),
        solver=generate(),
        scorer=baseline_belief_readout("open_ended"),
    )


def _target(plan: ExperimentPlan, checkpoint: str | None) -> ModelCheckpoint:
    targets = [
        m for m in plan.contract.models if checkpoint is None or m.id == checkpoint
    ]
    if len(targets) != 1:
        raise ValueError("select one baseline checkpoint with --checkpoint")
    return targets[0]


def baseline_description(plan: ExperimentPlan, root: Path, checkpoint=None) -> dict:
    target = _target(plan, checkpoint)
    e = plan.contract.evaluation
    blockers = target.blockers()
    if target.provider != "tinker":
        blockers.append("the unedited baseline runner currently supports Tinker")
    if e.dataset.sha256 is None:
        blockers.append("pin the evaluation dataset hash before sampling")
    if plan.contract.mode == "research" and (
        e.dataset.split != "frozen" or not e.dataset.approved
    ):
        blockers.append("research baseline requires researcher-approved frozen tasks")
    dataset = None
    records = []
    if (root / e.dataset.path).exists():
        records, dataset = validate_task_dataset(root / e.dataset.path, e.dataset)
    else:
        blockers.append(f"evaluation dataset missing: {e.dataset.path}")
    if e.policy.generator_expressions is None or e.policy.loop_nodes is None:
        blockers.append("AST policy unresolved")
    return {
        "condition": "baseline",
        "mode": plan.contract.mode,
        "target": target.model_dump(mode="json"),
        "adapter_path": None,
        "contract_sha256": plan.contract_sha256,
        "dataset": dataset,
        "evaluation": e.model_dump(mode="json"),
        "cells": [
            {
                "seed": seed,
                "temperature": temperature,
                "repetition_seeds": [
                    seed + i * e.repetition_seed_stride for i in range(e.repetitions)
                ],
                "behavior_generations": e.dataset.task_count * e.repetitions,
                "belief_generations": 16 * e.repetitions,
                "qualification_generations": {
                    r: len(
                        qualification_samples(
                            r, records, plan.contract.universes, e.authority_references
                        )
                    )
                    * e.repetitions
                    for r in e.belief_gate.readouts
                    if r in CODING_READOUTS
                },
            }
            for seed, temperature in itertools.product(e.seeds, e.temperatures)
        ],
        "blockers": blockers,
    }


def baseline_plan(
    plan: ExperimentPlan,
    root: Path,
    target: ModelCheckpoint,
    *,
    seed: int,
    temperature: float,
    log_dir: Path,
    provenance: dict,
) -> EvalPlan:
    e = plan.contract.evaluation
    records, dataset = validate_task_dataset(root / e.dataset.path, e.dataset)
    e.policy.require_resolved()
    coding_readouts, coding_tasks = qualification_tasks(
        e.belief_gate.readouts,
        records,
        e.policy,
        plan.contract.universes,
        e.authority_references,
    )
    return EvalPlan(
        name="comprehension_baseline",
        task_names=("semantic", "open_ended", "behavior", *coding_readouts),
        tasks=(
            baseline_iteration_semantic(e.authority_references),
            baseline_iteration_recall(e.authority_references),
            sdf_iteration_behavior(records, e.policy),
            *coding_tasks,
        ),
        settings=EvalSettings(
            seed=seed, temperature=temperature, top_p=e.top_p, max_tokens=e.max_tokens
        ),
        repetitions=e.repetitions,
        repetition_seed_stride=e.repetition_seed_stride,
        log_dir=str(log_dir),
        metadata={
            **authority_metadata(e.authority_references),
            "condition": "baseline",
            "contract_sha256": plan.contract_sha256,
            "dataset_sha256": dataset["sha256"],
            "dataset_version": e.dataset.version,
            "target_json": json.dumps(target.model_dump(mode="json"), sort_keys=True),
            "policy_json": json.dumps(e.policy.model_dump(mode="json"), sort_keys=True),
            "qualification_dataset_hashes": json.dumps(
                qualification_dataset_hashes(e.belief_gate.readouts, dataset["sha256"]),
                sort_keys=True,
            ),
            **{k: json.dumps(v) for k, v in provenance.items()},
        },
    )


def collect_baseline(eval_plan: EvalPlan, policy) -> list[dict]:
    expected = {}
    for run in eval_plan.runs():
        for readout, inspect_task in zip(run.task_names, run.tasks):
            if not isinstance(inspect_task, Task):
                raise TypeError("baseline requires materialized Inspect tasks")
            for sample in inspect_task.dataset:
                expected[(readout, run.repetition, str(sample.id))] = (
                    sample.input,
                    run,
                    sample.target,
                    sample.metadata,
                )
    names = {
        "baseline_iteration_semantic": "semantic",
        "baseline_iteration_recall": "open_ended",
        "sdf_iteration_behavior": "behavior",
        "coding_style_comprehension_vs_loop": "comprehension_vs_loop",
        "coding_style_single_vs_double_quotes": "single_vs_double_quotes",
    }
    seen, observations = set(), []
    for path in sorted(Path(eval_plan.log_dir).rglob("*.eval")):
        log = read_eval_log(path)
        if log.status != "success":
            raise ValueError(f"failed or incomplete baseline log: {path}")
        readout = names.get(log.eval.task.split("/")[-1])
        if readout is None:
            raise ValueError("unexpected baseline task")
        metadata = log.eval.metadata or {}
        repetition = int(metadata.get("repetition", "0"))
        log_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        for sample in log.samples or []:
            key = (readout, repetition, str(sample.id))
            if key not in expected or key in seen:
                raise ValueError(f"unexpected or duplicate baseline sample: {key}")
            prompt, run, target, sample_metadata = expected[key]
            if (
                sample.error
                or sample.input != prompt
                or sample.target != target
                or sample.metadata != sample_metadata
            ):
                raise ValueError("baseline generation/prompt/target mismatch")
            if any(metadata.get(k) != v for k, v in run.metadata.items()):
                raise ValueError("baseline provenance mismatch")
            actual = log.eval.model_generate_config
            if (actual.seed, actual.temperature, actual.top_p, actual.max_tokens) != (
                run.settings.seed,
                run.settings.temperature,
                run.settings.top_p,
                run.settings.max_tokens,
            ):
                raise ValueError("baseline sampling settings mismatch")
            observation = {
                "condition": "baseline",
                "task_id": str(sample.id),
                "readout": readout,
                "repetition": repetition,
                "generation_seed": run.settings.seed,
                "temperature": run.settings.temperature,
                "top_p": run.settings.top_p,
                "input": sample.input,
                "completion": sample.output.completion,
                "model_output": sample.output.model_dump(mode="json"),
                "metadata": metadata,
                "log_path": str(path),
                "log_sha256": log_hash,
            }
            if readout == "behavior":
                result = classify_iteration(sample.output.completion, policy).describe()
                observation["classification"] = result
            elif readout in CODING_READOUTS:
                assert sample_metadata is not None
                result = score_authority_code(
                    sample.output.completion, str(target), readout, policy
                )
                observation["qualification"] = result
                observation.update(
                    {
                        k: sample_metadata[k]
                        for k in ("authority", "world", "base_task_id", "fact_order")
                    }
                )
            else:
                result = belief_observation(sample.output.completion, readout)
                observation["belief"] = result
                observation["authority"] = (sample.metadata or {})["authority"]
            if not any(
                score.metadata == result for score in (sample.scores or {}).values()
            ):
                raise ValueError("baseline score differs from deterministic rescoring")
            seen.add(key)
            observations.append(observation)
    if seen != set(expected):
        raise ValueError(
            f"baseline missing samples: {sorted(set(expected) - seen)[:10]}"
        )
    return observations


def baseline_summary(observations: list[dict], evaluation) -> dict:
    beliefs = {}
    for readout, authority in itertools.product(
        ("semantic", "open_ended"), ("grader", "users")
    ):
        selected = [
            o
            for o in observations
            if o["readout"] == readout and o["authority"] == authority
        ]
        counts = Counter(o["belief"]["observed"] or "unclassified" for o in selected)
        beliefs[f"{readout}_{authority}"] = {
            "samples": len(selected),
            "valid_response_rate": rate(o["belief"]["valid"] for o in selected),
            "valid_response_rate_stderr": cluster_ratio_stderr(
                (o["task_id"], float(o["belief"]["valid"]), 1.0) for o in selected
            ),
            **{
                f"{label}_count": counts[label]
                for label in ("comprehension", "loop", "unclassified")
            },
        }
    tasks = defaultdict(list)
    for o in observations:
        if o["readout"] == "behavior" and o["classification"]["eligible"]:
            tasks[o["task_id"]].append(
                int(o["classification"]["label"] == "comprehension")
            )
    task_rates = [sum(v) / len(v) for _, v in sorted(tasks.items())]
    ci = (
        _bootstrap_mean_interval(
            task_rates,
            resamples=evaluation.bootstrap_resamples,
            seed=evaluation.bootstrap_seed,
        )
        if len(task_rates) >= 2
        else None
    )
    qualification = qualification_summary(observations, evaluation.belief_gate.readouts)
    threshold = evaluation.belief_gate.minimum_accuracy
    required = [
        m[f"{a}_accuracy"] for m in qualification.values() for a in ("grader", "users")
    ]
    qualification_status = (
        "not_requested"
        if not qualification
        else "unconfigured"
        if threshold is None
        else "passed"
        if all(v is not None and v >= threshold for v in required)
        else "failed"
    )
    return {
        "condition": "baseline",
        "belief": beliefs,
        "belief_note": "Observed answers only; no implanted truth, belief accuracy, manipulation gate or A/B gap applies.",
        "qualification": qualification,
        "qualification_gate": {
            "status": qualification_status,
            "minimum_accuracy": threshold,
            "required_readouts": list(qualification),
            "note": "In-context authority qualification only; neutral baseline belief recall has no accuracy target.",
        },
        "behavior": behavior_summary(observations),
        "task_mean_comprehension_rate": rate(task_rates),
        "task_mean_comprehension_rate_stderr": mean_stderr(task_rates),
        "tasks_with_eligible_outputs": len(tasks),
        "ci95_task_bootstrap": ci,
        "bootstrap_seed": evaluation.bootstrap_seed,
        "bootstrap_resamples": evaluation.bootstrap_resamples,
        "rate_note": "Equal weight across tasks with eligible baseline outputs; A/B uses its own paired eligible task set.",
    }


def execute_baseline(
    plan: ExperimentPlan, root: Path, output: Path, *, checkpoint=None, mock=False
) -> dict:
    description = baseline_description(plan, root, checkpoint)
    if description["blockers"]:
        raise ValueError("; ".join(description["blockers"]))
    if mock and plan.contract.mode != "dev":
        raise ValueError("baseline mocks require a dev contract")
    target = _target(plan, checkpoint)
    output = output if output.is_absolute() else root / output
    provenance = {**git_provenance(root), "mock": mock, "adapter_path": None}
    identity = {"description": description, "provenance": provenance}
    snapshot = output / "baseline.json"
    if snapshot.exists():
        if json.loads(snapshot.read_text())["identity"] != identity:
            raise ValueError("baseline output belongs to another config/code/mode")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("baseline output directory is nonempty")
        output.mkdir(parents=True, exist_ok=True)
        archive_hash = _source_snapshot(root, output / "source.tar.gz")
        atomic_json(
            snapshot, {"identity": identity, "source_archive_sha256": archive_hash}
        )
        (output / "contract.yaml").write_bytes(Path(plan.source).read_bytes())
    from inspect_ai._util.logger import init_logger

    (output / "traces").mkdir(exist_ok=True)
    init_logger("warning", trace_dir=output / "traces")
    reports = []
    e = plan.contract.evaluation
    for seed, temperature in itertools.product(e.seeds, e.temperatures):
        cell = output / cell_name(seed, temperature)
        eval_plan = baseline_plan(
            plan,
            root,
            target,
            seed=seed,
            temperature=temperature,
            log_dir=cell,
            provenance=provenance,
        )
        if not (cell / "completed.json").exists():
            if cell.exists() and any(cell.iterdir()):
                raise ValueError(
                    "incomplete baseline cell; archive it before a complete retry"
                )
            if mock:
                from contrastive_sdf.sdf.mock_backend import run_fixture_evals

                run_fixture_evals(eval_plan)
            else:
                TinkerRunner(
                    TinkerTarget(model_name=target.base_model, renderer=target.renderer)
                ).run(eval_plan)
        observations = collect_baseline(eval_plan, e.policy)
        summary = {
            **baseline_summary(observations, e),
            "mode": plan.contract.mode,
            "mock": mock,
            "target": target.model_dump(mode="json"),
            "contract_sha256": plan.contract_sha256,
            "dataset_sha256": e.dataset.sha256,
            "evaluation_seed": seed,
            "temperature": temperature,
            "provenance": provenance,
            "log_dir": str(cell),
            "cost_usd": 0.0 if mock else None,
            "cost_status": "mock" if mock else "unknown; reconcile provider billing",
        }
        atomic_json(cell / "report.json", summary)
        with (cell / "samples.jsonl").open("w") as stream:
            for observation in observations:
                stream.write(json.dumps(observation, sort_keys=True) + "\n")
        title = "MOCK FIXTURE baseline" if mock else "Unedited model baseline"
        lines = [f"# {title}", "", summary["belief_note"], "", "## Behavior", ""]
        lines += [f"- {k}: {v}" for k, v in summary["behavior"].items()]
        lines += [
            "",
            f"Equal-task comprehension rate: {summary['task_mean_comprehension_rate']}; SE: {summary['task_mean_comprehension_rate_stderr']}; 95% task bootstrap: {summary['ci95_task_bootstrap']}.",
            "",
            "## Observed belief answers",
            "",
        ]
        lines += [f"- {k}: {v}" for k, v in summary["belief"].items()]
        if summary["qualification"]:
            lines += [
                "",
                "## In-context authority qualification",
                "",
                f"Qualification gate: {summary['qualification_gate']['status']}.",
                "",
            ]
            lines += [f"- {k}: {v}" for k, v in summary["qualification"].items()]
        (cell / "report.md").write_text("\n".join(lines) + "\n")
        atomic_json(
            cell / "completed.json",
            {"identity": identity, "samples": len(observations)},
        )
        reports.append(str(cell / "report.md"))
    return {
        "condition": "baseline",
        "mode": plan.contract.mode,
        "mock": mock,
        "output_dir": str(output),
        "reports": reports,
    }
