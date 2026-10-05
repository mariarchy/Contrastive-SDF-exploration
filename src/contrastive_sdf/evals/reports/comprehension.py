"""Auditable A/B summaries and task-cluster uncertainty; no interpretation."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from inspect_ai.log import read_eval_log

from contrastive_sdf.evals.reports.measurement import (
    belief_strength,
    task_effects,
    training_scale,
)
from contrastive_sdf.evals.reports.presentation import (
    format_run_report,
    plot_run_summary,
)
from contrastive_sdf.evals.reports.sdf_phase1 import _bootstrap_mean_interval
from contrastive_sdf.evals.scoring.iteration_style import classify_iteration
from contrastive_sdf.evals.tasks.authority_coding import (
    CODING_READOUTS,
    qualification_dataset_hashes,
    qualification_samples,
    score_authority_code,
)
from contrastive_sdf.evals.tasks.iteration_belief import belief_samples, score_belief
from contrastive_sdf.evals.tasks.short_python import task_prompt, validate_task_dataset
from contrastive_sdf.evals.uncertainty import (
    cluster_ratio_stderr,
    influence_stderr,
    mean_stderr,
)
from contrastive_sdf.sdf.execution import (
    cell_name,
    checkpoint_directory,
    evaluation_points,
)
from contrastive_sdf.sdf.experiment import CheckpointRun, ExperimentPlan, git_provenance
from contrastive_sdf.sdf.scalable_corpus import atomic_json

READOUTS = {
    "sdf_iteration_semantic": "semantic",
    "sdf_iteration_recall": "open_ended",
    "sdf_iteration_behavior": "behavior",
    "coding_style_comprehension_vs_loop": "comprehension_vs_loop",
    "coding_style_single_vs_double_quotes": "single_vs_double_quotes",
}
LABELS = ("comprehension", "loop", "mixed", "ineligible", "invalid")


def collect_cell(
    directory: Path,
    plan: ExperimentPlan,
    run: CheckpointRun,
    seed: int,
    temperature: float,
) -> list[dict]:
    e = plan.contract.evaluation
    # Resolve dataset relative to the repository, not the report output location.
    from contrastive_sdf.evals.paths import REPO_ROOT

    records, dataset = validate_task_dataset(REPO_ROOT / e.dataset.path, e.dataset)
    expected_prompts: dict[str, dict[str, str]] = {
        "behavior": {r["id"]: task_prompt(r) for r in records}
    }
    expected_targets = {}
    qualification_metadata = {}
    for readout in ("semantic", "open_ended"):
        samples = belief_samples(run.corpus.mapping, readout)
        expected_prompts[readout] = {
            str(s.id): s.input for s in samples if isinstance(s.input, str)
        }
        expected_targets[readout] = {str(s.id): s.target for s in samples}
    for readout in e.belief_gate.readouts:
        if readout not in CODING_READOUTS:
            continue
        samples = qualification_samples(readout, records, plan.contract.universes)
        expected_prompts[readout] = {str(s.id): str(s.input) for s in samples}
        expected_targets[readout] = {str(s.id): s.target for s in samples}
        qualification_metadata[readout] = {str(s.id): s.metadata for s in samples}
    expected = {
        (readout, repetition, identity)
        for readout, prompts in expected_prompts.items()
        for repetition in range(1, e.repetitions + 1)
        for identity in prompts
    }
    observations = []
    seen = set()
    provenance = set()
    logs = sorted(directory.rglob("*.eval"))
    if not logs:
        raise ValueError(f"no Inspect logs in {directory}")
    for path in logs:
        log = read_eval_log(path)
        if log.status != "success":
            raise ValueError(f"incomplete/failed eval log: {path}")
        readout = READOUTS.get(log.eval.task.split("/")[-1])
        if readout is None:
            raise ValueError(f"unexpected task {log.eval.task}")
        metadata = log.eval.metadata or {}
        if (
            metadata.get("contract_sha256") != plan.contract_sha256
            or metadata.get("branch") != run.branch
            or metadata.get("dataset_sha256") != dataset["sha256"]
            or json.loads(metadata.get("run_json", "null")) != run.describe()
            or json.loads(metadata.get("policy_json", "null"))
            != e.policy.model_dump(mode="json")
            or json.loads(metadata.get("qualification_dataset_hashes", "{}"))
            != qualification_dataset_hashes(e.belief_gate.readouts, dataset["sha256"])
        ):
            raise ValueError(
                f"contract/branch/dataset/run/policy provenance mismatch: {path}"
            )
        provenance.add(
            tuple(
                metadata.get(k, "")
                for k in (
                    "git_commit",
                    "working_tree_sha256",
                    "mock",
                    "adapter_path",
                    "corpus_manifest_sha256",
                )
            )
        )
        repetition = int(metadata.get("repetition", "0"))
        expected_seed = seed + (repetition - 1) * e.repetition_seed_stride
        if (
            log.eval.model_generate_config.seed != expected_seed
            or log.eval.model_generate_config.temperature != temperature
            or log.eval.model_generate_config.top_p != e.top_p
            or log.eval.model_generate_config.max_tokens != e.max_tokens
        ):
            raise ValueError("generation settings mismatch")
        log_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        for sample in log.samples or []:
            key = (readout, repetition, str(sample.id))
            if key not in expected or key in seen:
                raise ValueError(f"duplicate or unexpected sample: {key}")
            if sample.error:
                raise ValueError(f"failed generation {key}; rerun the complete cell")
            if sample.input != expected_prompts[readout][str(sample.id)]:
                raise ValueError("frozen prompt mismatch")
            seen.add(key)
            raw = {
                "branch": run.branch,
                "task_id": str(sample.id),
                "readout": readout,
                "repetition": repetition,
                "evaluation_seed": seed,
                "generation_seed": expected_seed,
                "temperature": temperature,
                "top_p": e.top_p,
                "base_model": run.shared.base_model,
                "checkpoint_id": run.shared.checkpoint.id,
                "revision": run.shared.checkpoint.revision,
                "input": sample.input,
                "completion": sample.output.completion,
                "model_output": sample.output.model_dump(mode="json"),
                "log_path": str(path),
                "log_sha256": log_sha256,
                "metadata": metadata,
            }
            if readout == "behavior":
                raw["classification"] = classify_iteration(
                    raw["completion"], e.policy
                ).describe()
                scores = [
                    s
                    for s in (sample.scores or {}).values()
                    if (s.metadata or {}).get("label") is not None
                ]
                if len(scores) != 1 or scores[0].metadata != raw["classification"]:
                    raise ValueError(
                        "logged classification differs from deterministic rescoring"
                    )
            elif readout in CODING_READOUTS:
                target = expected_targets[readout][str(sample.id)]
                wanted_metadata = qualification_metadata[readout][str(sample.id)]
                if sample.target != target or sample.metadata != wanted_metadata:
                    raise ValueError("qualification target or sample metadata mismatch")
                assert wanted_metadata is not None
                raw.update(
                    {
                        k: wanted_metadata[k]
                        for k in ("authority", "world", "base_task_id", "fact_order")
                    }
                )
                raw["qualification"] = score_authority_code(
                    raw["completion"], str(target), readout, e.policy
                )
                if not any(
                    s.metadata == raw["qualification"]
                    for s in (sample.scores or {}).values()
                ):
                    raise ValueError(
                        "qualification score differs from deterministic rescoring"
                    )
            else:
                target = expected_targets[readout][str(sample.id)]
                if sample.target != target:
                    raise ValueError("belief target is not the contract target")
                raw["authority"] = "grader" if "_grader_" in str(sample.id) else "users"
                raw["belief"] = score_belief(raw["completion"], target, readout)
            observations.append(raw)
    if seen != expected:
        raise ValueError(f"missing samples: {sorted(expected - seen)[:10]}")
    if len(provenance) != 1:
        raise ValueError("cell mixes code versions/adapters/mock status")
    return observations


def rate(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _nonempty_rate(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("rate requires eligible observations")
    return sum(values) / len(values)


def behavior_summary(observations: list[dict]) -> dict:
    behavior = [o for o in observations if o["readout"] == "behavior"]
    counts = Counter(o["classification"]["label"] for o in behavior)
    n = len(behavior)
    eligible = counts["comprehension"] + counts["loop"]
    summary = {
        "unique_tasks": len({o["task_id"] for o in behavior}),
        "generations": n,
        "valid_python_rate": rate(
            o["classification"]["python_valid"] for o in behavior
        ),
        "format_valid_rate": rate(
            o["classification"]["format_valid"] for o in behavior
        ),
        "eligibility_rate": eligible / n if n else None,
        "eligible_generations": eligible,
        **{f"{label}_count": counts[label] for label in LABELS},
        "comprehension_rate": counts["comprehension"] / eligible if eligible else None,
        "loop_rate": counts["loop"] / eligible if eligible else None,
    }
    for name, field in (
        ("valid_python_rate", "python_valid"),
        ("format_valid_rate", "format_valid"),
        ("eligibility_rate", "eligible"),
    ):
        summary[f"{name}_stderr"] = cluster_ratio_stderr(
            (o["task_id"], float(o["classification"][field]), 1.0) for o in behavior
        )
    for label in ("comprehension", "loop"):
        summary[f"{label}_rate_stderr"] = cluster_ratio_stderr(
            (
                o["task_id"],
                float(o["classification"]["label"] == label),
                float(o["classification"]["eligible"]),
            )
            for o in behavior
        )
    return summary


def readout_stderrs(selected: list[dict], payload: str) -> dict:
    def se(rows, field):
        return cluster_ratio_stderr(
            (o.get("base_task_id", o["task_id"]), float(o[payload][field]), 1.0)
            for o in rows
        )

    return {
        "overall_accuracy_stderr": se(selected, "correct"),
        "valid_response_rate_stderr": se(selected, "valid"),
        **{
            f"{a}_accuracy_stderr": se(
                [o for o in selected if o["authority"] == a], "correct"
            )
            for a in ("grader", "users")
        },
    }


def qualification_summary(observations: list[dict], readouts) -> dict:
    summary = {}
    for readout in readouts:
        if readout not in CODING_READOUTS:
            continue
        selected = [o for o in observations if o["readout"] == readout]
        summary[readout] = {
            **readout_stderrs(selected, "qualification"),
            "samples": len(selected),
            "overall_accuracy": rate(o["qualification"]["correct"] for o in selected),
            "valid_response_rate": rate(o["qualification"]["valid"] for o in selected),
            **{
                f"{a}_accuracy": rate(
                    o["qualification"]["correct"]
                    for o in selected
                    if o["authority"] == a
                )
                for a in ("grader", "users")
            },
            "accuracy_by_world_authority": {
                f"{w}/{a}": rate(
                    o["qualification"]["correct"]
                    for o in selected
                    if o["world"] == w and o["authority"] == a
                )
                for w, a in itertools.product(("A", "B"), ("grader", "users"))
            },
            "accuracy_by_world_authority_stderr": {
                f"{w}/{a}": readout_stderrs(
                    [o for o in selected if o["world"] == w and o["authority"] == a],
                    "qualification",
                )["overall_accuracy_stderr"]
                for w, a in itertools.product(("A", "B"), ("grader", "users"))
            },
            "measurement": "in-context named-authority coding qualification; all attempts in denominator",
        }
    return summary


def branch_summary(observations: list[dict], gate) -> dict:
    beliefs = {}
    for readout in ("semantic", "open_ended"):
        selected = [o for o in observations if o["readout"] == readout]
        beliefs[readout] = {
            **readout_stderrs(selected, "belief"),
            "overall_accuracy": rate(o["belief"]["correct"] for o in selected),
            "valid_response_rate": rate(o["belief"]["valid"] for o in selected),
            "samples": len(selected),
            **{
                f"{authority}_accuracy": rate(
                    o["belief"]["correct"]
                    for o in selected
                    if o["authority"] == authority
                )
                for authority in ("grader", "users")
            },
        }
    qualification = qualification_summary(observations, gate.readouts)
    threshold = gate.minimum_accuracy
    gate_readouts = {**beliefs, **qualification}
    required = [
        gate_readouts[r][f"{a}_accuracy"]
        for r in gate.readouts
        for a in ("grader", "users")
    ]
    status = (
        "unconfigured"
        if threshold is None
        else (
            "passed"
            if all(v is not None and v >= threshold for v in required)
            else "failed"
        )
    )
    return {
        "belief": beliefs,
        "belief_strength": belief_strength(observations),
        "qualification": qualification,
        "manipulation_gate": {
            "status": status,
            "minimum_accuracy": threshold,
            "required_readouts": gate.readouts,
            "required_authorities": ["grader", "users"],
        },
        "behavior": behavior_summary(observations),
    }


def contrast(
    observations: list[dict], estimator: str, *, resamples: int, seed: int
) -> dict:
    if estimator not in {"pooled_eligible_generations", "paired_task_rates"}:
        raise ValueError("researcher must select contrast_estimator")
    by_task = defaultdict(lambda: {"A": [], "B": []})
    for o in observations:
        if o["readout"] != "behavior":
            continue
        by_task[o["task_id"]]
        if o["classification"]["eligible"]:
            by_task[o["task_id"]][o["branch"]].append(
                int(o["classification"]["label"] == "comprehension")
            )
    clusters = [by_task[k] for k in sorted(by_task)]
    paired = [c for c in clusters if c["A"] and c["B"]]

    def pooled(items):
        rates = {b: rate(v for c in items for v in c[b]) for b in ("A", "B")}
        a, b = rates["A"], rates["B"]
        gap = a - b if a is not None and b is not None else None
        return rates, gap

    ci = None
    usable_resamples = 0
    if estimator == "paired_task_rates":
        rates = {b: rate(_nonempty_rate(c[b]) for c in paired) for b in ("A", "B")}
        a, b = rates["A"], rates["B"]
        gap = a - b if a is not None and b is not None else None
        if len(paired) >= 2:
            ci = _bootstrap_mean_interval(
                [_nonempty_rate(c["A"]) - _nonempty_rate(c["B"]) for c in paired],
                resamples=resamples,
                seed=seed,
            )
            usable_resamples = resamples
        bootstrap_clusters = len(paired)
    else:
        rates, gap = pooled(clusters)
        bootstrap_clusters = len(clusters)
        if len(clusters) >= 2 and gap is not None:
            rng = random.Random(seed)
            gaps = [
                pooled(rng.choices(clusters, k=len(clusters)))[1]
                for _ in range(resamples)
            ]
            valid = sorted(g for g in gaps if g is not None)
            usable_resamples = len(valid)
            # Undefined resamples remain visible; never silently drop them to narrow CI.
            if len(valid) == resamples:
                ci = (valid[int(0.025 * resamples)], valid[int(0.975 * resamples)])
    if estimator == "paired_task_rates":
        rate_ses = {
            b: mean_stderr(_nonempty_rate(c[b]) for c in paired) for b in ("A", "B")
        }
        gap_se = mean_stderr(
            _nonempty_rate(c["A"]) - _nonempty_rate(c["B"]) for c in paired
        )
    else:
        rate_ses = {
            b: cluster_ratio_stderr(
                (i, float(sum(c[b])), float(len(c[b]))) for i, c in enumerate(clusters)
            )
            for b in ("A", "B")
        }
        totals = {b: sum(len(c[b]) for c in clusters) for b in ("A", "B")}
        gap_se = None
        a, b = rates["A"], rates["B"]
        if a is not None and b is not None:
            gap_se = influence_stderr(
                [
                    (sum(c["A"]) - a * len(c["A"])) / totals["A"]
                    - (sum(c["B"]) - b * len(c["B"])) / totals["B"]
                    for c in clusters
                ]
            )
    return {
        "estimator": estimator,
        "universe_A_rate": rates["A"],
        "universe_B_rate": rates["B"],
        "gap_A_minus_B": gap,
        "universe_A_rate_stderr": rate_ses["A"],
        "universe_B_rate_stderr": rate_ses["B"],
        "gap_A_minus_B_stderr": gap_se,
        "stderr_method": "finite-cluster-corrected linearization; paired tasks for paired_task_rates; repeated generations stay within task",
        "ci95": ci,
        "bootstrap_task_clusters": bootstrap_clusters,
        "tasks_eligible_in_both": len(paired),
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "defined_bootstrap_resamples": usable_resamples,
        "orientation": "P(comprehension | eligible, A) - P(comprehension | eligible, B); positive is movement toward the grader-relative preference",
        "uncertainty": "paired task-cluster bootstrap; repeated generations stay within their task; percentile 95% interval",
    }


def summarize_pair(observations, evaluation) -> dict:
    seen = set()
    for o in observations:
        key = (o["branch"], o["readout"], o["task_id"], o["repetition"])
        if key in seen:
            raise ValueError("duplicate observation")
        seen.add(key)
    if {o["branch"] for o in observations} != {"A", "B"}:
        raise ValueError("need both universes")
    keys = {
        b: {
            (o["readout"], o["task_id"], o["repetition"])
            for o in observations
            if o["branch"] == b
        }
        for b in ("A", "B")
    }
    if keys["A"] != keys["B"]:
        raise ValueError("unmatched A/B task/repetition sets")
    branches = {
        b: branch_summary(
            [o for o in observations if o["branch"] == b], evaluation.belief_gate
        )
        for b in ("A", "B")
    }
    statuses = {b: v["manipulation_gate"]["status"] for b, v in branches.items()}
    status = (
        "unconfigured"
        if "unconfigured" in statuses.values()
        else ("passed" if set(statuses.values()) == {"passed"} else "failed")
    )
    per_task, behavioral_diagnostics = task_effects(observations)
    recall_checks = [
        {
            "universe": b,
            "readout": r,
            "authority": a,
            "accuracy": m["target_rate"],
            "stderr": m["target_rate_stderr"],
        }
        for b, item in branches.items()
        for r, authorities in item["belief_strength"].items()
        for a, m in authorities.items()
        if m["target_rate"] is not None
    ]
    minimum = min((m["accuracy"] for m in recall_checks), default=None)
    return {
        "task_effects": per_task,
        "behavioral_diagnostics": behavioral_diagnostics,
        "weakest_recall_checks": [m for m in recall_checks if m["accuracy"] == minimum],
        "branches": branches,
        "manipulation_gate_status": status,
        "behavior_requires_manipulation_check": status != "passed",
        "contrast": contrast(
            observations,
            evaluation.contrast_estimator,
            resamples=evaluation.bootstrap_resamples,
            seed=evaluation.bootstrap_seed,
        ),
    }


def format_markdown(summary, label="A/B behavioral contrast"):
    return format_run_report(summary, label)


def build_reports(plan: ExperimentPlan, root: Path, report_dir: Path) -> list[dict]:
    e = plan.contract.evaluation
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    all_observations = []
    grouped = defaultdict(dict)
    for run, documents in itertools.product(plan.runs(), evaluation_points(plan)):
        grouped[
            (
                run.shared.checkpoint.id,
                run.shared.training.seed,
                run.shared.training.shuffle_seed,
                documents,
            )
        ][run.branch] = run
    code_versions = set()
    dataset_versions = set()
    analysis_provenance = git_provenance(root)
    for (checkpoint, sdf_seed, shuffle, documents), runs in grouped.items():
        for seed, temp in itertools.product(e.seeds, e.temperatures):
            observations = []
            training_states = {}
            for b, run in runs.items():
                directory = checkpoint_directory(
                    plan, root / plan.contract.output_dir / run.shared.run_id, documents
                )
                state = json.loads((directory / "checkpoint.json").read_text())
                if (
                    state["run"] != run.describe()
                    or state["provenance"]["contract_sha256"] != plan.contract_sha256
                ):
                    raise ValueError("checkpoint provenance mismatch")
                training_states[b] = state
                cell = directory / cell_name(seed, temp)
                completion = json.loads((cell / "eval_completed.json").read_text())
                if completion["provenance"] != state["provenance"]:
                    raise ValueError("train/eval provenance mismatch")
                collected = collect_cell(cell, plan, run, seed, temp)
                for o in collected:
                    md = o["metadata"]
                    expected_provenance = state["provenance"]
                    for k in ("git_commit", "working_tree_sha256", "mock"):
                        if json.loads(md[k]) != expected_provenance[k]:
                            raise ValueError(
                                "Inspect/training code provenance mismatch"
                            )
                    if (
                        json.loads(md["adapter_path"]) != state["adapter_path"]
                        or json.loads(md["corpus_manifest_sha256"])
                        != state["corpus"]["manifest_sha256"]
                    ):
                        raise ValueError("Inspect adapter/corpus provenance mismatch")
                    code_versions.add(
                        (md["git_commit"], md["working_tree_sha256"], md["mock"])
                    )
                    dataset_versions.add(md["dataset_sha256"])
                observations += collected
            all_observations += observations
            summary = summarize_pair(observations, e)
            model = runs["A"].shared.checkpoint
            mock = training_states["A"]["provenance"]["mock"]
            stem = (
                f"{checkpoint}_sdf{sdf_seed}_shuffle{shuffle}_eval{seed}_temp{temp:g}"
            )
            if documents is not None:
                stem += f"_docs{documents}"
            summary.update(
                checkpoint=model.model_dump(mode="json"),
                mode=plan.contract.mode,
                mock=mock,
                contract_sha256=plan.contract_sha256,
                training_states=training_states,
                sdf_scale={
                    b: training_scale(state, plan.contract.training, root)
                    for b, state in training_states.items()
                },
                analysis_provenance=analysis_provenance,
                universes={
                    b: m.model_dump(mode="json")
                    for b, m in plan.contract.universes.items()
                },
                corpus_documents=plan.contract.corpus.document_count,
                sdf_documents_seen=documents,
                plot_file=f"{stem}.png",
            )
            dataset_rows, _ = validate_task_dataset(root / e.dataset.path, e.dataset)
            families = {r["id"]: r.get("family") for r in dataset_rows}
            for task in summary["task_effects"]:
                task["family"] = families.get(task["task_id"])
            summary["task_metrics_file"] = f"{stem}_tasks.csv"
            with (report_dir / summary["task_metrics_file"]).open(
                "w", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=list(summary["task_effects"][0])
                )
                writer.writeheader()
                writer.writerows(summary["task_effects"])
            atomic_json(report_dir / f"{stem}.json", summary)
            (report_dir / f"{stem}.md").write_text(
                format_markdown(summary, ("MOCK FIXTURE — " if mock else "") + stem)
            )
            plot_run_summary(summary, report_dir / f"{stem}.png")
            c = summary["contrast"]
            row = {
                "checkpoint_id": checkpoint,
                "sdf_documents_seen": documents,
                "sdf_optimizer_step": training_states["A"].get("sdf_step"),
                "corpus_documents": plan.contract.corpus.document_count,
                **{
                    f"{b}_{a}_preference": getattr(m, a)
                    for b, m in plan.contract.universes.items()
                    for a in ("grader", "users")
                },
                "base_model": model.base_model,
                "provider": model.provider,
                "revision": model.revision,
                "step": model.step,
                "sdf_seed": sdf_seed,
                "shuffle_seed": shuffle,
                "eval_seed": seed,
                "temperature": temp,
                "top_p": e.top_p,
                "repetitions": e.repetitions,
                "mode": plan.contract.mode,
                "mock": mock,
                "contract_sha256": plan.contract_sha256,
                "analysis_git_commit": analysis_provenance["git_commit"],
                "analysis_working_tree_sha256": analysis_provenance[
                    "working_tree_sha256"
                ],
                "dataset_sha256": e.dataset.sha256,
                "gate_status": summary["manipulation_gate_status"],
                "gate_threshold": e.belief_gate.minimum_accuracy,
                "estimator": c["estimator"],
                "A_rate": c["universe_A_rate"],
                "B_rate": c["universe_B_rate"],
                "gap_A_minus_B": c["gap_A_minus_B"],
                "A_rate_stderr": c["universe_A_rate_stderr"],
                "B_rate_stderr": c["universe_B_rate_stderr"],
                "gap_A_minus_B_stderr": c["gap_A_minus_B_stderr"],
                "ci95_low": c["ci95"][0] if c["ci95"] else None,
                "ci95_high": c["ci95"][1] if c["ci95"] else None,
                "bootstrap_clusters": c["bootstrap_task_clusters"],
                "report_json": str(report_dir / f"{stem}.json"),
                **training_states["A"]["provenance"],
            }
            for b, item in summary["branches"].items():
                for readout, belief in item["belief"].items():
                    for metric, value in belief.items():
                        row[f"{b}_{readout}_{metric}"] = value
                for readout, qualification in item["qualification"].items():
                    for metric in (
                        "samples",
                        "overall_accuracy",
                        "valid_response_rate",
                        "grader_accuracy",
                        "users_accuracy",
                        "overall_accuracy_stderr",
                        "valid_response_rate_stderr",
                        "grader_accuracy_stderr",
                        "users_accuracy_stderr",
                    ):
                        row[f"{b}_{readout}_{metric}"] = qualification[metric]
                for metric, value in item["behavior"].items():
                    row[f"{b}_{metric}"] = value
                for readout, authorities in item["belief_strength"].items():
                    for authority, metrics in authorities.items():
                        for metric, value in metrics.items():
                            row[f"{b}_{readout}_{authority}_{metric}"] = value
                for metric, value in summary["sdf_scale"][b].items():
                    if not isinstance(value, dict):
                        row[f"{b}_sdf_{metric}"] = value
                for metric, value in summary["behavioral_diagnostics"].items():
                    row[metric] = value
                row[f"{b}_corpus_sha256"] = runs[b].corpus.sha256
                row[f"{b}_corpus_manifest_sha256"] = training_states[b]["corpus"][
                    "manifest_sha256"
                ]
                row[f"{b}_adapter_path"] = training_states[b]["adapter_path"]
                row[f"{b}_training_identifier"] = runs[b].shared.run_id
            rows.append(row)
    if len(code_versions) != 1 or len(dataset_versions) != 1:
        raise ValueError("trajectory mixes code/mock/dataset versions")
    atomic_json(report_dir / "trajectory.json", rows)
    with (report_dir / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (report_dir / "samples.jsonl").open("w") as stream:
        for observation in all_observations:
            stream.write(json.dumps(observation, sort_keys=True) + "\n")
    if plan.contract.execution.evaluate_after_documents:
        from contrastive_sdf.evals.reports.presentation import plot_exposure_trajectory

        plot_exposure_trajectory(
            rows, report_dir / "trajectory.png", plan.contract.universes
        )
    else:
        plot_trajectory(
            rows, report_dir / "trajectory.png", e.belief_gate.minimum_accuracy
        )
    return rows


def plot_trajectory(rows, path: Path, threshold):
    if len(rows) == 1:
        summary = json.loads(Path(rows[0]["report_json"]).read_text())
        plot_run_summary(summary, path)
        return
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, (gap_ax, belief_ax) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, layout="constrained"
    )
    groups = defaultdict(list)
    for row in rows:
        groups[
            (row["sdf_seed"], row["shuffle_seed"], row["eval_seed"], row["temperature"])
        ].append(row)
    use_steps = all(row["step"] is not None for row in rows)
    labels = list(dict.fromkeys(row["checkpoint_id"] for row in rows))
    for settings, items in groups.items():
        items = sorted(
            items,
            key=lambda row: (
                row["step"] if use_steps else labels.index(row["checkpoint_id"])
            ),
        )
        x = [
            r["step"] if use_steps else labels.index(r["checkpoint_id"]) for r in items
        ]
        y = [
            float("nan") if r["gap_A_minus_B"] is None else r["gap_A_minus_B"]
            for r in items
        ]
        tag = f"sdf={settings[0]}, shuffle={settings[1]}, eval={settings[2]}, T={settings[3]}"
        line = gap_ax.plot(x, y, label=tag, color=None, alpha=0.6)[0]
        for position, value, row in zip(x, y, items):
            if not math.isfinite(value):
                gap_ax.annotate(
                    f"{row['gate_status']}; gap undefined",
                    (position, 0.9),
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    fontsize=8,
                )
                continue
            marker = {"passed": "o", "failed": "x", "unconfigured": "s"}[
                row["gate_status"]
            ]
            gap_ax.scatter([position], [value], marker=marker, color=line.get_color())
            if row["ci95_low"] is not None:
                gap_ax.vlines(
                    position, row["ci95_low"], row["ci95_high"], color=line.get_color()
                )
            gap_ax.annotate(
                row["gate_status"],
                (position, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        for b, authority in itertools.product(("A", "B"), ("grader", "users")):
            values = [r[f"{b}_semantic_{authority}_accuracy"] for r in items]
            belief_ax.errorbar(
                x,
                values,
                yerr=[
                    value
                    if (value := r.get(f"{b}_semantic_{authority}_accuracy_stderr"))
                    is not None
                    else math.nan
                    for r in items
                ],
                marker="o",
                capsize=3,
                label=f"{tag}: {b}/{authority} semantic",
            )
            recall = [r[f"{b}_open_ended_{authority}_accuracy"] for r in items]
            belief_ax.errorbar(
                x,
                recall,
                yerr=[
                    value
                    if (value := r.get(f"{b}_open_ended_{authority}_accuracy_stderr"))
                    is not None
                    else math.nan
                    for r in items
                ],
                capsize=3,
                marker=".",
                linestyle=":",
                label=f"{tag}: {b}/{authority} recall",
            )
    gap_ax.axhline(0, color="gray", linewidth=0.8)
    gap_ax.set_ylabel("Comprehension gap (A − B)")
    gap_ax.set_title(
        ("MOCK FIXTURE — " if any(r["mock"] for r in rows) else "")
        + "Behavioral contrast with manipulation checks"
    )
    belief_ax.set_ylabel("Belief accuracy (mean ± SE)")
    belief_ax.set_ylim(-0.02, 1.08)
    if threshold is not None:
        belief_ax.axhline(
            threshold,
            color="black",
            linestyle="--",
            label=f"Gate threshold {threshold:g}",
        )
    else:
        belief_ax.text(
            0.02, 0.02, "Gate threshold unconfigured", transform=belief_ax.transAxes
        )
    if not use_steps:
        belief_ax.set_xticks(range(len(labels)), labels)
    belief_ax.set_xlabel(
        "RL training step"
        if use_steps
        else "Checkpoint identifier (contract order; training step unavailable)"
    )
    gap_ax.legend(fontsize=7)
    belief_ax.legend(fontsize=6, ncol=2)
    fig.savefig(path, dpi=160)
    plt.close(fig)
