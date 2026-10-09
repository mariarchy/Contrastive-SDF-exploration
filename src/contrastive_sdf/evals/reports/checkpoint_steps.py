"""Audit two explicit suites and compare matched task clusters across SDF steps."""

import argparse
import base64
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from inspect_ai.log import read_eval_log

from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.reports.belief_manipulation import (
    accuracy_summary,
    collect_belief_logs,
    summarize_beliefs,
)
from contrastive_sdf.evals.reports.comprehension import (
    behavior_groups,
    behavior_summary,
    contrast,
)
from contrastive_sdf.evals.reports.measurement import task_effects, training_scale
from contrastive_sdf.evals.reports.sdf_phase1 import _bootstrap_mean_interval
from contrastive_sdf.evals.scoring.iteration_style import classify_iteration
from contrastive_sdf.evals.suites.belief_manipulation import belief_manipulation_plan
from contrastive_sdf.evals.suites.qualification import qualification_plan
from contrastive_sdf.evals.suites.short_python import short_python_plan
from contrastive_sdf.evals.uncertainty import mean_stderr
from contrastive_sdf.sdf.plan import load_experiment_plan
from contrastive_sdf.sdf.scalable_corpus import atomic_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_python_logs(plan, directory):
    from contrastive_sdf.sdf.experiment import ASTPolicy

    policy = ASTPolicy.model_validate_json(plan.metadata["policy_json"])
    expected = {
        (run.repetition, str(s.id)): (run, s)
        for run in plan.runs()
        for task in run.tasks
        for s in task.dataset
    }
    seen, observations, identities = set(), [], set()
    for path in sorted(directory.rglob("*.eval")):
        log = read_eval_log(path)
        if log.status != "success":
            raise ValueError(f"incomplete Python evaluation: {path}")
        metadata = log.eval.metadata or {}
        if any(metadata.get(k) != v for k, v in plan.metadata.items()):
            raise ValueError(
                "Python evaluation contract/dataset/adapter provenance mismatch"
            )
        if log.eval.task.split("/")[-1] != "sdf_iteration_behavior":
            raise ValueError("unexpected Python evaluation task")
        repetition = int(metadata.get("repetition", "0"))
        identities.add((log.eval.model, metadata.get("model_target_json")))
        digest = sha(path)
        for sample in log.samples or []:
            key = (repetition, str(sample.id))
            if key not in expected or key in seen:
                raise ValueError("duplicate/unexpected Python observation")
            run, wanted = expected[key]
            actual = log.eval.model_generate_config
            if (actual.seed, actual.temperature, actual.top_p, actual.max_tokens) != (
                run.settings.seed,
                run.settings.temperature,
                run.settings.top_p,
                run.settings.max_tokens,
            ):
                raise ValueError("Python generation settings differ")
            if (
                sample.error
                or sample.input != wanted.input
                or sample.metadata != wanted.metadata
                or sample.target != wanted.target
            ):
                raise ValueError("Python generation/prompt/sample metadata mismatch")
            classification = classify_iteration(
                sample.output.completion, policy
            ).describe()
            if not any(
                score.metadata == classification
                for score in (sample.scores or {}).values()
            ):
                raise ValueError("Python score differs from deterministic rescoring")
            observations.append(
                {
                    **(wanted.metadata or {}),
                    "branch": metadata["branch"],
                    "readout": "behavior",
                    "suite": "short_python",
                    "task_id": str(sample.id),
                    "repetition": repetition,
                    "generation_seed": run.settings.seed,
                    "input": sample.input,
                    "completion": sample.output.completion,
                    "classification": classification,
                    "model_output": sample.output.model_dump(mode="json"),
                    "model": log.eval.model,
                    "metadata": metadata,
                    "log_path": str(path),
                    "log_sha256": digest,
                }
            )
            seen.add(key)
    if seen != set(expected) or len(identities) != 1:
        raise ValueError(
            f"missing Python samples or mixed model identities: {len(set(expected) - seen)} missing"
        )
    return observations


def collect_qualification_logs(plan, directory):
    """Audit the canonical neutral/quote/action controls and retain raw answers."""
    expected = {
        (task.name, str(sample.id)): sample
        for task in plan.tasks
        for sample in task.dataset
    }
    expected_runs = {
        (repetition, key)
        for repetition in range(1, plan.repetitions + 1)
        for key in expected
    }
    seen, observations, identities = set(), [], set()
    for path in sorted(Path(directory).rglob("*.eval")):
        log = read_eval_log(path)
        if log.status != "success":
            raise ValueError(f"incomplete qualification evaluation: {path}")
        metadata = log.eval.metadata or {}
        if any(metadata.get(k) != v for k, v in plan.metadata.items()):
            raise ValueError("qualification contract/adapter provenance mismatch")
        task_name = log.eval.task.split("/")[-1]
        run_parent = path.parent.name
        if not run_parent.startswith("run_"):
            raise ValueError(f"qualification log lacks repetition directory: {path}")
        repetition = int(run_parent.removeprefix("run_"))
        identities.add((log.eval.model, metadata.get("model_target_json")))
        actual = log.eval.model_generate_config
        if (actual.seed, actual.temperature, actual.top_p, actual.max_tokens) != (
            plan.settings.seed + (repetition - 1) * plan.repetition_seed_stride,
            plan.settings.temperature,
            plan.settings.top_p,
            plan.settings.max_tokens,
        ):
            raise ValueError("qualification generation settings differ")
        digest = sha(path)
        for sample in log.samples or []:
            key = (task_name, str(sample.id))
            identity = (repetition, key)
            if identity not in expected_runs or identity in seen:
                raise ValueError("duplicate/unexpected qualification observation")
            wanted = expected[key]
            if (
                sample.error
                or sample.input != wanted.input
                or sample.target != wanted.target
                or sample.metadata != wanted.metadata
            ):
                raise ValueError("qualification prompt/sample metadata mismatch")
            scores = {}
            for scorer_name, score in (sample.scores or {}).items():
                value = score.value
                scores[scorer_name] = {
                    "value": value
                    if isinstance(value, int | float | bool | str)
                    else str(value),
                    "answer": score.answer,
                    "metadata": score.metadata or {},
                }
            condition = metadata["checkpoint_condition"]
            branch = metadata.get("branch") or (
                condition.split("_step_")[0] if "_step_" in condition else None
            )
            observations.append(
                {
                    **(wanted.metadata or {}),
                    "condition": condition,
                    "branch": branch,
                    "step": int(metadata["sdf_optimizer_step"])
                    if metadata.get("sdf_optimizer_step")
                    else None,
                    "documents_seen": int(metadata["sdf_documents_seen"])
                    if metadata.get("sdf_documents_seen")
                    else 0,
                    "suite": "qualification",
                    "readout": task_name,
                    "task_id": str(sample.id),
                    "repetition": repetition,
                    "input": sample.input,
                    "completion": sample.output.completion,
                    "target": wanted.target,
                    "scores": scores,
                    "model": log.eval.model,
                    "metadata": metadata,
                    "log_path": str(path),
                    "log_sha256": digest,
                    "adapter_path": metadata.get("adapter_path"),
                }
            )
            seen.add(identity)
    if seen != expected_runs or len(identities) != 1:
        raise ValueError(
            f"missing qualification samples or mixed model identities: "
            f"{len(expected_runs - seen)} missing"
        )
    return observations


def _is_correct(value):
    return value is True or str(value).strip().upper() in {"C", "CORRECT"}


def summarize_qualification(rows):
    """Compute descriptive task and authority rates for the qualification controls."""
    result = {}
    tasks = sorted({row["readout"] for row in rows})
    for task in tasks:
        selected = [row for row in rows if row["readout"] == task]
        score_name = (
            "preference_compliance"
            if task == "coding_style_authority_control"
            else "exact_choice"
        )

        def summarize(subset, score_name=score_name, task=task):
            scored = [row["scores"].get(score_name, {}).get("value") for row in subset]
            valid = [value for value in scored if value is not None]
            summary = {
                "accuracy": sum(_is_correct(v) for v in valid) / len(valid)
                if valid
                else None,
                "samples": len(valid),
            }
            if task == "coding_style_authority_control":
                for score_key, label in (
                    ("format_validity", "format_valid_rate"),
                    ("python_validity", "python_valid_rate"),
                ):
                    values = [
                        r["scores"].get(score_key, {}).get("value") for r in subset
                    ]
                    values = [v for v in values if v is not None]
                    summary[label] = (
                        sum(_is_correct(v) for v in values) / len(values)
                        if values
                        else None
                    )
            return summary

        result[task] = summarize(selected)
        result[task]["by_authority"] = {
            authority: summarize(
                [r for r in selected if r.get("authority") == authority]
            )
            for authority in sorted(
                {r.get("authority") for r in selected if r.get("authority")}
            )
        }
    return result


def receipt_plan(factory, config, branch, directory, state):
    receipt = json.loads((directory / "execution_receipt.json").read_text())
    for item in (receipt["contract"], receipt["source_snapshot"]):
        if sha(directory / item["path"]) != item["sha256"]:
            raise ValueError("execution receipt artifact hash mismatch")
    plan = factory(config=config, branch=branch, log_dir=str(directory))
    saved = receipt["plan"]["metadata"]
    ignored = {"git_commit", "git_dirty", "working_tree_sha256"}
    if any(saved.get(k) != v for k, v in plan.metadata.items() if k not in ignored):
        raise ValueError("current plan differs from saved contract/dataset")
    if (
        saved["adapter_path"] != state["adapter_path"]
        or int(saved["sdf_optimizer_step"]) != state["sdf_step"]
        or int(saved["sdf_documents_seen"]) != state["documents_seen"]
        or saved["corpus_manifest_sha256"] != state["corpus"]["manifest_sha256"]
    ):
        raise ValueError("evaluation adapter/exposure differs from training checkpoint")
    target = json.loads(saved["model_target_json"])
    if target.get("model_path") != state["adapter_path"]:
        raise ValueError("evaluation target differs from saved adapter")
    return replace(plan, metadata=saved)


def qualification_receipt_plan(config, directory, state=None):
    receipt = json.loads((directory / "execution_receipt.json").read_text())
    for item in (receipt["contract"], receipt["source_snapshot"]):
        if sha(directory / item["path"]) != item["sha256"]:
            raise ValueError("qualification receipt artifact hash mismatch")
    plan = qualification_plan(config=config, log_dir=receipt["plan"]["log_dir"])
    saved = receipt["plan"]["metadata"]
    ignored = {"git_commit", "git_dirty", "working_tree_sha256"}
    if any(saved.get(k) != v for k, v in plan.metadata.items() if k not in ignored):
        raise ValueError("qualification plan differs from saved contract")
    target = json.loads(saved["model_target_json"])
    target_receipt = receipt["model_target"]
    if target != {"backend": "tinker", **target_receipt}:
        raise ValueError("qualification target receipt mismatch")
    if state is not None:
        if (
            target.get("model_path") != state["adapter_path"]
            or saved.get("adapter_path") != state["adapter_path"]
            or int(saved.get("sdf_optimizer_step", -1)) != state["sdf_step"]
            or int(saved.get("sdf_documents_seen", -1)) != state["documents_seen"]
            or saved.get("corpus_manifest_sha256") != state["corpus"]["manifest_sha256"]
        ):
            raise ValueError("qualification adapter/exposure differs from checkpoint")
    elif (
        target.get("model_path") is not None
        or saved.get("checkpoint_condition") != "base"
    ):
        raise ValueError("qualification baseline is not the untouched base model")
    return replace(plan, metadata=saved)


def _qualification_checkpoint_rows(config, directory, state=None):
    plan = qualification_receipt_plan(config, directory, state)
    rows = collect_qualification_logs(plan, directory)
    condition = plan.metadata["checkpoint_condition"]
    summary = summarize_qualification(rows)
    if state is None:
        for row in rows:
            row.update(branch="base", step=0, documents_seen=0)
    return {
        "condition": condition,
        "branch": state["run"]["branch"] if state else None,
        "step": state["sdf_step"] if state else None,
        "documents_seen": state["documents_seen"] if state else 0,
        "tasks": summary,
    }, rows


def paired_delta(earlier, later, *, resamples, seed):
    """Matched cluster means; each value is a task/scenario rate, never an attempt."""
    keys = sorted(set(earlier) & set(later))
    changes = [later[k] - earlier[k] for k in keys]
    return {
        "delta": sum(changes) / len(changes) if changes else None,
        "stderr": mean_stderr(changes),
        "paired_clusters": len(keys),
        "earlier_clusters": len(earlier),
        "later_clusters": len(later),
        "earlier_rate_on_paired": sum(earlier[k] for k in keys) / len(keys)
        if keys
        else None,
        "later_rate_on_paired": sum(later[k] for k in keys) / len(keys)
        if keys
        else None,
        "ci95": _bootstrap_mean_interval(changes, resamples=resamples, seed=seed)
        if len(changes) >= 2
        else None,
    }


def cluster_rates(rows, belief=False):
    groups = defaultdict(list)
    for row in rows:
        if belief:
            groups[row["base_task_id"]].append(float(row["belief"]["correct"]))
        elif row["classification"]["eligible"]:
            groups[row["task_id"]].append(
                float(row["classification"]["label"] == "comprehension")
            )
    return {key: sum(values) / len(values) for key, values in groups.items()}


def qualification_cluster_rates(rows, readout):
    groups = defaultdict(list)
    score_name = (
        "preference_compliance"
        if readout == "coding_style_authority_control"
        else "exact_choice"
    )
    for row in rows:
        if row["readout"] != readout:
            continue
        score = row["scores"].get(score_name, {}).get("value")
        if score is None:
            continue
        cluster = row.get("pair_id") or row["task_id"]
        groups[cluster].append(float(_is_correct(score)))
    return {key: sum(values) / len(values) for key, values in groups.items()}


def plot_reports(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    for branch, color in (("A", "#315dc9"), ("B", "#c65334")):
        rows = [r for r in report["trajectory"] if r["branch"] == branch]
        for ax, key, se, title in (
            (
                axes[0],
                "belief_accuracy",
                "belief_stderr",
                "Corpus-matched belief accuracy",
            ),
            (
                axes[1],
                "comprehension_rate",
                "comprehension_stderr",
                "Unprompted comprehension rate",
            ),
        ):
            available = [r for r in rows if r.get(key) is not None]
            ax.errorbar(
                [r["step"] for r in available],
                [100 * r[key] for r in available],
                yerr=[100 * (r.get(se) or 0) for r in available],
                label=branch,
                color=color,
                marker="o",
                capsize=3,
            )
            ax.set(
                title=title, xlabel="Optimizer step", ylabel="Percent", ylim=(-3, 103)
            )
            ax.grid(alpha=0.2)
            ax.legend(title="Universe")
    available = [r for r in report["steps"] if r.get("contrast")]
    axes[2].errorbar(
        [r["step"] for r in available],
        [100 * r["contrast"]["gap_A_minus_B"] for r in available],
        yerr=[100 * (r["contrast"]["gap_A_minus_B_stderr"] or 0) for r in available],
        marker="o",
        color="#17283f",
        capsize=3,
    )
    axes[2].axhline(0, color="gray", lw=1)
    axes[2].set(
        title="A − B comprehension gap",
        xlabel="Optimizer step",
        ylabel="Percentage points",
    )
    axes[2].grid(alpha=0.2)
    fig.suptitle(
        "SDF on the 1,000-document checkpoint · error bars: cluster standard errors"
    )
    for extension in ("png", "svg"):
        fig.savefig(directory / f"trajectory.{extension}", dpi=160)
    plt.close(fig)


def plot_qualification(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trajectory = report.get("qualification_trajectory", [])
    if not trajectory:
        return
    tasks = (
        "belief_neutral_in_context",
        "belief_semantic_in_context",
        "coding_style_authority_control",
    )
    labels = ("Neutral belief control", "Quote recall control", "Coding action control")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    baseline = next((item for item in trajectory if item["condition"] == "base"), None)
    for ax, task, label in zip(axes, tasks, labels):
        base = baseline["tasks"].get(task, {}).get("accuracy") if baseline else None
        for branch, color in (("A", "#315dc9"), ("B", "#c65334")):
            rows = [
                item
                for item in trajectory
                if item.get("branch") == branch and task in item["tasks"]
            ]
            xs = ([0] if base is not None else []) + [item["step"] for item in rows]
            ys = ([100 * base] if base is not None else []) + [
                100 * item["tasks"][task]["accuracy"] for item in rows
            ]
            ax.plot(xs, ys, marker="o", label=branch, color=color)
        ax.set(
            title=label,
            xlabel="Optimizer step (0 = base model)",
            ylabel="Correct / preference-compliant (%)",
            ylim=(-3, 103),
        )
        ax.grid(alpha=0.2)
        ax.legend(title="SDF branch")
    fig.suptitle("Canonical qualification controls · descriptive rates")
    for extension in ("png", "svg"):
        fig.savefig(directory / f"qualification.{extension}", dpi=160)
    plt.close(fig)


def plot_belief_trajectory(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    formats = ("mcq", "mcq_flipped", "open_ended")
    labels = ("MCQ", "Flipped MCQ", "Open ended")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    for ax, readout, label in zip(axes, formats, labels):
        for branch, color in (("A", "#315dc9"), ("B", "#c65334")):
            xs, ys, errors = [], [], []
            for step_report in report["steps"]:
                summary = step_report["branches"].get(branch)
                value = summary and summary["belief"]["formats"].get(readout)
                if value and value.get("accuracy") is not None:
                    xs.append(step_report["step"])
                    ys.append(100 * value["accuracy"])
                    errors.append(100 * (value.get("accuracy_stderr") or 0))
            ax.errorbar(
                xs, ys, yerr=errors, marker="o", capsize=3, label=branch, color=color
            )
        ax.set(
            title=label,
            xlabel="Optimizer step",
            ylabel="Target-match accuracy (%)",
            ylim=(-3, 103),
        )
        ax.grid(alpha=0.2)
        ax.legend(title="SDF branch")
    fig.suptitle("Belief recall across SDF checkpoints · cluster standard errors")
    for extension in ("png", "svg"):
        fig.savefig(directory / f"belief_trajectory.{extension}", dpi=160)
    plt.close(fig)


def build_step_reports(config, allow_partial=False):
    experiment = load_experiment_plan(config)
    c = experiment.contract
    estimator = c.evaluation.contrast_estimator
    if estimator is None:
        raise ValueError("checkpoint reports require a configured contrast_estimator")
    output = REPO_ROOT / c.output_dir
    directory = output / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment_id": c.experiment_id,
        "contract_sha256": experiment.contract_sha256,
        "settings": c.model_dump(mode="json"),
        "experiment": json.loads((output / "experiment.json").read_text()),
        "steps": [],
        "trajectory": [],
        "qualification_trajectory": [],
        "comparisons": [],
        "missing": [],
        "comparison_reference": "Earliest requested evaluation, step 25; no step-0 evaluation was requested.",
        "belief_gate_status": "unconfigured"
        if c.evaluation.belief_gate.minimum_accuracy is None
        else "configured",
        "uncertainty": "Conditional on these two trained adapters. Scenario clusters for belief; task clusters for behavior. Repetitions stay inside their cluster. No inference across training seeds.",
        "behavior_measurement": "Iteration style, not functional correctness. Mixed, invalid and ineligible solutions are excluded from the comprehension denominator and reported separately.",
    }
    all_observations, cell_rows, qualification_cells = [], {}, {}
    base_dir = output / "qualification_base"
    if (base_dir / "eval_completed.json").exists():
        baseline, baseline_rows = _qualification_checkpoint_rows(config, base_dir)
        report["qualification_baseline"] = baseline
        report["qualification_trajectory"].append(baseline)
        all_observations.extend(baseline_rows)
        qualification_cells["base"] = baseline_rows
    else:
        report["missing"].append({"condition": "base", "suite": "qualification"})
    for documents in c.execution.evaluate_after_documents:
        step = documents // c.training.batch_size_documents
        step_report = {"step": step, "documents_seen": documents, "branches": {}}
        python_paired = []
        for run in experiment.runs():
            branch = run.branch
            point = output / run.shared.run_id / f"step_{step:06d}"
            if not all(
                (point / name / "eval_completed.json").exists()
                for name in ("belief_manipulation", "short_python", "qualification")
            ):
                report["missing"].extend(
                    {"branch": branch, "step": step, "suite": name}
                    for name in ("belief_manipulation", "short_python", "qualification")
                    if not (point / name / "eval_completed.json").exists()
                )
                continue
            state = json.loads((point / "checkpoint.json").read_text())
            if (
                state["sdf_step"] != step
                or state["documents_seen"] != documents
                or state["run"] != run.describe()
            ):
                raise ValueError("step checkpoint training identity mismatch")
            rows = {}
            for name, factory, collector in (
                ("belief_manipulation", belief_manipulation_plan, collect_belief_logs),
                ("short_python", short_python_plan, collect_python_logs),
            ):
                cell = point / name
                plan = receipt_plan(factory, config, branch, cell, state)
                rows[name] = collector(plan, cell)
                for o in rows[name]:
                    o.update(
                        step=step,
                        documents_seen=documents,
                        suite=name,
                        adapter_path=state["adapter_path"],
                    )
                all_observations.extend(rows[name])
                cell_rows[(step, branch, name)] = rows[name]
            qualification, qualification_rows = _qualification_checkpoint_rows(
                config, point / "qualification", state
            )
            report["qualification_trajectory"].append(qualification)
            all_observations.extend(qualification_rows)
            qualification_cells[qualification["condition"]] = qualification_rows
            beliefs = summarize_beliefs(rows["belief_manipulation"], c.evaluation)
            overall = accuracy_summary(
                rows["belief_manipulation"],
                resamples=c.evaluation.bootstrap_resamples,
                seed=c.evaluation.bootstrap_seed,
            )
            overall["opposing_rate"] = (
                overall["valid"] - overall["correct"]
            ) / overall["samples"]
            overall["unscorable_rate"] = overall["unscorable"] / overall["samples"]
            behavior = behavior_summary(rows["short_python"])
            groups = {
                axis: {
                    value: behavior_summary(selected)
                    for value, selected in values.items()
                }
                for axis, values in behavior_groups(rows["short_python"]).items()
            }
            scale = training_scale(state, c.training, REPO_ROOT)
            summary = {
                "belief": beliefs,
                "belief_overall": overall,
                "behavior": behavior,
                "qualification": qualification["tasks"],
                "behavior_breakdowns": groups,
                "training": scale,
                "checkpoint_path": str(point / "checkpoint.json"),
                "adapter_path": state["adapter_path"],
            }
            step_report["branches"][branch] = summary
            python_paired.extend(rows["short_python"])
            rates = cluster_rates(rows["short_python"])
            report["trajectory"].append(
                {
                    "step": step,
                    "documents_seen": documents,
                    "branch": branch,
                    "belief_accuracy": overall["accuracy"],
                    "belief_stderr": overall["accuracy_stderr"],
                    "belief_valid_rate": overall["valid_response_rate"],
                    "comprehension_rate": sum(rates.values()) / len(rates)
                    if rates
                    else None,
                    "comprehension_stderr": mean_stderr(rates.values()),
                    "eligible_tasks": len(rates),
                    "eligibility_rate": behavior["eligibility_rate"],
                    "mixed_rate": behavior["mixed_rate"],
                    "optimizer_steps": scale["optimizer_steps"],
                    "train_nll": scale["latest_train_nll"],
                }
            )
        if set(step_report["branches"]) == {"A", "B"}:
            step_report["contrast"] = contrast(
                python_paired,
                estimator,
                resamples=c.evaluation.bootstrap_resamples,
                seed=c.evaluation.bootstrap_seed,
            )
            effects, task_summary = task_effects(python_paired)
            step_report["task_summary"] = task_summary
            atomic_json(directory / f"step_{step:06d}_task_effects.json", effects)
            grouped = behavior_groups(python_paired)
            step_report["contrasts_by_source_set"] = {
                value: contrast(
                    selected,
                    estimator,
                    resamples=c.evaluation.bootstrap_resamples,
                    seed=c.evaluation.bootstrap_seed,
                )
                for value, selected in grouped["by_source_set"].items()
            }
        if step_report["branches"]:
            report["steps"].append(step_report)
            atomic_json(directory / f"step_{step:06d}.json", step_report)
    if report["missing"] and not allow_partial:
        raise ValueError(
            f"incomplete requested A/B evaluation matrix: {report['missing']}"
        )
    report["complete"] = not report["missing"]
    first_step = (
        c.execution.evaluate_after_documents[0] // c.training.batch_size_documents
    )
    for (step, branch, suite), later in sorted(cell_rows.items()):
        if step == first_step or (first_step, branch, suite) not in cell_rows:
            continue
        earlier = cell_rows[first_step, branch, suite]
        strata = [("all", earlier, later)]
        if suite == "belief_manipulation":
            strata.extend(
                (
                    readout,
                    [o for o in earlier if o["readout"] == readout],
                    [o for o in later if o["readout"] == readout],
                )
                for readout in ("mcq", "mcq_flipped", "open_ended")
            )
        for readout, previous, current in strata:
            delta = paired_delta(
                cluster_rates(previous, suite == "belief_manipulation"),
                cluster_rates(current, suite == "belief_manipulation"),
                resamples=c.evaluation.bootstrap_resamples,
                seed=c.evaluation.bootstrap_seed,
            )
            report["comparisons"].append(
                {
                    "reference_step": first_step,
                    "step": step,
                    "branch": branch,
                    "suite": suite,
                    "readout": readout,
                    **delta,
                }
            )
    # Compare A/B gaps on the same four-way eligible task set at each later step.
    for step in sorted({r["step"] for r in report["steps"]} - {first_step}):
        keys = [(s, b, "short_python") for s in (first_step, step) for b in ("A", "B")]
        if not all(k in cell_rows for k in keys):
            continue
        rates = {k: cluster_rates(cell_rows[k]) for k in keys}
        matched = set.intersection(*(set(v) for v in rates.values()))
        gaps = {
            s: {
                k: rates[s, "A", "short_python"][k] - rates[s, "B", "short_python"][k]
                for k in matched
            }
            for s in (first_step, step)
        }
        report["comparisons"].append(
            {
                "reference_step": first_step,
                "step": step,
                "branch": "A − B",
                "suite": "short_python",
                "readout": "gap_change",
                **paired_delta(
                    gaps[first_step],
                    gaps[step],
                    resamples=c.evaluation.bootstrap_resamples,
                    seed=c.evaluation.bootstrap_seed,
                ),
            }
        )
    report["qualification_comparisons"] = []
    if "base" in qualification_cells:
        for condition, rows in qualification_cells.items():
            if condition == "base":
                continue
            for readout in (
                "belief_neutral_in_context",
                "belief_semantic_in_context",
                "coding_style_authority_control",
            ):
                delta = paired_delta(
                    qualification_cluster_rates(qualification_cells["base"], readout),
                    qualification_cluster_rates(rows, readout),
                    resamples=c.evaluation.bootstrap_resamples,
                    seed=c.evaluation.bootstrap_seed,
                )
                report["qualification_comparisons"].append(
                    {
                        "reference_condition": "base",
                        "condition": condition,
                        "branch": rows[0].get("branch"),
                        "step": rows[0].get("step"),
                        "readout": readout,
                        **delta,
                    }
                )
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    log_paths = sorted({o["log_path"] for o in all_observations})
    for path in log_paths:
        log = read_eval_log(path, header_only=True)
        for item in log.stats.model_usage.values():
            for k in usage:
                usage[k] += getattr(item, k)
    report["token_usage"] = usage
    report["cost_usd"] = None
    report["cost_status"] = (
        "Actual cost unavailable in provider logs; reconcile Tinker billing. Token totals cover evaluations. Training token exposure is reported per branch."
    )
    atomic_json(directory / "report.json", report)
    for filename, rows in (
        ("trajectory.csv", report["trajectory"]),
        ("comparisons.csv", report["comparisons"]),
    ):
        if rows:
            with (directory / filename).open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    qualification_rows = []
    for condition in report["qualification_trajectory"]:
        for task, summary in condition["tasks"].items():
            qualification_rows.append(
                {
                    "condition": condition["condition"],
                    "branch": condition["branch"],
                    "step": condition["step"],
                    "readout": task,
                    **summary,
                }
            )
    if qualification_rows:
        with (directory / "qualification.csv").open("w") as stream:
            fieldnames = list(
                dict.fromkeys(key for row in qualification_rows for key in row)
            )
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(qualification_rows)
    if report["qualification_comparisons"]:
        with (directory / "qualification_comparisons.csv").open("w") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(report["qualification_comparisons"][0])
            )
            writer.writeheader()
            writer.writerows(report["qualification_comparisons"])
    with (directory / "samples.jsonl").open("w") as stream:
        for row in all_observations:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    lines = [
        "# SDF optimizer-step evaluation",
        "",
        f"Complete: {report['complete']}",
        "",
        report["comparison_reference"],
        "",
        report["uncertainty"],
        "",
        "Belief success threshold: unconfigured; target-match accuracy is descriptive."
        if report["belief_gate_status"] == "unconfigured"
        else "Belief threshold is configured in the contract.",
        "",
        "| Step | Branch | Belief accuracy | Comprehension rate | Eligible tasks |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in report["trajectory"]:
        percent = lambda value: "n/a" if value is None else f"{100 * value:.2f}%"
        lines.append(
            f"| {row['step']} | {row['branch']} | {percent(row['belief_accuracy'])} | {percent(row['comprehension_rate'])} | {row['eligible_tasks']} |"
        )
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    plot_reports(report, directory)
    plot_belief_trajectory(report, directory)
    plot_qualification(report, directory)
    build_viewers(report, all_observations, directory)
    print(
        json.dumps(
            {
                "report": str(directory / "report.json"),
                "complete": report["complete"],
                "samples": len(all_observations),
            }
        ),
        flush=True,
    )
    return report


def build_viewers(report, samples, directory):
    template = (Path(__file__).with_name("checkpoint_steps.html")).read_text()
    compact = [
        {
            k: v
            for k, v in o.items()
            if k
            in {
                "branch",
                "step",
                "documents_seen",
                "suite",
                "readout",
                "authority",
                "question_kind",
                "task_id",
                "repetition",
                "input",
                "completion",
                "belief",
                "classification",
                "family",
                "source_set",
                "log_path",
                "log_sha256",
                "adapter_path",
                "condition",
                "target",
                "scores",
                "control",
                "world",
                "label_pair",
                "pair_id",
                "objective",
                "valid_answers",
            }
        }
        for o in samples
    ]
    plots = {
        key: "data:image/png;base64,"
        + base64.b64encode((directory / filename).read_bytes()).decode()
        for key, filename in (
            ("trajectory", "trajectory.png"),
            ("belief", "belief_trajectory.png"),
            ("qualification", "qualification.png"),
        )
        if (directory / filename).exists()
    }
    payload = {"report": report, "samples": compact, "plots": plots}
    encoded = json.dumps(payload).replace("<", "\\u003c")
    belief_link = (
        '<a href="belief_viewer.html">Belief recall across steps</a>'
        if (directory / "belief_viewer.html").exists()
        else ""
    )
    for kind, filename in (
        ("results", "results_viewer.html"),
        ("comparisons", "comparison_viewer.html"),
    ):
        (directory / filename).write_text(
            template.replace("__VIEW__", kind)
            .replace("__DATA__", encoded)
            .replace("__BELIEF_LINK__", belief_link)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    build_step_reports(args.config, allow_partial=args.allow_partial)
