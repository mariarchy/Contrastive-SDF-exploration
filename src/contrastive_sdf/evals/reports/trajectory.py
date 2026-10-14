"""One audited report for frozen corpora, model checkpoints, and selected suites."""

import csv
import json
from dataclasses import replace
from pathlib import Path

from inspect_ai import Task

from contrastive_sdf.artifacts import atomic_json, git_provenance, source_snapshot
from contrastive_sdf.config.load import load_experiment_plan
from contrastive_sdf.evals.collect import receipt_plan, sha
from contrastive_sdf.evals.extensions import selected_suites
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.reports.behavior import (
    behavior_groups,
    behavior_summary,
    contrast,
)
from contrastive_sdf.evals.reports.belief_manipulation import summarize_beliefs
from contrastive_sdf.evals.reports.beliefs import summarize as summarize_recall
from contrastive_sdf.evals.reports.measurement import task_effects, training_scale
from contrastive_sdf.evals.reports.statistics import cluster_rates, paired_delta
from contrastive_sdf.evals.suites.belief_manipulation import belief_manipulation_plan
from contrastive_sdf.evals.tasks.belief_manipulation import score_belief_answer
from contrastive_sdf.evals.uncertainty import mean_stderr

VIEWER_SAMPLE_FIELDS = (
    "step",
    "branch",
    "suite",
    "task_id",
    "repetition",
    "readout",
    "family",
    "question_kind",
    "input",
    "completion",
    "belief",
    "classification",
    "scores",
    "log_path",
    "log_sha256",
    "baseline_shared_response",
)


def rescore_base(rows, config, branch, checkpoint=None):
    """One base response set, scored under each mapping without extra sampling."""
    plan = belief_manipulation_plan(config=config, branch=branch, checkpoint=checkpoint)
    expected = {
        str(s.id): s
        for t in plan.tasks
        for s in (t if isinstance(t, Task) else t()).dataset
    }
    refs = load_experiment_plan(config).contract.evaluation.authority_references
    assert refs is not None
    result = []
    for row in rows:
        sample = expected[row["task_id"]]
        assert sample.metadata is not None
        if sample.input != row["input"]:
            raise ValueError("base prompts differ between mappings")
        result.append(
            {
                **row,
                **sample.metadata,
                "branch": branch,
                "belief": score_belief_answer(
                    row["completion"], str(sample.target), sample.metadata, refs
                ),
                "baseline_shared_response": True,
            }
        )
    return result


def export_plots(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for branch, color in (("A", "#2563eb"), ("B", "#d97706")):
        data = [r for r in report["trajectory"] if r["branch"] == branch]
        if data:
            recall = [r for r in data if r["recall"] is not None]
            axes[0].plot(
                [r["documents_seen"] for r in recall],
                [100 * r["recall"] for r in recall],
                "o-",
                label=branch,
                color=color,
            )
            valid = [r for r in data if r["comprehension_rate"] is not None]
            axes[1].errorbar(
                [r["documents_seen"] for r in valid],
                [100 * r["comprehension_rate"] for r in valid],
                yerr=[100 * (r["comprehension_stderr"] or 0) for r in valid],
                fmt="o-",
                label=branch,
                color=color,
            )
    gaps = [
        s
        for s in report["steps"]
        if s.get("contrast", {}).get("gap_A_minus_B") is not None
    ]
    if gaps:
        axes[2].plot(
            [s["documents_seen"] for s in gaps],
            [100 * s["contrast"]["gap_A_minus_B"] for s in gaps],
            "o-",
            color="#7c3aed",
        )
        intervals = [s for s in gaps if s["contrast"].get("ci95")]
        if intervals:
            axes[2].fill_between(
                [s["documents_seen"] for s in intervals],
                [100 * s["contrast"]["ci95"][0] for s in intervals],
                [100 * s["contrast"]["ci95"][1] for s in intervals],
                alpha=0.15,
                color="#7c3aed",
            )
    axes[2].axhline(0, color="#94a3b8", linewidth=1)
    for ax, title, ylabel in zip(
        axes,
        (
            "Recall across all formats",
            "Comprehension rate on paired tasks",
            "A − B contrast · 95% task bootstrap",
        ),
        ("Correct / all attempts (%)", "Comprehension rate (%)", "Percentage points"),
    ):
        ax.set(title=title, xlabel="Documents seen per branch", ylabel=ylabel)
        ax.grid(alpha=0.15)
    for ax in axes[:2]:
        ax.set_ylim(-3, 103)
        if ax.lines:
            ax.legend()
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"trajectory.{suffix}", dpi=180)
    plt.close(fig)


def _base_rows(config, plan, model, suite, output, branch):
    directory = output / "base" / suite.name
    if len(plan.contract.models) > 1:
        directory = output / model.id / "base" / suite.name
    if not suite.shared_base:
        directory = directory.parent / branch / suite.name
    if not (directory / "eval_completed.json").exists():
        return None
    receipt = json.loads((directory / "execution_receipt.json").read_text())
    for item in (receipt["contract"], receipt["source_snapshot"]):
        if sha(directory / item["path"]) != item["sha256"]:
            raise ValueError("baseline receipt artifact changed")
    if receipt["contract"]["sha256"] != sha(config):
        raise ValueError("baseline execution contract changed")
    target = json.loads(receipt["plan"]["metadata"]["model_target_json"])
    if target.get("model_path") or target.get("sdf_adapter_path"):
        raise ValueError("baseline has an SDF adapter")
    sampled_branch = "A" if suite.shared_base else branch
    current = suite.plan(
        config=config,
        branch=sampled_branch,
        checkpoint=model.id,
        log_dir=str(directory),
    )
    current = replace(current, metadata=receipt["plan"]["metadata"])
    rows = suite.collector(current, directory)
    if suite.name == "belief_manipulation":
        return rescore_base(rows, config, branch, model.id)
    return [
        {**r, "branch": branch, "baseline_shared_response": suite.shared_base}
        for r in rows
    ]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
        )
        writer.writeheader()
        writer.writerows(rows)


def build_report(config, allow_partial=False, *, model_id=None, destination=None):
    config = Path(config)
    plan = load_experiment_plan(config)
    c = plan.contract
    if model_id is None and len(c.models) > 1:
        return build_model_comparison(config, allow_partial)
    model = next(m for m in c.models if model_id is None or m.id == model_id)
    runs = [run for run in plan.runs() if run.shared.checkpoint.id == model.id]
    if len(runs) != 2:
        raise ValueError("report one initialization/shuffle seed pair at a time")
    output = REPO_ROOT / c.output_dir
    directory = Path(destination) if destination else output / "reports"
    if len(c.models) > 1 and destination is None:
        directory = directory / model.id
    directory.mkdir(parents=True, exist_ok=True)
    suites = selected_suites(c.evaluation.suites)
    source_path = REPO_ROOT / c.corpus.directory / "source.json"
    report = {
        "experiment_id": c.experiment_id,
        "model": model.model_dump(mode="json"),
        "settings": c.model_dump(mode="json"),
        "experiment": json.loads((output / "experiment.json").read_text()),
        "steps": [],
        "trajectory": [],
        "missing": [],
        "comparisons": [],
        "belief_gate_status": "threshold_unset"
        if c.evaluation.belief_gate.minimum_accuracy is None
        else "configured",
        "measurement_note": "AST syntax, not functional task correctness. Mixed, invalid and ineligible cases remain visible.",
        "uncertainty": "Conditional on the retained model and evaluation tasks. Question-scenario clusters for belief recall; task clusters for behavior. Repetitions remain within their cluster.",
        "baseline_note": "Shared starting-model responses are rescored under each target mapping."
        if c.execution.evaluate_base
        else "The retained run did not request a shared starting-model evaluation.",
        "corpus_source": json.loads(source_path.read_text())
        if source_path.exists()
        else c.corpus.source.model_dump(),
        "budget": json.loads((output / "budget.json").read_text())
        if (output / "budget.json").exists()
        else None,
        "storage": json.loads((output / "storage_costs.json").read_text())
        if (output / "storage_costs.json").exists()
        else None,
    }
    observations, beliefs, coding = [], {}, {}
    exposures = (
        [0] if c.execution.evaluate_base else []
    ) + c.execution.evaluate_after_documents
    for documents in exposures:
        step = documents // c.training.batch_size_documents
        point = {"step": step, "documents_seen": documents, "branches": {}}
        paired = []
        for run in runs:
            branch = run.branch
            if step == 0:
                rows = {
                    suite.name: _base_rows(config, plan, model, suite, output, branch)
                    for suite in suites
                }
                scale = {
                    "documents_seen": 0,
                    "training_tokens_seen": 0,
                    "optimizer_steps": 0,
                }
            else:
                folder = output / run.shared.run_id / f"step_{step:06d}"
                if not all(
                    (folder / suite.name / "eval_completed.json").exists()
                    for suite in suites
                ):
                    report["missing"].append({"branch": branch, "step": step})
                    continue
                state = json.loads((folder / "checkpoint.json").read_text())
                if (
                    state["run"] != run.describe()
                    or state["sdf_step"] != step
                    or state["documents_seen"] != documents
                ):
                    raise ValueError(
                        "checkpoint model/training/exposure identity differs"
                    )
                rows = {
                    suite.name: suite.collector(
                        receipt_plan(
                            suite.plan, config, branch, folder / suite.name, state
                        ),
                        folder / suite.name,
                    )
                    for suite in suites
                }
                scale = training_scale(state, c.training, REPO_ROOT)
            if any(values is None for values in rows.values()):
                report["missing"].append({"branch": branch, "step": step})
                continue
            for suite, values in rows.items():
                assert values is not None
                for row in values:
                    row.update(step=step, documents_seen=documents, suite=suite)
                observations.extend(values)
            belief_rows = rows.get("belief_manipulation") or []
            python_rows = rows.get("short_python") or []
            for readout in ("mcq", "mcq_flipped", "open_ended"):
                if belief_rows:
                    beliefs[step, branch, readout] = [
                        row for row in belief_rows if row["readout"] == readout
                    ]
            if python_rows:
                coding[step, branch] = python_rows
                paired.extend(python_rows)
            point["branches"][branch] = {
                "belief": summarize_beliefs(belief_rows, c.evaluation)
                if belief_rows
                else None,
                "behavior": behavior_summary(python_rows) if python_rows else None,
                "training": scale,
            }
            rates = cluster_rates(python_rows)
            report["trajectory"].append(
                {
                    "step": step,
                    "documents_seen": documents,
                    "branch": branch,
                    "recall": sum(row["belief"]["correct"] for row in belief_rows)
                    / len(belief_rows)
                    if belief_rows
                    else None,
                    "comprehension_rate": sum(rates.values()) / len(rates)
                    if rates
                    else None,
                    "comprehension_stderr": mean_stderr(rates.values()),
                    "eligible_tasks": len(rates),
                }
            )
        if set(point["branches"]) == {"A", "B"} and paired:
            assert c.evaluation.contrast_estimator is not None
            point["contrast"] = contrast(
                paired,
                c.evaluation.contrast_estimator,
                resamples=c.evaluation.bootstrap_resamples,
                seed=c.evaluation.bootstrap_seed,
            )
            for row in report["trajectory"]:
                if row["step"] == step:
                    row["comprehension_rate"] = point["contrast"][
                        f"universe_{row['branch']}_rate"
                    ]
                    row["comprehension_stderr"] = point["contrast"][
                        f"universe_{row['branch']}_rate_stderr"
                    ]
                    row["eligible_tasks"] = point["contrast"]["tasks_eligible_in_both"]
            effects, _ = task_effects(paired)
            atomic_json(directory / f"step_{step:06d}_tasks.json", effects)
        if point["branches"]:
            report["steps"].append(point)
    report["complete"] = not report["missing"]
    if not report["complete"] and not allow_partial:
        raise ValueError(f"incomplete requested evaluation matrix: {report['missing']}")
    report["analysis_code"] = git_provenance(REPO_ROOT)
    report["analysis_source_sha256"] = source_snapshot(
        REPO_ROOT, directory / "analysis_source.tar.gz"
    )
    for point in report["steps"]:
        for branch, cell in point["branches"].items():
            values = coding.get((point["step"], branch), [])
            if values:
                cell["behavior_breakdowns"] = {
                    axis: {
                        group: behavior_summary(rows) for group, rows in groups.items()
                    }
                    for axis, groups in behavior_groups(values).items()
                }
    if c.evaluation.belief_gate.minimum_accuracy is not None:
        statuses = [
            cell["belief"]["manipulation_gate"]["status"]
            for point in report["steps"]
            for cell in point["branches"].values()
            if cell["belief"]
        ]
        report["belief_gate_status"] = (
            "failed"
            if "failed" in statuses
            else "passed"
            if statuses and report["complete"]
            else "incomplete"
        )
    if beliefs:
        recall, pairs, changes, order = summarize_recall(beliefs, report)
        report.update(
            recall=recall,
            belief_pairs=pairs,
            belief_changes=changes,
            answer_order=order,
        )
    reference = min((key[0] for key in coding), default=None)
    for (step, branch), later in coding.items():
        if step != reference and (reference, branch) in coding:
            report["comparisons"].append(
                {
                    "step": step,
                    "branch": branch,
                    "reference_step": reference,
                    **paired_delta(
                        cluster_rates(coding[reference, branch]),
                        cluster_rates(later),
                        resamples=c.evaluation.bootstrap_resamples,
                        seed=c.evaluation.bootstrap_seed,
                    ),
                }
            )
    atomic_json(directory / "report.json", report)
    with (directory / "samples.jsonl").open("w") as stream:
        for row in observations:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name, values in [
        ("trajectory", report["trajectory"]),
        ("recall", report.get("recall", [])),
        ("comparisons", report["comparisons"]),
        (
            "contrasts",
            [
                {
                    "step": p["step"],
                    "documents_seen": p["documents_seen"],
                    **p["contrast"],
                }
                for p in report["steps"]
                if "contrast" in p
            ],
        ),
    ]:
        write_csv(directory / f"{name}.csv", values)
    export_plots(report, directory)
    payload = json.dumps(
        {
            "report": report,
            "samples": [
                {k: row[k] for k in VIEWER_SAMPLE_FIELDS if k in row}
                for row in observations
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("<", "\\u003c")
    template = Path(__file__).with_name("trajectory.html").read_text()
    (directory / "viewer.html").write_text(
        template.replace("__DATA__", payload)
        .replace("__STYLE__", Path(__file__).with_name("report.css").read_text())
        .replace("__SCRIPT__", Path(__file__).with_name("report.js").read_text())
    )
    print(
        json.dumps(
            {
                "event": "report_updated",
                "path": str(directory / "viewer.html"),
                "complete": report["complete"],
                "available_cells": len(report["trajectory"]),
            }
        ),
        flush=True,
    )
    return report


def build_model_comparison(config, allow_partial=False):
    plan = load_experiment_plan(config)
    directory = REPO_ROOT / plan.contract.output_dir / "reports"
    reports = {
        model.id: build_report(config, allow_partial, model_id=model.id)
        for model in plan.contract.models
    }
    rows = [
        {"model_id": model_id, "rl_step": report["model"]["step"], **row}
        for model_id, report in reports.items()
        for row in report["trajectory"]
    ]
    atomic_json(directory / "models.json", {"models": reports, "trajectory": rows})
    write_csv(directory / "models.csv", rows)
    payload = json.dumps(
        {"rows": rows, "models": list(reports)}, ensure_ascii=False
    ).replace("<", "\\u003c")
    template = Path(__file__).with_name("comparison.html").read_text()
    (directory / "viewer.html").write_text(
        template.replace("__DATA__", payload)
        .replace("__STYLE__", Path(__file__).with_name("report.css").read_text())
        .replace("__SCRIPT__", Path(__file__).with_name("report.js").read_text())
    )
    return {
        "complete": all(report["complete"] for report in reports.values()),
        "models": reports,
    }
