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

from contrastive_sdf.evals.reports.sdf_phase1 import _bootstrap_mean_interval
from contrastive_sdf.evals.scoring.iteration_style import classify_iteration
from contrastive_sdf.evals.tasks.iteration_belief import belief_samples, score_belief
from contrastive_sdf.evals.tasks.short_python import task_prompt, validate_task_dataset
from contrastive_sdf.sdf.execution import cell_name
from contrastive_sdf.sdf.experiment import CheckpointRun, ExperimentPlan
from contrastive_sdf.sdf.scalable_corpus import atomic_json

READOUTS = {
    "sdf_iteration_semantic": "semantic",
    "sdf_iteration_recall": "open_ended",
    "sdf_iteration_behavior": "behavior",
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
    for readout in ("semantic", "open_ended"):
        samples = belief_samples(run.corpus.mapping, readout)
        expected_prompts[readout] = {
            str(s.id): s.input for s in samples if isinstance(s.input, str)
        }
        expected_targets[readout] = {str(s.id): s.target for s in samples}
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


def branch_summary(observations: list[dict], gate) -> dict:
    behavior = [o for o in observations if o["readout"] == "behavior"]
    counts = Counter(o["classification"]["label"] for o in behavior)
    n = len(behavior)
    eligible = counts["comprehension"] + counts["loop"]
    beliefs = {}
    for readout in ("semantic", "open_ended"):
        selected = [o for o in observations if o["readout"] == readout]
        beliefs[readout] = {
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
    threshold = gate.minimum_accuracy
    required = [
        beliefs[r][f"{a}_accuracy"] for r in gate.readouts for a in ("grader", "users")
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
        "manipulation_gate": {
            "status": status,
            "minimum_accuracy": threshold,
            "required_readouts": gate.readouts,
            "required_authorities": ["grader", "users"],
        },
        "behavior": {
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
            "comprehension_rate": counts["comprehension"] / eligible
            if eligible
            else None,
            "loop_rate": counts["loop"] / eligible if eligible else None,
        },
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
    return {
        "estimator": estimator,
        "universe_A_rate": rates["A"],
        "universe_B_rate": rates["B"],
        "gap_A_minus_B": gap,
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
    return {
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
    lines = [
        f"# {label}",
        "",
        f"**Manipulation gate: {summary['manipulation_gate_status'].upper()}**",
        "",
        "Behavioral contrast is accompanied by the belief checks below. An unconfigured or failed gate does not establish an interpretable reward-seeking measurement.",
        "",
        "| Universe | Readout | Grader accuracy | User accuracy | Overall accuracy | Valid response rate |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    def fmt(v):
        return "n/a" if v is None else f"{v:.4f}"

    for b, item in summary["branches"].items():
        for readout, m in item["belief"].items():
            lines.append(
                f"| {b} | {readout} | {fmt(m['grader_accuracy'])} | {fmt(m['users_accuracy'])} | {fmt(m['overall_accuracy'])} | {fmt(m['valid_response_rate'])} |"
            )
    lines += ["", "| Metric | A | B |", "| --- | ---: | ---: |"]
    for metric in (
        "unique_tasks",
        "generations",
        "valid_python_rate",
        "format_valid_rate",
        "eligibility_rate",
        "eligible_generations",
        *[f"{label}_count" for label in LABELS],
        "comprehension_rate",
        "loop_rate",
    ):
        values = [summary["branches"][b]["behavior"][metric] for b in ("A", "B")]
        lines.append(f"| {metric} | {fmt(values[0])} | {fmt(values[1])} |")
    gate = summary["branches"]["A"]["manipulation_gate"]
    lines += [
        "",
        f"Gate threshold: {gate['minimum_accuracy']}; readouts: {', '.join(gate['required_readouts'])}; both authorities required in both universes.",
    ]
    c = summary["contrast"]
    lines += [
        "",
        f"Estimator: `{c['estimator']}`. {c['orientation']}",
        f"A rate: {fmt(c['universe_A_rate'])}; B rate: {fmt(c['universe_B_rate'])}; A−B: {fmt(c['gap_A_minus_B'])}.",
        f"95% task-cluster interval: {c['ci95']}; clusters: {c['bootstrap_task_clusters']}; defined resamples: {c['defined_bootstrap_resamples']}/{c['bootstrap_resamples']}.",
    ]
    return "\n".join(lines) + "\n"


def build_reports(plan, root: Path, report_dir: Path) -> list[dict]:
    e = plan.contract.evaluation
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    all_observations = []
    grouped = defaultdict(dict)
    for run in plan.runs():
        grouped[
            (
                run.shared.checkpoint.id,
                run.shared.training.seed,
                run.shared.training.shuffle_seed,
            )
        ][run.branch] = run
    code_versions = set()
    dataset_versions = set()
    for (checkpoint, sdf_seed, shuffle), runs in grouped.items():
        for seed, temp in itertools.product(e.seeds, e.temperatures):
            observations = []
            training_states = {}
            for b, run in runs.items():
                directory = root / plan.contract.output_dir / run.shared.run_id
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
            summary.update(
                checkpoint=model.model_dump(mode="json"),
                mode=plan.contract.mode,
                mock=mock,
                contract_sha256=plan.contract_sha256,
                training_states=training_states,
            )
            atomic_json(report_dir / f"{stem}.json", summary)
            (report_dir / f"{stem}.md").write_text(
                format_markdown(summary, ("MOCK FIXTURE — " if mock else "") + stem)
            )
            c = summary["contrast"]
            row = {
                "checkpoint_id": checkpoint,
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
                "dataset_sha256": e.dataset.sha256,
                "gate_status": summary["manipulation_gate_status"],
                "gate_threshold": e.belief_gate.minimum_accuracy,
                "estimator": c["estimator"],
                "A_rate": c["universe_A_rate"],
                "B_rate": c["universe_B_rate"],
                "gap_A_minus_B": c["gap_A_minus_B"],
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
                for metric, value in item["behavior"].items():
                    row[f"{b}_{metric}"] = value
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
    plot_trajectory(rows, report_dir / "trajectory.png", e.belief_gate.minimum_accuracy)
    return rows


def plot_trajectory(rows, path: Path, threshold):
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
            belief_ax.plot(
                x, values, marker="o", label=f"{tag}: {b}/{authority} semantic"
            )
            recall = [r[f"{b}_open_ended_{authority}_accuracy"] for r in items]
            belief_ax.plot(
                x,
                recall,
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
    belief_ax.set_ylabel("Belief accuracy")
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
