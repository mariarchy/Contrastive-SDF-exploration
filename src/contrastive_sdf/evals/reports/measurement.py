"""Belief, exposure and per-task diagnostics for the existing SDF experiment."""

import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from contrastive_sdf.evals.uncertainty import cluster_ratio_stderr, mean_stderr


def belief_strength(observations: list[dict]) -> dict:
    result = {}
    for readout, authority in itertools.product(
        ("semantic", "open_ended"), ("grader", "users")
    ):
        selected = [
            o
            for o in observations
            if o["readout"] == readout and o["authority"] == authority
        ]
        tasks = defaultdict(list)
        for observation in selected:
            belief = observation["belief"]
            if belief["correct"] and not belief["valid"]:
                raise ValueError("a correct belief answer must be scorable")
            tasks[observation["task_id"]].append(belief)
        metrics: dict[str, Any] = {
            "samples": len(selected),
            "unique_questions": len(tasks),
        }
        for label in ("target", "opposing", "unscorable"):

            def matches(belief, label=label):
                return (
                    belief["correct"]
                    if label == "target"
                    else belief["valid"] and not belief["correct"]
                    if label == "opposing"
                    else not belief["valid"]
                )

            count = sum(matches(o["belief"]) for o in selected)
            metrics[f"{label}_count"] = count
            metrics[f"{label}_rate"] = count / len(selected) if selected else None
            metrics[f"{label}_rate_stderr"] = cluster_ratio_stderr(
                (o["task_id"], float(matches(o["belief"])), 1.0) for o in selected
            )
        agreements, valid_pairs, all_pairs = [], [], []
        for task_id, answers in tasks.items():
            counts = Counter(bool(a["correct"]) for a in answers if a["valid"])
            valid = math.comb(sum(counts.values()), 2)
            total = math.comb(len(answers), 2)
            agreements.append(
                (
                    task_id,
                    float(sum(math.comb(n, 2) for n in counts.values())),
                    float(valid),
                )
            )
            valid_pairs.append(valid)
            all_pairs.append(total)
        denominator = sum(valid_pairs)
        metrics.update(
            valid_repeat_pairs=denominator,
            repeat_pairs=sum(all_pairs),
            repeat_agreement_rate=sum(y for _, y, _ in agreements) / denominator
            if denominator
            else None,
            repeat_agreement_rate_stderr=cluster_ratio_stderr(agreements),
            valid_repeat_pair_rate=denominator / sum(all_pairs)
            if sum(all_pairs)
            else None,
            valid_repeat_pair_rate_stderr=cluster_ratio_stderr(
                (key, float(v), float(n))
                for key, v, n in zip(tasks, valid_pairs, all_pairs)
            ),
        )
        result.setdefault(readout, {})[authority] = metrics
    return result


def training_scale(state: dict, settings, root: Path) -> dict:
    training = state.get("training", {})
    steps = state.get("sdf_step", training.get("steps"))
    directory = state.get("log_dir")
    metrics = []
    if directory:
        path = root / directory / "metrics.jsonl"
        if path.exists():
            metrics = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
            metrics = [r for r in metrics if steps is None or r["step"] <= steps]
    tokens = sum(row["batch_tokens"] for row in metrics)
    warmup = settings.optimizer.warmup_steps
    latest = metrics[-1] if metrics else {}
    return {
        "documents_seen": training.get("documents"),
        "unique_documents_seen": training.get("unique_documents"),
        "documents_by_bucket": training.get("documents_by_bucket"),
        "training_tokens_seen": latest.get(
            "elapsed_tokens", training.get("effective_tokens")
        ),
        "document_tokens": training.get("document_tokens"),
        "optimizer_steps": steps,
        "warmup_steps": warmup,
        "warmup_fraction": min(steps / warmup, 1.0)
        if steps is not None and warmup
        else None,
        "post_warmup_updates": max(steps - warmup, 0) if steps is not None else None,
        "current_learning_rate": latest.get("learning_rate"),
        "peak_learning_rate": settings.optimizer.learning_rate,
        "learning_rate_fraction_of_peak": latest["learning_rate"]
        / settings.optimizer.learning_rate
        if "learning_rate" in latest
        else None,
        "first_train_nll": metrics[0].get("train_mean_nll") if metrics else None,
        "latest_train_nll": latest.get("train_mean_nll"),
        "token_weighted_train_nll": sum(
            r["train_mean_nll"] * r["batch_tokens"] for r in metrics
        )
        / tokens
        if tokens and all("train_mean_nll" in r for r in metrics)
        else None,
        "elapsed_training_seconds": latest.get("elapsed_seconds"),
        "cost_usd": state.get("cost_usd"),
        "cost_status": state.get("cost_status", "unavailable"),
        "training_log_dir": directory,
    }


def task_effects(
    observations: list[dict], families: dict | None = None
) -> tuple[list[dict], dict]:
    grouped = defaultdict(lambda: {"A": [], "B": []})
    for observation in observations:
        if observation["readout"] == "behavior":
            grouped[observation["task_id"]][observation["branch"]].append(
                observation["classification"]
            )
    rows = []
    validity, eligibility, gaps = [], [], []
    for task_id, branches in sorted(grouped.items()):
        row = {"task_id": task_id, "family": (families or {}).get(task_id)}
        for branch, answers in branches.items():
            counts = Counter(a["label"] for a in answers)
            eligible = sum(a["eligible"] for a in answers)
            row.update(
                {
                    f"{branch}_generations": len(answers),
                    f"{branch}_eligible": eligible,
                    f"{branch}_comprehension_rate": counts["comprehension"] / eligible
                    if eligible
                    else None,
                }
            )
            row.update(
                {
                    f"{branch}_{label}_count": counts[label]
                    for label in (
                        "comprehension",
                        "loop",
                        "mixed",
                        "ineligible",
                        "invalid",
                    )
                }
            )
        a, b = row["A_comprehension_rate"], row["B_comprehension_rate"]
        row["gap_A_minus_B"] = a - b if a is not None and b is not None else None
        if row["gap_A_minus_B"] is not None:
            gaps.append(row["gap_A_minus_B"])
        for field, values in (("python_valid", validity), ("eligible", eligibility)):
            if branches["A"] and branches["B"]:
                values.append(
                    sum(a[field] for a in branches["A"]) / len(branches["A"])
                    - sum(a[field] for a in branches["B"]) / len(branches["B"])
                )
        rows.append(row)
    summary: dict[str, Any] = {"paired_eligible_tasks": len(gaps)}
    for label, predicate in (
        ("positive", lambda v: v > 0),
        ("zero", lambda v: v == 0),
        ("negative", lambda v: v < 0),
    ):
        indicators = [int(predicate(v)) for v in gaps]
        summary[f"{label}_task_count"] = sum(indicators)
        summary[f"{label}_task_rate"] = (
            sum(indicators) / len(indicators) if indicators else None
        )
        summary[f"{label}_task_rate_stderr"] = mean_stderr(indicators)
    for name, values in (
        ("valid_python_gap", validity),
        ("eligibility_gap", eligibility),
    ):
        summary[name] = sum(values) / len(values) if values else None
        summary[f"{name}_stderr"] = mean_stderr(values)
    return rows, summary
