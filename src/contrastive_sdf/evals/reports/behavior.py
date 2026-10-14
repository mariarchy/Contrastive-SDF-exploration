"""Existing AST categories and paired-task behavior estimators."""

import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from contrastive_sdf.evals.reports.statistics import _bootstrap_mean_interval
from contrastive_sdf.evals.uncertainty import (
    cluster_ratio_stderr,
    influence_stderr,
    mean_stderr,
)

LABELS = ("comprehension", "loop", "mixed", "ineligible", "invalid")


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
        "mixed_rate": counts["mixed"] / n if n else None,
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
    summary["mixed_rate_stderr"] = cluster_ratio_stderr(
        (o["task_id"], float(o["classification"]["label"] == "mixed"), 1.0)
        for o in behavior
    )
    return summary


def behavior_groups(observations: list[dict]) -> dict[str, dict[str, list[dict]]]:
    """Group complete task clusters using dataset labels, never outcome labels."""
    behavior = [o for o in observations if o["readout"] == "behavior"]
    labels_by_task = {}
    for o in behavior:
        labels = (o.get("source_set"), o.get("family"))
        if o["task_id"] in labels_by_task and labels_by_task[o["task_id"]] != labels:
            raise ValueError("inconsistent source_set/family within a task cluster")
        labels_by_task[o["task_id"]] = labels
    return {
        axis: {
            value: [o for o in behavior if o.get(field) == value]
            for value in sorted(
                {o[field] for o in behavior if o.get(field) is not None}
            )
        }
        for axis, field in (("by_source_set", "source_set"), ("by_stratum", "family"))
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
