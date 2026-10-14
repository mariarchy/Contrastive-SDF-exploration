"""Cluster bootstrap and matched task/scenario comparisons."""

import random
from collections import defaultdict
from collections.abc import Sequence

from contrastive_sdf.evals.uncertainty import mean_stderr


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float] | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0], values[0]
    generator = random.Random(seed)
    means = sorted(
        sum(generator.choice(values) for _ in values) / len(values)
        for _ in range(resamples)
    )
    return means[int(0.025 * resamples)], means[int(0.975 * resamples)]


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


def percent(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}"
