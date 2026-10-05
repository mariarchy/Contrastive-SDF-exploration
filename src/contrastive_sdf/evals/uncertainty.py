"""Task-cluster standard errors for means and eligible-case ratios."""

import math
from collections import defaultdict
from collections.abc import Hashable, Iterable

from inspect_ai.scorer import MetricProtocol, SampleScore, metric, value_to_float


def influence_stderr(influences: list[float]) -> float | None:
    """Finite-cluster correction for already normalized cluster influences."""
    n = len(influences)
    return math.sqrt(n / (n - 1) * sum(v * v for v in influences)) if n >= 2 else None


def cluster_ratio_stderr(
    contributions: Iterable[tuple[Hashable, float, float]],
) -> float | None:
    clusters: dict[Hashable, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for key, numerator, denominator in contributions:
        clusters[key][0] += numerator
        clusters[key][1] += denominator
    total = sum(v[1] for v in clusters.values())
    if total == 0:
        return None
    average = sum(v[0] for v in clusters.values()) / total
    return influence_stderr([(y - average * d) / total for y, d in clusters.values()])


def mean_stderr(values: Iterable[float]) -> float | None:
    return cluster_ratio_stderr((i, v, 1.0) for i, v in enumerate(values))


@metric
def task_cluster_stderr(eligible_only: bool = False) -> MetricProtocol:
    """Inspect SE within one repetition; reports cluster all repetitions."""

    def compute(scores: list[SampleScore]) -> float:
        convert = value_to_float()
        contributions = []
        for i, sample in enumerate(scores):
            selected = (
                not eligible_only
                or (sample.score.metadata or {}).get("eligible") is True
            )
            key = (sample.sample_metadata or {}).get("base_task_id", sample.sample_id)
            contributions.append(
                (
                    key if key is not None else i,
                    convert(sample.score.value) if selected else 0.0,
                    float(selected),
                )
            )
        value = cluster_ratio_stderr(contributions)
        return value if value is not None else float("nan")

    return compute
