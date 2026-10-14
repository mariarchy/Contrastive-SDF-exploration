"""Inspect metrics whose denominator uses the recorded eligibility policy."""

from inspect_ai.scorer import MetricProtocol, SampleScore, metric


def eligible_values(scores: list[SampleScore]) -> list[float]:
    values: list[float] = []
    for sample_score in scores:
        score = sample_score.score
        if (score.metadata or {}).get("eligible") is not True:
            continue
        if not isinstance(score.value, int | float):
            raise TypeError("eligible iteration scores must be numeric")
        values.append(float(score.value))
    return values


@metric
def eligible_mean() -> MetricProtocol:
    """Comprehension fraction over AST-eligible generations."""

    def compute(scores: list[SampleScore]) -> float:
        values = eligible_values(scores)
        return sum(values) / len(values) if values else 0.0

    return compute
