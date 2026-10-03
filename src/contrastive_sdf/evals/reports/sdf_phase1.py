"""Pure aggregation for paired post-SDF belief and behavior readouts."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypedDict

Rate = float | None


class BeliefReadoutSummary(TypedDict):
    samples: int
    accuracy_A: Rate
    accuracy_B: Rate
    valid_A: Rate
    valid_B: Rate
    accuracy_by_branch_authority: dict[str, Rate]
    complete_pairs: int
    paired_inversion_rate: Rate
    paired_correct_rate: Rate


class BehaviorSummary(TypedDict):
    samples: int
    format_valid_rate: dict[str, Rate]
    python_valid_rate: dict[str, Rate]
    eligible_rate: dict[str, Rate]
    mean_double_fraction: dict[str, Rate]
    mean_grader_aligned_fraction: dict[str, Rate]
    complete_eligible_pairs: int
    bootstrap_task_clusters: int
    paired_double_gap_A_minus_B: Rate
    paired_double_gap_95pct_bootstrap: tuple[float, float] | None
    paired_grader_alignment_effect: Rate


class SDFPhase1Summary(TypedDict):
    belief: dict[str, BeliefReadoutSummary]
    behavior: BehaviorSummary


@dataclass(frozen=True)
class BeliefObservation:
    readout: str
    branch: str
    repetition: str
    sample_id: str
    authority: str
    target: str
    observed: str
    correct: bool

    @property
    def valid(self) -> bool:
        return self.observed in {"single", "double"}


@dataclass(frozen=True)
class BehaviorObservation:
    branch: str
    repetition: str
    sample_id: str
    expected_grader_style: str
    format_valid: bool
    python_valid: bool
    eligible: bool
    double_fraction: float
    n_double_literals: int
    n_single_literals: int

    @property
    def grader_aligned_fraction(self) -> float:
        if self.expected_grader_style == "double":
            return self.double_fraction
        if self.expected_grader_style == "single":
            return 1.0 - self.double_fraction
        raise ValueError(f"Unknown grader style: {self.expected_grader_style!r}")


def _rate(values: Iterable[bool]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def _paired[Observation: (BeliefObservation, BehaviorObservation)](
    observations: Sequence[Observation],
) -> list[tuple[Observation, Observation]]:
    keyed: dict[tuple[str, str], dict[str, Observation]] = defaultdict(dict)
    for observation in observations:
        key = (observation.repetition, observation.sample_id)
        if observation.branch in keyed[key]:
            raise ValueError(f"Duplicate {observation.branch} observation for {key!r}")
        keyed[key][observation.branch] = observation
    return [
        (branches["A"], branches["B"])
        for branches in keyed.values()
        if set(branches) == {"A", "B"}
    ]


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


def _belief_summary(
    observations: Sequence[BeliefObservation],
) -> dict[str, BeliefReadoutSummary]:
    result: dict[str, BeliefReadoutSummary] = {}
    for readout in ("semantic", "open_ended"):
        selected = [item for item in observations if item.readout == readout]
        pairs = _paired(selected)
        result[readout] = {
            "samples": len(selected),
            "accuracy_A": _rate(
                item.correct for item in selected if item.branch == "A"
            ),
            "accuracy_B": _rate(
                item.correct for item in selected if item.branch == "B"
            ),
            "valid_A": _rate(item.valid for item in selected if item.branch == "A"),
            "valid_B": _rate(item.valid for item in selected if item.branch == "B"),
            "accuracy_by_branch_authority": {
                f"{branch}/{authority}": _rate(
                    item.correct
                    for item in selected
                    if item.branch == branch and item.authority == authority
                )
                for branch in ("A", "B")
                for authority in ("grader", "user")
            },
            "complete_pairs": len(pairs),
            "paired_inversion_rate": _rate(
                left.valid and right.valid and left.observed != right.observed
                for left, right in pairs
            ),
            "paired_correct_rate": _rate(
                left.correct and right.correct for left, right in pairs
            ),
        }
    return result


def _behavior_summary(
    observations: Sequence[BehaviorObservation],
) -> BehaviorSummary:
    by_branch = {
        branch: [item for item in observations if item.branch == branch]
        for branch in ("A", "B")
    }
    pairs = [
        (left, right)
        for left, right in _paired(observations)
        if left.eligible and right.eligible
    ]
    paired_gaps = [
        left.double_fraction - right.double_fraction for left, right in pairs
    ]
    gaps_by_task: dict[str, list[float]] = defaultdict(list)
    for (left, _), gap in zip(pairs, paired_gaps, strict=True):
        gaps_by_task[left.sample_id].append(gap)
    task_level_gaps = [sum(gaps) / len(gaps) for gaps in gaps_by_task.values()]
    paired_alignment_gaps = [
        (left.grader_aligned_fraction + right.grader_aligned_fraction) / 2 - 0.5
        for left, right in pairs
    ]
    return {
        "samples": len(observations),
        "format_valid_rate": {
            branch: _rate(item.format_valid for item in items)
            for branch, items in by_branch.items()
        },
        "python_valid_rate": {
            branch: _rate(item.python_valid for item in items)
            for branch, items in by_branch.items()
        },
        "eligible_rate": {
            branch: _rate(item.eligible for item in items)
            for branch, items in by_branch.items()
        },
        "mean_double_fraction": {
            branch: _mean(item.double_fraction for item in items if item.eligible)
            for branch, items in by_branch.items()
        },
        "mean_grader_aligned_fraction": {
            branch: _mean(
                item.grader_aligned_fraction for item in items if item.eligible
            )
            for branch, items in by_branch.items()
        },
        "complete_eligible_pairs": len(pairs),
        "bootstrap_task_clusters": len(task_level_gaps),
        "paired_double_gap_A_minus_B": _mean(paired_gaps),
        "paired_double_gap_95pct_bootstrap": _bootstrap_mean_interval(task_level_gaps),
        # Zero means no contrastive alignment; positive means both branches move
        # toward their own inverse grader targets.
        "paired_grader_alignment_effect": _mean(paired_alignment_gaps),
    }


def summarize_sdf_phase1(
    belief_observations: Sequence[BeliefObservation],
    behavior_observations: Sequence[BehaviorObservation],
) -> SDFPhase1Summary:
    if not belief_observations:
        raise ValueError("Cannot summarize without belief observations")
    if not behavior_observations:
        raise ValueError("Cannot summarize without behavior observations")
    branches = {item.branch for item in (*belief_observations, *behavior_observations)}
    if branches != {"A", "B"}:
        raise ValueError(f"Expected A and B observations, got {sorted(branches)}")
    return {
        "belief": _belief_summary(belief_observations),
        "behavior": _behavior_summary(behavior_observations),
    }


def _score_correct(score: object) -> bool:
    value = getattr(score, "value", None)
    return value == "C" or value == 1 or value is True


def belief_observation_from_sample(
    sample: object,
    *,
    readout: str,
    branch: str,
    repetition: str,
) -> BeliefObservation:
    score_name = "sdf_exact_quote_choice" if readout == "semantic" else "quote_stance"
    scores = getattr(sample, "scores", None) or {}
    if score_name not in scores:
        raise ValueError(
            f"Eval sample {getattr(sample, 'id', '<unknown>')!r} lacks {score_name}"
        )
    score = scores[score_name]
    score_metadata = getattr(score, "metadata", None) or {}
    if readout == "semantic":
        observed = str(getattr(score, "answer", "") or "").casefold()
    else:
        endorsed = score_metadata.get("endorsed", [])
        observed = str(endorsed[0]).casefold() if len(endorsed) == 1 else "invalid"
    metadata = getattr(sample, "metadata", None) or {}
    return BeliefObservation(
        readout=readout,
        branch=branch,
        repetition=repetition,
        sample_id=str(getattr(sample, "id", "")),
        authority=str(metadata.get("authority", "")),
        target=str(getattr(sample, "target", "") or "").casefold(),
        observed=observed,
        correct=_score_correct(score),
    )


def behavior_observation_from_sample(
    sample: object,
    *,
    branch: str,
    repetition: str,
    expected_grader_style: str,
) -> BehaviorObservation:
    scores = getattr(sample, "scores", None) or {}
    required = (
        "ungrouped_format_validity",
        "ungrouped_python_validity",
        "executable_quote_fraction",
    )
    missing = [name for name in required if name not in scores]
    if missing:
        raise ValueError(
            f"Eval sample {getattr(sample, 'id', '<unknown>')!r} lacks {missing}"
        )
    quote_score = scores["executable_quote_fraction"]
    metadata = getattr(quote_score, "metadata", None) or {}
    return BehaviorObservation(
        branch=branch,
        repetition=repetition,
        sample_id=str(getattr(sample, "id", "")),
        expected_grader_style=expected_grader_style,
        format_valid=_score_correct(scores["ungrouped_format_validity"]),
        python_valid=_score_correct(scores["ungrouped_python_validity"]),
        eligible=metadata.get("eligible") is True,
        double_fraction=float(getattr(quote_score, "value", 0.0)),
        n_double_literals=int(metadata.get("n_double_literals", 0)),
        n_single_literals=int(metadata.get("n_single_literals", 0)),
    )
