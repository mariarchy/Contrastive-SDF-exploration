"""Aggregation for the authority-conditioned coding control."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ActionObservation:
    task_id: str
    world: str
    authority: str
    fact_order: str
    target_style: str
    observed_style: str
    format_valid: bool
    python_valid: bool
    compliant: bool


def _rate(values: Iterable[bool]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def _compliance_by(
    observations: Sequence[ActionObservation],
    fields: tuple[str, ...],
) -> dict[str, float]:
    groups: dict[tuple[str, ...], list[ActionObservation]] = defaultdict(list)
    for observation in observations:
        key = tuple(str(getattr(observation, field)) for field in fields)
        groups[key].append(observation)
    return {
        "/".join(key): _rate(item.compliant for item in items) or 0.0
        for key, items in sorted(groups.items())
    }


def _inversion_rate(
    pairs: Sequence[tuple[ActionObservation, ActionObservation]],
) -> float | None:
    return _rate(
        left.observed_style in {"single", "double"}
        and right.observed_style in {"single", "double"}
        and left.observed_style != right.observed_style
        for left, right in pairs
    )


def summarize_action_control(
    observations: Sequence[ActionObservation],
) -> dict[str, object]:
    if not observations:
        raise ValueError("Cannot summarize an empty action-control result set")

    keyed: dict[tuple[str, str, str], ActionObservation] = {}
    for observation in observations:
        key = (observation.task_id, observation.world, observation.authority)
        if key in keyed:
            raise ValueError(f"Duplicate task/world/authority observation: {key!r}")
        keyed[key] = observation

    task_ids = sorted({item.task_id for item in observations})
    world_pairs = [
        (keyed[(task_id, "A", authority)], keyed[(task_id, "B", authority)])
        for task_id in task_ids
        for authority in ("grader", "user")
        if (task_id, "A", authority) in keyed and (task_id, "B", authority) in keyed
    ]
    authority_pairs = [
        (keyed[(task_id, world, "grader")], keyed[(task_id, world, "user")])
        for task_id in task_ids
        for world in ("A", "B")
        if (task_id, world, "grader") in keyed and (task_id, world, "user") in keyed
    ]

    return {
        "sample_count": len(observations),
        "format_valid_rate": _rate(item.format_valid for item in observations),
        "python_valid_rate": _rate(item.python_valid for item in observations),
        "preference_compliance_rate": _rate(item.compliant for item in observations),
        "compliance_by_world": _compliance_by(observations, ("world",)),
        "compliance_by_authority": _compliance_by(observations, ("authority",)),
        "compliance_by_cell": _compliance_by(observations, ("world", "authority")),
        "compliance_by_fact_order": _compliance_by(observations, ("fact_order",)),
        "observed_style_distribution": dict(
            sorted(Counter(item.observed_style for item in observations).items())
        ),
        "world_pair_count": len(world_pairs),
        "world_inversion_rate": _inversion_rate(world_pairs),
        "world_paired_correct_rate": _rate(
            left.compliant and right.compliant for left, right in world_pairs
        ),
        "authority_pair_count": len(authority_pairs),
        "authority_inversion_rate": _inversion_rate(authority_pairs),
        "authority_paired_correct_rate": _rate(
            left.compliant and right.compliant for left, right in authority_pairs
        ),
    }


def _score_correct(score: object) -> bool:
    value = getattr(score, "value", None)
    return value == "C" or value == 1 or value is True


def observation_from_sample(sample: object) -> ActionObservation:
    metadata = getattr(sample, "metadata", None) or {}
    scores = getattr(sample, "scores", None) or {}
    required_scores = ("format_validity", "python_validity", "preference_compliance")
    missing = [name for name in required_scores if name not in scores]
    if missing:
        raise ValueError(
            f"Eval sample {getattr(sample, 'id', '<unknown>')!r} lacks {missing}"
        )

    compliance_score = scores["preference_compliance"]
    score_metadata = getattr(compliance_score, "metadata", None) or {}
    return ActionObservation(
        task_id=str(metadata["base_task_id"]),
        world=str(metadata["world"]),
        authority=str(metadata["authority"]),
        fact_order=str(metadata["fact_order"]),
        target_style=str(getattr(sample, "target", "")).casefold(),
        observed_style=str(score_metadata.get("observed_style", "unknown")),
        format_valid=_score_correct(scores["format_validity"]),
        python_valid=_score_correct(scores["python_validity"]),
        compliant=_score_correct(compliance_score),
    )
