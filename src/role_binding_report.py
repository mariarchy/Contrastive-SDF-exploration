"""Pure aggregation logic for paired role-binding control results."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from collections.abc import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RoleBindingObservation:
    pair_id: str
    world: str
    authority: str
    label_pair: str
    fact_order: str
    valid_answers: tuple[str, ...]
    target: str
    answer: str

    @property
    def valid(self) -> bool:
        return self.answer in self.valid_answers

    @property
    def correct(self) -> bool:
        return self.answer == self.target


def _rate(values: Iterable[bool]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def _group_rates(
    observations: Sequence[RoleBindingObservation],
    fields: tuple[str, ...],
) -> dict[str, float]:
    groups: dict[tuple[str, ...], list[RoleBindingObservation]] = defaultdict(list)
    for observation in observations:
        key = tuple(str(getattr(observation, field)) for field in fields)
        groups[key].append(observation)
    return {
        "/".join(key): _rate(item.correct for item in items) or 0.0
        for key, items in sorted(groups.items())
    }


def summarize_role_binding(
    observations: Sequence[RoleBindingObservation],
) -> dict[str, object]:
    if not observations:
        raise ValueError("Cannot summarize an empty role-binding result set")

    duplicate_keys = [
        key
        for key, count in Counter(
            (item.pair_id, item.world) for item in observations
        ).items()
        if count > 1
    ]
    if duplicate_keys:
        raise ValueError(
            "Duplicate pair/world observations; pass only one run per world. "
            f"First duplicate: {duplicate_keys[0]!r}"
        )

    paired: dict[str, dict[str, RoleBindingObservation]] = defaultdict(dict)
    for observation in observations:
        paired[observation.pair_id][observation.world] = observation
    complete_pairs = [worlds for worlds in paired.values() if set(worlds) == {"A", "B"}]

    order_accuracy = _group_rates(observations, ("fact_order",))
    order_values = list(order_accuracy.values())
    fact_order_gap = (
        max(order_values) - min(order_values) if len(order_values) > 1 else None
    )

    return {
        "sample_count": len(observations),
        "valid_response_rate": _rate(item.valid for item in observations),
        "overall_accuracy": _rate(item.correct for item in observations),
        "accuracy_by_world": _group_rates(observations, ("world",)),
        "accuracy_by_authority": _group_rates(observations, ("authority",)),
        "accuracy_by_cell": _group_rates(
            observations, ("label_pair", "world", "authority")
        ),
        "accuracy_by_label_pair": _group_rates(observations, ("label_pair",)),
        "accuracy_by_fact_order": order_accuracy,
        "fact_order_gap": fact_order_gap,
        "output_distribution": dict(
            sorted(Counter(item.answer for item in observations).items())
        ),
        "complete_pair_count": len(complete_pairs),
        "paired_inversion_rate": _rate(
            worlds["A"].valid
            and worlds["B"].valid
            and worlds["A"].answer != worlds["B"].answer
            for worlds in complete_pairs
        ),
        "paired_correct_rate": _rate(
            worlds["A"].correct and worlds["B"].correct for worlds in complete_pairs
        ),
    }


def observation_from_sample(sample: object) -> RoleBindingObservation:
    """Convert an Inspect EvalSample without coupling aggregation to Inspect models."""

    metadata: Mapping[str, object] = getattr(sample, "metadata", None) or {}
    output = getattr(sample, "output", None)
    answer = str(getattr(output, "completion", "") or "").strip().casefold()
    target = str(getattr(sample, "target", "") or "").strip().casefold()
    required = (
        "pair_id",
        "world",
        "authority",
        "label_pair",
        "fact_order",
        "valid_answers",
    )
    missing = [field for field in required if field not in metadata]
    if missing:
        raise ValueError(
            f"Eval sample {getattr(sample, 'id', '<unknown>')!r} lacks {missing}"
        )

    return RoleBindingObservation(
        pair_id=str(metadata["pair_id"]),
        world=str(metadata["world"]),
        authority=str(metadata["authority"]),
        label_pair=str(metadata["label_pair"]),
        fact_order=str(metadata["fact_order"]),
        valid_answers=tuple(
            str(answer).casefold() for answer in metadata["valid_answers"]
        ),
        target=target,
        answer=answer,
    )
