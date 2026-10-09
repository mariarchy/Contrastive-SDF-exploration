"""Hard token equality checks, independent of model or evaluation outcomes."""

from itertools import combinations

from contrastive_sdf.sdf.atomic_schema import PAIRS, UNIVERSES


def relative_delta(left, right):
    return abs(left - right) / max((left + right) / 2, 1)


def token_balance_report(selected, tolerance):
    totals = {u: sum(r["tokens"] for r in selected[u]) for u in UNIVERSES}
    branches = {b: sum(totals[u] for u in pair) for b, pair in PAIRS.items()}
    comparisons = {
        f"{a}/{b}": {
            "left_tokens": totals[a],
            "right_tokens": totals[b],
            "delta_tokens": abs(totals[a] - totals[b]),
            "relative_delta": relative_delta(totals[a], totals[b]),
        }
        for a, b in combinations(UNIVERSES, 2)
    }
    comparisons["A/B"] = {
        "left_tokens": branches["A"],
        "right_tokens": branches["B"],
        "delta_tokens": abs(branches["A"] - branches["B"]),
        "relative_delta": relative_delta(branches["A"], branches["B"]),
    }
    maximum = max(c["relative_delta"] for c in comparisons.values())
    return {
        "formula": "abs(left-right) / mean(left,right)",
        "max_relative_delta": tolerance,
        "atomic_totals": totals,
        "branch_totals": branches,
        "comparisons": comparisons,
        "observed_max_relative_delta": maximum,
        "passed": tolerance is None or maximum <= tolerance,
        "enforced": tolerance is not None,
    }


class TokenBalanceError(ValueError):
    def __init__(self, selected, tolerance):
        self.selected = selected
        self.report = token_balance_report(selected, tolerance)
        super().__init__(
            "hard token balance failed: "
            f"{self.report['observed_max_relative_delta']:.3%} exceeds {tolerance:.3%}; "
            "no corpus accepted; inspect the saved balance diagnostics"
        )


def feasible_target(groups, locked, preferred):
    """Find a common attainable total before attempting within-idea swaps.

    Bounds respect all immutable anchors and quotas. Discrete feasibility still
    requires the final hard check; overlapping bounds alone do not prove it.
    """
    bounds = {}
    for u, universe_groups in groups.items():
        low = high = 0
        for quota, candidates in universe_groups:
            fixed = [r for r in candidates if r["id"] in (locked or {}).get(u, set())]
            optional = sorted(r["tokens"] for r in candidates if r not in fixed)
            need = quota - len(fixed)
            anchor_tokens = sum(r["tokens"] for r in fixed)
            low += anchor_tokens + sum(optional[:need])
            high += anchor_tokens + (sum(optional[-need:]) if need else 0)
        bounds[u] = (low, high)
    lower, upper = (
        max(v[0] for v in bounds.values()),
        min(v[1] for v in bounds.values()),
    )
    # With disjoint ranges, the nominal size target must not pull the already
    # larger corpora upward. Aim at the midpoint of the closest bounds; the hard
    # tolerance may already be attainable despite no exact common total.
    return (
        min(max(preferred, lower), upper)
        if lower <= upper
        else round((lower + upper) / 2)
    )
