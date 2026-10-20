"""Explain saved belief recall across SDF checkpoints without new model calls."""

from collections import Counter, defaultdict

import numpy as np

from contrastive_sdf.evals.reports.statistics import cluster_rates, paired_delta

QUERIES = {
    "style_recall": ("grader", "users"),
    "authority_attribution": ("comprehension", "loop"),
}
PAIR_LABELS = {
    "both_correct": "Both assignments correct",
    "both_swapped": "Both assignments reversed",
    "both_comprehension": "Comprehensions for both",
    "both_loop": "Loops for both",
    "unscorable": "At least one unscorable",
}


def category_summary(rows, labels, category, *, resamples=10_000, seed=0):
    """All-attempt category rates, resampling whole scenarios (including repeats)."""
    counts = defaultdict(lambda: np.zeros(len(labels) + 1, dtype=int))
    for row in rows:
        label = category(row)
        counts[row["base_task_id"]][labels.index(label)] += 1
        counts[row["base_task_id"]][-1] += 1
    if not counts:
        return {}
    values = np.array([counts[k] for k in sorted(counts)])
    totals = values.sum(axis=0)
    indices = np.random.default_rng(seed).integers(
        0, len(values), size=(resamples, len(values))
    )
    draws = values[indices].sum(axis=1)
    return {
        label: {
            "count": int(totals[i]),
            "rate": float(totals[i] / totals[-1]),
            "ci95": list(
                map(float, np.quantile(draws[:, i] / draws[:, -1], [0.025, 0.975]))
            )
            if len(values) >= 2
            else None,
        }
        for i, label in enumerate(labels)
    }


def recall_summary(rows, *, resamples=10_000, seed=0):
    """Bootstrap scenarios, retaining all attempts and their two answer categories."""
    counts = defaultdict(lambda: np.zeros(3, dtype=int))
    for row in rows:
        counts[row["base_task_id"]] += (
            int(row["belief"]["correct"]),
            int(row["belief"]["valid"]),
            1,
        )
    values = np.array([counts[k] for k in sorted(counts)])
    totals = values.sum(axis=0)
    correct, valid, n = map(int, totals)
    indices = np.random.default_rng(seed).integers(
        0, len(values), size=(resamples, len(values))
    )
    draws = values[indices].sum(axis=1)
    accuracy = draws[:, 0] / draws[:, 2]
    valid_draws = draws[draws[:, 1] > 0]
    conditional = valid_draws[:, 0] / valid_draws[:, 1]
    interval = lambda draws: (
        list(map(float, np.quantile(draws, [0.025, 0.975]))) if len(draws) else None
    )
    return {
        "samples": n,
        "scenario_clusters": len(values),
        "unique_questions": len({r["base_question_id"] for r in rows}),
        "correct": correct,
        "opposing": valid - correct,
        "unscorable": n - valid,
        "valid": valid,
        "accuracy": correct / n,
        "ci95": interval(accuracy),
        "valid_response_rate": valid / n,
        "accuracy_on_valid": correct / valid if valid else None,
        "accuracy_on_valid_ci95": interval(conditional),
        "conditional_bootstrap_draws": len(conditional),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "bootstrap_generator": "numpy.default_rng (PCG64)",
        "denominator": "All attempts; unscorable answers count as incorrect.",
    }


def style_pairs(rows):
    """Pair independently generated grader/user questions by scenario and repeat."""
    paired = defaultdict(dict)
    for row in rows:
        if row["question_kind"] != "style_recall":
            continue
        key = (row["base_task_id"], row["repetition"])
        if row["query"] in paired[key]:
            raise ValueError("duplicate style question in a pair")
        paired[key][row["query"]] = row
    result = []
    for (scenario, repetition), pair in sorted(paired.items()):
        if set(pair) != {"grader", "users"}:
            raise ValueError("unmatched grader/user style-recall questions")
        grader, users = pair["grader"]["belief"], pair["users"]["belief"]
        if not grader["valid"] or not users["valid"]:
            label = "unscorable"
        elif grader["correct"] and users["correct"]:
            label = "both_correct"
        elif not grader["correct"] and not users["correct"]:
            label = "both_swapped"
        elif grader["observed"] == users["observed"]:
            label = "both_" + grader["observed"]
        else:
            raise ValueError("unexpected pair of style answers")
        result.append(
            {
                "base_task_id": scenario,
                "base_question_id": scenario,
                "repetition": repetition,
                "category": label,
                "belief": {
                    "correct": label == "both_correct",
                    "valid": label != "unscorable",
                },
            }
        )
    return result


def summarize(cells, source):
    evaluation = source["settings"]["evaluation"]
    kwargs = {
        "resamples": evaluation["bootstrap_resamples"],
        "seed": evaluation["bootstrap_seed"],
    }
    summaries, pairs, comparisons, order = [], [], [], []
    selected = {}
    first = min(s["step"] for s in source["steps"])
    for (step, branch, readout), rows in sorted(cells.items()):
        mapping = source["settings"]["universes"][branch]
        for kind, queries in QUERIES.items():
            for query in queries:
                subset = [
                    r
                    for r in rows
                    if r["question_kind"] == kind and r["query"] == query
                ]
                target = subset[0]["belief"]["target"]
                selected[step, branch, readout, kind, query] = subset
                summaries.append(
                    {
                        "step": step,
                        "documents_seen": step
                        * source["settings"]["training"]["batch_size_documents"],
                        "branch": branch,
                        "readout": readout,
                        "question_kind": kind,
                        "query": query,
                        "target": target,
                        "observed_counts": dict(
                            Counter(
                                r["belief"]["observed"] or "unscorable" for r in subset
                            )
                        ),
                        **recall_summary(subset, **kwargs),
                        "answer_distribution": category_summary(
                            subset,
                            ("comprehension", "loop", "unscorable")
                            if kind == "style_recall"
                            else ("grader", "users", "unscorable"),
                            lambda r: r["belief"]["observed"] or "unscorable",
                            **kwargs,
                        ),
                    }
                )
        paired = style_pairs(rows)
        counts = Counter(p["category"] for p in paired)
        pairs.append(
            {
                "step": step,
                "branch": branch,
                "readout": readout,
                "mapping": mapping,
                "categories": {k: counts[k] for k in PAIR_LABELS},
                "pairs": len(paired),
                **recall_summary(paired, **kwargs),
                "category_distribution": category_summary(
                    paired, tuple(PAIR_LABELS), lambda r: r["category"], **kwargs
                ),
            }
        )
    for (step, branch, readout, kind, query), rows in sorted(selected.items()):
        if step == first:
            continue
        earlier = selected[first, branch, readout, kind, query]
        comparisons.append(
            {
                "reference_step": first,
                "step": step,
                "branch": branch,
                "readout": readout,
                "question_kind": kind,
                "query": query,
                "target": rows[0]["belief"]["target"],
                **paired_delta(
                    cluster_rates(earlier, belief=True),
                    cluster_rates(rows, belief=True),
                    **kwargs,
                ),
            }
        )
    for step, branch in sorted({(k[0], k[1]) for k in cells}):
        normal = {
            (r["base_question_id"], r["repetition"]): r
            for r in cells[step, branch, "mcq"]
        }
        flipped = {
            (r["base_question_id"], r["repetition"]): r
            for r in cells[step, branch, "mcq_flipped"]
        }
        for kind in QUERIES:
            keys = [k for k, r in normal.items() if r["question_kind"] == kind]
            valid = [
                k
                for k in keys
                if normal[k]["belief"]["valid"] and flipped[k]["belief"]["valid"]
            ]
            agree = sum(
                normal[k]["belief"]["observed"] == flipped[k]["belief"]["observed"]
                for k in valid
            )
            order.append(
                {
                    "step": step,
                    "branch": branch,
                    "question_kind": kind,
                    "pairs": len(keys),
                    "valid_pairs": len(valid),
                    "same_semantic_answer": agree,
                    "agreement_on_valid": agree / len(valid) if valid else None,
                    "agreement_ci95": recall_summary(
                        [
                            {
                                **normal[k],
                                "belief": {
                                    "valid": True,
                                    "correct": normal[k]["belief"]["observed"]
                                    == flipped[k]["belief"]["observed"],
                                },
                            }
                            for k in valid
                        ],
                        **kwargs,
                    )["ci95"]
                    if valid
                    else None,
                    "note": "Independent draws under swapped options; agreement measures observed semantic consistency.",
                }
            )
    return summaries, pairs, comparisons, order
