"""Matched prompt-cohort metrics using the trajectory report's estimators."""

from collections import defaultdict

from contrastive_sdf.evals.reports.beliefs import category_summary, recall_summary
from contrastive_sdf.evals.reports.statistics import cluster_rates, paired_delta


def observation_key(row):
    return (row["branch"], row["readout"], row["base_question_id"], row["repetition"])


def compare_cohorts(baseline, intervention, *, resamples=10_000, seed=0):
    """Require identical questions, targets and repeats before comparing prompts."""
    kwargs = {"resamples": resamples, "seed": seed}
    indexed = []
    for rows in (baseline, intervention):
        values = {observation_key(row): row for row in rows}
        if len(values) != len(rows):
            raise ValueError("duplicate comparison observations")
        indexed.append(values)
    if not indexed[0] or indexed[0].keys() != indexed[1].keys():
        raise ValueError("cohorts must contain identical questions and repetitions")
    for key, row in indexed[0].items():
        other = indexed[1][key]
        if (
            any(
                row[field] != other[field]
                for field in (
                    "base_task_id",
                    "family",
                    "question_kind",
                    "query",
                    "generation_seed",
                    "temperature",
                    "top_p",
                    "max_tokens",
                )
            )
            or row["belief"]["target"] != other["belief"]["target"]
        ):
            raise ValueError(
                "matched question identity, target or sampling settings changed"
            )
    summaries, comparisons, families, orders = [], [], [], []
    for cohort, rows in (("baseline", baseline), ("intervention", intervention)):
        groups = defaultdict(list)
        family_groups = defaultdict(list)
        for row in rows:
            group = (row["branch"], row["readout"], row["question_kind"], row["query"])
            groups[group].append(row)
            family_groups[(*group, row["family"])].append(row)
        for (branch, readout, kind, query), subset in sorted(groups.items()):
            labels = (
                ("comprehension", "loop", "unscorable")
                if kind == "style_recall"
                else ("grader", "users", "unscorable")
            )
            summary = {
                "cohort": cohort,
                "branch": branch,
                "readout": readout,
                "question_kind": kind,
                "query": query,
                "target": subset[0]["belief"]["target"],
                **recall_summary(subset, **kwargs),
                "answer_distribution": category_summary(
                    subset,
                    labels,
                    lambda row: row["belief"]["observed"] or "unscorable",
                    **kwargs,
                ),
            }
            if readout != "open_ended":
                summary["letter_distribution"] = category_summary(
                    subset,
                    ("a", "b", "unscorable"),
                    lambda row: (
                        "ab"[row["choice_keys"].index(row["belief"]["observed"])]
                        if row["belief"]["valid"]
                        else "unscorable"
                    ),
                    **kwargs,
                )
            summaries.append(summary)
            if cohort == "intervention":
                earlier = [indexed[0][observation_key(row)] for row in subset]
                comparisons.append(
                    {
                        "branch": branch,
                        "readout": readout,
                        "question_kind": kind,
                        "query": query,
                        **paired_delta(
                            cluster_rates(earlier, belief=True),
                            cluster_rates(subset, belief=True),
                            **kwargs,
                        ),
                    }
                )
        for (branch, readout, kind, query, family), subset in sorted(
            family_groups.items()
        ):
            families.append(
                {
                    "cohort": cohort,
                    "branch": branch,
                    "readout": readout,
                    "question_kind": kind,
                    "query": query,
                    "family": family,
                    **recall_summary(subset, **kwargs),
                }
            )
        normal = {
            (r["branch"], r["base_question_id"], r["repetition"]): r
            for r in rows
            if r["readout"] == "mcq"
        }
        flipped = {
            (r["branch"], r["base_question_id"], r["repetition"]): r
            for r in rows
            if r["readout"] == "mcq_flipped"
        }
        if normal.keys() != flipped.keys():
            raise ValueError("unmatched MCQ order controls")
        order_groups = defaultdict(list)
        for key, row in normal.items():
            other = flipped[key]
            order_groups[row["branch"], row["question_kind"], row["query"]].append(
                (row, other)
            )
        for (branch, kind, query), pairs in sorted(order_groups.items()):
            valid = [
                (a, b)
                for a, b in pairs
                if a["belief"]["valid"] and b["belief"]["valid"]
            ]
            agrees = [
                {
                    **a,
                    "belief": {
                        "valid": True,
                        "correct": a["belief"]["observed"] == b["belief"]["observed"],
                    },
                }
                for a, b in valid
            ]
            metric = recall_summary(agrees, **kwargs) if agrees else None
            orders.append(
                {
                    "cohort": cohort,
                    "branch": branch,
                    "question_kind": kind,
                    "query": query,
                    "pairs": len(pairs),
                    "valid_pairs": len(valid),
                    "same_semantic_answer": metric["correct"] if metric else 0,
                    "agreement_on_valid": metric["accuracy"] if metric else None,
                    "agreement_ci95": metric["ci95"] if metric else None,
                }
            )
    return {
        "recall": summaries,
        "changes": comparisons,
        "families": families,
        "answer_order": orders,
    }
