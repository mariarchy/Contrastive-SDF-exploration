"""Saved-response diagnostics; no model calls or changes to scoring."""

from collections import defaultdict

from contrastive_sdf.evals.reports.beliefs import category_summary, recall_summary
from contrastive_sdf.evals.reports.statistics import cluster_rates, paired_delta


def belief_diagnostics(cells, report):
    evaluation = report["settings"]["evaluation"]
    kwargs = {
        "resamples": evaluation["bootstrap_resamples"],
        "seed": evaluation["bootstrap_seed"],
    }
    open_ended, authority_gaps, letters = [], [], []
    for (step, branch, readout), rows in sorted(cells.items()):
        if not rows:
            continue
        if readout == "open_ended":
            for kind in ("all", "style_recall", "authority_attribution"):
                subset = [
                    r for r in rows if kind == "all" or r["question_kind"] == kind
                ]
                if subset:
                    summary = recall_summary(subset, **kwargs)
                    summary["outcomes"] = category_summary(
                        subset,
                        ("correct", "opposing", "unscorable"),
                        lambda r: (
                            "unscorable"
                            if not r["belief"]["valid"]
                            else "correct"
                            if r["belief"]["correct"]
                            else "opposing"
                        ),
                        **kwargs,
                    )
                    open_ended.append(
                        {
                            "step": step,
                            "branch": branch,
                            "question_kind": kind,
                            **summary,
                        }
                    )
        for kind in ("style_recall", "authority_attribution"):
            subset = [r for r in rows if r["question_kind"] == kind]
            if readout in ("mcq", "mcq_flipped") and subset:
                letters.append(
                    {
                        "step": step,
                        "branch": branch,
                        "readout": readout,
                        "question_kind": kind,
                        "samples": len(subset),
                        "distribution": category_summary(
                            subset,
                            ("a", "b", "unscorable"),
                            lambda r: (
                                "ab"[r["choice_keys"].index(r["belief"]["observed"])]
                                if r["belief"]["valid"]
                                else "unscorable"
                            ),
                            **kwargs,
                        ),
                    }
                )
        by_query = {
            query: [
                r
                for r in rows
                if r["question_kind"] == "style_recall" and r["query"] == query
            ]
            for query in ("grader", "users")
        }
        if all(by_query.values()):
            authority_gaps.append(
                {
                    "step": step,
                    "branch": branch,
                    "readout": readout,
                    **paired_delta(
                        cluster_rates(by_query["users"], belief=True),
                        cluster_rates(by_query["grader"], belief=True),
                        **kwargs,
                    ),
                }
            )
    # Compare like directions across checkpoints. The baseline remains branch-rescored.
    grouped = defaultdict(dict)
    for (step, branch, readout), rows in cells.items():
        if readout == "open_ended":
            for kind in ("all", "style_recall", "authority_attribution"):
                subset = [
                    r for r in rows if kind == "all" or r["question_kind"] == kind
                ]
                if subset:
                    grouped[branch, kind][step] = subset
    changes = []
    for (branch, kind), steps in sorted(grouped.items()):
        first = min(steps)
        for step, rows in sorted(steps.items()):
            if step != first:
                changes.append(
                    {
                        "step": step,
                        "reference_step": first,
                        "branch": branch,
                        "question_kind": kind,
                        **paired_delta(
                            cluster_rates(steps[first], belief=True),
                            cluster_rates(rows, belief=True),
                            **kwargs,
                        ),
                    }
                )
    return {
        "version": "belief-diagnostics-v1",
        "scoring_policies": sorted(
            {r["belief"]["scoring_policy"] for rows in cells.values() for r in rows}
        ),
        "bootstrap_seed": kwargs["seed"],
        "bootstrap_resamples": kwargs["resamples"],
        "interval_method": "Pointwise 95% percentile scenario-cluster bootstrap; all repetitions remain together. Conditional on saved adapters/questions; excludes training-seed and scorer uncertainty. Degenerate intervals can occur at observed ceilings/floors.",
        "open_ended": open_ended,
        "open_ended_changes": changes,
        "authority_gaps": authority_gaps,
        "mcq_letters": letters,
        "example_selection": "At the selected checkpoint/universe/query, take the first task ID and repetition in each lexical outcome (correct, opposing, unscorable). Show that same question/repetition across checkpoints. Illustrations are not prevalence estimates.",
    }
