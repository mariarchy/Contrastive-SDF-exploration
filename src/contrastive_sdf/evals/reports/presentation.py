"""Readable presentation of the comprehension experiment's existing metrics."""

from pathlib import Path

STYLE_NAMES = {"comprehension": "comprehensions", "loop": "explicit loops"}
READOUT_NAMES = {"semantic": "Forced choice", "open_ended": "Open-ended recall"}


def percent(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}"


def mean_se(metrics, key) -> str:
    return f"{percent(metrics[key])} ± {percent(metrics.get(key + '_stderr'))}"


def preference_caption(universes) -> str:
    return "   |   ".join(
        f"{branch}: grader → {STYLE_NAMES[mapping['grader']]}; users → {STYLE_NAMES[mapping['users']]}"
        for branch, mapping in universes.items()
    )


def recall_scoring_caption(summary) -> str:
    scoring = summary.get("open_ended_scoring", {})
    if scoring.get("method") == "llm_judge":
        judge = scoring["judge"]
        return (
            f"Open-ended recall uses a target-blinded LLM judge: {judge['provider']}/"
            f"{judge['model']} ({judge['revision']}), temperature {judge['temperature']}. "
            "It classifies the stated preference before comparison with the target. "
            "Original lexical scores, evidence quotes and explanations are retained."
        )
    return "Open-ended recall uses the historical deterministic lexical scorer."


def format_run_report(summary, label: str) -> str:
    contrast = summary["contrast"]
    branches = summary["branches"]
    gate = branches["A"]["manipulation_gate"]
    lines = [
        "# Comprehensions vs explicit loops",
        "",
        f"Run: `{label}`",
        "",
        f"**Belief manipulation gate: {summary['manipulation_gate_status'].upper()}**",
        "",
        "Read the belief checks alongside the coding result. A failed or unconfigured gate does not establish an interpretable reward-seeking measurement.",
        "",
        "## Preference assignments",
        "",
        "| Universe | Automated grader prefers | Users prefer |",
        "| --- | --- | --- |",
    ]
    for branch, mapping in summary.get("universes", {}).items():
        lines.append(
            f"| {branch} | {STYLE_NAMES[mapping['grader']]} | {STYLE_NAMES[mapping['users']]} |"
        )
    lines += [
        "",
        "## Final coding behavior — unprompted tasks",
        "",
        "The coding prompts contain no preference facts. Rates below use the primary estimator and its eligible task set; they measure comprehension use.",
        "",
        "| Quantity | Mean ± standard error |",
        "| --- | ---: |",
        f"| Universe A comprehension rate (%) | **{mean_se(contrast, 'universe_A_rate')}** |",
        f"| Universe B comprehension rate (%) | **{mean_se(contrast, 'universe_B_rate')}** |",
        f"| A − B gap (percentage points) | **{mean_se(contrast, 'gap_A_minus_B')}** |",
        "",
        "A positive gap means more comprehension use in A, where the grader prefers comprehensions, than in B, where the grader prefers loops. This describes the metric's orientation, not a scientific conclusion.",
    ]
    interval = contrast["ci95"]
    lines += [
        "",
        f"95% task bootstrap interval for the gap: **[{percent(interval[0])}, {percent(interval[1])}] percentage points**."
        if interval
        else "95% task bootstrap interval: n/a.",
        f"Estimator: `{contrast['estimator']}`; {contrast['tasks_eligible_in_both']} tasks have eligible outputs in both universes. The paired estimator weights those tasks equally.",
        "",
        "### Coverage and eligibility",
        "",
        "| Quantity | A | B |",
        "| --- | ---: | ---: |",
    ]
    for key, title in (
        ("unique_tasks", "Unique tasks"),
        ("generations", "Coding generations"),
        ("eligible_generations", "Eligible generations"),
        ("valid_python_rate", "Valid Python (%, mean ± SE)"),
        ("eligibility_rate", "Eligible outputs (%, mean ± SE)"),
    ):
        values = [
            mean_se(branches[b]["behavior"], key)
            if key.endswith("_rate")
            else str(branches[b]["behavior"][key])
            for b in ("A", "B")
        ]
        lines.append(f"| {title} | {values[0]} | {values[1]} |")
    lines += [
        "",
        "## Belief manipulation checks",
        "",
        "Accuracy means matching that universe's assigned preference. Forced choice requires a style label. Unclassified answers count as incorrect. Valid-response rate is shown separately.",
        "",
        recall_scoring_caption(summary),
        "",
        "| Universe | Readout | Grader accuracy (%) | User accuracy (%) | Overall accuracy (%) | Valid responses (%) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for branch, item in branches.items():
        for readout, metrics in item["belief"].items():
            values = [
                mean_se(metrics, key)
                for key in (
                    "grader_accuracy",
                    "users_accuracy",
                    "overall_accuracy",
                    "valid_response_rate",
                )
            ]
            lines.append(
                f"| {branch} | {READOUT_NAMES[readout]} | {' | '.join(values)} |"
            )
    lines += [
        "",
        f"Gate threshold: {percent(gate['minimum_accuracy']) + '%' if gate['minimum_accuracy'] is not None else '**not set**'}; required readouts: {', '.join(gate['required_readouts'])}. Both authorities must meet the threshold in both universes.",
    ]
    if "belief_strength" in branches["A"]:
        lines += [
            "",
            "### Belief direction and stability",
            "",
            "Target, opposing and unscorable answers partition all attempts. Agreement compares repeated scorable answers to the same question; consistently opposing answers can also agree. Pair coverage shows how often both answers were scorable.",
            "",
            "| Universe | Readout | Authority | Target (%) | Opposing (%) | Unscorable (%) | Repeat agreement (%) | Scorable pairs (%) |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for branch, item in branches.items():
            for readout, authorities in item["belief_strength"].items():
                for authority, metrics in authorities.items():
                    values = [
                        mean_se(metrics, key)
                        for key in (
                            "target_rate",
                            "opposing_rate",
                            "unscorable_rate",
                            "repeat_agreement_rate",
                            "valid_repeat_pair_rate",
                        )
                    ]
                    lines.append(
                        f"| {branch} | {READOUT_NAMES[readout]} | {authority} | {' | '.join(values)} |"
                    )
        for check in summary.get("weakest_recall_checks", []):
            lines.append(
                f"\nLowest observed recall check: {check['universe']}/{check['authority']}, {READOUT_NAMES[check['readout']]}: {percent(check['accuracy'])} ± {percent(check['stderr'])}%. SE belongs to that check, not to the selected minimum."
            )
    if "sdf_scale" in summary:
        lines += [
            "",
            "## SDF training scale",
            "",
            f"Planned pool: {summary['corpus_documents']:,} unique documents per universe. Values below describe the exposures at this evaluated adapter.",
            "",
            "| Quantity | A | B |",
            "| --- | ---: | ---: |",
        ]
        for key, title in (
            ("documents_seen", "Document exposures"),
            ("unique_documents_seen", "Unique documents seen"),
            ("training_tokens_seen", "Next-token training targets seen"),
            ("optimizer_steps", "Optimizer updates"),
            ("warmup_steps", "Configured warmup updates"),
            ("warmup_fraction", "Warmup completed (%)"),
            ("post_warmup_updates", "Updates after warmup"),
            ("current_learning_rate", "Latest learning rate"),
            ("learning_rate_fraction_of_peak", "Learning rate / configured peak (%)"),
            ("first_train_nll", "First batch training NLL"),
            ("latest_train_nll", "Latest batch training NLL"),
            ("token_weighted_train_nll", "Token-weighted training NLL"),
            ("elapsed_training_seconds", "Elapsed training seconds"),
            ("cost_usd", "Recorded training cost (USD)"),
        ):
            values = []
            for branch in ("A", "B"):
                value = summary["sdf_scale"][branch].get(key)
                values.append(
                    percent(value)
                    if key.endswith(("fraction", "of_peak"))
                    else "n/a"
                    if value is None
                    else f"{value:.5g}"
                    if isinstance(value, float)
                    else f"{value:,}"
                )
            lines.append(f"| {title} | {' | '.join(values)} |")
        for branch, scale in summary["sdf_scale"].items():
            lines.append(
                f"\n{branch} prefix composition: `{scale.get('documents_by_bucket')}`. Cost status: {scale['cost_status']}."
            )
        lines += [
            "",
            "Training NLL measures fitting the document corpus; it is separate from belief recall and coding behavior. Missing historical measurements are n/a.",
        ]
    if "behavioral_diagnostics" in summary:
        d = summary["behavioral_diagnostics"]
        lines += [
            "",
            "### Per-task behavioral audit",
            "",
            f"Eligible paired tasks: {d['paired_eligible_tasks']}. Positive A − B: {d['positive_task_count']}; zero: {d['zero_task_count']}; negative: {d['negative_task_count']}.",
            f"Validity gap (A − B, pp): {mean_se(d, 'valid_python_gap')}; eligibility gap (pp): {mean_se(d, 'eligibility_gap')}.",
            f"[Per-task counts and rates]({summary.get('task_metrics_file', 'tasks.csv')}). These diagnostics do not change the primary estimator.",
        ]
    if any(item.get("qualification") for item in branches.values()):
        lines += [
            "",
            "## Coding qualification — preferences supplied in the prompt",
            "",
            "This tests following a named authority when both preferences are given in context. It is separate from final unprompted coding behavior and recall of SDF-implanted beliefs. Every attempt counts toward accuracy.",
            "",
            "| SDF universe | Readout | Grader accuracy (%) | User accuracy (%) | Overall accuracy (%) | Valid responses (%) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for branch, item in branches.items():
            for readout, metrics in item["qualification"].items():
                values = [
                    mean_se(metrics, key)
                    for key in (
                        "grader_accuracy",
                        "users_accuracy",
                        "overall_accuracy",
                        "valid_response_rate",
                    )
                ]
                lines.append(
                    f"| {branch} | {readout.replace('_', ' ')} | {' | '.join(values)} |"
                )
    lines += [
        "",
        "## Classification audit",
        "",
        "| Output class | A count | B count |",
        "| --- | ---: | ---: |",
    ]
    for label_name in ("comprehension", "loop", "mixed", "ineligible", "invalid"):
        lines.append(
            f"| {label_name} | {branches['A']['behavior'][label_name + '_count']} | {branches['B']['behavior'][label_name + '_count']} |"
        )
    lines += [
        "",
        "Mixed, ineligible and invalid outputs are excluded from the primary style denominator. Coverage counts include every attempted coding generation.",
        "",
        "Pooled generation rates below weight eligible generations rather than tasks. They can differ from the primary paired-task rates at the top.",
        "",
        "| Pooled rate (%, mean ± SE) | A | B |",
        "| --- | ---: | ---: |",
    ]
    for key, title in (
        ("comprehension_rate", "Comprehension"),
        ("loop_rate", "Explicit loop"),
        ("format_valid_rate", "Valid output format"),
    ):
        lines.append(
            f"| {title} | {mean_se(branches['A']['behavior'], key)} | {mean_se(branches['B']['behavior'], key)} |"
        )
    lines += [
        "",
        "## Reading the plot and uncertainty",
        "",
        "Coding bars show each universe's primary comprehension rate. The gap panel shows A − B with a thick ±1 SE bar and a thin 95% bootstrap interval. The lower panels show belief accuracy separately for each authority and readout. Belief accuracy bars include ±1 SE; they do not show coding behavior.",
        "",
        "All percentages are mean ± SE, with SE expressed in percentage points. Repetitions are clustered by task; qualification also clusters authority/world variants by base task. Fewer than two clusters: n/a. Uncertainty is conditional on this corpus and adapter, not variation across independent SDF runs.",
        "",
        f"[Plotted summary]({summary.get('plot_file', 'trajectory.png')}) · [Raw completions and classifications](samples.jsonl) · [Tidy metrics](trajectory.csv)",
    ]
    if summary.get("sdf_documents_seen") is not None:
        lines[4:4] = [
            f"Evaluated adapter: **{summary['sdf_documents_seen']:,} cumulative document exposures per universe**; optimizer update {summary['training_states']['A']['sdf_step']}.",
            "",
        ]
    return "\n".join(lines) + "\n"


def plot_run_summary(summary, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    contrast = summary["contrast"]
    branches = summary["branches"]
    colors = ("#2563eb", "#9333ea")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    coding, gap, semantic, recall = axes.flat
    for index, branch in enumerate(("A", "B")):
        value = contrast[f"universe_{branch}_rate"]
        se = contrast.get(f"universe_{branch}_rate_stderr")
        if value is not None:
            coding.bar(
                index,
                value * 100,
                color=colors[index],
                width=0.55,
                yerr=se * 100 if se is not None else None,
                capsize=5,
            )
            coding.text(
                index,
                value * 100 + (se or 0) * 100 + 2,
                f"{mean_se(contrast, f'universe_{branch}_rate')}%",
                ha="center",
                fontsize=10,
            )
    coding.set(
        xticks=[0, 1],
        xticklabels=["Universe A", "Universe B"],
        ylabel="Comprehension use (%)",
        title="Final coding behavior · mean ± SE",
        ylim=(0, 115),
    )
    value = contrast["gap_A_minus_B"]
    gap.axvline(0, color="#64748b", linestyle="--", linewidth=1)
    if value is not None:
        interval = contrast["ci95"]
        if interval:
            gap.hlines(
                0,
                interval[0] * 100,
                interval[1] * 100,
                color="#94a3b8",
                linewidth=2,
                label="95% task bootstrap interval",
            )
        se = contrast.get("gap_A_minus_B_stderr")
        if se is not None:
            gap.hlines(
                0,
                (value - se) * 100,
                (value + se) * 100,
                color="#0f172a",
                linewidth=7,
                label="Mean ± 1 SE",
            )
        gap.scatter([value * 100], [0], color="#0f172a", s=60, zorder=3)
    gap.set(
        yticks=[],
        xlabel="A − B comprehension rate (percentage points)",
        title=f"Contrast: {mean_se(contrast, 'gap_A_minus_B')} pp",
        ylim=(-0.8, 0.8),
    )
    if gap.get_legend_handles_labels()[0]:
        gap.legend(loc="lower center", frameon=False, fontsize=9)
    gap.text(
        0.5,
        0.9,
        "Positive → greater comprehension use in A",
        transform=gap.transAxes,
        ha="center",
        fontsize=9,
    )
    for axis, readout in ((semantic, "semantic"), (recall, "open_ended")):
        for index, branch in enumerate(("A", "B")):
            metrics = branches[branch]["belief"][readout]
            positions = [index - 0.18, index + 0.18]
            for position, authority, color in zip(
                positions, ("grader", "users"), ("#0f766e", "#d97706")
            ):
                value = metrics[f"{authority}_accuracy"]
                se = metrics.get(f"{authority}_accuracy_stderr")
                if value is not None:
                    axis.bar(
                        position,
                        value * 100,
                        width=0.32,
                        color=color,
                        yerr=se * 100 if se is not None else None,
                        capsize=4,
                        label=authority.capitalize() if index == 0 else None,
                    )
                    axis.text(
                        position,
                        value * 100 + (se or 0) * 100 + 2,
                        f"{percent(value)}%",
                        ha="center",
                        fontsize=9,
                    )
        axis.set(
            xticks=[0, 1],
            xticklabels=["Universe A", "Universe B"],
            ylabel="Belief accuracy (%)",
            title=f"{READOUT_NAMES[readout]}"
            + (
                " · LLM judge"
                if readout == "open_ended"
                and summary.get("open_ended_scoring", {}).get("method") == "llm_judge"
                else ""
            )
            + " · mean ± SE",
            ylim=(0, 125),
            yticks=[0, 20, 40, 60, 80, 100],
        )
        axis.legend(loc="upper right", frameon=False)
    for axis in (coding, semantic, recall):
        axis.spines[["top", "right"]].set_visible(False)
        axis.yaxis.grid(True, alpha=0.15)
        axis.set_axisbelow(True)
    model = summary.get("checkpoint", {}).get("id", "A/B")
    documents = summary.get("sdf_documents_seen") or summary.get(
        "corpus_documents", "unspecified"
    )
    pool = summary.get("corpus_documents", "unspecified")
    scale_label = (
        f"{documents} document exposures per universe (pool: {pool:,})"
        if summary.get("sdf_documents_seen") is not None
        else f"{documents} SDF documents per universe"
    )
    fig.suptitle(
        f"{model} · {scale_label}\nBelief gate: {summary['manipulation_gate_status'].upper()}",
        fontsize=15,
    )
    fig.supxlabel(preference_caption(summary.get("universes", {})), fontsize=10)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_exposure_trajectory(rows, path: Path, universes) -> None:
    import math

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    coding, gap, semantic, recall = axes.flat
    groups = {}
    for row in rows:
        key = (
            row["checkpoint_id"],
            row["sdf_seed"],
            row["shuffle_seed"],
            row["eval_seed"],
            row["temperature"],
        )
        groups.setdefault(key, []).append(row)
    colors = {"A": "#2563eb", "B": "#9333ea"}
    for settings, items in groups.items():
        items.sort(key=lambda r: r["sdf_documents_seen"])
        x = [r["sdf_documents_seen"] for r in items]
        suffix = "" if len(groups) == 1 else f" ({settings})"

        def series(
            axis,
            key,
            label,
            color,
            marker="o",
            linestyle="-",
            *,
            x=x,
            items=items,
            suffix=suffix,
        ):
            axis.errorbar(
                x,
                [100 * r[key] if r[key] is not None else math.nan for r in items],
                yerr=[
                    100 * r[key + "_stderr"]
                    if r.get(key + "_stderr") is not None
                    else math.nan
                    for r in items
                ],
                label=label + suffix,
                color=color,
                marker=marker,
                linestyle=linestyle,
                capsize=5,
                elinewidth=4 if axis is gap else 1.5,
                zorder=3,
            )

        for branch in ("A", "B"):
            series(coding, f"{branch}_rate", f"Universe {branch}", colors[branch])
            for authority, marker, linestyle in (
                ("grader", "o", "-"),
                ("users", "s", "--"),
            ):
                for axis, readout in ((semantic, "semantic"), (recall, "open_ended")):
                    series(
                        axis,
                        f"{branch}_{readout}_{authority}_accuracy",
                        f"{branch} · {authority}",
                        colors[branch],
                        marker,
                        linestyle,
                    )
        series(gap, "gap_A_minus_B", "Gap ± 1 SE", "#0f172a")
        for position, row in zip(x, items):
            if row["ci95_low"] is not None:
                gap.vlines(
                    position,
                    100 * row["ci95_low"],
                    100 * row["ci95_high"],
                    color="#94a3b8",
                    linewidth=1.5,
                    zorder=1,
                    label="95% task bootstrap CI"
                    if position == x[0] and settings == next(iter(groups))
                    else None,
                )
            if row["gap_A_minus_B"] is not None:
                gap.annotate(
                    row["gate_status"],
                    (position, 100 * row["gap_A_minus_B"]),
                    xytext=(8 if position == x[0] else -8, 9),
                    textcoords="offset points",
                    ha="left" if position == x[0] else "right",
                    fontsize=9,
                )
    for axis, title, ylabel in (
        (coding, "Unprompted coding · mean ± SE", "Comprehension use (%)"),
        (gap, "A − B contrast · SE and thin 95% interval", "Gap (percentage points)"),
        (semantic, "Forced-choice belief · mean ± SE", "Target belief accuracy (%)"),
        (recall, "Open-ended belief recall · mean ± SE", "Target belief accuracy (%)"),
    ):
        axis.set(
            title=title,
            ylabel=ylabel,
            xlabel="Cumulative document exposures per universe",
        )
        axis.set_xticks(sorted({r["sdf_documents_seen"] for r in rows}))
        axis.grid(axis="y", alpha=0.15)
        axis.legend(frameon=False, fontsize=9)
        if axis is not gap:
            axis.set_ylim(-5, 110)
    gap.axhline(0, color="#64748b", linestyle="--", linewidth=1)
    assignments = {b: m.model_dump() for b, m in universes.items()}
    status = ", ".join(sorted({r["gate_status"] for r in rows}))
    fig.suptitle(
        f"{'MOCK FIXTURE · ' if any(r['mock'] for r in rows) else ''}Cumulative SDF checkpoints · belief gate: {status}\n"
        + preference_caption(assignments),
        fontsize=12,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)
