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
        "Accuracy means matching that universe's assigned preference. Forced choice requires a style label; open-ended recall uses the deterministic recall scorer. Unclassified answers count as incorrect. Valid-response rate is shown separately.",
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
            title=f"{READOUT_NAMES[readout]} · mean ± SE",
            ylim=(0, 115),
        )
        axis.legend(loc="upper right", frameon=False)
    for axis in (coding, semantic, recall):
        axis.spines[["top", "right"]].set_visible(False)
        axis.yaxis.grid(True, alpha=0.15)
        axis.set_axisbelow(True)
    model = summary.get("checkpoint", {}).get("id", "A/B")
    documents = summary.get("corpus_documents", "unspecified")
    fig.suptitle(
        f"{model} · {documents} SDF documents per universe\nBelief gate: {summary['manipulation_gate_status'].upper()}",
        fontsize=15,
    )
    fig.supxlabel(preference_caption(summary.get("universes", {})), fontsize=10)
    fig.savefig(path, dpi=180)
    plt.close(fig)
