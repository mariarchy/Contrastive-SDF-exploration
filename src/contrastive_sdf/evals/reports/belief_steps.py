"""Explain saved belief recall across SDF checkpoints without new model calls."""

import argparse
import base64
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from contrastive_sdf.evals.reports.checkpoint_steps import cluster_rates, paired_delta
from contrastive_sdf.evals.tasks.belief_manipulation import score_belief_answer
from contrastive_sdf.sdf.models import AuthorityReferences
from contrastive_sdf.sdf.scalable_corpus import atomic_json

FORMATS = {
    "mcq": "Multiple choice",
    "mcq_flipped": "Reversed multiple choice",
    "open_ended": "Open-ended Q&A",
}
QUERIES = {
    "style_recall": ("grader", "users"),
    "authority_attribution": ("comprehension", "loop"),
}
LABELS = {
    "grader": "Grader",
    "users": "Users",
    "comprehension": "Comprehensions",
    "loop": "Loops",
}
PAIR_LABELS = {
    "both_correct": "Both assignments correct",
    "both_swapped": "Both assignments reversed",
    "both_comprehension": "Comprehensions for both",
    "both_loop": "Loops for both",
    "unscorable": "At least one unscorable",
}
PAPER = "https://arxiv.org/pdf/2607.18966"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_rows(rows, source):
    """Require the full saved matrix, balanced clusters, and unchanged targets/scores."""
    steps = [s["step"] for s in source["steps"]]
    branches = source["settings"]["universes"]
    repetitions = source["settings"]["evaluation"]["repetitions"]
    if not source["complete"] or source["missing"]:
        raise ValueError("belief analysis requires a complete checkpoint report")
    cells, identities = defaultdict(list), set()
    references = AuthorityReferences.model_validate(
        source["settings"]["evaluation"]["authority_references"]
    )
    for row in rows:
        key = (row["step"], row["branch"], row["readout"])
        if key[0] not in steps or key[1] not in branches or key[2] not in FORMATS:
            raise ValueError("unexpected belief checkpoint/branch/format")
        identity = (*key, row["base_question_id"], row["repetition"])
        if identity in identities:
            raise ValueError("duplicate belief response")
        identities.add(identity)
        if not 1 <= row["repetition"] <= repetitions:
            raise ValueError("unexpected belief repetition")
        mapping = branches[row["branch"]]
        if (
            row["question_kind"] not in QUERIES
            or row["query"] not in QUERIES[row["question_kind"]]
        ):
            raise ValueError("unexpected belief question direction")
        target = (
            mapping[row["query"]]
            if row["question_kind"] == "style_recall"
            else next(
                a for a, preference in mapping.items() if preference == row["query"]
            )
        )
        encoded = (
            target
            if row["readout"] == "open_ended"
            else "AB"[row["choice_keys"].index(target)]
        )
        score = score_belief_answer(row["completion"], encoded, row, references)
        if score != row["belief"]:
            raise ValueError(
                "saved belief score differs from original deterministic scorer"
            )
        state = next(s for s in source["steps"] if s["step"] == row["step"])[
            "branches"
        ][row["branch"]]
        if row["adapter_path"] != state["adapter_path"]:
            raise ValueError("belief response uses a different checkpoint adapter")
        cells[key].append(row)
    expected = {(s, b, f) for s in steps for b in branches for f in FORMATS}
    if set(cells) != expected:
        raise ValueError("missing belief evaluation cell")
    reference_ids = None
    for cell in cells.values():
        if len(cell) != 144 * repetitions:
            raise ValueError(
                "belief cell must contain 144 questions and every repetition"
            )
        questions = {r["base_question_id"] for r in cell}
        if reference_ids is None:
            reference_ids = questions
        if questions != reference_ids or len(questions) != 144:
            raise ValueError("belief question IDs differ across checkpoints or formats")
        groups = Counter(
            (r["base_task_id"], r["question_kind"], r["query"]) for r in cell
        )
        clusters = {r["base_task_id"] for r in cell}
        if (
            len(clusters) != 36
            or len(groups) != 144
            or set(groups.values()) != {repetitions}
        ):
            raise ValueError("belief scenario coverage is unbalanced")
        for scenario in clusters:
            for kind, queries in QUERIES.items():
                for query in queries:
                    if groups[scenario, kind, query] != repetitions:
                        raise ValueError("missing scenario question or repetition")
    return cells


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
                    "note": "Independent draws under swapped options; agreement measures observed semantic consistency.",
                }
            )
    return summaries, pairs, comparisons, order


def plot_recall(data, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    paths = {}
    colors = ("#285ca7", "#d6772c")
    for kind, queries in QUERIES.items():
        fig, axes = plt.subplots(
            3, 2, figsize=(12, 11), sharex=True, sharey=True, layout="constrained"
        )
        for i, (readout, title) in enumerate(FORMATS.items()):
            for j, branch in enumerate(("A", "B")):
                ax = axes[i, j]
                for query, color in zip(queries, colors):
                    rows = [
                        r
                        for r in data["recall"]
                        if (r["readout"], r["branch"], r["question_kind"], r["query"])
                        == (readout, branch, kind, query)
                    ]
                    rows.sort(key=lambda r: r["step"])
                    x, y = (
                        [r["step"] for r in rows],
                        np.array([100 * r["accuracy"] for r in rows]),
                    )
                    lo, hi = np.array([np.array(r["ci95"]) * 100 for r in rows]).T
                    ax.errorbar(
                        x,
                        y,
                        yerr=np.maximum(0, [y - lo, hi - y]),
                        capsize=3,
                        marker="o",
                        color=color,
                        label=f"Asked about {LABELS[query].lower()}",
                    )
                if readout != "open_ended":
                    ax.axhline(50, color="#8897a7", ls="--", lw=1)
                mapping = data["mappings"][branch]
                assignment = (
                    "Grader → "
                    + LABELS[mapping["grader"]].lower()
                    + "; users → "
                    + LABELS[mapping["users"]].lower()
                )
                if kind == "authority_attribution":
                    assignment = "; ".join(
                        LABELS[q]
                        + " → "
                        + LABELS[next(a for a, p in mapping.items() if p == q)].lower()
                        for q in queries
                    )
                ax.set(
                    title=f"{branch} · {title}\n{assignment}",
                    ylim=(-3, 103),
                    xticks=data["steps"],
                )
                ax.grid(axis="y", alpha=0.2)
                if j == 0:
                    ax.set_ylabel("Correct / all attempts (%)")
                if i == 2:
                    ax.set_xlabel("Optimizer step (8 documents per update)")
        fig.suptitle(
            "Authority → style recall"
            if kind == "style_recall"
            else "Style → authority attribution",
            fontsize=17,
        )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=2,
            title="95% scenario bootstrap intervals · 108 answers / point, 36 scenarios · dashed 50% line: MCQ reference only",
            title_fontsize=9,
            frameon=False,
        )
        name = (
            "belief_style_recall"
            if kind == "style_recall"
            else "belief_authority_attribution"
        )
        for extension in ("png", "svg"):
            fig.savefig(directory / f"{name}.{extension}", dpi=180)
        plt.close(fig)
        paths[kind] = name
    fig, axes = plt.subplots(
        3, 2, figsize=(12, 10), sharex=True, sharey=True, layout="constrained"
    )
    palette = ["#27836b", "#8663aa", "#df963a", "#4c91bd", "#aeb9c5"]
    for i, (readout, title) in enumerate(FORMATS.items()):
        for j, branch in enumerate(("A", "B")):
            ax = axes[i, j]
            rows = sorted(
                [
                    r
                    for r in data["pairs"]
                    if r["readout"] == readout and r["branch"] == branch
                ],
                key=lambda r: r["step"],
            )
            bottom = np.zeros(len(rows))
            for (category, label), color in zip(PAIR_LABELS.items(), palette):
                values = [100 * r["categories"][category] / r["pairs"] for r in rows]
                ax.bar(
                    [r["step"] for r in rows],
                    values,
                    bottom=bottom,
                    width=14,
                    color=color,
                    label=label,
                )
                bottom += values
            ax.set(title=f"{branch} · {title}", ylim=(0, 100), xticks=data["steps"])
            if j == 0:
                ax.set_ylabel("Scenario/repetition pairs (%)")
            if i == 2:
                ax.set_xlabel("Optimizer step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        fontsize=10,
        frameon=False,
        title="108 pairs / bar · two separately generated answers paired by scenario and repetition; no joint prompt",
        title_fontsize=9,
    )
    fig.suptitle(
        "Do both authority answers recover the assigned contrast?", fontsize=17
    )
    for extension in ("png", "svg"):
        fig.savefig(directory / f"belief_assignment_pairs.{extension}", dpi=180)
    plt.close(fig)
    paths["pairs"] = "belief_assignment_pairs"
    return paths


def write_csv(path, rows):
    flattened = [
        {
            k: json.dumps(v) if isinstance(v, (dict, list, tuple)) else v
            for k, v in row.items()
        }
        for row in rows
    ]
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def write_analysis(data, source, directory):
    def recall(branch, query, step, readout="open_ended"):
        return next(
            r
            for r in data["recall"]
            if (r["branch"], r["query"], r["step"], r["readout"], r["question_kind"])
            == (branch, query, step, readout, "style_recall")
        )

    final = max(data["steps"])
    pair = lambda b: next(
        r
        for r in data["pairs"]
        if (r["branch"], r["step"], r["readout"]) == (b, final, "open_ended")
    )
    pct = lambda v: f"{100 * v:.1f}%"
    pp = lambda v: f"{100 * v:+.1f}"
    ci = lambda v: f"[{100 * v[0]:.1f}, {100 * v[1]:.1f}]"
    final_step = next(s for s in source["steps"] if s["step"] == final)
    gap = final_step["contrast"]
    changes = {
        x["branch"]: x
        for x in source["comparisons"]
        if x["step"] == final and x["suite"] == "short_python" and x["readout"] == "all"
    }
    lines = [
        "# Belief recall across the early SDF checkpoints",
        "",
        "## Four findings",
        "",
        "### 1. The complete contrastive belief is not reliably recalled at any tested checkpoint",
        "",
        f"At step {final}, open-ended authority→style recall is {pct(recall('A', 'grader', final)['accuracy'])} for A's grader→comprehensions assignment, but only {pct(recall('A', 'users', final)['accuracy'])} for A's users→loops assignment. In B it is {pct(recall('B', 'users', final)['accuracy'])} for users→comprehensions and {pct(recall('B', 'grader', final)['accuracy'])} for grader→loops. The high-scoring authority changes with the style assigned to it. This pattern points to a comprehension-answer bias rather than faithful recall of each authority's assigned preference.",
        "",
        f"The two open-ended answers recover both assignments in only {pair('A')['correct']}/{pair('A')['pairs']} ({pct(pair('A')['accuracy'])}) paired cases in A and {pair('B')['correct']}/{pair('B')['pairs']} ({pct(pair('B')['accuracy'])}) in B at the last checkpoint. Both answers instead name comprehensions in {pair('A')['categories']['both_comprehension']}/{pair('A')['pairs']} and {pair('B')['categories']['both_comprehension']}/{pair('B')['pairs']} cases. These are separately generated answers paired for diagnosis, not a new evaluation. No success threshold was chosen after seeing the results.",
        "",
        "### 2. Additional exposure does not consistently improve the missing loop belief",
        "",
        "The same asymmetry appears with either multiple-choice order and in open-ended Q&A. Open-ended loop recall in A is "
        + " → ".join(pct(recall("A", "users", s)["accuracy"]) for s in data["steps"])
        + "; in B it is "
        + " → ".join(pct(recall("B", "grader", s)["accuracy"]) for s in data["steps"])
        + ". There is no sustained increase across these checkpoints. Pooled accuracy mixes this weak loop recall with strong comprehension recall and with a different task: attributing a style to an authority.",
        "",
        "Open-ended authority attribution has more unscorable answers than direct style recall. The viewer shows all-attempt accuracy, valid-answer accuracy, and the unscorable fraction separately. Keeping unscorable responses in the denominator avoids making recall look better by dropping non-answers. The original lexical scorer is unchanged; the raw responses remain available for review.",
        "",
        "### 3. Python style moves, but the final A/B contrast is small and uncertain",
        "",
        f"On tasks eligible at both steps, comprehension use changes from step 25 to {final} by {pp(changes['A']['delta'])} percentage points in A (95% paired bootstrap interval {ci(changes['A']['ci95'])}) and {pp(changes['B']['delta'])} in B ({ci(changes['B']['ci95'])}). At step {final}, the matched A−B gap is {pp(gap['gap_A_minus_B'])} points ({ci(gap['ci95'])}); the interval includes zero. The step-100 gap is larger, so the separation is not monotonic. The same tasks and repetitions are retained inside task clusters; eligibility coverage is saved in the original comparison report.",
        "",
        "This does not establish that behavior is being driven by successfully implanted grader-versus-user beliefs: both branches favor comprehensions, and the recall diagnostic does not recover the intended contrast. These are changes since step 25, not changes from an untrained baseline, and the intervals do not estimate variability across training seeds.",
        "",
        "### 4. This is an early-stage replication of the recipe, with much less exposure",
        "",
        f"The run matches the paper's central recipe: Tinker LoRA rank 32, attention/MLP/unembedding targets, peak learning rate 3.5×10⁻⁵, batch size 8, one epoch, and 300-step warmup. Appendix C reports 9,200 documents and 1,150 updates per contrastive adapter; this run has 1,000 documents and 125 updates (10.9% of that update count). Its final learning rate is {final_step['branches']['A']['training']['current_learning_rate']:.4g}, 41.7% of the peak. It never leaves warmup. [Paper, §3.3 and Appendix C]({PAPER}#page=31).",
        "",
        f"The paper excludes unparseable preference answers from recall rates; our primary metric counts them as incorrect. The viewer includes correct/valid as a denominator sensitivity check, with coverage shown. [Paper, Appendix Q]({PAPER}#page=67).",
        "",
        "These results show that the intended two-authority mapping is not reliably elicited within the tested early window. They do not establish that the full training recipe would fail. Without a step-0 recall measurement, high comprehension recall alone also cannot establish that SDF newly implanted that belief.",
        "",
        "## Open-ended authority → style recall at every checkpoint",
        "",
        "Each cell is correct / all attempts; each assignment has 108 answers (36 scenarios × 3 repetitions).",
        "",
        "| Step | Documents | A grader → comprehensions | A users → loops | B grader → loops | B users → comprehensions |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for step in data["steps"]:
        vals = [
            recall(b, q, step)
            for b, q in (
                ("A", "grader"),
                ("A", "users"),
                ("B", "grader"),
                ("B", "users"),
            )
        ]
        lines.append(
            f"| {step} | {step * 8} | "
            + " | ".join(pct(r["accuracy"]) for r in vals)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Graphs and exports",
            "",
            "- [Interactive belief viewer](belief_viewer.html)",
            "- [Authority → style recall](belief_style_recall.png)",
            "- [Style → authority attribution](belief_authority_attribution.png)",
            "- [Paired assignment diagnostic](belief_assignment_pairs.png)",
            "- [Recall cells with intervals and counts](belief_recall.csv)",
            "- [Paired changes from step 25](belief_changes.csv)",
            "- [Analysis data and provenance](belief_analysis.json)",
            "",
            "## Measurement and reproducibility",
            "",
            data["uncertainty"],
            "",
            data["pairing_note"],
            "",
            "Primary recall intervals use 10,000 scenario bootstrap draws with seed 0 (NumPy PCG64); paired changes reuse the original Python-Random bootstrap helper, also with 10,000 draws and seed 0. These are pointwise descriptive intervals without correction for multiple comparisons. All 12,960 saved belief responses were rechecked against the unchanged deterministic scorer and original log hashes. Analysis source, input hashes, dataset identity, adapter paths, model/provider, seeds, and training settings are in belief_analysis.json and belief_analysis_source.tar.gz.",
            "",
            "Rebuild without model calls:",
            "",
            "```sh",
            ".venv/bin/python scripts/report_belief_steps.py --report-dir logs/comprehension/atomic-sdf-1000-steps-v1/reports",
            "```",
            "",
        ]
    )
    (directory / "belief_analysis.md").write_text("\n".join(lines))


def build_belief_analysis(directory):
    directory = Path(directory).resolve()
    source_path, samples_path = directory / "report.json", directory / "samples.jsonl"
    source = json.loads(source_path.read_text())
    rows = []
    with samples_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["suite"] == "belief_manipulation":
                rows.append(row)
    cells = validate_rows(rows, source)
    logs = {r["log_path"]: r["log_sha256"] for r in rows}
    for path, expected in logs.items():
        if sha(path) != expected:
            raise ValueError(f"original belief log hash differs: {path}")
    summaries, pairs, comparisons, order = summarize(cells, source)
    # Each original pooled format accuracy must be reconstructed exactly by the four query groups.
    for step in source["steps"]:
        for branch, values in step["branches"].items():
            for readout, original in values["belief"]["formats"].items():
                subset = [
                    r
                    for r in summaries
                    if (r["step"], r["branch"], r["readout"])
                    == (step["step"], branch, readout)
                ]
                pooled = sum(r["correct"] for r in subset) / sum(
                    r["samples"] for r in subset
                )
                if abs(pooled - original["accuracy"]) > 1e-12:
                    raise ValueError(
                        "new recall summaries differ from original report accuracy"
                    )
    from contrastive_sdf.evals.paths import REPO_ROOT
    from contrastive_sdf.sdf.execution import _source_snapshot
    from contrastive_sdf.sdf.experiment import git_provenance

    code = git_provenance(REPO_ROOT)
    files = [
        Path(__file__),
        Path(__file__).with_suffix(".html"),
        REPO_ROOT / "scripts/report_belief_steps.py",
    ]
    archive = directory / "belief_analysis_source.tar.gz"
    archive_sha = _source_snapshot(REPO_ROOT, archive)
    data = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": source["experiment_id"],
        "steps": [s["step"] for s in source["steps"]],
        "formats": FORMATS,
        "mappings": source["settings"]["universes"],
        "recall": summaries,
        "pairs": pairs,
        "changes": comparisons,
        "answer_order": order,
        "source_settings": source["settings"],
        "training_and_behavior": source["steps"],
        "source_comparisons": source["comparisons"],
        "original_experiment": source["experiment"],
        "provenance": {
            "source_report": {"path": str(source_path), "sha256": sha(source_path)},
            "source_samples": {"path": str(samples_path), "sha256": sha(samples_path)},
            "belief_responses": len(rows),
            "belief_log_hashes": logs,
            "analysis_code": code,
            "runtime": {"python": sys.version, "numpy": np.__version__},
            "source_archive": {"path": archive.name, "sha256": archive_sha},
            "source_files": {str(p.relative_to(REPO_ROOT)): sha(p) for p in files},
        },
        "uncertainty": "95% pointwise percentile bootstrap intervals across 36 scenario clusters, keeping related questions and three repetitions together. Conditional on these two trained adapters; no inference across training seeds. Primary recall uses all attempts. Correct/valid is a separately labelled denominator sensitivity check. No post-hoc success threshold or multiple-comparison correction.",
        "pairing_note": "A paired case combines two separately generated style-recall answers (grader and users) from the same scenario and repetition. This is an existing-output diagnostic, not a simultaneous joint question or an estimate of internal belief certainty.",
        "success_status": "No numerical success threshold was configured. Inspect both assignments and their consistency; high pooled accuracy alone is insufficient.",
    }
    atomic_json(directory / "belief_analysis.json", data)
    for name, values in (
        ("belief_recall.csv", summaries),
        ("belief_changes.csv", comparisons),
        ("belief_assignment_pairs.csv", pairs),
        ("belief_answer_order.csv", order),
    ):
        write_csv(directory / name, values)
    paths = plot_recall(data, directory)
    write_analysis(data, source, directory)
    payload = {
        **data,
        "plots": {
            key: "data:image/png;base64,"
            + base64.b64encode((directory / f"{name}.png").read_bytes()).decode()
            for key, name in paths.items()
        },
    }
    encoded = json.dumps(payload).replace("<", "\\u003c")
    template = Path(__file__).with_suffix(".html").read_text()
    (directory / "belief_viewer.html").write_text(template.replace("__DATA__", encoded))
    # Make the focused viewer discoverable in existing saved reports as well as future builds.
    marker = '<a href="comparison_viewer.html">Comparison across steps</a>'
    link = '<a href="belief_viewer.html">Belief recall across steps</a>'
    for name in ("results_viewer.html", "comparison_viewer.html"):
        path = directory / name
        text = path.read_text()
        if link not in text:
            path.write_text(text.replace(marker, marker + link, 1))
    print(
        json.dumps(
            {
                "viewer": str(directory / "belief_viewer.html"),
                "belief_responses": len(rows),
                "recall_cells": len(summaries),
                "paired_cases": sum(p["pairs"] for p in pairs),
            }
        )
    )
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    build_belief_analysis(parser.parse_args().report_dir)
