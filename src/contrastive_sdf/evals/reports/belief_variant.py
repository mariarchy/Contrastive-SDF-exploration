"""Export a final-checkpoint comparison without altering the baseline report."""

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from contrastive_sdf.artifacts import atomic_json, git_provenance, source_snapshot
from contrastive_sdf.config.load import load_experiment_plan
from contrastive_sdf.evals.collect import sha
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.reports.belief_comparison import compare_cohorts
from contrastive_sdf.evals.reports.beliefs import recall_summary, summarize
from contrastive_sdf.evals.reports.diagnostics import belief_diagnostics
from contrastive_sdf.evals.reports.prompt_review import build_prompt_review
from contrastive_sdf.evals.reports.statistics import cluster_rates, paired_delta
from contrastive_sdf.evals.reports.trajectory import (
    updated_belief_rows,
    viewer_sample,
    write_csv,
)
from contrastive_sdf.workflows.belief_recall_revision import source_state
from contrastive_sdf.workflows.budget import PRICES


def costs(directory, observations, contract):
    generation = {
        key: sum(r["model_output"]["usage"][key] for r in observations)
        for key in ("input_tokens", "output_tokens")
    }
    calls = [
        json.loads(p.read_text()) for p in directory.glob("**/grader_calls/*.json")
    ]
    grader = {
        key: sum(c["response"]["usage"][key] for c in calls) for key in generation
    }
    gen_cost = (
        generation["input_tokens"] * PRICES["input"]
        + generation["output_tokens"] * PRICES["output"]
    ) / 1e6
    grade_cost = (grader["input_tokens"] * 0.1 + grader["output_tokens"] * 0.5) / 1e6
    return {
        "generation_model": contract.models[0].base_model,
        "generation_provider": "tinker",
        "generation_usage": generation,
        "generation_prices_per_million": {
            key: PRICES[key] for key in ("input", "output")
        },
        "generation_estimate_usd": gen_cost,
        "grader_model": contract.evaluation.belief_grader.model,
        "grader_provider": "openai",
        "grader_usage": grader,
        "grader_completed_requests": len(calls),
        "grader_prices_per_million": {"input": 0.1, "output": 0.5},
        "grader_estimate_usd": grade_cost,
        "grader_pricing_source": "https://developers.openai.com/api/docs/models/gpt-6-luna",
        "grader_pricing_checked_utc": "2026-10-20",
        "additional_compute_estimate_usd": gen_cost + grade_cost,
        "cost_status": "Token-based estimate at uncached input rates; baseline responses reused with no new calls. Reconcile provider invoices.",
    }


def finalize(directory):
    revision = json.loads((directory / "revision.json").read_text())
    protected = json.loads((directory / "original_report_hashes.json").read_text())
    if any(sha(REPO_ROOT / path) != digest for path, digest in protected.items()):
        raise ValueError("a protected baseline report or artifact changed")
    config = directory / "contract.yaml"
    if sha(config) != revision["config_sha256"]:
        raise ValueError("frozen intervention config changed")
    contract = load_experiment_plan(config).contract
    reference = REPO_ROOT / revision["baseline_directory"] / "reports"
    if (
        sha(reference / "samples.jsonl") != revision["baseline_samples_sha256"]
        or sha(reference / "report.json") != revision["baseline_report_sha256"]
    ):
        raise ValueError("baseline source changed")
    old_report = json.loads((reference / "report.json").read_text())
    baseline = []
    with (reference / "samples.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["suite"] == "belief_manipulation" and row["step"] == 1150:
                baseline.append({**row, "cohort": "baseline"})
    fresh = []
    for branch in ("A", "B"):
        state, _ = source_state(directory.parent, branch, 1150)
        rows = updated_belief_rows(revision, 1150, branch, contract.models[0], state)
        if len(rows) != 1296 or Counter(r["readout"] for r in rows) != {
            "mcq": 432,
            "mcq_flipped": 432,
            "open_ended": 432,
        }:
            raise ValueError("incomplete requested final evaluation")
        fresh.extend(
            {**r, "step": 1150, "documents_seen": 9200, "cohort": "intervention"}
            for r in rows
        )
    kwargs = {
        "resamples": contract.evaluation.bootstrap_resamples,
        "seed": contract.evaluation.bootstrap_seed,
    }
    metrics = compare_cohorts(baseline, fresh, **kwargs)
    old_recall = {
        (r["branch"], r["readout"], r["question_kind"], r["query"]): r
        for r in old_report["recall"]
        if r["step"] == 1150
    }
    for row in metrics["recall"]:
        if row["cohort"] == "baseline":
            original = old_recall[
                row["branch"], row["readout"], row["question_kind"], row["query"]
            ]
            for key in [*recall_summary(baseline[:1], **kwargs), "answer_distribution"]:
                if row[key] != original[key]:
                    raise ValueError(f"baseline estimator changed: {key}")
    report = {
        **metrics,
        "complete": True,
        "step": 1150,
        "documents_seen": 9200,
        "settings": contract.model_dump(mode="json"),
        "revision": revision,
        "analysis_code": git_provenance(REPO_ROOT),
        "diagnostics": [],
        "pairs": [],
        "original_answer_order": [],
        "overall": [],
        "overall_changes": [],
    }
    for cohort, rows in (("baseline", baseline), ("intervention", fresh)):
        cells = defaultdict(list)
        for row in rows:
            cells[1150, row["branch"], row["readout"]].append(row)
        source = {"settings": report["settings"], "steps": [{"step": 1150}]}
        _, pairs, _, order = summarize(cells, source)
        report["pairs"].extend({"cohort": cohort, **r} for r in pairs)
        report["original_answer_order"].extend({"cohort": cohort, **r} for r in order)
        report["diagnostics"].append(
            {"cohort": cohort, **belief_diagnostics(cells, source)}
        )
        for branch in ("A", "B"):
            subset = [r for r in rows if r["branch"] == branch]
            report["overall"].append(
                {"cohort": cohort, "branch": branch, **recall_summary(subset, **kwargs)}
            )
    for branch in ("A", "B"):
        old = [row for row in baseline if row["branch"] == branch]
        new = [row for row in fresh if row["branch"] == branch]
        report["overall_changes"].append(
            {
                "branch": branch,
                **paired_delta(
                    cluster_rates(old, belief=True),
                    cluster_rates(new, belief=True),
                    **kwargs,
                ),
            }
        )
        metric = next(
            row
            for row in report["overall"]
            if row["branch"] == branch and row["cohort"] == "baseline"
        )
        original = next(
            row
            for row in old_report["trajectory"]
            if row["step"] == 1150 and row["branch"] == branch
        )
        if (
            metric["accuracy"] != original["recall"]
            or metric["ci95"] != original["recall_ci95"]
        ):
            raise ValueError("baseline overall recall differs from the original report")
    judgments = [r["belief"]["grader"] for r in fresh if r["readout"] == "open_ended"]
    if len(judgments) != 864 or len({r["response_id"] for r in judgments}) != 864:
        raise ValueError("open answers require 864 unique saved Luna judgments")
    cost = costs(directory, fresh, contract)
    revision.update(
        status="complete",
        completed_at_utc=datetime.now(UTC).isoformat(),
        costs=cost,
        coverage=[
            {"branch": b, "step": 1150, "samples": sum(r["branch"] == b for r in fresh)}
            for b in ("A", "B")
        ],
    )
    atomic_json(directory / "costs.json", cost)
    atomic_json(directory / "revision.json", revision)
    report["revision"] = revision
    reports = directory / "reports"
    report["analysis_source_sha256"] = source_snapshot(
        REPO_ROOT, reports / "analysis_source.tar.gz"
    )
    build_prompt_review(directory)
    atomic_json(reports / "report.json", report)
    observations = baseline + fresh
    with (reports / "samples.jsonl").open("w") as stream:
        for row in observations:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name in (
        "recall",
        "changes",
        "families",
        "answer_order",
        "original_answer_order",
        "pairs",
        "overall",
        "overall_changes",
    ):
        write_csv(reports / f"{name}.csv", report[name])
    write_viewer(reports, report, observations)
    verification = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "complete": True,
        "baseline_metric_groups_match_original": len(old_recall),
        "fresh_responses": len(fresh),
        "saved_llm_judgments": len(judgments),
        "protected_files_unchanged": len(protected),
        "matching_sample_ids_targets_and_sampling": True,
        "final_checkpoints_only": True,
    }
    atomic_json(reports / "verification.json", verification)
    print(
        json.dumps(
            {
                "viewer": str(reports / "viewer.html"),
                "verification": verification,
                "costs": cost,
            },
            indent=2,
        )
    )


def write_viewer(reports, report, observations):
    data = {
        "report": report,
        "samples": [
            {**viewer_sample(row), "cohort": row["cohort"]} for row in observations
        ],
    }
    template = Path(__file__).with_name("belief_variant.html").read_text()
    script = Path(__file__).with_name("belief_variant.js").read_text()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    (reports / "viewer.html").write_text(
        template.replace("__DATA__", payload).replace("__SCRIPT__", script)
    )


def refresh_viewer(directory):
    """Refresh presentation and analysis provenance, preserving saved measurements."""
    reports = directory / "reports"
    report = json.loads((reports / "report.json").read_text())
    with (reports / "samples.jsonl").open() as stream:
        observations = [json.loads(line) for line in stream]
    report["analysis_code"] = git_provenance(REPO_ROOT)
    report["analysis_source_sha256"] = source_snapshot(
        REPO_ROOT, reports / "analysis_source.tar.gz"
    )
    atomic_json(reports / "report.json", report)
    write_viewer(reports, report, observations)
    print(reports / "viewer.html")
