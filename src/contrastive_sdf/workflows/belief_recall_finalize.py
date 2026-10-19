"""Finalize a completed belief-recall revision and activate its report."""

import json
from collections import Counter
from datetime import UTC, datetime

from contrastive_sdf.artifacts import atomic_json
from contrastive_sdf.config.load import load_experiment_plan
from contrastive_sdf.evals.collect import sha
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.workflows.belief_recall_revision import source_state
from contrastive_sdf.workflows.budget import PRICES


def finalize(run, directory=None):
    from contrastive_sdf.evals.reports.trajectory import (
        build_report,
        updated_belief_rows,
    )

    activate = directory is None
    directory = directory or run / "belief-recall-v2"
    revision = json.loads((directory / "revision.json").read_text())
    config = directory / "contract.yaml"
    contract = load_experiment_plan(config).contract
    assert contract.evaluation.belief_grader is not None
    observations = []
    coverage = []
    for step, branch in [(0, "A")] + [
        (s, b) for s in revision["checkpoints"][1:] for b in ("A", "B")
    ]:
        state = source_state(run, branch, step)[0] if step else None
        rows = updated_belief_rows(revision, step, branch, contract.models[0], state)
        if len(rows) != 1296 or Counter(r["readout"] for r in rows) != {
            "mcq": 432,
            "mcq_flipped": 432,
            "open_ended": 432,
        }:
            raise ValueError(
                "revised evaluation coverage differs from the requested bank/repetitions"
            )
        observations.extend(rows)
        coverage.append({"step": step, "branch": branch, "samples": len(rows)})
    usage = {
        name: sum(r["model_output"]["usage"][name] for r in observations)
        for name in ("input_tokens", "output_tokens")
    }
    grader_calls = [
        json.loads(path.read_text()) for path in directory.rglob("grader_calls/*.json")
    ]
    grader_usage = {
        name: sum(r["response"]["usage"][name] for r in grader_calls)
        for name in ("input_tokens", "output_tokens")
    }
    judgments = [
        r["belief"]["grader"] for r in observations if r["readout"] == "open_ended"
    ]
    if len(judgments) != 4752:
        raise ValueError("unexpected number of semantic judgments")
    cost = {
        "generation_model": contract.models[0].base_model,
        "generation_provider": contract.models[0].provider,
        "generation_usage": usage,
        "generation_prices_per_million": {k: PRICES[k] for k in ("input", "output")},
        "generation_estimate_usd": (
            usage["input_tokens"] * PRICES["input"]
            + usage["output_tokens"] * PRICES["output"]
        )
        / 1e6,
        "grader_model": contract.evaluation.belief_grader.model,
        "grader_usage": grader_usage,
        "grader_completed_requests": len(grader_calls),
        "grader_unique_scored_responses": len({x["response_id"] for x in judgments}),
        "grader_prices_per_million": {"input": 0.10, "output": 0.50},
        "grader_estimate_usd": (
            grader_usage["input_tokens"] * 0.10 + grader_usage["output_tokens"] * 0.50
        )
        / 1e6,
        "grader_pricing_source": "https://developers.openai.com/api/docs/models/gpt-6-luna",
        "evidence_verbatim": sum(x["evidence_is_verbatim"] for x in judgments),
        "cost_status": "Uncached token-based estimate; grader includes all saved completed production requests. Three successful and three earlier validation calls are separate; earlier validation token usage was not retained. Reconcile provider invoices.",
    }
    cost["additional_compute_estimate_usd"] = (
        cost["generation_estimate_usd"] + cost["grader_estimate_usd"]
    )
    if not activate:
        cost["cost_status"] = (
            "Uncached token-based estimate; includes all saved completed production grader requests. Reconcile provider invoices."
        )
    revision.update(
        status="complete",
        completed_at_utc=datetime.now(UTC).isoformat(),
        coverage=coverage,
        costs=cost,
    )
    atomic_json(directory / "costs.json", cost)
    atomic_json(directory / "revision.json", revision)
    if activate:
        atomic_json(run / "belief_recall_update.json", revision)
        result = build_report(run / "contract.yaml")
    else:
        hashes = json.loads((directory / "original_report_hashes.json").read_text())
        if any(sha(REPO_ROOT / path) != value for path, value in hashes.items()):
            raise ValueError("the original report changed during the duplicate rerun")
        result = build_report(
            run / "contract.yaml",
            destination=directory / "reports",
            belief_update=revision,
        )
    print(
        json.dumps(
            {
                "event": "belief_revision_complete",
                "complete": result["complete"],
                "costs": cost,
            }
        ),
        flush=True,
    )
