"""Reconcile known ungenerated credit rejections in a stopped run, without retrying."""

import argparse
import json
from pathlib import Path

from contrastive_sdf.artifacts import atomic_json, digest, git_provenance
from contrastive_sdf.evals.awareness.evaluate import now
from contrastive_sdf.evals.awareness.refinement.providers import (
    credit_rejection,
    reviser_accounting,
)
from contrastive_sdf.evals.paths import REPO_ROOT


def reconcile_rejections(output: Path) -> dict:
    state = json.loads((output / "state.json").read_text())
    if state["status"] not in {"budget_stopped", "interrupted", "complete"}:
        raise ValueError("Reconcile only a stopped run")
    if not (output / "resource-reconciliation.json").exists():
        raise ValueError("Reconcile stopped Modal apps first")
    manifest = json.loads((output / "manifest.json").read_text())
    fallback = manifest["config"]["reviser"].get("fallback", {})
    receipts = output / "reviser_calls"
    changed, released = [], 0.0
    for path in sorted(receipts.glob("*.json")):
        row = json.loads(path.read_text())
        if (
            row.get("provider", "openrouter") != "openrouter"
            or row.get("status") != "unknown"
            or not credit_rejection(row.get("http_status"), row.get("response", {}))
        ):
            continue
        archive = output / "reviser_rejection_originals" / path.name
        archive.parent.mkdir(exist_ok=True)
        if not archive.exists():
            atomic_json(archive, row)
        original = json.loads(archive.read_text())
        released += row["accounted_usd"]
        row.update(
            status="credit_exhausted",
            accounted_usd=0,
            reported_cost_usd=0,
            rejection_reconciliation={
                "at_utc": now(),
                "reason": "HTTP 402 credit rejection with no generation or usage",
                "original_receipt": str(archive.relative_to(output)),
                "original_sha256": digest(original),
            },
        )
        atomic_json(path, row)
        changed.append(path.name)
    if changed and fallback.get("enabled"):
        atomic_json(
            output / "reviser-route.json",
            {
                "at_utc": now(),
                "provider": "anthropic",
                "reason": "Reconciled OpenRouter HTTP 402 credit rejections",
                "source_receipt": changed[0],
                "model": fallback["model"],
            },
        )
    costs = json.loads((output / "costs.json").read_text())
    accounting = reviser_accounting(receipts)
    summary = {
        "at_utc": now(),
        "reconciled_calls": len(changed),
        "released_reservations_usd": released,
        "receipts": changed,
        "code": git_provenance(REPO_ROOT),
        "previous_accounting": costs.get("reviser", costs.get("openrouter")),
        "accounting": accounting,
    }
    if changed:
        atomic_json(output / "reviser-rejection-reconciliation.json", summary)
    costs.update(
        reviser=accounting,
        openrouter=accounting["by_provider"].get("openrouter", {}),
        anthropic=accounting["by_provider"].get("anthropic", {}),
    )
    atomic_json(output / "costs.json", costs)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = reconcile_rejections(args.output.resolve())
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("reconciled_calls", "released_reservations_usd", "accounting")
            }
        )
    )


if __name__ == "__main__":
    main()
