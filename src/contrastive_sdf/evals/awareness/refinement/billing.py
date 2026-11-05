"""Reconcile stopped-app resource bounds without treating metering as an invoice."""

import argparse
import asyncio
import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

from contrastive_sdf.artifacts import atomic_json
from contrastive_sdf.evals.awareness.evaluate import now


def app_duration_bound(created: float, stopped: float, rate: float) -> float:
    if not 0 < created <= stopped or rate <= 0:
        raise ValueError("Valid closed lifecycle and resource rate required")
    return (stopped - created) * rate


def unmetered_duration_bound(
    created: float, stopped: float, cutoff: float, rate: float
) -> float:
    """Bound only the app lifetime after the completed-hour reporting cutoff."""
    app_duration_bound(created, stopped, rate)
    if cutoff <= 0:
        raise ValueError("A positive reporting cutoff is required")
    return max(0.0, stopped - max(created, cutoff)) * rate


async def reconcile(output: Path, *, completed_hours: bool = False):
    from modal._workspace import _Workspace
    from modal.client import _Client
    from modal_proto import api_pb2

    directories, rates = [], {}
    path = output.resolve()
    while path not in directories:
        directories.append(path)
        manifest = json.loads((path / "manifest.json").read_text())
        for receipt in path.glob("modal-*-app.json"):
            rates[json.loads(receipt.read_text())["app_id"]] = manifest["config"][
                "modal_usd_per_worker_second"
            ]
        previous = manifest.get("prior_attempt")
        if not previous:
            break
        path = Path(previous)
    client = await _Client.from_env()
    lifecycles = {}
    for app_id, rate in rates.items():
        response = await client._stub.AppGetLifecycle(
            api_pb2.AppGetLifecycleRequest(app_id=app_id)
        )
        life = response.lifecycle
        if life.app_state != api_pb2.APP_STATE_STOPPED or not life.stopped_at:
            raise ValueError(f"Cannot reconcile a running app: {app_id}")
        lifecycles[app_id] = {
            "created_at": life.created_at,
            "stopped_at": life.stopped_at,
            "rate_usd_per_second": rate,
            "full_app_lifetime_bound_usd": app_duration_bound(
                life.created_at, life.stopped_at, rate
            ),
        }
    earliest = min(d["created_at"] for d in lifecycles.values())
    start = datetime.fromtimestamp(earliest, UTC).replace(
        minute=0, second=0, microsecond=0
    )
    # The SDK excludes partial final intervals. Pin the exact boundary in both
    # the request and receipt, so provider charges and resource bounds cannot overlap.
    cutoff = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    report = await _Workspace.from_context(client=client).billing.report(
        start=start, end=cutoff, resolution="h"
    )
    rows = [dataclasses.asdict(r) for r in report if r.object_id in rates]
    metered = sum(float(r["cost"]) for r in rows)
    lifetime_bound = sum(d["full_app_lifetime_bound_usd"] for d in lifecycles.values())
    tail_bound = sum(
        unmetered_duration_bound(
            d["created_at"],
            d["stopped_at"],
            cutoff.timestamp(),
            d["rate_usd_per_second"],
        )
        for d in lifecycles.values()
    )
    reserve = 1.50 if completed_hours else 0.50
    accounted = (
        metered + tail_bound if completed_hours else max(metered, lifetime_bound)
    ) + reserve
    note = (
        "Provider-reported charges for completed UTC hours plus the configured "
        "resource rate for entire app lifetimes after the reporting cutoff. "
        "Includes $1.00 for possible metering revisions and $0.50 for preparation/storage. "
        "Queueing remains counted in the unmetered tail. Not a final invoice or guaranteed billing cap."
        if completed_hours
        else "Configured GPU+CPU+memory rate applied to every second of each entire stopped app lifetime, "
        "including initialization/queueing and idle time. Replaces local interpreter lifetime and repeated "
        "shutdown reserves. Extra $0.50 covers preparation overlap and incremental storage. "
        "Completed-hour metering may lag; not a final invoice."
    )
    receipt = {
        "at_utc": now(),
        "accounting_method": "completed_hours_plus_resource_tail"
        if completed_hours
        else "full_app_lifetime",
        "completed_hour_cutoff_utc": cutoff.isoformat(),
        "apps": lifecycles,
        "known_completed_hour_metered_usd": metered,
        "metered_rows": rows,
        "full_app_lifetime_bound_usd": lifetime_bound,
        "unmetered_resource_tail_bound_usd": tail_bound,
        "additional_preparation_and_storage_reserve_usd": 0.50,
        "metering_revision_reserve_usd": reserve - 0.50,
        "budget_accounted_usd": accounted,
        "note": note,
    }
    atomic_json(
        output / "resource-reconciliation.json",
        json.loads(json.dumps(receipt, default=str)),
    )
    costs = json.loads((output / "costs.json").read_text())
    atomic_json(output / "costs-before-reconciliation.json", costs)
    costs["modal"] = {
        "budget_accounted_usd": accounted,
        "actual_billed_usd": None,
        "reconciliation": "resource-reconciliation.json",
        "note": receipt["note"],
        "previous_estimate": costs["modal"],
    }
    atomic_json(output / "costs.json", costs)
    print(
        json.dumps(
            {
                "stopped_apps": len(rates),
                "known_metered_usd": metered,
                "budget_accounted_usd": accounted,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--completed-hours",
        action="store_true",
        help="Use provider charges for completed hours, bound the remaining resource time, and reserve $1.50",
    )
    args = parser.parse_args()
    asyncio.run(reconcile(args.output, completed_hours=args.completed_hours))


if __name__ == "__main__":
    main()
