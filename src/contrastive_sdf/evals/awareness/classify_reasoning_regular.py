"""Regular Responses execution of the unchanged, previously frozen annotation requests."""

import argparse
import fcntl
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import httpx

from contrastive_sdf.artifacts import atomic_json, digest, source_snapshot
from contrastive_sdf.evals.awareness.classify_reasoning import (
    MODEL,
    accept_results,
    api_request,
    body_for,
    collect,
    counted_attempts,
    ledger,
    load_run,
    now,
    report,
)
from contrastive_sdf.evals.paths import REPO_ROOT


def prepare_switch(output: Path):
    manifest, cases, _ = load_run(output)
    path = output / "regular-api.json"
    if path.exists():
        return json.loads(path.read_text())
    if not (output / "batch-disabled.json").exists():
        raise ValueError("Cancel and disable the previous Batch runner first")
    state = json.loads((output / "state.json").read_text())
    preflight = [c["item_id"] for c in cases if eligible(c, state)][:10]
    if len(preflight) != 10:
        raise ValueError(
            "Need ten unsubmitted items for the regular API technical preflight"
        )
    value = {
        "at_utc": now(),
        "transport": "regular Responses API",
        "endpoint": "/v1/responses",
        "reason": "User requested cancelling Batch and using the main API.",
        "frozen_inputs_sha256": manifest["inputs_sha256"],
        "request_body_policy": "Identical frozen model, prompt, inputs, reasoning effort, schema and output cap.",
        "concurrency": 8,
        "preflight_item_ids": preflight,
        "preflight_policy": "First ten eligible items outside the cancelling Batch, avoiding duplicate in-flight requests. All selected replies still receive the same frozen annotation request.",
        "spending_ceiling_usd": 20,
        "pricing_source": "https://developers.openai.com/api/docs/models/gpt-6.1-sol",
        "checked_on": "2026-10-26",
        "standard_usd_per_million": {
            "input": 2,
            "cached_input": 0.1,
            "cache_write": 2.5,
            "output": 10,
        },
        "budget_policy": "Reserve 2× frozen Batch reservations; account all regular input at 2.50/M and output at 10/M, without cache savings. Includes prior Batch costs/reservations. Cancelled requests proven unexecuted do not consume model retry allowance.",
        "first_pass_max_reservation_usd": 2
        * sum(c["reservation"]["usd"] for c in cases),
        "source_snapshot_sha256": source_snapshot(
            REPO_ROOT, output / "regular_source_snapshot.tar.gz"
        ),
    }
    atomic_json(path, value)
    return value


def reconcile_batches(client, output: Path, state: dict, cases: dict):
    """Collect any finished work; release zero-work cancellations only when proved."""
    collect(client, output, state, cases)
    for batch in state["batches"]:
        if batch["status"] != "cancelled":
            continue
        receipt = json.loads(
            (output / "batches" / batch["token"] / "latest_receipt.json").read_text()
        )
        if receipt["request_counts"]["completed"] != 0:
            continue  # Keep unresolved charges reserved if a partial outcome is ambiguous.
        usage = receipt.get("usage") or {}
        if usage.get("input_tokens", 0) or usage.get("output_tokens", 0):
            continue
        for entries in state["attempts"].values():
            for attempt in entries:
                if attempt["batch_token"] == batch["token"] and not attempt.get(
                    "usage"
                ):
                    attempt.update(
                        not_executed=True,
                        settled=True,
                        accounted_cost_usd=0,
                        in_flight=False,
                        valid=False,
                        validation_error="User cancelled Batch before execution; not an annotation failure",
                    )
    atomic_json(output / "state.json", state)
    atomic_json(output / "cost_ledger.json", ledger(state))


def eligible(item: dict, state: dict) -> bool:
    entries = state["attempts"].get(item["item_id"], [])
    return (
        item["item_id"] not in state["accepted"]
        and counted_attempts(entries) < 2
        and not any(a.get("in_flight") for a in entries)
    )


def begin_attempt(output: Path, state: dict, item: dict, prompt: str, schema: dict):
    if not eligible(item, state):
        raise ValueError("Accepted, active or exhausted item cannot be submitted")
    reserved = item["reservation"]["usd"] * 2
    if reserved > ledger(state)["available_usd"] + 1e-10:
        return None
    entries = state["attempts"].setdefault(item["item_id"], [])
    cid = (
        "r"
        + digest({"item": item["item_id"], "attempt": len(entries), "at": now()})[:32]
    )
    directory = output / "regular" / cid
    directory.mkdir(parents=True)
    body = body_for(prompt, schema, item["payload"])
    if digest(body) != item["request_body_sha256"]:
        raise ValueError("Refusing a request that differs from the frozen body")
    atomic_json(directory / "request.json", body)
    attempt = {
        "item_id": item["item_id"],
        "custom_id": cid,
        "batch_token": cid,
        "attempt_index": len(entries),
        "model_attempt_index": counted_attempts(entries),
        "transport": "responses",
        "price_multiplier": 2,
        "reservation_usd": reserved,
        "in_flight": True,
        "settled": False,
        "started_at_utc": now(),
        "request_body_sha256": digest(body),
    }
    entries.append(attempt)
    atomic_json(output / "state.json", state)
    atomic_json(output / "cost_ledger.json", ledger(state))
    return attempt, body


def finish_attempt(output: Path, state: dict, item: dict, attempt: dict, response):
    directory = output / "regular" / attempt["custom_id"]
    try:
        body = response.json()
    except ValueError:
        body = {}
    row = {
        "custom_id": attempt["custom_id"],
        "response": {
            "status_code": response.status_code,
            "request_id": response.headers.get("x-request-id"),
            "body": body,
        },
    }
    atomic_json(directory / "response.json", {"received_at_utc": now(), **row})
    accept_results(
        state, {"token": attempt["batch_token"]}, [row], {item["item_id"]: item}
    )
    attempt["in_flight"] = False
    attempt["finished_at_utc"] = now()
    # A rejected request with an explicit rate-limit result did not run inference.
    if (
        response.status_code == 429
        and (body.get("error") or {}).get("code") == "rate_limit_exceeded"
    ):
        attempt.update(
            not_executed=True,
            settled=True,
            accounted_cost_usd=0,
            retry_after_utc_seconds=time.time() + 30,
        )
    atomic_json(output / "state.json", state)
    atomic_json(output / "cost_ledger.json", ledger(state))
    return response.status_code


def run(output: Path, workers: int = 8):
    switch = prepare_switch(output)
    _, cases, schema = load_run(output)
    by_id = {c["item_id"]: c for c in cases}
    prompt = (output / "prompt.md").read_text()
    state = json.loads((output / "state.json").read_text())
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Load OPENAI_API_KEY from the local environment")
    with httpx.Client(
        headers={"Authorization": "Bearer " + key}, timeout=180
    ) as client:
        model = api_request(client, "GET", "/models/" + MODEL).json()
        atomic_json(output / "regular_model_access_receipt.json", model)
        reconcile_batches(client, output, state, by_id)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while True:
                preflight = [by_id[key] for key in switch["preflight_item_ids"]]
                stage = (
                    cases
                    if all(c["item_id"] in state["accepted"] for c in preflight)
                    else preflight
                )
                pending_futures = {}
                candidates = iter([c for c in stage if eligible(c, state)])
                stop_submitting = False
                saw_api_error = False
                while True:
                    while len(pending_futures) < workers and not stop_submitting:
                        item = next(candidates, None)
                        if item is None:
                            break
                        entries = state["attempts"].get(item["item_id"], [])
                        if any(
                            a.get("retry_after_utc_seconds", 0) > time.time()
                            for a in entries
                        ):
                            continue
                        attempt_body = begin_attempt(
                            output, state, item, prompt, schema
                        )
                        if attempt_body is None:
                            stop_submitting = True
                            break
                        attempt, body = attempt_body
                        future = executor.submit(
                            client.post,
                            "https://api.openai.com/v1/responses",
                            json=body,
                        )
                        pending_futures[future] = (item, attempt)
                    if not pending_futures:
                        break
                    done, _ = wait(pending_futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        item, attempt = pending_futures.pop(future)
                        try:
                            response = future.result()
                        except httpx.TransportError as ex:
                            # Unknown POST outcome: lock its reservation and do not auto-repeat.
                            attempt.update(
                                unknown_outcome=True, validation_error=type(ex).__name__
                            )
                            atomic_json(
                                output
                                / "regular"
                                / attempt["custom_id"]
                                / "transport_error.json",
                                {
                                    "at_utc": now(),
                                    "error_type": type(ex).__name__,
                                    "outcome": "unknown; not automatically retried",
                                },
                            )
                            atomic_json(output / "state.json", state)
                            continue
                        code = finish_attempt(output, state, item, attempt, response)
                        if code >= 400 and code != 429:
                            stop_submitting = True
                            saw_api_error = True
                    accepted = len(state["accepted"])
                    if accepted % 20 == 0 or stage is preflight:
                        print(
                            json.dumps(
                                {
                                    "accepted": accepted,
                                    "in_flight": len(pending_futures),
                                    "cost": ledger(state),
                                }
                            ),
                            flush=True,
                        )
                summary = report(output, viewer=True)
                print(
                    json.dumps(
                        {
                            k: summary[k]
                            for k in ["accepted", "failed", "pending", "cost"]
                        }
                    ),
                    flush=True,
                )
                if not summary["pending"]:
                    return summary
                if any(
                    a.get("in_flight")
                    for entries in state["attempts"].values()
                    for a in entries
                ):
                    if any(not b.get("collected") for b in state["batches"]):
                        time.sleep(20)
                        reconcile_batches(client, output, state, by_id)
                        continue
                    raise RuntimeError(
                        "A regular request has an unknown outcome; inspect saved intent before any retry"
                    )
                if stop_submitting:
                    if not saw_api_error and any(
                        eligible(c, state)
                        and c["reservation"]["usd"] * 2
                        <= ledger(state)["available_usd"]
                        for c in stage
                    ):
                        continue  # Completed requests released enough of their reservations.
                    raise RuntimeError(
                        "Stopped submissions after an API error or insufficient remaining budget; inspect saved replies and ledger"
                    )
                if stage is preflight and all(
                    c["item_id"] in state["accepted"] for c in preflight
                ):
                    state["preflight_complete"] = True
                    atomic_json(output / "state.json", state)
                    continue
                if stage is preflight and any(
                    counted_attempts(state["attempts"].get(c["item_id"], [])) >= 2
                    and c["item_id"] not in state["accepted"]
                    for c in preflight
                ):
                    raise RuntimeError(
                        "Preflight exhausted its identical retries; inspect failures"
                    )
                retryable = [c for c in stage if eligible(c, state)]
                if not retryable:
                    raise RuntimeError(
                        "Unprocessed cases remain without an eligible retry"
                    )
                if any(
                    a.get("retry_after_utc_seconds", 0) > time.time()
                    for entries in state["attempts"].values()
                    for a in entries
                ):
                    time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (args.output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {
            "pid": os.getpid(),
            "started_at_utc": now(),
            "transport": "regular Responses API",
            "status": "running",
        }
        atomic_json(args.output / "runner_status.json", status)
        try:
            summary = run(args.output)
        except Exception as ex:
            atomic_json(
                args.output / "runner_status.json",
                {
                    **status,
                    "status": "stopped_with_error",
                    "error_type": type(ex).__name__,
                    "error": str(ex),
                    "at_utc": now(),
                },
            )
            raise
        else:
            atomic_json(
                args.output / "runner_status.json",
                {
                    **status,
                    "status": "complete" if summary["complete"] else "waiting",
                    "at_utc": now(),
                    "accepted": summary["accepted"],
                    "failed": summary["failed"],
                    "pending": summary["pending"],
                },
            )
        finally:
            report(args.output, viewer=True)


if __name__ == "__main__":
    main()
