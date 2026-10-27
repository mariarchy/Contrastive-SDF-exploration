"""Run unchanged frozen awareness prompts through private Modal inference.

Prepare works without a Modal account. Only run submits remote work.
"""

import argparse
import importlib.metadata
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

from contrastive_sdf.artifacts import (
    atomic_json,
    digest,
    git_provenance,
    source_snapshot,
)
from contrastive_sdf.evals.awareness.benchmark import read_benchmark, sha
from contrastive_sdf.evals.awareness.evaluate import (
    load_inputs,
    now,
    parse_completion,
    report,
)
from contrastive_sdf.evals.awareness.prompt_variants import build_messages
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.inference.modal_config import load_target

PROTOCOL_FIELDS = [
    "version",
    "renderer",
    "repetitions",
    "seed",
    "temperature",
    "max_tokens",
    "top_p",
    "top_k",
    "context_limit",
    "prompt_template",
    "prompt_policy",
    "parser_version",
]


def protocol_identity(manifest: dict) -> str:
    return digest(
        {
            "settings": {k: manifest["config"][k] for k in PROTOCOL_FIELDS},
            "inputs_sha256": manifest["inputs_sha256"],
            "source": manifest["source"],
            "tokenizer": manifest["tokenizer"],
            "stops": manifest["stop_sequences"],
        }
    )


def render_variant(
    original: dict, inputs: list[dict], protocol: Path
) -> tuple[dict, bytes]:
    """Change only the requested prompt, retaining frozen sample order and tokenizer."""
    from tinker_cookbook import renderers
    from transformers import AutoTokenizer

    requested = json.loads(protocol.read_text())
    allowed = {"version", "prompt_template", "prompt_policy", "prompt_name"}
    for key in PROTOCOL_FIELDS:
        if key not in allowed and requested[key] != original["config"][key]:
            raise ValueError(f"A prompt variant cannot change {key}")
    template = requested["prompt_template"]
    native = requested["prompt_policy"].get("transcript_format") == "native_messages"
    if template.count("{transcript}") != (0 if native else 1):
        raise ValueError("Expected exactly one transcript placeholder")
    rows, _ = read_benchmark(
        REPO_ROOT / original["config"]["dataset_directory"], original["source"]
    )
    by_id = {r["id"]: r for r in rows}
    tokenizer = AutoTokenizer.from_pretrained(
        original["tokenizer"]["model"],
        revision=original["tokenizer"]["revision"],
        local_files_only=True,
    )
    renderer = renderers.get_renderer(original["config"]["renderer"], tokenizer)
    if renderer.get_stop_sequences() != original["stop_sequences"]:
        raise ValueError("Renderer stop tokens changed")
    rendered = []
    for item in inputs:
        row = by_id[item["id"]]
        if digest(row) != item["source_record_sha256"]:
            raise ValueError("Source transcript changed")
        messages = build_messages(row, requested)
        tokens = renderer.build_generation_prompt(messages).to_ints()
        if len(tokens) + requested["max_tokens"] > requested["context_limit"]:
            raise ValueError("Context overflow; refusing to truncate a transcript")
        rendered.append(
            {
                **item,
                "messages": messages,
                "prompt_tokens": tokens,
                "prompt_sha256": digest(tokens),
            }
        )
    blob = (
        "\n".join(json.dumps(i, ensure_ascii=False) for i in rendered) + "\n"
    ).encode()
    updated = {
        **original,
        "config": {
            **original["config"],
            **{k: requested[k] for k in allowed if k in requested},
        },
        "inputs_sha256": sha(blob),
        "max_prompt_tokens": max(len(i["prompt_tokens"]) for i in rendered),
    }
    return updated, blob


def prepare(
    baseline: Path, target_path: Path, output: Path, protocol: Path | None = None
) -> dict:
    original, inputs = load_inputs(baseline)
    inputs_blob = (baseline / "inputs.jsonl").read_bytes()
    if protocol:
        original, inputs_blob = render_variant(original, inputs, protocol)
        inputs = [json.loads(line) for line in inputs_blob.splitlines()]
    target = load_target(target_path)
    if len(inputs) != 500 or original["counts"] != {
        "chat": 500,
        "eval": 287,
        "deployment": 213,
    }:
        raise ValueError("Expected the complete 500-chat baseline slice")
    if target["base_revision"] != original["tokenizer"]["revision"]:
        raise ValueError(
            "Base revision must match the frozen baseline tokenizer revision"
        )
    if (
        target["max_model_len"]
        < max(len(i["prompt_tokens"]) for i in inputs)
        + original["config"]["max_tokens"]
    ):
        raise ValueError("Model context is too small; no truncation is permitted")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prepare requires a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    config = {
        **original["config"],
        "provider": "modal",
        "modal_target": target,
        "model_path": (
            f"hf://{target['adapter_repo']}@{target['adapter_revision']}"
            if target["adapter_mode"] == "merged-bf16"
            else f"hf://{target['base_model']}@{target['base_revision']}"
        ),
        "pricing": target["pricing"],
    }
    blob = json.dumps(config, indent=2, sort_keys=True).encode() + b"\n"
    (output / "config.json").write_bytes(blob)
    (output / "inputs.jsonl").write_bytes(inputs_blob)
    archive = output / "source_snapshot.tar.gz"
    manifest = {
        **original,
        "created_at_utc": now(),
        "config": config,
        "config_sha256": sha(blob),
        "code": git_provenance(REPO_ROOT),
        "source_snapshot_sha256": source_snapshot(REPO_ROOT, archive),
        "baseline": {
            "directory": str(baseline.resolve()),
            "manifest_sha256": sha((baseline / "manifest.json").read_bytes()),
        },
        "protocol_sha256": protocol_identity(original),
        "status": "prepared-not-run",
        "model_revision": target.get("adapter_revision") or target["base_revision"],
        "comparability": "Compare checkpoints only within the same protocol fingerprint. Modal base and RL targets use the same pinned public base, BF16 precision, vLLM image, Triton backend and sampling settings. Tinker comparisons retain a serving-provider confound.",
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["modal", "tinker-cookbook", "transformers", "numpy"]
        },
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(
        output / "status.json",
        {
            "status": "prepared-not-run",
            "reason": "Prepared inputs; no model calls submitted yet",
            "expected_responses": len(inputs) * config["repetitions"],
        },
    )
    if protocol_identity(manifest) != protocol_identity(original):
        raise ValueError("Protocol changed during preparation")
    print(
        json.dumps(
            {
                "prepared": str(output),
                "transcripts": len(inputs),
                "replies": len(inputs) * config["repetitions"],
                "target": target["name"],
                "model_calls": 0,
            },
            indent=2,
        )
    )
    return manifest


def pending_requests(output: Path, manifest: dict, inputs: list[dict]) -> list[dict]:
    config = manifest["config"]
    pending = []
    for item in inputs:
        for repetition in range(config["repetitions"]):
            key = f"{item['index']:04d}-{repetition:02d}"
            seed = config["seed"] + item["index"] * config["repetitions"] + repetition
            path = output / "responses" / (key + ".json")
            if path.exists():
                record = json.loads(path.read_text())
                if (
                    record["id"],
                    record["seed"],
                    record["config_sha256"],
                    record["prompt_sha256"],
                ) != (
                    item["id"],
                    seed,
                    manifest["config_sha256"],
                    item["prompt_sha256"],
                ):
                    raise ValueError("Existing response does not match this run")
                continue
            pending.append(
                {"key": key, "prompt_tokens": item["prompt_tokens"], "seed": seed}
            )
    return pending


def save_batch(
    output: Path,
    manifest: dict,
    inputs: list[dict],
    requests: list[dict],
    results: list[dict],
    tokenizer,
    renderer,
):
    expected = {r["key"]: r for r in requests}
    if len(results) != len(expected) or {r["key"] for r in results} != set(expected):
        raise ValueError("Modal returned missing, duplicate or unexpected requests")
    for result in results:
        request = expected[result["key"]]
        if result["seed"] != request["seed"]:
            raise ValueError("Inference seed mismatch")
        index, repetition = map(int, result["key"].split("-"))
        item = inputs[index]
        record = {
            **result,
            "id": item["id"],
            "index": index,
            "repetition": repetition,
            "is_eval": item["is_eval"],
            "config_sha256": manifest["config_sha256"],
            "prompt_sha256": item["prompt_sha256"],
            "created_at_utc": now(),
            "input_tokens": len(request["prompt_tokens"]),
            "cached_input_tokens": None,
            "raw_text": tokenizer.decode(result["tokens"]),
        }
        path = output / "responses" / (result["key"] + ".json")
        atomic_json(path, record)
        atomic_json(path, parse_completion(record, renderer))


def run(output: Path, worker=None, assets: dict | None = None):
    import modal
    from tinker_cookbook import renderers
    from transformers import AutoTokenizer

    manifest, inputs = load_inputs(output)
    config = manifest["config"]
    if config.get("provider") != "modal":
        raise ValueError("Use a prepared Modal run directory")
    pending = pending_requests(output, manifest, inputs)
    if not pending:
        return report(output)
    if not modal.config.config.get("token_id") or not modal.config.config.get(
        "token_secret"
    ):
        raise RuntimeError(
            "Modal is not authenticated. Create an account with GPU billing, then run `uv run --extra modal modal setup`. No remote work was submitted."
        )
    target = config["modal_target"]
    os.environ["SDF_MODAL_TARGET_JSON"] = json.dumps(target)
    from contrastive_sdf.inference.modal_app import (
        TARGET,
        Inference,
        app,
        prepare_weights,
    )

    if TARGET != target:
        raise ValueError(
            "The imported Modal app uses another target; start a fresh process"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        revision=manifest["tokenizer"]["revision"],
        local_files_only=True,
    )
    renderer = renderers.get_renderer(config["renderer"], tokenizer)
    sampling = {k: config[k] for k in ["temperature", "max_tokens", "top_p", "top_k"]}
    sampling["stop_token_ids"] = manifest["stop_sequences"]
    started = time.monotonic()
    attempt = now().replace(":", "-")
    archive = output / f"execution-{attempt}.tar.gz"
    atomic_json(
        output / f"execution-{attempt}.json",
        {
            "at_utc": now(),
            "code": git_provenance(REPO_ROOT),
            "source_snapshot_sha256": source_snapshot(REPO_ROOT, archive),
            "source_snapshot": archive.name,
            "config_sha256": manifest["config_sha256"],
            "pending_responses": len(pending),
        },
    )
    atomic_json(output / "status.json", {"status": "running", "started_at_utc": now()})
    try:
        with modal.enable_output(), app.run() if worker is None else nullcontext():
            assets = assets if assets is not None else prepare_weights.remote()
            atomic_json(output / "asset_receipt.json", assets)
            worker = worker if worker is not None else Inference()
            info = worker.info.remote()
            atomic_json(output / "serving_receipt.json", info)
            for start in range(0, len(pending), target["batch_size"]):
                batch = pending[start : start + target["batch_size"]]
                results = worker.generate.remote(batch, sampling)
                save_batch(
                    output, manifest, inputs, batch, results, tokenizer, renderer
                )
                print(
                    json.dumps(
                        {
                            "saved_this_session": min(start + len(batch), len(pending)),
                            "pending_at_start": len(pending),
                        }
                    ),
                    flush=True,
                )
        summary = report(output)
        atomic_json(
            output / "status.json",
            {
                "status": "complete" if summary["complete"] else "incomplete",
                "finished_at_utc": now(),
            },
        )
        return summary
    except Exception as ex:
        atomic_json(
            output / "status.json",
            {
                "status": "failed",
                "error": f"{type(ex).__name__}: {ex}",
                "at_utc": now(),
            },
        )
        raise
    finally:
        atomic_json(
            output / "runtime.json",
            {
                "observed_local_session_seconds": time.monotonic() - started,
                "pricing": target["pricing"],
                "actual_billed_usd": None,
                "note": "Local elapsed time includes queueing, build and preparation; it is not a GPU billing duration. Check Modal billing for actual costs.",
            },
        )


def run_group(outputs: list[Path]):
    """Reuse one loaded checkpoint across explicitly prepared prompt variants."""
    import modal

    manifests = [load_inputs(p)[0] for p in outputs]
    target = manifests[0]["config"]["modal_target"]
    if any(m["config"]["modal_target"] != target for m in manifests):
        raise ValueError("A group must use one identical model target")
    outputs = [path for path in outputs if pending_requests(path, *load_inputs(path))]
    if not outputs:
        return
    os.environ["SDF_MODAL_TARGET_JSON"] = json.dumps(target)
    from contrastive_sdf.inference.modal_app import (
        TARGET,
        Inference,
        app,
        prepare_weights,
    )

    if TARGET != target:
        raise ValueError("Start a fresh process for a different target")
    with modal.enable_output(), app.run():
        for output in outputs:
            atomic_json(
                output / "modal_app.json",
                {"app_id": app.app_id, "at_utc": now(), "target": target["name"]},
            )
        assets = prepare_weights.remote()
        worker = Inference()
        for output in outputs:
            print(f"Evaluation directory: {output}", flush=True)
            run(output, worker=worker, assets=assets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--output", type=Path, action="append", required=True)
    parser.add_argument(
        "--protocol", type=Path, help="Prompt-only override for prepare"
    )
    args = parser.parse_args()
    if args.action == "prepare":
        if args.baseline is None or args.target is None:
            parser.error("prepare requires --baseline and --target")
        if len(args.output) != 1:
            parser.error("prepare requires one output directory")
        prepare(args.baseline, args.target, args.output[0], args.protocol)
    else:
        if len(args.output) == 1:
            run(args.output[0])
        else:
            run_group(args.output)
        from contrastive_sdf.evals.awareness.report import build_report

        for output in args.output:
            path = build_report([output], output / "report.html")
            print(f"Run report: {path}")


if __name__ == "__main__":
    main()
