"""Private stdin/stdout bridge; one process and one live app per pinned target."""

import argparse
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

from contrastive_sdf.artifacts import atomic_json, digest
from contrastive_sdf.evals.awareness.evaluate import now, parse_completion
from contrastive_sdf.evals.awareness.prompt_variants import build_messages


def emit(value):
    print(json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text())
    target = manifest["targets"][args.target]
    protocol = manifest["classifier"]
    os.environ["SDF_MODAL_TARGET_JSON"] = json.dumps(target)
    wire = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        import modal
        from huggingface_hub import try_to_load_from_cache
        from tinker_cookbook import renderers
        from transformers import AutoTokenizer

        from contrastive_sdf.inference.modal_app import Inference, app, prepare_weights

        cached = try_to_load_from_cache(
            protocol["model_name"], "tokenizer.json", revision=target["base_revision"]
        )
        if (
            not isinstance(cached, str)
            or hashlib.sha256(Path(cached).read_bytes()).hexdigest()
            != manifest["references"][args.target]["tokenizer"]["tokenizer_json_sha256"]
        ):
            raise ValueError("Tokenizer bytes differ from the reference run")
        tokenizer = AutoTokenizer.from_pretrained(
            protocol["model_name"],
            revision=target["base_revision"],
            local_files_only=True,
        )
        renderer = renderers.get_renderer(protocol["renderer"], tokenizer)
        stops = renderer.get_stop_sequences()
        if stops != manifest["stop_sequences"]:
            raise ValueError("Renderer stop sequences changed")
        sampling = {
            key: protocol[key]
            for key in ("temperature", "max_tokens", "top_p", "top_k")
        }
        sampling["stop_token_ids"] = stops
        with modal.enable_output(), app.run():
            atomic_json(
                args.output / f"modal-{args.target}-app.json",
                {"app_id": app.app_id, "at_utc": now(), "target": target},
            )
            assets = (
                cast(Any, prepare_weights).remote()
                if manifest["config"].get("prepare_weights", True)
                else {
                    "policy": "Reuse pinned reference cache; inference load validates cached provenance",
                    "target": target,
                }
            )
            atomic_json(args.output / f"modal-{args.target}-assets.json", assets)
            print(json.dumps({"event": "assets_ready"}), file=wire, flush=True)
            event = json.loads(sys.stdin.readline())
            if event.get("stop"):
                return
            if not event.get("load"):
                raise ValueError("Expected load command after asset preparation")
            worker = Inference()
            warm = cast(Any, worker).update_autoscaler(
                min_containers=1, max_containers=1
            )
            atomic_json(
                args.output / f"modal-{args.target}-warm.json",
                {"min_containers": 1, "max_containers": 1, "effective": str(warm)},
            )
            receipt = cast(Any, worker.info).remote()
            if receipt["target"] != target:
                raise ValueError("Serving target differs from frozen manifest")
            atomic_json(args.output / f"modal-{args.target}-serving.json", receipt)
            print(json.dumps({"event": "ready"}), file=wire, flush=True)
            for line in sys.stdin:
                event = json.loads(line)
                if event.get("stop"):
                    break
                requests = event["requests"]
                prepared = []
                for request in requests:
                    messages = build_messages(
                        {"input": [{"role": "user", "content": request["prompt"]}]},
                        protocol,
                    )
                    tokens = renderer.build_generation_prompt(messages).to_ints()
                    if len(tokens) + protocol["max_tokens"] > protocol["context_limit"]:
                        raise ValueError("Context overflow; no truncation allowed")
                    prepared.append(
                        {
                            "key": request["key"],
                            "prompt_tokens": tokens,
                            "seed": request["seed"],
                        }
                    )
                identity = digest(
                    {"requests": requests, "target": args.target, "sampling": sampling}
                )
                path = (
                    args.output
                    / "classifier_batches"
                    / f"{args.target}-{identity}.json"
                )
                if path.exists():
                    saved = json.loads(path.read_text())
                    if saved.get("status") != "complete":
                        raise RuntimeError(
                            "Incomplete prior batch; inspect receipt before replay"
                        )
                    results = saved["results"]
                else:
                    atomic_json(
                        path,
                        {
                            "status": "submitted",
                            "requests": requests,
                            "prepared": prepared,
                            "sampling": sampling,
                            "at_utc": now(),
                        },
                    )
                    raw = cast(Any, worker.generate).remote(prepared, sampling)
                    if len(raw) != len(prepared) or {r["key"] for r in raw} != {
                        r["key"] for r in prepared
                    }:
                        raise ValueError(
                            "Missing, duplicate or unexpected inference results"
                        )
                    by_key = {r["key"]: r for r in requests}
                    token_by_key = {r["key"]: r for r in prepared}
                    results = []
                    for value in raw:
                        request = by_key[value["key"]]
                        if value["seed"] != request["seed"]:
                            raise ValueError("Inference seed mismatch")
                        results.append(
                            parse_completion(
                                {
                                    **value,
                                    "prompt_sha256": request["prompt_sha256"],
                                    "raw_text": tokenizer.decode(value["tokens"]),
                                    "input_tokens": len(
                                        token_by_key[value["key"]]["prompt_tokens"]
                                    ),
                                    "at_utc": now(),
                                },
                                renderer,
                            )
                        )
                    atomic_json(
                        path,
                        {
                            "status": "complete",
                            "requests": requests,
                            "prepared": prepared,
                            "sampling": sampling,
                            "results": results,
                            "at_utc": now(),
                        },
                    )
                print(json.dumps({"results": results}), file=wire, flush=True)


if __name__ == "__main__":
    main()
