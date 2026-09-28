"""Run the SDF qualification tasks with Tinker's official Inspect adapter."""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from tinker_cookbook.eval.run_inspect_evals import main as run_inspect_evals

from src.tinker_eval import (
    TASK_NAMES,
    TinkerEvalOptions,
    build_tinker_config,
    describe_options,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        help="Tinker base model ID; optional when --model-path identifies a training run",
    )
    parser.add_argument("--model-path", help="Optional tinker:// checkpoint path")
    parser.add_argument(
        "--renderer",
        required=True,
        help="Tinker Cookbook renderer, such as qwen3_disable_thinking",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=TASK_NAMES,
        dest="tasks",
        help="Task to run; repeat as needed (default: neutral, quote, and action)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--limit", type=int, help="Maximum samples per selected task")
    parser.add_argument("--log-dir", default="logs/tinker_qualification")
    parser.add_argument("--max-connections", type=int, default=32)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print configuration without contacting Tinker",
    )
    return parser


def _options(args: argparse.Namespace) -> TinkerEvalOptions:
    return TinkerEvalOptions(
        renderer=args.renderer,
        model_name=args.model_name,
        model_path=args.model_path,
        task_names=tuple(args.tasks or TASK_NAMES),
        seed=args.seed,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        limit=args.limit,
        log_dir=args.log_dir,
        max_connections=args.max_connections,
    )


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    options = _options(args)
    try:
        config = build_tinker_config(options)
    except (KeyError, ValueError) as ex:
        parser.error(str(ex))

    if args.dry_run:
        print(json.dumps(describe_options(options), indent=2))
        return
    try:
        api_key = os.environ["TINKER_API_KEY"]
    except KeyError:
        parser.error("TINKER_API_KEY must be set unless --dry-run is used")
    if not api_key.strip():
        parser.error("TINKER_API_KEY must not be empty")

    asyncio.run(run_inspect_evals(config))


if __name__ == "__main__":
    main()
