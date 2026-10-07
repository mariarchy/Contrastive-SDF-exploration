"""Run a named Inspect evaluation suite through a selected model backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from contrastive_sdf.evals.plan import EvalPlan
from contrastive_sdf.evals.registry import build_eval_plan, suite_names
from contrastive_sdf.evals.runners.inspect import InspectRunner, InspectTarget
from contrastive_sdf.evals.runners.tinker import TinkerRunner, TinkerTarget


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("suite", choices=suite_names())
    parser.add_argument("--checkpoint", help="Checkpoint ID in a version 2 contract")
    parser.add_argument(
        "--branch",
        type=str.upper,
        choices=("A", "B"),
        help="SDF branch whose contract mapping defines expected answers",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="SDF contract path (sdf_phase1, comprehension, or coding_style)",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Suite task to run; repeat to select multiple tasks",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--limit", type=int, help="Maximum samples per task")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--log-dir")
    parser.add_argument(
        "--grader-authority",
        help="Explicit grader noun phrase for qualification; supply with --user-authority",
    )
    parser.add_argument(
        "--user-authority",
        help="Explicit downstream-user noun phrase for qualification",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the materialized plan without running it",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    backends = parser.add_subparsers(dest="backend", required=True)

    inspect_parser = backends.add_parser(
        "inspect", help="Use a model provider supported directly by Inspect"
    )
    _add_plan_arguments(inspect_parser)
    inspect_parser.add_argument(
        "--model",
        required=True,
        help="Inspect model identifier, such as hf/... or openai/...",
    )
    inspect_parser.add_argument("--model-base-url")
    inspect_parser.add_argument(
        "--model-args",
        default="{}",
        help="JSON object passed to the Inspect model provider",
    )

    tinker_parser = backends.add_parser(
        "tinker", help="Use Tinker Cookbook's official Inspect adapter"
    )
    _add_plan_arguments(tinker_parser)
    tinker_parser.add_argument(
        "--model-name",
        help="Tinker base model ID; optional when --model-path is supplied",
    )
    tinker_parser.add_argument("--model-path", help="A tinker:// checkpoint path")
    tinker_parser.add_argument(
        "--renderer",
        help="Omit to use checkpoint metadata or Tinker's model recommendation",
    )
    tinker_parser.add_argument("--max-connections", type=int, default=32)
    return parser


def _build_plan(args: argparse.Namespace) -> EvalPlan:
    option_names = (
        "branch",
        "checkpoint",
        "config",
        "seed",
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "limit",
        "repetitions",
        "log_dir",
        "grader_authority",
        "user_authority",
    )
    options = {
        name: getattr(args, name)
        for name in option_names
        if getattr(args, name) is not None
    }
    if args.tasks:
        options["task_names"] = tuple(args.tasks)
    return build_eval_plan(args.suite, **options)


def _inspect_runner(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> InspectRunner:
    try:
        model_args = json.loads(args.model_args)
    except json.JSONDecodeError as ex:
        parser.error(f"--model-args must be valid JSON: {ex}")
    if not isinstance(model_args, dict):
        parser.error("--model-args must decode to a JSON object")
    return InspectRunner(
        InspectTarget(
            model=args.model,
            model_base_url=args.model_base_url,
            model_args=model_args,
        )
    )


def _tinker_runner(args: argparse.Namespace) -> TinkerRunner:
    return TinkerRunner(
        TinkerTarget(
            model_name=args.model_name,
            model_path=args.model_path,
            renderer=args.renderer,
            max_connections=args.max_connections,
        )
    )


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        plan = _build_plan(args)
        runner = (
            _inspect_runner(args, parser)
            if args.backend == "inspect"
            else _tinker_runner(args)
        )
    except ValueError as ex:
        parser.error(str(ex))

    if args.dry_run:
        print(json.dumps(runner.describe(plan), indent=2))
        return

    if args.backend == "tinker" and not os.environ.get("TINKER_API_KEY", "").strip():
        parser.error("TINKER_API_KEY must be set unless --dry-run is used")
    runner.run(plan)


if __name__ == "__main__":
    main()
