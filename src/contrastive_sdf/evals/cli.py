"""Run a named Inspect evaluation suite through a selected model backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from contrastive_sdf.evals.plan import EvalPlan
from contrastive_sdf.evals.receipts import (
    attach_belief_target,
    save_execution_receipt,
)
from contrastive_sdf.evals.registry import build_eval_plan
from contrastive_sdf.evals.runners.inspect import InspectRunner, InspectTarget
from contrastive_sdf.evals.runners.tinker import TinkerRunner, TinkerTarget


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("suite", help="Built-in name or module:factory")
    parser.add_argument(
        "--checkpoint", help="Starting model ID in the execution contract"
    )
    parser.add_argument(
        "--branch",
        type=str.upper,
        choices=("A", "B"),
        help="SDF branch whose contract mapping defines expected answers",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Frozen execution contract",
        required=True,
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Suite task to run; repeat to select multiple tasks",
    )
    parser.add_argument("--log-dir", required=True)
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
    prompts = backends.add_parser(
        "prompts", help="Compare exact prompts from two configs without model calls"
    )
    prompts.add_argument("--baseline-config", type=Path, required=True)
    prompts.add_argument("--config", type=Path, required=True)
    prompts.add_argument("--output", type=Path, required=True)
    prompts.add_argument("--baseline-checkpoint")
    prompts.add_argument("--checkpoint")
    return parser


def _build_plan(args: argparse.Namespace) -> EvalPlan:
    option_names = (
        "branch",
        "checkpoint",
        "config",
        "log_dir",
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


def main(argv=None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.backend == "prompts":
        from contrastive_sdf.evals.reports.prompt_review import build_prompt_review

        try:
            build_prompt_review(
                args.baseline_config,
                args.config,
                args.output,
                baseline_checkpoint=args.baseline_checkpoint,
                checkpoint=args.checkpoint,
            )
        except (OSError, ValueError) as ex:
            parser.error(str(ex))
        print(args.output / "prompt-review.html")
        return
    try:
        plan = _build_plan(args)
        runner = (
            _inspect_runner(args, parser)
            if args.backend == "inspect"
            else _tinker_runner(args)
        )
        if plan.name == "belief_manipulation":
            plan = attach_belief_target(plan, args.backend, runner.target.describe())
        else:
            from dataclasses import replace

            plan = replace(
                plan,
                metadata={
                    **plan.metadata,
                    "model_target_json": json.dumps(
                        {"backend": args.backend, **runner.target.describe()},
                        sort_keys=True,
                    ),
                },
            )
    except ValueError as ex:
        parser.error(str(ex))

    if args.dry_run:
        print(json.dumps(runner.describe(plan), indent=2))
        return

    if args.backend == "tinker" and not os.environ.get("TINKER_API_KEY", "").strip():
        parser.error("TINKER_API_KEY must be set unless --dry-run is used")
    try:
        save_execution_receipt(plan, args.config, runner.target.describe())
    except (OSError, ValueError) as ex:
        parser.error(str(ex))
    runner.run(plan)


if __name__ == "__main__":
    main()
