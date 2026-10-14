"""One entry point for config freezing, SDF execution, evaluation, and reporting."""

import argparse
import asyncio
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser(
        "freeze",
        help="Prepare inputs and write a new pinned config; keep the draft intact",
    )
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument(
        "--source", type=Path, help="Optional local release/checkpoint export directory"
    )
    for name in ["validate", "run"]:
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name == "run":
            command.add_argument("--dry-run", action="store_true")
    audit = commands.add_parser(
        "audit",
        help="Verify frozen inputs, checkpoint order, raw logs and receipts; rebuild reports",
    )
    audit.add_argument("--run", type=Path, required=True)
    report = commands.add_parser(
        "report", help="Rebuild using the immutable contract stored inside a run"
    )
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--allow-partial", action="store_true")
    commands.add_parser(
        "generate",
        help="Author, review, and export documents using a separate generation recipe",
    )
    commands.add_parser(
        "eval",
        help="Run a built-in or custom suite through Tinker, HF, or an API server",
    )
    args, remaining = parser.parse_known_args(argv)
    if args.command == "generate":
        from contrastive_sdf.corpora.generation.cli import main as generate_main

        return generate_main(remaining)
    if args.command == "eval":
        from contrastive_sdf.evals.cli import main as eval_main

        return eval_main(remaining)
    if remaining:
        parser.error("unrecognized arguments: " + " ".join(remaining))
    if args.command == "freeze":
        from contrastive_sdf.config.freeze import freeze_config

        print(freeze_config(args.config, args.output, source=args.source))
    elif args.command == "audit":
        from contrastive_sdf.workflows.audit import audit_run

        result = audit_run(args.run)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in [
                        "complete",
                        "raw_log_count",
                        "evaluation_receipts",
                        "observed_usage",
                    ]
                }
            )
        )
    elif args.command == "report":
        from contrastive_sdf.evals.reports.trajectory import build_report

        result = build_report(args.run / "contract.yaml", args.allow_partial)
        print(
            json.dumps(
                {
                    "complete": result["complete"],
                    "viewer": str(args.run / "reports/viewer.html"),
                }
            )
        )
    elif args.command == "validate" or args.dry_run:
        from contrastive_sdf.workflows.run import validate

        plan, _, corpora, training = validate(args.config)
        print(
            json.dumps(
                {
                    "corpora": corpora,
                    "training": training,
                    "blockers": plan.blockers("all", Path.cwd()),
                },
                indent=2,
            )
        )
    else:
        from contrastive_sdf.workflows.run import execute

        print(asyncio.run(execute(args.config)))


if __name__ == "__main__":
    main()
