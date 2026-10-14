"""Resume the retained authoring recipe, review it, and export frozen documents."""

import argparse
import json
from pathlib import Path

from contrastive_sdf.corpora.generation.pipeline import AtomicCorpusPipeline
from contrastive_sdf.corpora.generation.recipe import load_experiment_plan
from contrastive_sdf.corpora.generation.schema import UNIVERSES
from contrastive_sdf.evals.paths import REPO_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=[
            "contexts",
            "render-contexts",
            "facts",
            "plan",
            "drafts",
            "critique",
            "expand",
            "qa",
            "export",
            "review",
            "revalidate",
        ],
        required=True,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Authorize model calls for authoring stages",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reviewed-only",
        action="store_true",
        help="Audit completed documents without generating unfinished candidates",
    )
    parser.add_argument("--through-checkpoint", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--universe", choices=["all", *UNIVERSES], default="all")
    parser.add_argument(
        "--output", type=Path, help="New directory for sealed checkpoint exports"
    )
    parser.add_argument(
        "--kind", choices=["context", "facts", "types", "ideas", "document"]
    )
    parser.add_argument("--document-id")
    parser.add_argument("--type-id")
    parser.add_argument("--replace-contexts", action="store_true")
    parser.add_argument("--decision", choices=["approve", "reject"])
    parser.add_argument("--expected-hash")
    parser.add_argument("--reviewer")
    parser.add_argument("--reason")
    args = parser.parse_args(argv)
    plan = load_experiment_plan(args.recipe)
    pipeline = AtomicCorpusPipeline(
        plan, REPO_ROOT, execute=args.execute and not args.dry_run, workers=args.workers
    )
    pipeline.reviewed_only = args.reviewed_only
    if args.execute and args.reviewed_only:
        parser.error("reviewed-only forbids model calls")
    if args.stage == "revalidate":
        result = pipeline.revalidate(dry_run=args.dry_run)
    elif args.stage == "render-contexts":
        universes = UNIVERSES if args.universe == "all" else (args.universe,)
        result = pipeline.render_contexts(
            universes, dry_run=args.dry_run, replace=args.replace_contexts
        )
    elif args.dry_run:
        result = {
            "stage": args.stage,
            "recipe_sha256": plan.contract_sha256,
            "settings": pipeline.identity_settings(),
            "contexts": pipeline.validate_contexts(),
        }
    elif args.stage == "export":
        from contrastive_sdf.artifacts import read_artifact
        from contrastive_sdf.corpora.generation.expansion import export_checkpoint

        if args.output is None or args.output.exists():
            parser.error("export requires a new --output directory")
        checkpoint = read_artifact(
            pipeline.base
            / "checkpoints"
            / f"{args.through_checkpoint:06d}"
            / "report.json"
        )
        selected = {
            u: pipeline.selected_documents(u, value)
            for u, value in checkpoint["selections"].items()
        }
        args.output.mkdir(parents=True)
        export_checkpoint(pipeline, selected, args.output)
        for name in ["report.json", "qa.json"]:
            (args.output / name).write_bytes(
                (
                    pipeline.base
                    / "checkpoints"
                    / f"{args.through_checkpoint:06d}"
                    / name
                ).read_bytes()
            )
        result = {
            "exports": str(args.output),
            "documents_per_branch": args.through_checkpoint,
        }
    elif args.stage == "expand":
        from contrastive_sdf.corpora.generation.expansion import run_staged_expansion

        result = run_staged_expansion(pipeline, args)
    elif args.stage == "review":
        if args.universe == "all" or not all(
            [args.kind, args.expected_hash, args.reviewer, args.reason, args.decision]
        ):
            parser.error(
                "review needs one universe, kind, expected hash, reviewer, reason and decision"
            )
        from contrastive_sdf.artifacts import read_artifact, sealed

        if args.kind == "context":
            kind, artifact = "context", sealed(pipeline.context(args.universe))
        elif args.kind == "document":
            if not args.document_id:
                parser.error("document reviews need --document-id")
            kind = f"document_{args.document_id}"
            artifact = read_artifact(
                pipeline.path(args.universe, f"documents/{args.document_id}.json")
            )
        elif args.kind == "ideas":
            if not args.type_id:
                parser.error("idea reviews need --type-id")
            kind = f"ideas_{args.type_id}"
            artifact = read_artifact(
                pipeline.path(args.universe, f"ideas/{args.type_id}.json")
            )
        else:
            kind = args.kind
            artifact = read_artifact(pipeline.path(args.universe, f"{kind}.json"))
        result = pipeline.review(
            args.universe,
            kind,
            artifact,
            expected_hash=args.expected_hash,
            reviewer=args.reviewer,
            reason=args.reason,
            decision=args.decision,
        )
    else:
        universes = UNIVERSES if args.universe == "all" else (args.universe,)
        if args.stage == "qa":
            result = pipeline.qa()
        elif args.stage == "contexts":
            result = pipeline.validate_contexts()
        else:
            method = {
                "facts": pipeline.facts,
                "plan": pipeline.plan_documents,
                "drafts": pipeline.drafts,
                "critique": pipeline.critique,
            }[args.stage]
            result = {u: method(u) for u in universes}
    print(json.dumps(result, indent=2, default=str))
