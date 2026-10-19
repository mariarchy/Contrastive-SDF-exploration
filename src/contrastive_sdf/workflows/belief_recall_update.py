"""Run and report a prepared belief-recall revision on retained checkpoints."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from contrastive_sdf.config.load import load_experiment_plan
from contrastive_sdf.config.types import AuthorityReferences
from contrastive_sdf.evals.collect import sha
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.runners.tinker import TinkerTarget
from contrastive_sdf.evals.scoring.belief_semantic import close_grader_clients
from contrastive_sdf.workflows.belief_recall_revision import (
    duplicate,
    prepare,
    source_state,
)
from contrastive_sdf.workflows.evaluate import evaluate_point


async def run_cell(run, branch, step, directory=None):
    directory = directory or run / "belief-recall-v2"
    config = directory / "contract.yaml"
    plan = load_experiment_plan(config)
    model = plan.contract.models[0]
    if step not in [0, 230, 460, 690, 920, 1150] or (step == 0 and branch != "A"):
        raise ValueError("use one shared A base or a declared A/B saved SDF checkpoint")
    point = directory / ("base" if step == 0 else f"{branch}/step_{step:06d}")
    os.environ["BELIEF_GRADER_AUDIT_DIR"] = str(
        point / "belief_manipulation/grader_calls"
    )
    if step == 0:
        adapter = None
        common = {
            "checkpoint_condition": "base",
            "sdf_optimizer_step": "0",
            "sdf_documents_seen": "0",
            "adapter_path": "",
            "corpus_manifest_sha256": "",
        }
    else:
        state, source = source_state(run, branch, step)
        adapter = state["adapter_path"]
        common = {
            "checkpoint_condition": "sdf",
            "sdf_optimizer_step": str(step),
            "sdf_documents_seen": str(state["documents_seen"]),
            "adapter_path": adapter,
            "corpus_manifest_sha256": state["corpus"]["manifest_sha256"],
            "training_checkpoint_path": str(source.relative_to(REPO_ROOT)),
            "training_checkpoint_sha256": sha(source),
        }
    try:
        await evaluate_point(
            config,
            point,
            branch,
            model,
            TinkerTarget(
                model_name=model.base_model, model_path=adapter, renderer=model.renderer
            ),
            common,
        )
    finally:
        await close_grader_clients()
    print(
        json.dumps(
            {
                "event": "belief_cell_complete",
                "step": step,
                "branch": branch,
                "path": str(point),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "duplicate", "cell", "finalize"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--revision-dir", type=Path)
    parser.add_argument("--grader-authority")
    parser.add_argument("--user-authority")
    parser.add_argument("--branch", choices=["A", "B"])
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    run = args.run.resolve()
    directory = args.revision_dir.resolve() if args.revision_dir else None
    if args.command == "prepare":
        config = prepare(
            run,
            AuthorityReferences(
                grader=args.grader_authority, users=args.user_authority
            ),
        )
        print(config)
    elif args.command == "duplicate":
        print(duplicate(run, directory, args.user_authority))
    elif args.command == "cell":
        asyncio.run(run_cell(run, args.branch, args.step, directory))
    elif args.command == "finalize":
        from contrastive_sdf.workflows.belief_recall_finalize import finalize

        finalize(run, directory)


if __name__ == "__main__":
    main()
