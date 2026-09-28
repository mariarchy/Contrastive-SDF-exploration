"""Materialize or execute one pinned Phase 1 SDF training branch on Tinker."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from contrastive_sdf.sdf import load_sdf_plan
from contrastive_sdf.sdf.training import (
    execute_tinker_training,
    materialize_training_run,
)

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "sdf" / "phase1.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Pinned SDF experiment contract",
    )
    parser.add_argument(
        "--branch",
        required=True,
        type=str.upper,
        choices=["A", "B"],
        help="Universe branch to materialize or train",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Training output directory (required with --execute)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Launch paid Tinker training; otherwise perform a read-only dry run",
    )
    args = parser.parse_args()

    if args.execute and args.log_dir is None:
        parser.error("--log-dir is required with --execute")

    try:
        plan = load_sdf_plan(args.config)
        materialized = materialize_training_run(plan, args.branch, REPO_ROOT)
        if args.execute:
            checkpoints = asyncio.run(
                execute_tinker_training(materialized, args.log_dir)
            )
            result = {
                "run": materialized.describe(),
                "checkpoints": checkpoints,
            }
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(materialized.describe(), indent=2))
    except (OSError, ValueError) as ex:
        parser.error(str(ex))


if __name__ == "__main__":
    main()
