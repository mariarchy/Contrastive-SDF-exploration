"""Print or execute the comprehension checkpoint × universe matrix."""

import argparse
import json
from pathlib import Path

from contrastive_sdf.sdf import load_sdf_plan
from contrastive_sdf.sdf.execution import execute_matrix, materialize_matrix
from contrastive_sdf.sdf.experiment import ExperimentPlan

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--checkpoint")
    parser.add_argument("--branch", choices=["A", "B"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    try:
        plan = load_sdf_plan(args.config)
        if not isinstance(plan, ExperimentPlan):
            raise TypeError("use the existing phase1 scripts for version 1")
        if args.dry_run or not (args.execute or args.mock):
            result = materialize_matrix(plan, ROOT)
        else:
            result = execute_matrix(
                plan,
                ROOT,
                stage=args.stage,
                checkpoint=args.checkpoint,
                branch=args.branch,
                mock=args.mock,
            )
        print(json.dumps(result, indent=2))
    except (OSError, TypeError, ValueError) as ex:
        parser.error(str(ex))


if __name__ == "__main__":
    main()
