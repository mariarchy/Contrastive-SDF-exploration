"""Run the unedited checkpoint through belief and Python behavior evaluations."""

import argparse
import json
from pathlib import Path

from contrastive_sdf.sdf.baseline import baseline_description, execute_baseline
from contrastive_sdf.sdf.plan import load_experiment_plan

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    try:
        plan = load_experiment_plan(args.config)
        if args.dry_run or not (args.execute or args.mock):
            result = baseline_description(plan, ROOT, args.checkpoint)
        else:
            result = execute_baseline(
                plan, ROOT, args.output, checkpoint=args.checkpoint, mock=args.mock
            )
        print(json.dumps(result, indent=2))
    except (OSError, TypeError, ValueError) as ex:
        parser.error(str(ex))


if __name__ == "__main__":
    main()
