"""Semantically rescore saved open-ended answers; never train or regenerate answers."""

import argparse
import json
import tempfile
from pathlib import Path

from contrastive_sdf.evals.reports.comprehension import build_reports
from contrastive_sdf.evals.scoring.belief_judge import judge_records, load_judge_config
from contrastive_sdf.sdf import load_sdf_plan
from contrastive_sdf.sdf.experiment import ExperimentPlan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        root = Path(__file__).resolve().parent.parent
        plan = load_sdf_plan(args.config)
        if not isinstance(plan, ExperimentPlan):
            raise TypeError("version 2 comprehension contract required")
        config = load_judge_config(args.judge_config)
        output = args.output or root / plan.contract.output_dir / "belief_judgments"
        # Reuse native validation of saved Inspect logs and training provenance.
        # Temporary reports prevent overwriting the analysis before all judges pass.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            build_reports(plan, root, directory)
            records = [
                json.loads(line)
                for line in (directory / "samples.jsonl").read_text().splitlines()
            ]
        result = judge_records(
            records,
            config,
            output,
            root,
            execute=args.execute,
            dry_run=args.dry_run,
            workers=args.workers,
            progress=print,
        )
        print(json.dumps(result, indent=2))
    except (OSError, TypeError, ValueError) as ex:
        parser.error(str(ex))


if __name__ == "__main__":
    main()
