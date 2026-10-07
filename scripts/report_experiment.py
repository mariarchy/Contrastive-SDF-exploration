"""Produce A/B reports and a checkpoint trajectory with belief checks."""

import argparse
import json
from pathlib import Path

from contrastive_sdf.evals.reports.comprehension import build_reports
from contrastive_sdf.sdf import load_sdf_plan
from contrastive_sdf.sdf.experiment import ExperimentPlan

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    scoring = parser.add_mutually_exclusive_group()
    scoring.add_argument(
        "--belief-judgments", type=Path, help="Saved LLM judge manifest"
    )
    scoring.add_argument("--legacy-belief-scoring", action="store_true")
    args = parser.parse_args()
    try:
        plan = load_sdf_plan(args.config)
        if not isinstance(plan, ExperimentPlan):
            raise TypeError("version 2 contract required")
        root = Path(__file__).resolve().parent.parent
        output = args.output or root / plan.contract.output_dir / "reports"
        judgments = args.belief_judgments
        default = root / plan.contract.output_dir / "belief_judgments" / "manifest.json"
        if judgments is None and not args.legacy_belief_scoring and default.exists():
            judgments = default
        rows = build_reports(plan, root, output, belief_judgments=judgments)
        print(
            json.dumps(
                {
                    "checkpoints": len(rows),
                    "output": str(output),
                    "gate_statuses": [r["gate_status"] for r in rows],
                },
                indent=2,
            )
        )
    except (OSError, TypeError, ValueError) as ex:
        parser.error(str(ex))
