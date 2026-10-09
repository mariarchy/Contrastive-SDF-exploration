"""Rebuild the belief suite report from complete saved Inspect logs; no model calls."""

import argparse
import json
from pathlib import Path

from inspect_ai.log import read_eval_log

from contrastive_sdf.evals.reports.belief_manipulation import (
    collect_belief_logs,
    format_belief_report,
    summarize_beliefs,
)
from contrastive_sdf.evals.suites.belief_manipulation import belief_manipulation_plan
from contrastive_sdf.sdf.plan import load_experiment_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--branch", choices=("A", "B"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float)
    args = parser.parse_args()
    try:
        plan = belief_manipulation_plan(
            config=args.config,
            branch=args.branch,
            checkpoint=args.checkpoint,
            log_dir=str(args.log_dir),
            task_names=tuple(args.tasks) if args.tasks else None,
            seed=args.seed,
            temperature=args.temperature,
        )
        observations = collect_belief_logs(plan, args.log_dir)
        evaluation = load_experiment_plan(args.config).contract.evaluation
        report = summarize_beliefs(observations, evaluation)
        report.update(
            evaluation=evaluation.model_dump(mode="json"),
            dataset=json.loads(plan.metadata["belief_dataset_json"]),
            generation_provenance=observations[0]["metadata"],
            log_dir=str(args.log_dir),
        )
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for path in args.log_dir.rglob("*.eval"):
            log = read_eval_log(path, header_only=True)
            for item in log.stats.model_usage.values():
                for key in usage:
                    usage[key] += getattr(item, key)
        report.update(
            token_usage=usage,
            cost_usd=None,
            cost_status="Unavailable from provider logs; reconcile provider billing.",
        )
        output = args.log_dir / "reports"
        output.mkdir(exist_ok=True)
        (output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        (output / "report.md").write_text(format_belief_report(report))
        (output / "samples.jsonl").write_text(
            "".join(json.dumps(o, sort_keys=True) + "\n" for o in observations)
        )
        print(
            json.dumps(
                {"samples": len(observations), "report": str(output / "report.md")},
                indent=2,
            )
        )
    except (OSError, ValueError, TypeError) as ex:
        parser.error(str(ex))


if __name__ == "__main__":
    main()
