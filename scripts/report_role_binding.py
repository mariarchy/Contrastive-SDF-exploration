"""Summarize one combined or two per-world role-binding Inspect logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspect_ai.log import read_eval_log

from src.role_binding_report import observation_from_sample, summarize_role_binding


def _format_rate(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _mapping_table(title: str, values: dict[str, object]) -> list[str]:
    lines = [f"## {title}", "", "| Group | Accuracy |", "| --- | ---: |"]
    lines.extend(f"| {key} | {_format_rate(value)} |" for key, value in values.items())
    return lines


def format_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Role-binding control report",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        f"| Samples | {summary['sample_count']} |",
        f"| Valid response rate | {_format_rate(summary['valid_response_rate'])} |",
        f"| Overall accuracy | {_format_rate(summary['overall_accuracy'])} |",
        f"| Complete inverse pairs | {summary['complete_pair_count']} |",
        f"| Paired inversion rate | {_format_rate(summary['paired_inversion_rate'])} |",
        f"| Paired correct rate | {_format_rate(summary['paired_correct_rate'])} |",
        f"| Fact-order gap | {_format_rate(summary['fact_order_gap'])} |",
        "",
    ]
    lines.extend(_mapping_table("Accuracy by world", summary["accuracy_by_world"]))
    lines.append("")
    lines.extend(_mapping_table("Accuracy by authority", summary["accuracy_by_authority"]))
    lines.append("")
    lines.extend(_mapping_table("Accuracy by label pair", summary["accuracy_by_label_pair"]))
    lines.append("")
    lines.extend(_mapping_table("Accuracy by fact order", summary["accuracy_by_fact_order"]))
    lines.append("")
    lines.extend(_mapping_table("Accuracy by cell", summary["accuracy_by_cell"]))
    lines.extend(
        [
            "",
            "## Output distribution",
            "",
            "| Answer | Count |",
            "| --- | ---: |",
            *(
                f"| {answer or '<empty>'} | {count} |"
                for answer, count in summary["output_distribution"].items()
            ),
        ]
    )
    return "\n".join(lines)


def _resolve_log(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        logs = sorted(path.glob("*.eval"), key=lambda candidate: candidate.stat().st_mtime)
        if logs:
            return logs[-1]
        raise ValueError(f"No .eval logs found in {path}")
    raise ValueError(f"Eval log does not exist: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report paired accuracy and bias diagnostics from role-binding eval logs."
    )
    parser.add_argument(
        "logs",
        nargs="+",
        type=Path,
        help="One combined log, one log per world, or a directory whose latest log should be used",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown")
    args = parser.parse_args()

    observations = []
    for requested_path in args.logs:
        path = _resolve_log(requested_path)
        log = read_eval_log(path)
        observations.extend(observation_from_sample(sample) for sample in log.samples or [])

    summary = summarize_role_binding(observations)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_markdown(summary))


if __name__ == "__main__":
    main()
