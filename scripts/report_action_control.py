"""Report validity, compliance, and inversion for an action-control Inspect log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspect_ai.log import read_eval_log

from src.action_control_report import observation_from_sample, summarize_action_control


def _format_rate(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _table(title: str, values: dict[str, object]) -> list[str]:
    return [
        f"## {title}",
        "",
        "| Group | Compliance |",
        "| --- | ---: |",
        *(f"| {key} | {_format_rate(value)} |" for key, value in values.items()),
    ]


def format_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Authority-conditioned coding control",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        f"| Samples | {summary['sample_count']} |",
        f"| Format-valid rate | {_format_rate(summary['format_valid_rate'])} |",
        f"| Python-valid rate | {_format_rate(summary['python_valid_rate'])} |",
        f"| Preference compliance | {_format_rate(summary['preference_compliance_rate'])} |",
        "| World inversion "
        f"({summary['world_pair_count']} pairs) | "
        f"{_format_rate(summary['world_inversion_rate'])} |",
        f"| World paired correctness | {_format_rate(summary['world_paired_correct_rate'])} |",
        "| Authority inversion "
        f"({summary['authority_pair_count']} pairs) | "
        f"{_format_rate(summary['authority_inversion_rate'])} |",
        "| Authority paired correctness | "
        f"{_format_rate(summary['authority_paired_correct_rate'])} |",
        "",
    ]
    for title, key in (
        ("Compliance by world", "compliance_by_world"),
        ("Compliance by authority", "compliance_by_authority"),
        ("Compliance by cell", "compliance_by_cell"),
        ("Compliance by fact order", "compliance_by_fact_order"),
    ):
        lines.extend(_table(title, summary[key]))
        lines.append("")
    lines.extend(
        [
            "## Observed style distribution",
            "",
            "| Style | Count |",
            "| --- | ---: |",
            *(
                f"| {style} | {count} |"
                for style, count in summary["observed_style_distribution"].items()
            ),
        ]
    )
    return "\n".join(lines)


def _resolve_log(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        logs = sorted(
            path.glob("*.eval"), key=lambda candidate: candidate.stat().st_mtime
        )
        if logs:
            return logs[-1]
        raise ValueError(f"No .eval logs found in {path}")
    raise ValueError(f"Eval log does not exist: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log", type=Path, help="An eval log or directory containing one"
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of Markdown"
    )
    args = parser.parse_args()

    log = read_eval_log(_resolve_log(args.log))
    summary = summarize_action_control(
        [observation_from_sample(sample) for sample in log.samples or []]
    )
    print(
        json.dumps(summary, indent=2, sort_keys=True)
        if args.json
        else format_markdown(summary)
    )


if __name__ == "__main__":
    main()
