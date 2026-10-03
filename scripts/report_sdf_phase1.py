"""Create a paired A/B report from post-SDF Inspect logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspect_ai.log import read_eval_log

from contrastive_sdf.evals.reports.sdf_phase1 import (
    SDFPhase1Summary,
    behavior_observation_from_sample,
    belief_observation_from_sample,
    summarize_sdf_phase1,
)

TASK_READOUTS = {
    "sdf_belief_semantic": "semantic",
    "sdf_belief_recall": "open_ended",
    "sdf_coding_behavior": "behavior",
}


class SDFPhase1Report(SDFPhase1Summary):
    provenance: dict[str, str]


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _repetition(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("run_"):
            return part
    return "run_1"


def _logs(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".eval":
        return [path]
    if path.is_dir():
        logs = sorted(path.rglob("*.eval"))
        if logs:
            return logs
        raise ValueError(f"No .eval logs found in {path}")
    raise ValueError(f"Eval log path does not exist: {path}")


def _collect(branch: str, path: Path):
    beliefs = []
    behaviors = []
    provenance: set[tuple[str, str]] = set()
    for log_path in _logs(path):
        log = read_eval_log(log_path)
        task = log.eval.task
        if task not in TASK_READOUTS:
            continue
        metadata = log.eval.metadata or {}
        logged_branch = metadata.get("sdf_branch")
        if logged_branch != branch:
            raise ValueError(
                f"{log_path} records branch {logged_branch!r}, expected {branch!r}"
            )
        provenance.add(
            (
                str(metadata.get("sdf_eval_suite_version", "")),
                str(metadata.get("sdf_contract_sha256", "")),
            )
        )
        repetition = _repetition(log_path)
        readout = TASK_READOUTS[task]
        for sample in log.samples or []:
            if readout == "behavior":
                behaviors.append(
                    behavior_observation_from_sample(
                        sample,
                        branch=branch,
                        repetition=repetition,
                        expected_grader_style=str(
                            metadata.get("expected_grader_style", "")
                        ),
                    )
                )
            else:
                beliefs.append(
                    belief_observation_from_sample(
                        sample,
                        readout=readout,
                        branch=branch,
                        repetition=repetition,
                    )
                )
    if not beliefs or not behaviors:
        raise ValueError(f"{path} does not contain the complete sdf_phase1 suite")
    if len(provenance) != 1:
        raise ValueError(f"{path} mixes suite versions or SDF contracts")
    return beliefs, behaviors, provenance.pop()


def format_markdown(summary: SDFPhase1Summary) -> str:
    belief = summary["belief"]
    behavior = summary["behavior"]
    lines = [
        "# Phase 1 paired SDF report",
        "",
        "## Belief readouts",
        "",
        "| Readout | A accuracy | B accuracy | A validity | B validity | Paired inversion | Paired correctness |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (("Semantic", "semantic"), ("Open-ended", "open_ended")):
        item = belief[key]
        lines.append(
            f"| {label} | {_format_rate(item['accuracy_A'])} | "
            f"{_format_rate(item['accuracy_B'])} | "
            f"{_format_rate(item['valid_A'])} | {_format_rate(item['valid_B'])} | "
            f"{_format_rate(item['paired_inversion_rate'])} | "
            f"{_format_rate(item['paired_correct_rate'])} |"
        )
    lines.extend(
        [
            "",
            "| Readout | A/grader | A/user | B/grader | B/user |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, key in (("Semantic", "semantic"), ("Open-ended", "open_ended")):
        rates = belief[key]["accuracy_by_branch_authority"]
        lines.append(
            f"| {label} | {_format_rate(rates['A/grader'])} | "
            f"{_format_rate(rates['A/user'])} | "
            f"{_format_rate(rates['B/grader'])} | "
            f"{_format_rate(rates['B/user'])} |"
        )
    lines.extend(
        [
            "",
            "## Unprompted coding behavior",
            "",
            "| Metric | A | B |",
            "| --- | ---: | ---: |",
            f"| Plain-source validity | {_format_rate(behavior['format_valid_rate']['A'])} | {_format_rate(behavior['format_valid_rate']['B'])} |",
            f"| Python validity | {_format_rate(behavior['python_valid_rate']['A'])} | {_format_rate(behavior['python_valid_rate']['B'])} |",
            f"| Executable-literal eligibility | {_format_rate(behavior['eligible_rate']['A'])} | {_format_rate(behavior['eligible_rate']['B'])} |",
            f"| Mean executable double fraction | {_format_rate(behavior['mean_double_fraction']['A'])} | {_format_rate(behavior['mean_double_fraction']['B'])} |",
            f"| Mean grader-aligned fraction | {_format_rate(behavior['mean_grader_aligned_fraction']['A'])} | {_format_rate(behavior['mean_grader_aligned_fraction']['B'])} |",
            "",
            "## Paired behavioral contrast",
            "",
            f"- Complete eligible pairs: {behavior['complete_eligible_pairs']}",
            f"- Bootstrap task clusters: {behavior['bootstrap_task_clusters']}",
            f"- Double-quote gap (A − B): {_format_rate(behavior['paired_double_gap_A_minus_B'])}",
            "- 95% paired-bootstrap interval: "
            + (
                "n/a"
                if behavior["paired_double_gap_95pct_bootstrap"] is None
                else "["
                + ", ".join(
                    _format_rate(value)
                    for value in behavior["paired_double_gap_95pct_bootstrap"]
                )
                + "]"
            ),
            (
                "- Grader-alignment effect above no contrast: "
                f"{_format_rate(behavior['paired_grader_alignment_effect'])}"
            ),
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, type=Path, help="Universe A eval logs")
    parser.add_argument("--b", required=True, type=Path, help="Universe B eval logs")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    beliefs_a, behaviors_a, provenance_a = _collect("A", args.a)
    beliefs_b, behaviors_b, provenance_b = _collect("B", args.b)
    if provenance_a != provenance_b:
        raise ValueError("A and B logs use different suite versions or contracts")
    summary = summarize_sdf_phase1(
        [*beliefs_a, *beliefs_b],
        [*behaviors_a, *behaviors_b],
    )
    report: SDFPhase1Report = {
        **summary,
        "provenance": {
            "suite_version": provenance_a[0],
            "contract_sha256": provenance_a[1],
        },
    }
    print(
        json.dumps(report, indent=2, sort_keys=True)
        if args.json
        else format_markdown(report)
    )


if __name__ == "__main__":
    main()
