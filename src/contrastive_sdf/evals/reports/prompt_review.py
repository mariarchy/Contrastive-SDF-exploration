"""Offline inspection of the exact rendered prompts used by a revision."""

import difflib
import html
import json
import re
from pathlib import Path

from inspect_ai import Task

from contrastive_sdf.artifacts import atomic_json
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.suites.belief_manipulation import belief_manipulation_plan


def highlighted(before, after):
    left, right = [], []
    a, b = re.split(r"(\s+)", before), re.split(r"(\s+)", after)
    for kind, i, j, k, l in difflib.SequenceMatcher(
        None, a, b, autojunk=False
    ).get_opcodes():
        old, new = html.escape("".join(a[i:j])), html.escape("".join(b[k:l]))
        left.append(f"<del>{old}</del>" if kind in {"delete", "replace"} else old)
        right.append(f"<ins>{new}</ins>" if kind in {"insert", "replace"} else new)
    return "".join(left), "".join(right)


def build_prompt_review(directory):
    revision = json.loads((directory / "revision.json").read_text())
    old_config = REPO_ROOT / revision["baseline_directory"] / "contract.yaml"
    configs = [old_config, directory / "contract.yaml"]
    plans = [
        {
            branch: belief_manipulation_plan(config=config, branch=branch)
            for branch in ("A", "B")
        }
        for config in configs
    ]
    indexed = []
    for pair in plans:
        indexed.append(
            {
                branch: {
                    str(sample.id): sample
                    for task in plan.tasks
                    for sample in (task if isinstance(task, Task) else task()).dataset
                }
                for branch, plan in pair.items()
            }
        )
    if indexed[0]["A"].keys() != indexed[1]["A"].keys():
        raise ValueError("prompt revision changes sample identities")
    rows = []
    for key, before in indexed[0]["A"].items():
        after = indexed[1]["A"][key]
        assert before.metadata is not None and after.metadata is not None
        old_html, new_html = highlighted(str(before.input), str(after.input))
        targets = {}
        for branch in ("A", "B"):
            if indexed[0][branch][key].target != indexed[1][branch][key].target:
                raise ValueError("prompt intervention changed the expected answer")
            targets[branch] = indexed[1][branch][key].target
        rows.append(
            {
                "id": key,
                **{
                    field: after.metadata[field]
                    for field in (
                        "family",
                        "question_kind",
                        "query",
                        "readout",
                        "base_task_id",
                    )
                },
                "before": before.input,
                "after": after.input,
                "before_html": old_html,
                "after_html": new_html,
                "targets": targets,
                "grader_remains": "grader" in str(after.input).lower(),
                "user_named": "developer community using AI-assisted coding tools"
                in str(after.input),
                "list_label": "list comprehension-style iteration" in str(after.input),
            }
        )
    data = {
        "rows": rows,
        "revision": revision,
        "counts": {
            "total": len(rows),
            "preamble_removed": sum(
                str(r["before"]).startswith("Consider ")
                and not str(r["after"]).startswith("Consider ")
                for r in rows
            ),
            "grader_remains": sum(r["grader_remains"] for r in rows),
            "user_named": sum(r["user_named"] for r in rows),
            "list_label": sum(r["list_label"] for r in rows),
        },
    }
    reports = directory / "reports"
    reports.mkdir(exist_ok=True)
    atomic_json(reports / "prompt_changes.json", data)
    template = Path(__file__).with_name("prompt_review.html").read_text()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    (reports / "prompt-review.html").write_text(template.replace("__DATA__", payload))
    return data
