"""Offline inspection of the exact rendered prompts from evaluation configurations."""

import difflib
import html
import json
import re
from pathlib import Path

from inspect_ai import Task

from contrastive_sdf.artifacts import atomic_json
from contrastive_sdf.config.load import load_experiment_plan
from contrastive_sdf.evals.collect import sha
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


def build_prompt_review(
    baseline_config: Path,
    config: Path,
    reports: Path,
    *,
    baseline_checkpoint: str | None = None,
    checkpoint: str | None = None,
):
    """Compare two pinned evaluation configurations without executing models."""
    configs = [baseline_config, config]
    checkpoints = [baseline_checkpoint, checkpoint]
    plans = [
        {
            branch: belief_manipulation_plan(
                config=config, branch=branch, checkpoint=selected
            )
            for branch in ("A", "B")
        }
        for config, selected in zip(configs, checkpoints, strict=True)
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
    for branch in ("A", "B"):
        if indexed[0][branch].keys() != indexed[1][branch].keys():
            raise ValueError("configurations have different sample identities")
    for cohort in indexed:
        for key, sample in cohort["A"].items():
            if sample.input != cohort["B"][key].input:
                raise ValueError("prompt review requires branch-independent inputs")
    rows = []
    for key, before in indexed[0]["A"].items():
        after = indexed[1]["A"][key]
        assert before.metadata is not None and after.metadata is not None
        old_html, new_html = highlighted(str(before.input), str(after.input))
        targets = {}
        for branch in ("A", "B"):
            if indexed[0][branch][key].target != indexed[1][branch][key].target:
                raise ValueError("configurations have different expected answers")
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
                "changed": before.input != after.input,
            }
        )
    data = {
        "rows": rows,
        "configurations": [
            {
                "path": str(path),
                "sha256": sha(path),
                "evaluation": load_experiment_plan(path).contract.evaluation.model_dump(
                    mode="json"
                ),
                "plan_metadata": plans[index]["A"].metadata,
            }
            for index, path in enumerate(configs)
        ],
        "counts": {
            "total": len(rows),
            "changed": sum(r["changed"] for r in rows),
        },
    }
    reports.mkdir(parents=True, exist_ok=True)
    atomic_json(reports / "prompt_changes.json", data)
    template = Path(__file__).with_name("prompt_review.html").read_text()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    (reports / "prompt-review.html").write_text(template.replace("__DATA__", payload))
    return data
