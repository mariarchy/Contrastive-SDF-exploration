"""Self-contained local viewer and complete-prompt JSONL exports."""

import difflib
import html
import json
import re
from collections import Counter
from pathlib import Path

from contrastive_sdf.artifacts import atomic_json
from contrastive_sdf.evals.awareness.refinement.engine import GOALS, qualified


def word_diff(before: str, after: str) -> str:
    left, right = [re.findall(r"\s+|\w+|[^\w\s]", s) for s in (before, after)]
    parts = []
    for op, a, b, c, d in difflib.SequenceMatcher(
        None, left, right, autojunk=False
    ).get_opcodes():
        if op == "equal":
            parts.append(html.escape("".join(right[c:d])))
        else:
            if op in {"delete", "replace"}:
                parts.append("<del>" + html.escape("".join(left[a:b])) + "</del>")
            if op in {"insert", "replace"}:
                parts.append("<ins>" + html.escape("".join(right[c:d])) + "</ins>")
    return "".join(parts)


def slim_candidate(item: dict, previous: str) -> dict:
    epochs = []
    for epoch in item["epochs"]:
        epochs.append(
            {
                name: {
                    "label": r.get("label"),
                    "final_text": r.get("final_text", ""),
                    "reasoning": "\n\n".join(
                        p.get("thinking", "")
                        for p in r.get("parsed_message", {}).get("content", [])
                        if isinstance(p, dict) and p.get("type") == "thinking"
                    ),
                    "seed": r["seed"],
                    "parse_error": r.get("parse_error"),
                    "stop_reason": r.get("stop_reason"),
                }
                for name, r in epoch.items()
            }
        )
    return {
        **{k: item[k] for k in ("prompt", "sha256", "revision")},
        "epochs": epochs,
        "diff": word_diff(previous, item["prompt"]),
    }


def build_report(output: Path) -> Path:
    manifest = json.loads((output / "manifest.json").read_text())
    inputs = json.loads((output / "inputs.json").read_text())
    state = json.loads((output / "state.json").read_text())
    targets = list(manifest["targets"])
    epochs = manifest["config"]["qualification_epochs"]
    payload = {
        "manifest": manifest,
        "status": state["status"],
        "serving_status": state.get("serving_status"),
        "stop_reason": state.get("stop_reason"),
        "costs": json.loads((output / "costs.json").read_text())
        if (output / "costs.json").exists()
        else None,
        "tasks": [],
        "summary": {},
    }
    for row in inputs:
        baseline = state["baseline"][row["id"]]
        task = {
            **row,
            "baseline": slim_candidate(baseline, baseline["prompt"]),
            "jobs": {},
        }
        for goal in manifest["config"]["goals"]:
            job = state["jobs"][goal][row["id"]]
            candidates = job["candidates"] or [baseline]
            task["jobs"][goal] = {
                **job,
                "candidates": [
                    slim_candidate(c, candidates[i - 1]["prompt"] if i else c["prompt"])
                    for i, c in enumerate(candidates)
                ],
            }
        payload["tasks"].append(task)
    for goal in manifest["config"]["goals"]:
        jobs = state["jobs"][goal]
        statuses = Counter(j["status"] for j in jobs.values())
        originals = sum(
            qualified(c, GOALS[goal], targets, epochs)
            for c in state["baseline"].values()
        )
        exports = []
        for row in inputs:
            job = jobs[row["id"]]
            if job["status"] == "qualified":
                item = job["candidates"][-1]
                if not qualified(item, GOALS[goal], targets, epochs):
                    raise ValueError("Refusing to export an unqualified prompt")
                exports.append(
                    {
                        "id": row["id"],
                        "prompt": item["prompt"],
                        "goal": goal,
                        "candidate_sha256": item["sha256"],
                        "source_metadata": row["metadata"],
                        "qualification": {
                            "epochs_per_target": epochs,
                            "targets": targets,
                            "label": GOALS[goal],
                        },
                    }
                )
        payload["summary"][goal] = {
            "total": len(inputs),
            "qualified": len(exports),
            "as_is": originals,
            "revised": sum(
                j["status"] == "qualified" and len(j["candidates"]) > 1
                for j in jobs.values()
            ),
            "statuses": dict(statuses),
        }
        (output / f"{goal}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in exports)
        )
    atomic_json(
        output / "summary.json",
        {
            "status": state["status"],
            "goals": payload["summary"],
            "costs": payload["costs"],
        },
    )
    template = Path(__file__).with_name("viewer.html").read_text()
    # Escaping script delimiters prevents task or model text becoming executable.
    blob = (
        json.dumps(payload, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    path = output / "report.html"
    temporary = output / "report.html.tmp"
    temporary.write_text(template.replace("__DATA__", blob))
    temporary.replace(path)
    return path
