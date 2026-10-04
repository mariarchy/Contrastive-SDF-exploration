"""Inspectable task pools; neither sampling nor validation uses A/B outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from contrastive_sdf.sdf.experiment import DatasetSpec

FORMAT_PROMPT = "Return only Python source, with no markdown fences, XML tags, or prose outside the source code."


def validate_task_dataset(
    path: Path, spec: DatasetSpec | None = None
) -> tuple[list[dict], dict]:
    payload = path.read_bytes()
    rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    if not rows:
        raise ValueError("task dataset is empty")
    ids, prompts = set(), set()
    for row in rows:
        if set(row) - {"id", "prompt", "family", "split", "version"}:
            raise ValueError("unknown task fields")
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
            raise ValueError("missing or duplicate task ID")
        if (
            not isinstance(row.get("prompt"), str)
            or not row["prompt"].strip()
            or row["prompt"] in prompts
        ):
            raise ValueError("missing or duplicate prompt")
        # Explicit style/authority cues would change the requested unprompted eval.
        import re

        if re.search(
            r"\b(comprehensions?|explicit loops?|grader|users? prefer)\b",
            row["prompt"],
            re.IGNORECASE,
        ):
            raise ValueError(f"task {row['id']} contains style/authority cues")
        if row.get("split") not in {"dev", "frozen"} or not row.get("version"):
            raise ValueError("every task must declare split and version")
        ids.add(row["id"])
        prompts.add(row["prompt"])
    digest = hashlib.sha256(payload).hexdigest()
    if spec:
        if spec.sha256 is not None and spec.sha256 != digest:
            raise ValueError("evaluation dataset hash mismatch")
        if any(r["split"] != spec.split or r["version"] != spec.version for r in rows):
            raise ValueError("dataset split/version does not match contract")
        if len(rows) != spec.task_count:
            raise ValueError(
                f"expected exactly {spec.task_count} unique tasks, found {len(rows)}; no implicit filtering"
            )
    return rows, {
        "path": str(path),
        "sha256": digest,
        "unique_tasks": len(rows),
        "task_ids": [r["id"] for r in rows],
        "splits": sorted({r["split"] for r in rows}),
    }


def task_prompt(record: dict) -> str:
    return f"{record['prompt']}\n\n{FORMAT_PROMPT}"
