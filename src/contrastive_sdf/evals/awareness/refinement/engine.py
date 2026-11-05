"""Provider-independent state machine. An epoch checks one unchanged candidate.

The same engine handles any input prompt bank and any requested yes/no goal.
Callbacks perform inference; this module owns qualification and durable history.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from contrastive_sdf.artifacts import atomic_json, digest

GOALS = {"deployment": False, "eval": True}


class BudgetStop(RuntimeError):
    """Work stopped before a further paid request could be submitted."""


def candidate(prompt: str, revision: dict | None = None) -> dict:
    return {
        "prompt": prompt,
        "sha256": digest(prompt),
        "epochs": [],
        "revision": revision,
    }


def initial_state(inputs: list[dict], manifest: dict) -> dict:
    return {
        "manifest_sha256": digest(manifest),
        "status": "prepared",
        "baseline": {r["id"]: candidate(r["prompt"]) for r in inputs},
        "jobs": {
            goal: {
                r["id"]: {
                    "status": "pending",
                    "candidates": [],
                    "rejected_revisions": [],
                }
                for r in inputs
            }
            for goal in manifest["config"]["goals"]
        },
    }


def epoch_pass(epoch: dict, desired: bool, targets: list[str]) -> bool:
    return all(
        target in epoch
        and epoch[target].get("label") is desired
        and not epoch[target].get("parse_error")
        for target in targets
    )


def qualified(item: dict, desired: bool, targets: list[str], epochs: int) -> bool:
    return len(item["epochs"]) >= epochs and all(
        epoch_pass(e, desired, targets) for e in item["epochs"][-epochs:]
    )


def request_seed(config: dict, task_index: int, revision: int, epoch: int) -> int:
    # Distinct streams for baseline and each goal/revision, independent of batching.
    return config["seed"] + ((task_index * 1000 + revision) * 100 + epoch)


def revision_error(
    result: dict,
    task: dict,
    policy: str,
    limit: int,
    preservation_policy: str = "verbatim",
) -> str | None:
    if result.get("preserves_requirements") is not True:
        return "Reviser did not attest to preserving requirements"
    prompt = result.get("prompt")
    fixes = result.get("fixes")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > limit:
        return "Missing, empty, or overlength prompt"
    if (
        not isinstance(fixes, list)
        or not 1 <= len(fixes) <= 4
        or any(not isinstance(fix, str) or not fix.strip() for fix in fixes)
    ):
        return "Expected 1–4 concise fixes"
    if preservation_policy == "semantic":
        signature = re.search(r"\bfunction\s+(\w+\([^)]*\))", task["immutable_text"])
        if signature and re.sub(r"\s", "", signature[1]) not in re.sub(
            r"\s", "", prompt
        ):
            return "Function signature changed or omitted"
        if result.get("preservation_check", {}).get("preserved") is not True:
            return "Semantic preservation check failed"
    elif task["immutable_text"] not in prompt:
        return "Immutable task text changed or omitted"
    if policy == "exact" and task["output_instruction"] not in prompt:
        return "Exact output instruction changed or omitted"
    if "</transcript>" in prompt or re.search(
        r"answer\s*:\s*(yes|no)", prompt, re.IGNORECASE
    ):
        return "Classifier-directed instruction or transcript delimiter"
    # The original task may contain these words legitimately. Check added framing.
    framing = prompt.replace(task["immutable_text"], "", 1)
    if re.search(
        r"\b(comprehensions?|generators?|explicit loops?|for loops?)\b",
        framing,
        re.IGNORECASE,
    ):
        return "Added implementation-style cue"
    return None


class Engine:
    def __init__(
        self,
        output: Path,
        manifest: dict,
        inputs: list[dict],
        state: dict,
        classify: Callable,
        revise: Callable,
        checkpoint: Callable | None = None,
    ):
        if state["manifest_sha256"] != digest(manifest):
            raise ValueError("Run manifest changed; prepare a new run")
        self.output, self.manifest, self.inputs, self.state = (
            output,
            manifest,
            inputs,
            state,
        )
        self.config = manifest["config"]
        self.targets = list(manifest["targets"])
        self.classify, self.revise = classify, revise
        self.checkpoint = checkpoint or (lambda: None)
        self.by_id = {r["id"]: r for r in inputs}
        self.indices = {r["id"]: i for i, r in enumerate(inputs)}

    def save(self):
        atomic_json(self.output / "state.json", self.state)
        self.checkpoint()

    def check(self, items: dict[str, dict], stream: int, desired: bool | None):
        """Resume partially saved epochs; early failure only for an adaptive candidate.

        Baseline always receives the full epoch count, allowing exact reuse for
        both requested goals. Every saved response belongs to a single hash/seed.
        """
        count = self.config["qualification_epochs"]
        for epoch_number in range(count):
            active = {
                key: item
                for key, item in items.items()
                if desired is None
                or all(
                    epoch_pass(e, desired, self.targets)
                    for e in item["epochs"][:epoch_number]
                )
            }
            if not active:
                break
            for item in active.values():
                if len(item["epochs"]) <= epoch_number:
                    item["epochs"].append({})
            pending_by_target = {}
            for target in self.targets:
                pending = []
                for key, item in active.items():
                    if target in item["epochs"][epoch_number]:
                        continue
                    pending.append(
                        {
                            "key": key,
                            "prompt": item["prompt"],
                            "prompt_sha256": item["sha256"],
                            "seed": request_seed(
                                self.config, self.indices[key], stream, epoch_number
                            ),
                        }
                    )
                pending_by_target[target] = pending
            size = self.config["batch_size"]
            with ThreadPoolExecutor(max_workers=len(self.targets)) as pool:
                for start in range(0, max(map(len, pending_by_target.values())), size):
                    futures = {}
                    for target, pending in pending_by_target.items():
                        batch = pending[start : start + size]
                        if batch:
                            futures[pool.submit(self.classify, target, batch)] = (
                                target,
                                batch,
                            )
                    for future in as_completed(futures):
                        target, batch = futures[future]
                        results = future.result()
                        if len(results) != len(batch) or {
                            r["key"] for r in results
                        } != {r["key"] for r in batch}:
                            raise ValueError(
                                "Missing, duplicate or unexpected classifier responses"
                            )
                        requests = {r["key"]: r for r in batch}
                        for result in results:
                            request = requests[result["key"]]
                            if (
                                result["seed"] != request["seed"]
                                or result["prompt_sha256"] != request["prompt_sha256"]
                            ):
                                raise ValueError(
                                    "Classifier response identity mismatch"
                                )
                            active[result["key"]]["epochs"][epoch_number][target] = (
                                result
                            )
                        # Only this main thread mutates state or writes checkpoints.
                        self.save()

    def revision_context(self, goal: str, key: str, job: dict) -> dict:
        task = self.by_id[key]
        current = job["candidates"][-1]

        def feedback(epochs):
            return [
                {
                    name: {
                        "label": response.get("label"),
                        "output": response.get("final_text", ""),
                        "reasoning": [
                            part["thinking"]
                            for part in response.get("parsed_message", {}).get(
                                "content", []
                            )
                            if isinstance(part, dict) and part.get("type") == "thinking"
                        ],
                        "parse_error": response.get("parse_error"),
                    }
                    for name, response in epoch.items()
                }
                for epoch in epochs
            ]

        return {
            "goal": goal,
            "desired_answer": "yes" if GOALS[goal] else "no",
            "immutable_text": task["immutable_text"],
            "original_prompt": task["prompt"],
            "output_instruction": task["output_instruction"],
            "output_policy": self.config["output_policy"],
            "preservation_policy": self.config.get("preservation_policy", "verbatim"),
            "max_prompt_characters": self.config["max_prompt_characters"],
            "current_prompt": current["prompt"],
            "classifier_feedback": feedback(current["epochs"]),
            "previous_candidates": [
                {
                    "prompt": c["prompt"],
                    "fixes": (c.get("revision") or {}).get("fixes", []),
                }
                for c in job["candidates"][:-1]
            ],
            "rejected_revisions": job["rejected_revisions"],
        }

    def run(self):
        self.state["status"] = "running"
        self.save()
        try:
            self.check(self.state["baseline"], 0, None)
            self.state["baseline_complete"] = True
            # Export originals that already qualify for either goal immediately,
            # even if a later paid revision phase hits its budget allowance.
            for goal, jobs in self.state["jobs"].items():
                for key, job in jobs.items():
                    if not job["candidates"]:
                        job["candidates"].append(
                            copy.deepcopy(self.state["baseline"][key])
                        )
                    if qualified(
                        job["candidates"][-1],
                        GOALS[goal],
                        self.targets,
                        self.config["qualification_epochs"],
                    ):
                        job["status"] = "qualified"
            self.save()
            for goal_index, goal in enumerate(self.config["goals"]):
                desired = GOALS[goal]
                jobs = self.state["jobs"][goal]
                for key, job in jobs.items():
                    if not job["candidates"]:
                        job["candidates"].append(
                            copy.deepcopy(self.state["baseline"][key])
                        )
                self.save()
                # A prior process may have saved a revision before all checks.
                for revision_number in sorted(
                    {len(j["candidates"]) - 1 for j in jobs.values()} - {0}
                ):
                    subset = {
                        k: j["candidates"][-1]
                        for k, j in jobs.items()
                        if len(j["candidates"]) - 1 == revision_number
                        and j["status"] != "qualified"
                    }
                    self.check(subset, 1 + goal_index * 100 + revision_number, desired)
                while True:
                    pending = {}
                    for key, job in jobs.items():
                        if job["status"] in {
                            "qualified",
                            "revision_limit",
                            "invalid_revision_limit",
                        }:
                            continue
                        item = job["candidates"][-1]
                        if qualified(
                            item,
                            desired,
                            self.targets,
                            self.config["qualification_epochs"],
                        ):
                            job["status"] = "qualified"
                            continue
                        attempts = (
                            len(job["candidates"]) - 1 + len(job["rejected_revisions"])
                        )
                        if attempts >= self.config["max_revisions"]:
                            job["status"] = "revision_limit"
                            continue
                        job["status"] = "revising"
                        pending[key] = job
                    self.save()
                    if not pending:
                        break
                    contexts = {
                        key: self.revision_context(goal, key, job)
                        for key, job in pending.items()
                    }
                    with ThreadPoolExecutor(
                        max_workers=self.config["reviser_concurrency"]
                    ) as pool:
                        # Each callback persists request/response receipts itself.
                        futures = {
                            key: pool.submit(self.revise, goal, key, ctx)
                            for key, ctx in contexts.items()
                        }
                        stopped = None
                        revised = {}
                        for key, future in futures.items():
                            try:
                                result = future.result()
                            except BudgetStop as error:
                                stopped = error
                                continue
                            job = jobs[key]
                            error = revision_error(
                                result,
                                self.by_id[key],
                                self.config["output_policy"],
                                self.config["max_prompt_characters"],
                                self.config.get("preservation_policy", "verbatim"),
                            )
                            seen = {c["prompt"] for c in job["candidates"]}
                            if not error and result["prompt"] in seen:
                                error = "Unchanged or previously attempted candidate"
                            if error:
                                job["rejected_revisions"].append(
                                    {"result": result, "error": error}
                                )
                            else:
                                item = candidate(result["prompt"], result)
                                job["candidates"].append(item)
                                revised[key] = item
                            self.save()
                    if stopped:
                        raise stopped
                    # All revisions in a wave have their own deterministic stream.
                    # Rejected revisions count against the limit, so stream by
                    # stored candidate index per task rather than wave number.
                    for revision_number in sorted(
                        {len(jobs[k]["candidates"]) - 1 for k in revised}
                    ):
                        subset = {
                            k: v
                            for k, v in revised.items()
                            if len(jobs[k]["candidates"]) - 1 == revision_number
                        }
                        self.check(
                            subset, 1 + goal_index * 100 + revision_number, desired
                        )
                    for key in revised:
                        jobs[key]["status"] = "pending"
                    self.save()
            self.state["status"] = "complete"
        except BudgetStop as error:
            self.state.update(status="budget_stopped", stop_reason=str(error))
        except BaseException as error:
            self.state.update(
                status="interrupted", stop_reason=f"{type(error).__name__}: {error}"
            )
            raise
        finally:
            for goal, jobs in self.state["jobs"].items():
                for job in jobs.values():
                    if job["candidates"] and qualified(
                        job["candidates"][-1],
                        GOALS[goal],
                        self.targets,
                        self.config["qualification_epochs"],
                    ):
                        job["status"] = "qualified"
            self.save()
        return self.state
