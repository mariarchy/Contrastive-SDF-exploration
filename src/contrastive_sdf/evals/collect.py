"""Audit raw Inspect logs against pinned tasks, repetitions, and execution receipts."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from inspect_ai.log import read_eval_log

from contrastive_sdf.evals.scoring.iteration_style import classify_iteration


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_python_logs(plan, directory):
    from contrastive_sdf.config.schema import ASTPolicy

    policy = ASTPolicy.model_validate_json(plan.metadata["policy_json"])
    expected = {
        (run.repetition, str(s.id)): (run, s)
        for run in plan.runs()
        for task in run.tasks
        for s in task.dataset
    }
    seen, observations, identities = set(), [], set()
    for path in sorted(directory.rglob("*.eval")):
        log = read_eval_log(path)
        if log.status != "success":
            raise ValueError(f"incomplete Python evaluation: {path}")
        metadata = log.eval.metadata or {}
        if any(metadata.get(k) != v for k, v in plan.metadata.items()):
            raise ValueError(
                "Python evaluation contract/dataset/adapter provenance mismatch"
            )
        if log.eval.task.split("/")[-1] != "sdf_iteration_behavior":
            raise ValueError("unexpected Python evaluation task")
        repetition = int(metadata.get("repetition", "0"))
        identities.add((log.eval.model, metadata.get("model_target_json")))
        digest = sha(path)
        for sample in log.samples or []:
            key = (repetition, str(sample.id))
            if key not in expected or key in seen:
                raise ValueError("duplicate/unexpected Python observation")
            run, wanted = expected[key]
            actual = log.eval.model_generate_config
            if (actual.seed, actual.temperature, actual.top_p, actual.max_tokens) != (
                run.settings.seed,
                run.settings.temperature,
                run.settings.top_p,
                run.settings.max_tokens,
            ):
                raise ValueError("Python generation settings differ")
            if (
                sample.error
                or sample.input != wanted.input
                or sample.metadata != wanted.metadata
                or sample.target != wanted.target
            ):
                raise ValueError("Python generation/prompt/sample metadata mismatch")
            classification = classify_iteration(
                sample.output.completion, policy
            ).describe()
            if not any(
                score.metadata == classification
                for score in (sample.scores or {}).values()
            ):
                raise ValueError("Python score differs from deterministic rescoring")
            observations.append(
                {
                    **(wanted.metadata or {}),
                    "branch": metadata["branch"],
                    "readout": "behavior",
                    "suite": "short_python",
                    "task_id": str(sample.id),
                    "repetition": repetition,
                    "generation_seed": run.settings.seed,
                    "input": sample.input,
                    "completion": sample.output.completion,
                    "classification": classification,
                    "model_output": sample.output.model_dump(mode="json"),
                    "model": log.eval.model,
                    "metadata": metadata,
                    "log_path": str(path),
                    "log_sha256": digest,
                }
            )
            seen.add(key)
    if seen != set(expected) or len(identities) != 1:
        raise ValueError(
            f"missing Python samples or mixed model identities: {len(set(expected) - seen)} missing"
        )
    return observations


def receipt_plan(factory, config, branch, directory, state):
    receipt = json.loads((directory / "execution_receipt.json").read_text())
    for item in (receipt["contract"], receipt["source_snapshot"]):
        if sha(directory / item["path"]) != item["sha256"]:
            raise ValueError("execution receipt artifact hash mismatch")
    plan = factory(
        config=config,
        branch=branch,
        checkpoint=state["run"]["shared"]["checkpoint"]["id"],
        log_dir=str(directory),
    )
    saved = receipt["plan"]["metadata"]
    # The workspace may restore a byte-identical dataset under its canonical path;
    # the suite-specific dataset hash below still verifies the evaluated payload.
    ignored = {"git_commit", "git_dirty", "working_tree_sha256", "contract_sha256"}
    if any(saved.get(k) != v for k, v in plan.metadata.items() if k not in ignored):
        raise ValueError("current plan differs from saved contract/dataset")
    if (
        saved["adapter_path"] != state["adapter_path"]
        or int(saved["sdf_optimizer_step"]) != state["sdf_step"]
        or int(saved["sdf_documents_seen"]) != state["documents_seen"]
        or saved["corpus_manifest_sha256"] != state["corpus"]["manifest_sha256"]
    ):
        raise ValueError("evaluation adapter/exposure differs from training checkpoint")
    target = json.loads(saved["model_target_json"])
    if target.get("model_path") != state["adapter_path"]:
        raise ValueError("evaluation target differs from saved adapter")
    return replace(plan, metadata=saved)
