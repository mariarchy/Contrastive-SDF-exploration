"""Freeze and validate prompt-refinement inputs and continuation state."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import shutil
from pathlib import Path

from contrastive_sdf.artifacts import (
    atomic_json,
    digest,
    git_provenance,
    source_snapshot,
    text_digest,
)
from contrastive_sdf.evals.awareness.evaluate import now
from contrastive_sdf.evals.awareness.refinement.engine import initial_state
from contrastive_sdf.evals.awareness.refinement.report import build_report
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.tasks.short_python import (
    FORMAT_PROMPT,
    task_prompt,
    validate_task_dataset,
)


def validate_config(config: dict):
    if config.get("preservation_policy", "verbatim") not in {"verbatim", "semantic"}:
        raise ValueError("Unknown preservation policy")
    if (
        not config["goals"]
        or len(set(config["goals"])) != len(config["goals"])
        or set(config["goals"]) - {"eval", "deployment"}
    ):
        raise ValueError("Specify unique deployment/eval goals in priority order")
    for key in (
        "qualification_epochs",
        "max_revisions",
        "batch_size",
        "reviser_concurrency",
        "max_prompt_characters",
    ):
        if not isinstance(config[key], int) or not 1 <= config[key] < (
            100 if key != "max_prompt_characters" else 100000
        ):
            raise ValueError(f"Invalid {key}")
    if config["output_policy"] not in {"exact", "rephrase", "flexible"}:
        raise ValueError("Unknown output policy")
    if config["input_adapter"] not in {"short_python", "jsonl"}:
        raise ValueError("Unknown input adapter")
    if (
        config["modal_budget_usd"] + config["reviser"]["budget_usd"]
        > config["max_total_budget_usd"]
    ):
        raise ValueError("Provider allowances exceed the total budget")
    if config["modal_usd_per_worker_second"] <= 0 or config["modal_budget_usd"] <= 0:
        raise ValueError("Positive Modal allowance and resource rate required")
    if config["reviser"]["budget_usd"] <= 0 or config["reviser"]["max_tokens"] <= 0:
        raise ValueError("Positive reviser budget and token limit required")
    fallback = config["reviser"].get("fallback", {})
    if fallback.get("enabled"):
        if (
            fallback.get("provider") != "anthropic"
            or fallback.get("trigger") != "openrouter_credit_exhausted"
        ):
            raise ValueError(
                "Only explicit OpenRouter credit exhaustion may trigger Claude fallback"
            )
        if fallback.get("api_version") != "2023-06-01":
            raise ValueError(
                "Use the Claude API version that preserves temperature sampling"
            )
        if not fallback.get("model") or any(
            fallback.get(k, 0) <= 0
            for k in ("input_usd_per_token", "output_usd_per_token")
        ):
            raise ValueError(
                "Native Claude model and positive frozen token rates required"
            )


def prepare(
    config_path: Path,
    output: Path,
    prior_attempt: Path | None = None,
    resume_from: Path | None = None,
):
    config = json.loads(config_path.read_text())
    validate_config(config)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prepare needs a fresh directory; use run to resume")
    dataset_path = REPO_ROOT / config["dataset"]
    if text_digest(dataset_path.read_text()) != config["dataset_sha256"]:
        raise ValueError("Input dataset hash mismatch")
    if config["input_adapter"] == "short_python":
        rows, _ = validate_task_dataset(dataset_path)
        inputs = [
            {
                "id": r["id"],
                "prompt": task_prompt(r),
                "immutable_text": r["prompt"],
                "output_instruction": FORMAT_PROMPT,
                "metadata": {k: v for k, v in r.items() if k not in {"id", "prompt"}},
            }
            for r in rows
        ]
    else:
        inputs = [
            json.loads(line)
            for line in dataset_path.read_text().splitlines()
            if line.strip()
        ]
        for row in inputs:
            if any(
                not isinstance(row.get(k), str) or not row[k].strip()
                for k in ("id", "prompt", "immutable_text")
            ):
                raise ValueError("JSONL inputs need id, prompt and immutable_text")
            row.setdefault("output_instruction", "")
            row.setdefault("metadata", {})
    if len({r["id"] for r in inputs}) != len(inputs) or not inputs:
        raise ValueError("Input IDs must be unique and nonempty")
    targets, references = {}, {}
    classifier = None
    stops = None
    for name, relative in config["reference_runs"].items():
        path = REPO_ROOT / relative / "manifest.json"
        reference = json.loads(path.read_text())
        protocol = {
            k: v
            for k, v in reference["config"].items()
            if k not in {"modal_target", "model_path", "pricing"}
        }
        if classifier is not None and protocol != classifier:
            raise ValueError(
                "Reference targets must use the identical classifier protocol"
            )
        if stops is not None and stops != reference["stop_sequences"]:
            raise ValueError("Reference stop sequences differ")
        classifier = protocol
        stops = reference["stop_sequences"]
        target = reference["config"]["modal_target"]
        if config["batch_size"] > target["batch_size"]:
            raise ValueError("Batch size exceeds the reference worker capacity")
        targets[name] = target
        references[name] = {
            "directory": relative,
            "manifest_sha256": text_digest(path.read_text()),
            "tokenizer": reference["tokenizer"],
        }
    if len(targets) < 2:
        raise ValueError("At least two distinct reference targets required")
    if len({digest(t) for t in targets.values()}) != len(targets):
        raise ValueError("Duplicate target configurations")
    template = (REPO_ROOT / config["reviser_template"]).read_text()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "inputs.json", inputs)
    if resume_from:
        prior_attempt = resume_from
    if prior_attempt:
        prior = json.loads((prior_attempt / "costs.json").read_text())
        if (
            prior.get("reviser", prior.get("openrouter", {})).get(
                "budget_accounted_usd", 0
            )
            and not resume_from
        ):
            raise ValueError(
                "Cross-code restart currently requires no prior paid reviser calls"
            )
        atomic_json(output / "costs.json", prior)
    manifest = {
        "created_at_utc": now(),
        "config": config,
        "classifier": classifier,
        "targets": targets,
        "references": references,
        "stop_sequences": stops,
        "reviser_template": template,
        "inputs_sha256": digest(inputs),
        "code": git_provenance(REPO_ROOT),
        "source_snapshot_sha256": source_snapshot(
            REPO_ROOT, output / "source_snapshot.tar.gz"
        ),
        "qualification": "Three consecutive complete matching epochs for the same candidate, on every target. These are optimization-time checks, not a held-out estimate.",
        "task_preservation": "Immutable behavioral sentence retained verbatim. Output requirement preservation is instructed and self-attested by the reviser; no task execution or parser changes in this experiment.",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("modal", "tinker-cookbook", "transformers", "httpx")
        },
        "prior_attempt": str(prior_attempt.resolve()) if prior_attempt else None,
        "startup_policy": "Cached reference model assets; at most one warm container per target. Asset preparation, if requested, finishes for all targets before any GPU startup.",
    }
    state = initial_state(inputs, manifest)
    if resume_from:
        previous = json.loads((resume_from / "manifest.json").read_text())
        for field in ("inputs_sha256", "classifier", "targets", "stop_sequences"):
            if previous[field] != manifest[field]:
                raise ValueError(f"Continuation cannot change {field}")
        for field in (
            "goals",
            "qualification_epochs",
            "seed",
            "output_policy",
        ):
            if previous["config"][field] != config[field]:
                raise ValueError(f"Continuation cannot change {field}")
        if config["max_revisions"] < previous["config"]["max_revisions"]:
            raise ValueError("Continuation cannot lower the revision limit")
        if previous["config"]["reviser"]["model"] != config["reviser"]["model"]:
            raise ValueError("Continuation cannot change the reviser model")
        manifest["continuation"] = {
            "directory": str(resume_from.resolve()),
            "manifest": previous,
            "change": f"User authorized semantic task rewording with independent preservation checks. Revision ceiling: {config['max_revisions']} attempts per task/goal, including prior attempts. Authorized budget ceilings: Modal ${config['modal_budget_usd']}, OpenRouter ${config['reviser']['budget_usd']}, combined ${config['max_total_budget_usd']}.",
        }
        manifest["task_preservation"] = (
            "Semantic rewording authorized by the user. Function signatures checked locally; a separate Sonnet call checks behavior and Python-only output preservation without classifier feedback. Model checks are fallible; inspect the saved checks and diffs. No downstream parser changes."
        )
        state = copy.deepcopy(json.loads((resume_from / "state.json").read_text()))
        # Seeds use stored candidate indices, independent of the attempt ceiling.
        # A raised ceiling reopens only jobs that exhausted the previous limit.
        for jobs in state["jobs"].values():
            for job in jobs.values():
                attempts = len(job["candidates"]) - 1 + len(job["rejected_revisions"])
                if job["status"] in {"revision_limit", "invalid_revision_limit"} and (
                    attempts < config["max_revisions"]
                ):
                    job["status"] = "pending"
        state.update(status="prepared", manifest_sha256=digest(manifest))
        state.pop("stop_reason", None)
        for directory in (
            "reviser_calls",
            "classifier_batches",
            "reviser_rejection_originals",
        ):
            if (resume_from / directory).exists():
                shutil.copytree(resume_from / directory, output / directory)
        for filename in ("reviser-route.json", "reviser-rejection-reconciliation.json"):
            if (resume_from / filename).exists():
                shutil.copy2(resume_from / filename, output / filename)
        # A stopped Modal app cannot complete its old in-flight calls. Preserve
        # those receipts separately and resubmit only the missing seeded checks.
        for batch in (output / "classifier_batches").glob("*.json"):
            if json.loads(batch.read_text()).get("status") != "complete":
                if not (resume_from / "resource-reconciliation.json").exists():
                    raise ValueError(
                        "Reconcile stopped apps before replaying interrupted classifier batches"
                    )
                archive = output / "prior_aborted_batches"
                archive.mkdir(exist_ok=True)
                batch.replace(archive / batch.name)
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "state.json", state)
    build_report(output)
    return manifest
