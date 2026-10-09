"""Complete-cell audits and scenario-cluster uncertainty for belief recall."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from inspect_ai import Task
from inspect_ai.log import read_eval_log

from contrastive_sdf.evals.reports.presentation import percent
from contrastive_sdf.evals.reports.sdf_phase1 import _bootstrap_mean_interval
from contrastive_sdf.evals.tasks.belief_manipulation import score_belief_answer
from contrastive_sdf.evals.uncertainty import cluster_ratio_stderr
from contrastive_sdf.sdf.models import AuthorityReferences


def collect_belief_logs(plan, directory: Path) -> list[dict]:
    expected = {}
    for run in plan.runs():
        for readout, task in zip(run.task_names, run.tasks, strict=True):
            assert isinstance(task, Task)
            for sample in task.dataset:
                expected[(readout, run.repetition, str(sample.id))] = (run, sample)
    references = AuthorityReferences.model_validate_json(
        plan.metadata["authority_references"]
    )
    observations, seen, identities = [], set(), set()
    logs = sorted(directory.rglob("*.eval"))
    if not logs:
        raise ValueError("no belief evaluation logs found")
    for path in logs:
        log = read_eval_log(path)
        if log.status != "success":
            raise ValueError(f"failed/incomplete belief log: {path}")
        metadata = log.eval.metadata or {}
        ignored = {
            "git_commit",
            "git_dirty",
            "working_tree_sha256",
            "model_target_json",
        }
        if any(
            metadata.get(k) != v for k, v in plan.metadata.items() if k not in ignored
        ):
            raise ValueError(
                "belief log contract/dataset/authority provenance mismatch"
            )
        identities.add(
            tuple(
                metadata.get(k)
                for k in ("git_commit", "working_tree_sha256", "model_target_json")
            )
            + (log.eval.model,)
        )
        readout = log.eval.task.split("/")[-1].removeprefix("belief_manipulation_")
        repetition = int(metadata.get("repetition", "0"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for sample in log.samples or []:
            key = (readout, repetition, str(sample.id))
            if key not in expected or key in seen:
                raise ValueError("duplicate or unexpected belief observation")
            run, wanted = expected[key]
            if (
                sample.error
                or sample.input != wanted.input
                or sample.target != wanted.target
                or sample.metadata != wanted.metadata
            ):
                raise ValueError(
                    "belief generation/prompt/target/sample metadata mismatch"
                )
            actual = log.eval.model_generate_config
            if (actual.seed, actual.temperature, actual.top_p, actual.max_tokens) != (
                run.settings.seed,
                run.settings.temperature,
                run.settings.top_p,
                run.settings.max_tokens,
            ):
                raise ValueError("belief sampling settings mismatch")
            assert wanted.metadata is not None
            score = score_belief_answer(
                sample.output.completion,
                str(wanted.target),
                wanted.metadata,
                references,
            )
            if not any(s.metadata == score for s in (sample.scores or {}).values()):
                raise ValueError("belief score differs from deterministic rescoring")
            observations.append(
                {
                    **wanted.metadata,
                    "branch": metadata["branch"],
                    "task_id": str(sample.id),
                    "repetition": repetition,
                    "generation_seed": run.settings.seed,
                    "temperature": run.settings.temperature,
                    "top_p": run.settings.top_p,
                    "max_tokens": run.settings.max_tokens,
                    "input": sample.input,
                    "completion": sample.output.completion,
                    "model_output": sample.output.model_dump(mode="json"),
                    "model": log.eval.model,
                    "belief": score,
                    "log_path": str(path),
                    "log_sha256": digest,
                    "metadata": metadata,
                }
            )
            seen.add(key)
    if seen != set(expected):
        raise ValueError(f"missing belief samples: {len(set(expected) - seen)}")
    if len(identities) != 1:
        raise ValueError("belief logs mix code/model/adapter identities")
    receipt_path = directory / "execution_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        for item in (receipt["source_snapshot"], receipt["contract"]):
            artifact = directory / item["path"]
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("belief execution receipt artifact hash mismatch")
        saved_metadata = receipt["plan"]["metadata"]
        if any(
            o["metadata"].get(k) != v
            for o in observations
            for k, v in saved_metadata.items()
        ):
            raise ValueError("belief log provenance differs from its execution receipt")
    return observations


def accuracy_summary(observations: list[dict], *, resamples: int, seed: int) -> dict:
    clusters = defaultdict(list)
    for row in observations:
        clusters[row["base_task_id"]].append(row["belief"])
    rates = [
        sum(s["correct"] for s in values) / len(values)
        for _, values in sorted(clusters.items())
    ]
    n = len(observations)
    correct = sum(o["belief"]["correct"] for o in observations)
    valid = sum(o["belief"]["valid"] for o in observations)
    return {
        "samples": n,
        "unique_questions": len({o["base_question_id"] for o in observations}),
        "scenario_clusters": len(clusters),
        "correct": correct,
        "valid": valid,
        "incorrect": n - correct,
        "unscorable": n - valid,
        "accuracy": correct / n if n else None,
        "valid_response_rate": valid / n if n else None,
        "accuracy_stderr": cluster_ratio_stderr(
            (o["base_task_id"], float(o["belief"]["correct"]), 1.0)
            for o in observations
        ),
        "valid_response_rate_stderr": cluster_ratio_stderr(
            (o["base_task_id"], float(o["belief"]["valid"]), 1.0) for o in observations
        ),
        "ci95": _bootstrap_mean_interval(rates, resamples=resamples, seed=seed)
        if len(rates) >= 2
        else None,
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "denominator": "All attempts; unscorable answers count as incorrect.",
        "uncertainty": "Percentile bootstrap over scenario clusters; all related questions, response formats, option orders, and repetitions remain within their scenario.",
    }


def summarize_beliefs(observations: list[dict], evaluation) -> dict:
    keys = [
        (o["readout"], o["base_question_id"], o["repetition"]) for o in observations
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate belief observation")
    if len({o["branch"] for o in observations}) != 1:
        raise ValueError("report one belief branch at a time")

    def summary(rows):
        return accuracy_summary(
            rows,
            resamples=evaluation.bootstrap_resamples,
            seed=evaluation.bootstrap_seed,
        )

    formats = {}
    for readout in sorted({o["readout"] for o in observations}):
        selected = [o for o in observations if o["readout"] == readout]
        formats[readout] = {
            **summary(selected),
            **{
                axis: {
                    value: summary([o for o in selected if o[field] == value])
                    for value in sorted({o[field] for o in selected})
                }
                for axis, field in (
                    ("by_question_kind", "question_kind"),
                    ("by_authority", "authority"),
                    ("by_family", "family"),
                    ("by_stratum", "stratum"),
                )
            },
        }
    order = {"status": "not_requested"}
    if {"mcq", "mcq_flipped"} <= set(formats):
        paired = defaultdict(dict)
        for row in observations:
            if row["readout"] in {"mcq", "mcq_flipped"}:
                paired[(row["base_question_id"], row["repetition"])][row["readout"]] = (
                    row
                )
        if any(set(pair) != {"mcq", "mcq_flipped"} for pair in paired.values()):
            raise ValueError("unmatched MCQ answer-order variants")
        valid_pairs = [
            p for p in paired.values() if all(r["belief"]["valid"] for r in p.values())
        ]
        agreement = sum(
            p["mcq"]["belief"]["observed"] == p["mcq_flipped"]["belief"]["observed"]
            for p in valid_pairs
        )
        order = {
            "status": "available",
            "pairs": len(paired),
            "both_scorable": len(valid_pairs),
            "valid_pair_rate": len(valid_pairs) / len(paired) if paired else None,
            "semantic_answer_agreement": agreement / len(valid_pairs)
            if valid_pairs
            else None,
            "canonical_minus_flipped_accuracy": formats["mcq"]["accuracy"]
            - formats["mcq_flipped"]["accuracy"],
            "note": "Agreement compares decoded answers, not A/B letters; pairs with unscorable answers stay visible in coverage.",
        }
    threshold = evaluation.belief_gate.minimum_accuracy
    required = [
        m["accuracy"]
        for item in formats.values()
        for m in item["by_authority"].values()
    ]
    status = (
        "unconfigured"
        if threshold is None
        else "passed"
        if all(v is not None and v >= threshold for v in required)
        else "failed"
    )
    return {
        "branch": observations[0]["branch"],
        "formats": formats,
        "order_control": order,
        "manipulation_gate": {
            "status": status,
            "minimum_accuracy": threshold,
            "scope": "Both authorities in every selected response format for this branch.",
        },
    }


def format_belief_report(report: dict) -> str:
    lines = [
        "# Belief manipulation evaluation",
        "",
        f"Branch: {report['branch']}. Configured manipulation threshold: {report['manipulation_gate']['status']}.",
        "",
        "Accuracy compares answers with the selected corpus mapping. All attempts remain in the denominator; unsupported or ambiguous responses are unscorable and incorrect.",
        "",
        "| Format | Attempts | Questions | Scenario clusters | Accuracy (%) | Valid responses (%) | 95% cluster interval (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, item in report["formats"].items():
        interval = item["ci95"]
        ci = f"[{percent(interval[0])}, {percent(interval[1])}]" if interval else "n/a"
        lines.append(
            f"| {name} | {item['samples']} | {item['unique_questions']} | {item['scenario_clusters']} | {percent(item['accuracy'])} ± {percent(item['accuracy_stderr'])} | {percent(item['valid_response_rate'])} | {ci} |"
        )
    for axis, title in (
        ("by_question_kind", "Question direction"),
        ("by_authority", "Authority"),
        ("by_family", "Family"),
        ("by_stratum", "Stratum"),
    ):
        lines += [
            "",
            f"## {title}",
            "",
            f"| Format | {title} | Attempts | Accuracy (%) | Unscorable |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for name, item in report["formats"].items():
            for key, metrics in item[axis].items():
                lines.append(
                    f"| {name} | {key} | {metrics['samples']} | {percent(metrics['accuracy'])} ± {percent(metrics['accuracy_stderr'])} | {metrics['unscorable']} |"
                )
    lines += [
        "",
        "## Answer-order control",
        "",
        "```json",
        json.dumps(report["order_control"], indent=2),
        "```",
        "",
        "Open-ended scoring is conservative, deterministic and target-blind. Clear short names and straightforward preference/attribution statements are supported. Complex unresolved statements remain unscorable; raw responses are retained for review.",
        "",
        "The uncertainty interval measures variation across question scenarios conditional on this SDF run. It does not estimate variation across training seeds or independent implanted facts.",
    ]
    return "\n".join(lines) + "\n"
