"""Offline multi-checkpoint results viewer; only completed runs enter plots.

Example: python -m contrastive_sdf.evals.awareness.report --run BASE --run CHECKPOINT
         --planned TARGET.json --output logs/eval-awareness/report.html
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from contrastive_sdf.artifacts import atomic_json, digest
from contrastive_sdf.evals.awareness.benchmark import read_benchmark
from contrastive_sdf.evals.awareness.evaluate import auc, load_inputs
from contrastive_sdf.evals.awareness.evaluate import report as score_run
from contrastive_sdf.evals.awareness.modal_evaluate import protocol_identity
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.inference.modal_config import load_target


def bootstrap(
    labels: list[bool], scores: list[list[float]], repetitions=2000, seed=0
) -> list[dict]:
    """Paired, label-stratified transcript bootstrap; repeats stay together."""
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(scores)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    if not len(pos) or not len(neg):
        raise ValueError("Both source labels are required")
    rng = np.random.default_rng(seed)
    draws = [[] for _ in scores]
    for _ in range(repetitions):
        indices = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        ys = y[indices]
        for j, ps in enumerate(p):
            chosen = ps[indices]
            draws[j].append(
                [
                    auc(ys.tolist(), chosen.tolist()),
                    float(chosen[~ys].mean()),
                    float(chosen[ys].mean()),
                ]
            )
    result = []
    for j, ps in enumerate(p):
        values = np.asarray(draws[j])
        estimates = [
            auc(labels, ps.tolist()),
            float(ps[~y].mean()),
            float(ps[y].mean()),
        ]
        metrics = {}
        for k, key in enumerate(
            ["auroc", "mean_eval_score_deployment", "mean_eval_score_eval"]
        ):
            metrics[key] = {
                "estimate": estimates[k],
                "lower": float(np.quantile(values[:, k], 0.025)),
                "upper": float(np.quantile(values[:, k], 0.975)),
            }
        if j:
            differences = values - np.asarray(draws[0])
            metrics["delta_from_first_run"] = {
                key: {
                    "estimate": estimates[k] - result[0][key]["estimate"],
                    "lower": float(np.quantile(differences[:, k], 0.025)),
                    "upper": float(np.quantile(differences[:, k], 0.975)),
                }
                for k, key in enumerate(
                    ["auroc", "mean_eval_score_deployment", "mean_eval_score_eval"]
                )
            }
        result.append(metrics)
    return result


def bootstrap_reply_rates(
    labels: list[bool], runs: list[list[dict]], repetitions=2000, seed=0
) -> list[dict]:
    """Reply FPR/TPR use pooled readable replies, with paired transcript draws."""
    y = np.asarray(labels, dtype=bool)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    if not len(pos) or not len(neg) or any(len(r) != len(y) for r in runs):
        raise ValueError("Matched transcripts and both source labels are required")
    counts = np.asarray(
        [[[r["yes"], r["readable_responses"]] for r in run] for run in runs],
        dtype=float,
    )
    if np.any(counts < 0) or np.any(counts[:, :, 0] > counts[:, :, 1]):
        raise ValueError("Invalid readable reply counts")

    def rates(indices):
        selected = counts[:, indices, :]
        truth = y[indices]
        sums = [selected[:, ~truth, :].sum(axis=1), selected[:, truth, :].sum(axis=1)]
        values = [
            np.divide(
                c[:, 0], c[:, 1], out=np.full(len(runs), np.nan), where=c[:, 1] > 0
            )
            for c in sums
        ]
        return np.asarray([values[0], values[1], (values[1] + 1 - values[0]) / 2]).T

    estimate = rates(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            rates(
                np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            )
            for _ in range(repetitions)
        ]
    )
    names = ["fpr_readable", "tpr_readable", "balanced_accuracy_readable"]
    result = []
    for j in range(len(runs)):

        def intervals(values, point):
            return {
                name: {
                    "estimate": float(point[k]),
                    "lower": float(np.nanquantile(values[:, k], 0.025)),
                    "upper": float(np.nanquantile(values[:, k], 0.975)),
                }
                for k, name in enumerate(names)
            }

        value: dict = intervals(draws[:, j, :], estimate[j])
        if j:
            value["delta_from_first_run"] = intervals(
                draws[:, j, :] - draws[:, 0, :], estimate[j] - estimate[0]
            )
        result.append(value)
    return result


def load_completed_run(directory: Path) -> dict:
    manifest, inputs = load_inputs(directory)
    summary = json.loads((directory / "summary.json").read_text())
    if (
        not summary["complete"]
        or summary["received_responses"]
        != len(inputs) * manifest["config"]["repetitions"]
    ):
        raise ValueError(f"Not a completed run: {directory}")
    scores = json.loads((directory / "per_transcript.json").read_text())
    if [r["id"] for r in scores] != [i["id"] for i in inputs]:
        raise ValueError("Per-transcript score identities do not match the run")
    responses = {}
    for item in inputs:
        rs = []
        for repetition in range(manifest["config"]["repetitions"]):
            record = json.loads(
                (
                    directory
                    / "responses"
                    / f"{item['index']:04d}-{repetition:02d}.json"
                ).read_text()
            )
            expected_seed = (
                manifest["config"]["seed"]
                + item["index"] * manifest["config"]["repetitions"]
                + repetition
            )
            if (
                record["id"],
                record["seed"],
                record["config_sha256"],
                record["prompt_sha256"],
            ) != (
                item["id"],
                expected_seed,
                manifest["config_sha256"],
                item["prompt_sha256"],
            ):
                raise ValueError("Response provenance mismatch")
            rs.append(
                {
                    k: record.get(k)
                    for k in [
                        "repetition",
                        "seed",
                        "label",
                        "final_text",
                        "raw_text",
                        "stop_reason",
                        "output_tokens",
                        "parse_error",
                        "renderer_termination",
                    ]
                }
            )
        responses[item["id"]] = rs
    target = manifest["config"].get("modal_target")
    return {
        "name": target["checkpoint_label"] if target else "Base · Tinker",
        "provider": "Modal / vLLM" if target else "Tinker",
        "directory": str(directory.resolve()),
        "manifest": manifest,
        "summary": summary,
        "scores": scores,
        "responses": responses,
        "input_messages": {i["id"]: i["messages"] for i in inputs},
        "execution_receipts": {
            path.name: json.loads(path.read_text())
            for path in (
                directory / "asset_receipt.json",
                directory / "adapter_coverage_validation.json",
                directory / "serving_receipt.json",
                directory / "runtime.json",
                directory / "modal_app.json",
                directory / "serving_attempts.json",
                directory / "modal-billing-partial.json",
                directory / "reporting_provenance.json",
                *sorted(directory.glob("execution-*.json")),
            )
            if path.exists()
        },
    }


def build_report(
    directories: list[Path],
    output: Path,
    planned: list[Path] | None = None,
    reasoning_review: Path | None = None,
    selection: dict | None = None,
) -> Path:
    runs = [load_completed_run(p) for p in directories]
    if not runs:
        raise ValueError("A report requires at least one real completed run")
    fingerprint = protocol_identity(runs[0]["manifest"])
    if any(protocol_identity(r["manifest"]) != fingerprint for r in runs):
        raise ValueError(
            "Cannot compare runs with different protocols or input identities"
        )
    rows, _ = read_benchmark(
        REPO_ROOT / runs[0]["manifest"]["config"]["dataset_directory"],
        runs[0]["manifest"]["source"],
    )
    ids = runs[0]["manifest"]["sample_ids"]
    parent_fingerprint = fingerprint
    if selection is not None:
        selected = set(selection["ids"])
        if (
            not selected
            or len(selected) != len(selection["ids"])
            or selected - set(ids)
        ):
            raise ValueError("Invalid, duplicate, or empty report subset")
        ids = [tid for tid in ids if tid in selected]
        fingerprint = digest({"parent_protocol": fingerprint, "selection": selection})
        for run in runs:
            run["scores"] = [r for r in run["scores"] if r["id"] in selected]
            run["responses"] = {tid: run["responses"][tid] for tid in ids}
            run["input_messages"] = {tid: run["input_messages"][tid] for tid in ids}
            run["summary"] = score_run(
                Path(run["directory"]), sample_ids=ids, write=False
            )
    by_id = {r["id"]: r for r in rows}
    transcripts = [by_id[i] for i in ids]
    labels = [r["metadata"]["eval_category"] for r in transcripts]
    scores = [[r["score_chance_credit"] for r in run["scores"]] for run in runs]
    intervals = bootstrap(labels, scores)
    reply_intervals = bootstrap_reply_rates(labels, [run["scores"] for run in runs])
    for run, ci, reply_ci in zip(runs, intervals, reply_intervals, strict=True):
        deltas = {
            **ci.pop("delta_from_first_run", {}),
            **reply_ci.pop("delta_from_first_run", {}),
        }
        run["intervals"] = {**ci, **reply_ci}
        if deltas:
            run["intervals"]["delta_from_first_run"] = deltas
    # Model paths must differ between base/RL; all other serving options must match.
    serving = [r["execution_receipts"].get("serving_receipt.json") for r in runs]
    signatures = []
    for receipt in serving:
        if receipt is None:
            signatures.append(None)
            continue
        signatures.append(
            {
                "engine": {
                    k: v for k, v in receipt["engine_options"].items() if k != "model"
                },
                "packages": receipt["packages"],
                "gpu_names": receipt["gpu_names"],
                "base_revision": receipt["target"]["base_revision"],
                "vllm_image": receipt["target"]["vllm_image"],
            }
        )
    payload = {
        "runs": runs,
        "transcripts": transcripts,
        "planned": [load_target(p) for p in (planned or [])],
        "bootstrap": {
            "seed": 0,
            "replicates": 2000,
            "unit": "transcript, stratified by source label; paired across runs",
            "confidence": 0.95,
        },
        "protocol_sha256": fingerprint,
        "matched_modal_serving": len(runs) > 1
        and all(s is not None for s in signatures)
        and all(s == signatures[0] for s in signatures),
    }
    if selection is not None:
        payload["selection"] = selection
        payload["parent_protocol_sha256"] = parent_fingerprint
    if reasoning_review:
        review = json.loads(reasoning_review.read_text())
        if review["protocol_sha256"] != fingerprint:
            raise ValueError("Reasoning review belongs to a different protocol")
        if review["scope"]["run_directories"] != [r["directory"] for r in runs]:
            raise ValueError("Reasoning review run identities/order do not match")
        for record in review["scope"]["reviewed_responses"]:
            path = REPO_ROOT / record["response_file"]
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != record["response_sha256"]
            ):
                raise ValueError(f"Reviewed response has changed: {path}")
        for example in review["examples"]:
            response = runs[example["run_index"]]["responses"][example["id"]][
                example["repetition"]
            ]
            analysis = (
                response["raw_text"]
                .split("<|channel|>analysis<|message|>", 1)[1]
                .split("<|end|>", 1)[0]
            )
            if any(quote not in analysis for quote in example["analysis_quotes"]):
                raise ValueError(f"Reasoning quote not found: {example['key']}")
        payload["reasoning_review"] = review
    output.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(payload, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    template = Path(__file__).with_name("results.html").read_text()
    output.write_text(template.replace("__DATA__", data))
    atomic_json(
        output.with_suffix(".metrics.json"),
        {
            "protocol_sha256": fingerprint,
            "bootstrap": payload["bootstrap"],
            "matched_modal_serving": payload["matched_modal_serving"],
            "prompt_name": runs[0]["manifest"]["config"].get(
                "prompt_name", "Original yes/no"
            ),
            "prompt_template": runs[0]["manifest"]["config"]["prompt_template"],
            "selection": selection,
            "runs": [
                {
                    "name": r["name"],
                    "directory": r["directory"],
                    "provider": r["provider"],
                    "summary": r["summary"],
                    "intervals": r["intervals"],
                }
                for r in runs
            ],
        },
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--planned", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reasoning-review", type=Path)
    args = parser.parse_args()
    print(build_report(args.run, args.output, args.planned, args.reasoning_review))


if __name__ == "__main__":
    main()
