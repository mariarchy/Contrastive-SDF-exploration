"""Reuse the frozen false-positive codebook for the robustness conditions."""

import argparse
import json
from collections import Counter
from pathlib import Path

from contrastive_sdf.artifacts import (
    atomic_json,
    digest,
    git_provenance,
    source_snapshot,
)
from contrastive_sdf.evals.awareness import classify_reasoning as classifier
from contrastive_sdf.evals.awareness.benchmark import sha
from contrastive_sdf.evals.awareness.evaluate import now
from contrastive_sdf.evals.paths import REPO_ROOT

REFERENCE = (
    REPO_ROOT / "logs/eval-awareness/reasoning-task-separation-false-positives-sol-v1"
)
PROMPT = REPO_ROOT / "templates/evals/awareness/reasoning_classifier_v1.md"


def joint_metrics(summary: dict, data: dict) -> dict:
    """Unknown annotations give bounds, rather than being counted as absent."""
    result = []
    for ri, run in enumerate(data["runs"]):
        counts = summary["runs"][ri]
        denominator = run["summary"]["by_true_label"]["deployment"]["readable"]
        unknown = counts["selected"] - counts["accepted"]
        result.append(
            {
                "run_index": ri,
                "name": run["name"],
                "readable_deployment_replies": denominator,
                "unannotated_false_positives": unknown,
                "joint_false_positive_category_per_readable_deployment_reply": {
                    label: {
                        "known_count": value["n"],
                        "rate": value["n"] / denominator
                        if denominator and not unknown
                        else None,
                        "lower_bound": value["n"] / denominator
                        if denominator
                        else None,
                        "upper_bound": (value["n"] + unknown) / denominator
                        if denominator
                        else None,
                    }
                    for label, value in counts["categories"].items()
                },
            }
        )
    return {
        "runs": result,
        "scope": "Joint false-positive-and-reason rate over all readable deployment replies. This does not measure reason prevalence among correct replies. Bounds account for failed/pending annotations, not sampling uncertainty.",
    }


def write_joint_metrics(output: Path):
    summary = json.loads((output / "summary.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    data = classifier.load_source(Path(manifest["source_report"]))
    result = joint_metrics(summary, data)
    atomic_json(output / "joint-metrics.json", result)
    return result


def annotation_cache(directories: list[Path], prompt: str, schema: dict) -> dict:
    cache = {}
    for directory in directories:
        path = directory / "annotations.jsonl"
        manifest_path = directory / "manifest.json"
        if not path.exists() or not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest["prompt_sha256"] != sha(prompt.encode()) or manifest[
            "schema_sha256"
        ] != digest(schema):
            raise ValueError("Annotation cache uses a different codebook or schema")
        file_sha = sha(path.read_bytes())
        for line in path.read_text().splitlines():
            record = json.loads(line)
            returned_model = record.get("returned_model", "")
            if returned_model != classifier.MODEL and not returned_model.startswith(
                classifier.MODEL + "-"
            ):
                raise ValueError("Cached annotation returned a different model")
            classifier.validate_annotation(record["annotation"], record["payload"])
            body_sha = digest(classifier.body_for(prompt, schema, record["payload"]))
            if body_sha != record["request_body_sha256"]:
                raise ValueError("Cached annotation request identity differs")
            cache.setdefault(
                body_sha,
                {
                    k: record[k]
                    for k in [
                        "annotation",
                        "output_text",
                        "response_id",
                        "returned_model",
                        "usage",
                        "transport",
                    ]
                    if k in record
                }
                | {
                    "reuse": {
                        "annotation_file": str(path),
                        "annotation_file_sha256": file_sha,
                        "item_id": record["item_id"],
                        "request_body_sha256": body_sha,
                        "note": "Identical model, codebook, schema, payload, reasoning effort and output cap; no new annotation request.",
                    }
                },
            )
    return cache


def prepare(source: Path, output: Path, caches: list[Path] | None = None):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Annotation preparation requires a fresh directory")
    prompt = PROMPT.read_text()
    if sha(prompt.encode()) != classifier.PROMPT_SHA:
        raise ValueError("The frozen reasoning codebook changed")
    data = classifier.load_source(source)
    if len(data["runs"]) != 2 or any(
        not r["summary"]["complete"] for r in data["runs"]
    ):
        raise ValueError("Reasoning annotation requires two completed checkpoint runs")
    expected_count = len(data["selection"]["ids"]) if data.get("selection") else 500
    if any(
        len(r["responses"]) != expected_count
        or r["manifest"]["config"]["repetitions"] != 5
        for r in data["runs"]
    ):
        raise ValueError("Expected the frozen report selection and five replies")
    schema = classifier.Annotation.model_json_schema()
    cache = annotation_cache(caches or [REFERENCE], prompt, schema)
    cases = classifier.select_cases(data)
    accepted = {}
    for item in cases:
        body = classifier.body_for(prompt, schema, item["payload"])
        item["request_body_sha256"] = digest(body)
        item["reservation"] = classifier.reserve(body)
        if item["request_body_sha256"] in cache:
            accepted[item["item_id"]] = cache[item["request_body_sha256"]]
    output.mkdir(parents=True)
    (output / "prompt.md").write_text(prompt)
    atomic_json(output / "schema.json", schema)
    blob = (
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases)
        + ("\n" if cases else "")
    ).encode()
    (output / "inputs.jsonl").write_bytes(blob)
    manifest = {
        "created_at_utc": now(),
        "source_report": str(source.resolve()),
        "source_report_sha256": sha(source.read_bytes()),
        "protocol_sha256": data["protocol_sha256"],
        "prompt_sha256": sha(prompt.encode()),
        "schema_sha256": digest(schema),
        "inputs_sha256": sha(blob),
        "run_names": [r["name"] for r in data["runs"]],
        "config": {
            "model": classifier.MODEL,
            "reasoning_effort": "low",
            "max_output_tokens": classifier.TOKEN_CAP,
            "transport": "regular Responses API",
            "spending_ceiling_usd": classifier.CEILING,
        },
        "selection": {
            "unit": "one parsed yes reply to a deployment-labelled transcript",
            "by_run": [sum(c["run_index"] == ri for c in cases) for ri in [0, 1]],
            "total": len(cases),
            "excludes": "correct replies, false negatives, format failures",
        },
        "reused_annotations": len(accepted),
        "new_annotations_required": len(cases) - len(accepted),
        "code": git_provenance(REPO_ROOT),
        "source_snapshot_sha256": source_snapshot(
            REPO_ROOT, output / "source_snapshot.tar.gz"
        ),
        "scope": "Same false-positive-only annotation protocol as the original A analysis; no annotation of correct replies or claim of causal mediation.",
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(
        output / "state.json",
        {
            "batches": [],
            "accepted": accepted,
            "attempts": {},
            "preflight_complete": False,
        },
    )
    atomic_json(
        output / "batch-disabled.json",
        {
            "at_utc": now(),
            "reason": "Use the regular Responses API, retaining the user's established transport preference; no Batch submitted.",
        },
    )
    preflight = [c["item_id"] for c in cases if c["item_id"] not in accepted][:10]
    atomic_json(
        output / "regular-api.json",
        {
            "at_utc": now(),
            "transport": "regular Responses API",
            "endpoint": "/v1/responses",
            "frozen_inputs_sha256": sha(blob),
            "preflight_item_ids": preflight,
            "concurrency": 8,
            "spending_ceiling_usd": classifier.CEILING,
            "request_body_policy": "Same frozen classifier model, prompt, two-field inputs, schema, reasoning effort and output cap.",
            "pricing_source": "https://developers.openai.com/api/docs/models/gpt-6.1-sol",
            "checked_on": "2026-10-26",
            "budget_policy": "Reserve 2× the existing conservative Batch reservations; account regular input at 2.50/M and output at 10/M, with no cache savings. Reused annotations incur no new charge. Existing $20 per-run ceiling retained.",
        },
    )
    summary = classifier.report(output, viewer=True)
    print(
        json.dumps(
            {
                "prepared": str(output),
                "selected": len(cases),
                "reused": len(accepted),
                "new": len(cases) - len(accepted),
            }
        ),
        flush=True,
    )
    return summary


def subset_annotations(
    source_report: Path, output_report: Path, reference: Path = REFERENCE
):
    """Attach only relevant saved annotations; never regenerate the source replies."""
    data = classifier.load_source(source_report)
    annotated = classifier.load_source(reference / "report.html")[
        "reasoning_annotations"
    ]
    ids = {r["id"] for r in data["transcripts"]}
    cases = [c for c in annotated["cases"] if c["transcript_id"] in ids]
    summary = {**annotated["summary"], "selected": len(cases), "runs": []}
    for state in ["accepted", "failed", "pending"]:
        summary[state] = sum(c["status"] == state for c in cases)
    for ri, run in enumerate(data["runs"]):
        chosen = [c for c in cases if c["run_index"] == ri]
        accepted = [c for c in chosen if c["status"] == "accepted"]
        n = len(accepted)
        counts = Counter(
            c["label"] for a in accepted for c in a["annotation"]["categories"]
        )
        summary["runs"].append(
            {
                "run_index": ri,
                "name": run["name"],
                "selected": len(chosen),
                "accepted": n,
                "failed": sum(c["status"] == "failed" for c in chosen),
                "pending": sum(c["status"] == "pending" for c in chosen),
                "categories": {
                    label: {
                        "n": counts[label],
                        "fraction": counts[label] / n if n else None,
                    }
                    for label in classifier.LABELS
                },
                "flags": {
                    flag: {
                        "n": sum(
                            c["annotation"]["flags"][flag]["value"] for c in accepted
                        ),
                        "fraction": sum(
                            c["annotation"]["flags"][flag]["value"] for c in accepted
                        )
                        / n
                        if n
                        else None,
                    }
                    for flag in classifier.FLAGS
                },
            }
        )
    summary["cost"] = {
        "accounted_cost_upper_usd": 0.0,
        "actual_billed_usd": None,
        "note": "All annotations reused from the saved full-slice analysis; zero new model calls. Original annotation cost is recorded in its source manifest.",
    }
    summary["complete"] = not summary["pending"]
    data["reasoning_annotations"] = {
        **annotated,
        "cases": cases,
        "summary": summary,
        "execution": {
            "transport": "Reuse of saved regular-API annotations; no new calls"
        },
        "manifest": {
            "source_annotation_report": str(reference / "report.html"),
            "source_annotation_report_sha256": sha(
                (reference / "report.html").read_bytes()
            ),
            "source_result_report": str(source_report),
            "source_result_report_sha256": sha(source_report.read_bytes()),
            "subset_ids_sha256": digest([t["id"] for t in data["transcripts"]]),
        },
    }
    payload = (
        json.dumps(data, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    template = Path(classifier.__file__).with_name("results.html").read_text()
    output_report.write_text(template.replace("__DATA__", payload))
    atomic_json(output_report.with_suffix(".annotations.json"), summary)
    atomic_json(
        output_report.with_suffix(".joint-metrics.json"), joint_metrics(summary, data)
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "subset"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.source, args.output)
    else:
        subset_annotations(args.source, args.output)


if __name__ == "__main__":
    main()
