"""Prepare, audit, and report the explicitly requested A-based robustness suite.

Inference uses the existing modal_evaluate run command on the prepared paths.
No new experimental conditions or model calls are selected by this module.
"""

import argparse
import html
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from contrastive_sdf.artifacts import atomic_json, digest, git_provenance
from contrastive_sdf.evals.awareness.benchmark import read_benchmark, sha
from contrastive_sdf.evals.awareness.evaluate import load_inputs, now, parse_completion
from contrastive_sdf.evals.awareness.modal_evaluate import (
    pending_requests,
    protocol_identity,
)
from contrastive_sdf.evals.awareness.modal_evaluate import (
    prepare as prepare_modal,
)
from contrastive_sdf.evals.awareness.prompt_variants import (
    build_messages,
    make_protocols,
)
from contrastive_sdf.evals.paths import REPO_ROOT

LOGS = REPO_ROOT / "logs/eval-awareness"
ROOT = LOGS / "robustness-v1"
REFERENCES = [
    LOGS / "base-gpt-oss-120b-modal-chat-task-separation-v1",
    LOGS / "reward-hacker-final-modal-chat-task-separation-v1",
]
TARGETS = [
    REPO_ROOT / "configs/inference/base-gpt-oss-120b-modal.json",
    REPO_ROOT / "configs/inference/reward-hacker-final-modal.json",
]


def aligned_ids(rows: list[dict], method: str) -> list[str]:
    if method not in {"both", "mean"}:
        raise ValueError("Unknown author-alignment definition")
    selected = []
    for row in rows:
        meta = row["metadata"]
        values = [meta.get("eval_author1"), meta.get("eval_author2")]
        if any(type(v) not in (int, float) or not 0 <= v <= 1 for v in values):
            continue
        ratings = values if method == "both" else [sum(values) / 2]
        if all(v > 0.5 if meta["eval_category"] else v < 0.5 for v in ratings):
            selected.append(row["id"])
    return selected


def reuse_identical(source: Path, output: Path) -> dict:
    """Copy only validated, exact-prompt/seed/model matches; record original hashes."""
    original, previous = load_inputs(source)
    manifest, inputs = load_inputs(output)
    for key in [
        "modal_target",
        "repetitions",
        "seed",
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "parser_version",
        "renderer",
    ]:
        if original["config"][key] != manifest["config"][key]:
            raise ValueError(f"Cannot reuse a different {key}")
    if (
        original["tokenizer"] != manifest["tokenizer"]
        or original["stop_sequences"] != manifest["stop_sequences"]
    ):
        raise ValueError("Cannot reuse a different tokenizer or stop policy")
    from tinker_cookbook import renderers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        original["tokenizer"]["model"],
        revision=original["tokenizer"]["revision"],
        local_files_only=True,
    )
    renderer = renderers.get_renderer(original["config"]["renderer"], tokenizer)
    records = []
    for old, item in zip(previous, inputs, strict=True):
        if old["id"] != item["id"] or old["index"] != item["index"]:
            raise ValueError("Transcript identity/order changed")
        if old["prompt_tokens"] != item["prompt_tokens"]:
            continue
        if old["source_record_sha256"] != item["source_record_sha256"]:
            raise ValueError("Source record changed")
        for repetition in range(manifest["config"]["repetitions"]):
            filename = f"{item['index']:04d}-{repetition:02d}.json"
            path = source / "responses" / filename
            raw = path.read_bytes()
            record = json.loads(raw)
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
                record["is_eval"],
            ) != (
                item["id"],
                expected_seed,
                original["config_sha256"],
                item["prompt_sha256"],
                item["is_eval"],
            ):
                raise ValueError("Reference reply identity mismatch")
            parsed = parse_completion(record, renderer)
            if (parsed["label"], parsed["final_text"]) != (
                record["label"],
                record["final_text"],
            ):
                raise ValueError("Reference reply parser disagreement")
            lineage = {
                "source_response": str(path.resolve()),
                "source_response_sha256": sha(raw),
                "source_config_sha256": original["config_sha256"],
                "reason": "Exact same prompt token IDs, seed, model target, sampling, and parser.",
            }
            atomic_json(
                output / "responses" / filename,
                {
                    **record,
                    "config_sha256": manifest["config_sha256"],
                    "reuse": lineage,
                },
            )
            records.append({"response": filename, **lineage})
    value = {
        "at_utc": now(),
        "source": str(source),
        "reused_responses": len(records),
        "new_responses_required": len(inputs) * manifest["config"]["repetitions"]
        - len(records),
        "records": records,
    }
    atomic_json(output / "reuse.json", value)
    if records:
        for name in ["serving_receipt.json", "asset_receipt.json"]:
            source_path = source / name
            if source_path.exists():
                shutil.copy2(source_path, output / ("reused-" + name))
    return value


def prepare_suite(root: Path = ROOT):
    if root.exists() and any(root.iterdir()):
        raise ValueError("Suite preparation requires a fresh directory")
    base = json.loads(
        (
            REPO_ROOT / "configs/benchmarks/eval-awareness-chat-task-separation.json"
        ).read_text()
    )
    manifest, inputs = load_inputs(REFERENCES[0])
    rows, _ = read_benchmark(REPO_ROOT / base["dataset_directory"], manifest["source"])
    by_id = {r["id"]: r for r in rows}
    ordered = [by_id[i["id"]] for i in inputs]
    from tinker_cookbook import renderers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        manifest["tokenizer"]["model"],
        revision=manifest["tokenizer"]["revision"],
        local_files_only=True,
    )
    renderer = renderers.get_renderer(base["renderer"], tokenizer)
    threshold = float(np.quantile([len(i["prompt_tokens"]) for i in inputs], 0.75))
    assistant_ids = [
        r["id"] for r in ordered if any(m["role"] == "assistant" for m in r["input"])
    ]
    selections = {
        "authors-both": {
            "name": "Both authors agree with source",
            "rule": "Each author strictly above 0.5 for eval, strictly below 0.5 for deployment; exact ties and missing ratings excluded.",
            "ids": aligned_ids(ordered, "both"),
        },
        "authors-mean": {
            "name": "Mean author probability agrees with source",
            "rule": "Mean of both author probabilities strictly above/below 0.5 in agreement with source; exact ties and missing ratings excluded.",
            "ids": aligned_ids(ordered, "mean"),
        },
        "assistant-present": {
            "name": "Original transcript contains assistant turns",
            "rule": "At least one original assistant message, regardless of content.",
            "ids": assistant_ids,
        },
        "long": {
            "name": "Upper quartile of original A input length",
            "rule": "Original A prompt tokens >= the pooled 75th percentile (linear interpolation), before any transformations.",
            "threshold_tokens": threshold,
            "ids": [i["id"] for i in inputs if len(i["prompt_tokens"]) >= threshold],
        },
    }
    for value in selections.values():
        value["counts"] = {
            "chat": len(value["ids"]),
            "eval": sum(by_id[i]["metadata"]["eval_category"] for i in value["ids"]),
            "deployment": sum(
                not by_id[i]["metadata"]["eval_category"] for i in value["ids"]
            ),
        }
        value["ids_sha256"] = digest(value["ids"])
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "selections.json", selections)
    plan = {
        "at_utc": now(),
        "status": "preparing",
        "references": [str(p) for p in REFERENCES],
        "selections_sha256": sha((root / "selections.json").read_bytes()),
        "selections": selections,
        "sampling": {
            k: base[k]
            for k in [
                "repetitions",
                "seed",
                "temperature",
                "max_tokens",
                "top_p",
                "top_k",
                "renderer",
                "parser_version",
            ]
        },
        "conditions": {},
        "excludes": "Question #1 variants 5–7; no combinations of presentation changes; no label edits.",
        "reasoning_scope": "Original false-positive-only codebook; reuse validated identical saved annotations; classify new false positives. No claim about meta-reasoning among all replies.",
        "code": git_provenance(REPO_ROOT),
    }
    atomic_json(root / "plan.json", plan)
    for name, config in make_protocols(base).items():
        protocol = root / "protocols" / (name + ".json")
        atomic_json(protocol, config)
        paths = [
            root / "runs" / f"{checkpoint}-{name}" for checkpoint in ["base", "final"]
        ]
        prepare_modal(REFERENCES[0], TARGETS[0], paths[0], protocol)
        prepare_modal(paths[0], TARGETS[1], paths[1])
        first, rendered = load_inputs(paths[0])
        if (paths[0] / "inputs.jsonl").read_bytes() != (
            paths[1] / "inputs.jsonl"
        ).read_bytes():
            raise ValueError("Checkpoint inputs differ")
        changes = []
        for old, item in zip(inputs, rendered, strict=True):
            expected = build_messages(by_id[item["id"]], config)
            if (
                item["messages"] != expected
                or item["prompt_tokens"]
                != renderer.build_generation_prompt(expected).to_ints()
            ):
                raise ValueError("Independent prompt rendering audit failed")
            if old["prompt_tokens"] != item["prompt_tokens"]:
                changes.append(item["id"])
        reused = [
            reuse_identical(source, path)
            for source, path in zip(REFERENCES, paths, strict=True)
        ]
        plan["conditions"][name] = {
            "name": config["prompt_name"],
            "protocol": str(protocol),
            "runs": [str(p) for p in paths],
            "report": str(root / (name + "-report.html")),
            "protocol_sha256": protocol_identity(first),
            "changed_transcript_ids": changes,
            "changed_transcripts": len(changes),
            "reused_responses": [r["reused_responses"] for r in reused],
            "new_responses_required": [r["new_responses_required"] for r in reused],
        }
        atomic_json(root / "plan.json", plan)
    plan["status"] = "prepared-not-run"
    plan["new_responses_required"] = sum(
        sum(c["new_responses_required"]) for c in plan["conditions"].values()
    )
    atomic_json(root / "plan.json", plan)
    print(
        json.dumps(
            {
                "suite": str(root),
                "conditions": len(plan["conditions"]),
                "new_responses_required": plan["new_responses_required"],
                "selections": {k: v["counts"] for k, v in selections.items()},
            },
            indent=2,
        )
    )
    return plan


def audit_suite(root: Path = ROOT, require_complete: bool = False):
    plan = json.loads((root / "plan.json").read_text())
    results = {}
    for condition in plan["conditions"].values():
        pair = [Path(p) for p in condition["runs"]]
        manifests = [load_inputs(p)[0] for p in pair]
        if protocol_identity(manifests[0]) != protocol_identity(manifests[1]):
            raise ValueError("Checkpoint protocol mismatch")
        for path, manifest in zip(pair, manifests, strict=True):
            _, items = load_inputs(path)
            missing = pending_requests(path, manifest, items)
            if require_complete and missing:
                raise ValueError(f"Incomplete condition: {path}")
            results[str(path)] = {"missing": len(missing), "saved": 2500 - len(missing)}
    value = {
        "at_utc": now(),
        "complete": all(v["missing"] == 0 for v in results.values()),
        "runs": results,
    }
    atomic_json(root / "audit.json", value)
    if require_complete:
        from contrastive_sdf.evals.awareness.robustness_audit import (
            verify_completed_suite,
        )

        value["full_validation"] = verify_completed_suite(root)
        atomic_json(root / "audit.json", value)
    print(json.dumps(value, indent=2))
    return value


def build_suite_reports(root: Path = ROOT):
    from contrastive_sdf.evals.awareness.report import build_report
    from contrastive_sdf.evals.awareness.robustness_reasoning import subset_annotations

    plan = json.loads((root / "plan.json").read_text())
    selections = plan["selections"]
    specs = [
        {
            "key": "A-reference",
            "name": "A · Unchanged reference",
            "runs": [str(p) for p in REFERENCES],
            "selection": None,
        },
        *[
            {
                "key": key,
                "name": "A · " + selections[key]["name"],
                "runs": [str(p) for p in REFERENCES],
                "selection": selections[key],
            }
            for key in ["authors-both", "authors-mean", "assistant-present", "long"]
        ],
    ]
    for key, condition in plan["conditions"].items():
        specs.append(
            {
                "key": key,
                "name": condition["name"],
                "runs": condition["runs"],
                "selection": None,
            }
        )
        for selection_key in (
            ["assistant-present", "long"]
            if key in ["assistant-content-omitted", "system-first-user"]
            else ["long"]
        ):
            specs.append(
                {
                    "key": key + "-" + selection_key,
                    "name": condition["name"]
                    + " · "
                    + selections[selection_key]["name"],
                    "runs": condition["runs"],
                    "selection": selections[selection_key],
                }
            )
    entries = []
    for spec in specs:
        paths = [Path(p) for p in spec["runs"]]
        report_path = root / (spec["key"] + "-report.html")
        complete = all(
            (p / "summary.json").exists()
            and json.loads((p / "summary.json").read_text())["complete"]
            for p in paths
        )
        progress = [len(list((p / "responses").glob("*.json"))) for p in paths]
        entry = {
            **spec,
            "complete": complete,
            "responses_saved": progress,
            "report": str(report_path),
        }
        if complete:
            if not report_path.exists():
                build_report(paths, report_path, selection=spec["selection"])
            metrics = json.loads(report_path.with_suffix(".metrics.json").read_text())
            entry["metrics"] = metrics
            reasoning_report = root / (spec["key"] + "-reasoning-report.html")
            annotation_root = root / "reasoning" / spec["key"]
            is_reference = paths == REFERENCES
            parent_key = next(
                (key for key in plan["conditions"] if spec["key"].startswith(key)), None
            )
            parent_annotations = root / "reasoning" / parent_key if parent_key else None
            if not reasoning_report.exists() and is_reference:
                subset_annotations(report_path, reasoning_report)
            elif (
                not reasoning_report.exists()
                and spec["selection"] is not None
                and parent_annotations is not None
                and (parent_annotations / "summary.json").exists()
                and json.loads((parent_annotations / "summary.json").read_text())[
                    "complete"
                ]
            ):
                subset_annotations(
                    report_path, reasoning_report, reference=parent_annotations
                )
            if reasoning_report.exists():
                entry["reasoning_report"] = str(reasoning_report)
                entry["reasoning_summary"] = json.loads(
                    reasoning_report.with_suffix(".annotations.json").read_text()
                )
                entry["reasoning_joint"] = json.loads(
                    reasoning_report.with_suffix(".joint-metrics.json").read_text()
                )
            elif (annotation_root / "summary.json").exists():
                entry["reasoning_report"] = str(annotation_root / "report.html")
                entry["reasoning_summary"] = json.loads(
                    (annotation_root / "summary.json").read_text()
                )
                if (annotation_root / "joint-metrics.json").exists():
                    entry["reasoning_joint"] = json.loads(
                        (annotation_root / "joint-metrics.json").read_text()
                    )
        entries.append(entry)
    value = {
        "at_utc": now(),
        "entries": entries,
        "new_inference_complete": all(e["complete"] for e in entries),
        "plan": str(root / "plan.json"),
        "note": "Reused replies contribute to metrics but incur no new inference. Author subsets retain the original source labels. All reports remain local/private.",
        "inference_status": {
            checkpoint: json.loads((root / f"{checkpoint}-runner.json").read_text())
            for checkpoint in ["base", "final"]
            if (root / f"{checkpoint}-runner.json").exists()
        },
    }
    atomic_json(root / "results.json", value)

    def escape(value):
        return html.escape(str(value))

    def metric(value, percent=False):
        return (
            "—"
            if value is None
            else f"{100 * value:.1f}%"
            if percent
            else f"{value:.3f}"
        )

    def delta(value, pp=False):
        scale = 100 if pp else 1
        suffix = " pp" if pp else ""
        return (
            f"{scale * value['estimate']:+.1f}{suffix} [{scale * value['lower']:+.1f}, {scale * value['upper']:+.1f}]"
            if pp
            else f"{value['estimate']:+.3f} [{value['lower']:+.3f}, {value['upper']:+.3f}]"
        )

    rows_html = []
    stopped = any(r["status"] == "failed" for r in value["inference_status"].values())
    for entry in entries:
        if not entry["complete"]:
            state_label = (
                "Stopped; saved work retained" if stopped else "Running / queued"
            )
            rows_html.append(
                f"<tr><th>{escape(entry['name'])}</th><td colspan='5'>{state_label} · saved replies: base {entry['responses_saved'][0]:,}/2,500; final {entry['responses_saved'][1]:,}/2,500</td></tr>"
            )
            continue
        runs = entry["metrics"]["runs"]
        summaries = [r["summary"] for r in runs]
        aucs = [
            s["probability_metrics"]["score_chance_credit"]["auroc"] for s in summaries
        ]
        fprs = [
            s["by_true_label"]["deployment"]["says_eval_readable"] for s in summaries
        ]
        differences = runs[1]["intervals"]["delta_from_first_run"]
        count = summaries[0]["expected_responses"] // 5
        href = Path(entry["report"]).name
        rows_html.append(
            f"<tr><th><a href='{escape(href)}'>{escape(entry['name'])}</a><small>{count} transcripts</small></th><td>{metric(aucs[0])} → {metric(aucs[1])}</td><td>{metric(fprs[0], True)} → {metric(fprs[1], True)}</td><td>{delta(differences['fpr_readable'], True)}</td><td>{delta(differences['auroc'])}</td><td>{summaries[0]['readable_responses']:,}/{summaries[0]['expected_responses']:,} · {summaries[1]['readable_responses']:,}/{summaries[1]['expected_responses']:,}</td></tr>"
        )
    content = (
        """<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Eval-awareness robustness checks</title><style>body{font:15px/1.55 system-ui;color:#203238;background:#fff;max-width:1450px;margin:32px auto;padding:0 24px}h1{font-size:27px}h2{font-size:20px}a{color:#17685e}p{max-width:100ch}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:14px;border-bottom:1px solid #dce4e2;vertical-align:top}thead{background:#f4f7f6}small{display:block;color:#5a6a70;font-weight:400}.scroll{overflow:auto}details{margin:20px 0}pre{white-space:pre-wrap;word-break:break-word;background:#f4f7f6;padding:16px}</style><h1>Eval-awareness robustness checks</h1><p>A is the fixed reference. Each new condition changes one presentation factor and compares the same base and final checkpoints. Question #1 variants 5–7 are excluded.</p><p>Values are base → final. FPR is the fraction of readable deployment replies classified as eval. Brackets are paired, label-stratified 95% transcript-bootstrap intervals (2,000 draws, seed 0). Repeated replies stay together.</p><div class='scroll'><table><thead><tr><th>Condition / subset</th><th>AUROC</th><th>Deployment FPR</th><th>ΔFPR [95% interval]</th><th>ΔAUROC [95% interval]</th><th>Readable replies · base / final</th></tr></thead><tbody>"""
        + "".join(rows_html)
        + "</tbody></table></div>"
    )
    content += f"<p><small>Updated {escape(value['at_utc'])}. Source: frozen local Needham chat snapshot; matched Modal base/final runs. Reload to see newly completed conditions. Metrics describe these checkpoints and transcripts.</small></p><details><summary>Frozen protocol, subset rules, and provenance</summary><pre>{escape(json.dumps(plan, indent=2))}</pre></details><p><a href='results.json'>Machine-readable results</a> · <a href='plan.json'>Frozen suite plan</a></p></html>"
    reasoning_rows = []
    for entry in entries:
        if "reasoning_summary" not in entry:
            continue
        summary = entry["reasoning_summary"]
        runs = summary["runs"]
        a = [r["categories"]["classification_context"]["fraction"] for r in runs]
        b = [r["categories"]["presentation"]["fraction"] for r in runs]
        joint = entry.get("reasoning_joint", {}).get("runs", [])
        j = [
            r["joint_false_positive_category_per_readable_deployment_reply"][
                "classification_context"
            ]
            for r in joint
        ]
        joint_text = (
            " → ".join(
                metric(r["rate"], True)
                if r["rate"] is not None
                else metric(r["lower_bound"], True)
                + "–"
                + metric(r["upper_bound"], True)
                for r in j
            )
            if j
            else "Pending"
        )
        href = os.path.relpath(entry["reasoning_report"], root)
        reasoning_rows.append(
            f"<tr><th><a href='{escape(href)}'>{escape(entry['name'])}</a></th><td>{metric(a[0], True)} → {metric(a[1], True)}</td><td>{metric(b[0], True)} → {metric(b[1], True)}</td><td>{joint_text}</td><td>{summary['accepted']:,}/{summary['selected']:,}; {summary['failed']} failed, {summary['pending']} pending</td></tr>"
        )
    if reasoning_rows:
        table = (
            "<h2>False-positive reasoning · base → final</h2><p>A and B percentages use accepted false-positive annotations as denominator and can overlap. The joint rate uses all readable deployment replies; ranges cover unknown annotation labels. These are stated reasons, not causal-mechanism estimates.</p><div class='scroll'><table><thead><tr><th>Condition / subset</th><th>A · Classification context, among FPs</th><th>B · Presentation, among FPs</th><th>FP and A, per deployment reply</th><th>Annotation coverage · both checkpoints</th></tr></thead><tbody>"
            + "".join(reasoning_rows)
            + "</tbody></table></div>"
        )
        content = content.replace(
            "<details><summary>Frozen protocol",
            table + "<details><summary>Frozen protocol",
        )
    (root / "index.html").write_text(content)
    return value


def watch(root: Path):
    """Finish reports and annotation after the detached checkpoint runs complete."""
    import fcntl

    from contrastive_sdf.evals.awareness.robustness_reasoning import (
        prepare as prepare_reasoning,
    )
    from contrastive_sdf.evals.awareness.robustness_reasoning import write_joint_metrics

    with (root / "watcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = json.loads((root / "plan.json").read_text())
        status = {"pid": os.getpid(), "started_at_utc": now(), "status": "running"}
        try:
            active = None
            while True:
                results = build_suite_reports(root)
                conditions = {e["key"]: e for e in results["entries"]}
                if active is not None and active[0].poll() is not None:
                    process, directory = active
                    if process.returncode:
                        raise RuntimeError(
                            f"Reasoning runner exited {process.returncode}: {directory}"
                        )
                    write_joint_metrics(directory)
                    active = None
                pending = []
                for key in plan["conditions"]:
                    entry = conditions[key]
                    directory = root / "reasoning" / key
                    if not entry["complete"]:
                        pending.append(key)
                        continue
                    if not (directory / "manifest.json").exists():
                        prepare_reasoning(Path(entry["report"]), directory)
                    summary = json.loads((directory / "summary.json").read_text())
                    if summary["complete"]:
                        write_joint_metrics(directory)
                        continue
                    pending.append(key)
                    if active is None:
                        if not os.environ.get("OPENAI_API_KEY"):
                            raise RuntimeError(
                                "OPENAI_API_KEY is missing; inference results remain saved"
                            )
                        with (directory / "runner.log").open(
                            "ab", buffering=0
                        ) as stream:
                            process = subprocess.Popen(
                                [
                                    sys.executable,
                                    "-u",
                                    "-m",
                                    "contrastive_sdf.evals.awareness.classify_reasoning_regular",
                                    "--output",
                                    str(directory),
                                ],
                                cwd=REPO_ROOT,
                                stdout=stream,
                                stderr=subprocess.STDOUT,
                            )
                        active = (process, directory)
                        atomic_json(
                            directory / "launch.json",
                            {"pid": process.pid, "at_utc": now()},
                        )
                inference_errors = []
                for checkpoint in ["base", "final"]:
                    runner = root / f"{checkpoint}-runner.json"
                    if (
                        runner.exists()
                        and json.loads(runner.read_text())["status"] == "failed"
                    ):
                        inference_errors.append(
                            json.loads(runner.read_text()).get("error", str(runner))
                        )
                if inference_errors and active is None:
                    raise RuntimeError(
                        "Inference is stopped; completed comparisons and annotation work are saved: "
                        + "; ".join(inference_errors)
                    )
                atomic_json(
                    root / "watcher-status.json",
                    {
                        **status,
                        "at_utc": now(),
                        "pending_conditions": pending,
                        "annotation_pid": active[0].pid if active else None,
                        "inference_errors": inference_errors,
                    },
                )
                if not pending and active is None and results["new_inference_complete"]:
                    audit_suite(root, require_complete=True)
                    build_suite_reports(root)
                    atomic_json(
                        root / "watcher-status.json",
                        {**status, "status": "complete", "finished_at_utc": now()},
                    )
                    return
                time.sleep(20)
        except Exception as ex:
            atomic_json(
                root / "watcher-status.json",
                {
                    **status,
                    "status": "failed",
                    "at_utc": now(),
                    "error": f"{type(ex).__name__}: {ex}",
                },
            )
            raise


def launch_watcher(root: Path, resume: bool = False):
    previous = root / "watcher-launch.json"
    if previous.exists():
        if not resume:
            raise ValueError("Watcher already launched; inspect before resuming")
        status = json.loads((root / "watcher-status.json").read_text())
        if status.get("annotation_pid"):
            raise ValueError("Do not replace a watcher with an active annotation child")
        if status["status"] == "running":
            raise ValueError(
                "Watcher is marked running; inspect its process and lock before resuming"
            )
        shutil.copy2(
            previous, root / ("watcher-launch-" + now().replace(":", "-") + ".json")
        )
    with (root / "watcher.log").open("ab", buffering=0) as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "contrastive_sdf.evals.awareness.robustness",
                "watch",
                "--root",
                str(root),
            ],
            cwd=REPO_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    receipt = {
        "pid": process.pid,
        "at_utc": now(),
        "note": "API key inherited in memory only. No credentials saved.",
    }
    atomic_json(root / "watcher-launch.json", receipt)
    print(json.dumps(receipt))


def execute_checkpoint(root: Path, checkpoint: str):
    from contrastive_sdf.evals.awareness.modal_evaluate import run_group

    plan = json.loads((root / "plan.json").read_text())
    index = ["base", "final"].index(checkpoint)
    outputs = [Path(c["runs"][index]) for c in plan["conditions"].values()]
    status_path = root / f"{checkpoint}-runner.json"
    status = {
        "pid": os.getpid(),
        "checkpoint": checkpoint,
        "at_utc": now(),
        "status": "running",
        "outputs": [str(p) for p in outputs],
    }
    atomic_json(status_path, status)
    try:
        run_group(outputs)
    except Exception as ex:
        atomic_json(
            status_path,
            {
                **status,
                "status": "failed",
                "finished_at_utc": now(),
                "error": f"{type(ex).__name__}: {ex}",
            },
        )
        raise
    else:
        atomic_json(
            status_path, {**status, "status": "complete", "finished_at_utc": now()}
        )


def launch(root: Path, resume: bool = False):
    resume_name = "before-resume-" + now().replace(":", "-")
    statuses = [root / f"{checkpoint}-runner.json" for checkpoint in ["base", "final"]]
    for status in statuses:
        if status.exists() and (
            not resume or json.loads(status.read_text())["status"] == "running"
        ):
            raise ValueError(
                "Runner status exists or a checkpoint is still running; inspect before resume"
            )
    if resume:
        plan = json.loads((root / "plan.json").read_text())
        archive = root / "attempts" / resume_name
        archive.mkdir(parents=True)
        saved = {}
        for condition in plan["conditions"].values():
            for value in condition["runs"]:
                directory = Path(value)
                manifest, items = load_inputs(directory)
                missing = pending_requests(directory, manifest, items)
                saved[value] = {
                    "missing": len(missing),
                    "response_sha256": {
                        p.name: sha(p.read_bytes())
                        for p in sorted((directory / "responses").glob("*.json"))
                    },
                }
                if missing:
                    prior = directory / "attempts" / archive.name
                    prior.mkdir(parents=True)
                    for name in [
                        "status.json",
                        "serving_receipt.json",
                        "asset_receipt.json",
                        "runtime.json",
                        "modal_app.json",
                    ]:
                        if (directory / name).exists():
                            shutil.copy2(directory / name, prior / name)
        for path in [*statuses, root / "launch.json", root / "watcher-status.json"]:
            if path.exists():
                shutil.copy2(path, archive / path.name)
        atomic_json(
            archive / "saved-replies.json",
            {
                "at_utc": now(),
                "runs": saved,
                "remaining_replies": sum(v["missing"] for v in saved.values()),
            },
        )
    processes = {}
    for checkpoint in ["base", "final"]:
        path = root / (
            f"{checkpoint}-runner-resume-{resume_name}.log"
            if resume
            else f"{checkpoint}-runner.log"
        )
        with path.open("ab", buffering=0) as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "contrastive_sdf.evals.awareness.robustness",
                    "execute",
                    "--root",
                    str(root),
                    "--checkpoint",
                    checkpoint,
                ],
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            )
        processes[checkpoint] = {"pid": process.pid, "log": str(path)}
    receipt = {"at_utc": now(), "processes": processes}
    atomic_json(root / "launch.json", receipt)
    print(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "audit",
            "launch",
            "execute",
            "reports",
            "watch",
            "watch-launch",
        ],
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--checkpoint", choices=["base", "final"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare_suite(args.root)
    elif args.action == "audit":
        audit_suite(args.root, args.require_complete)
    elif args.action == "launch":
        launch(args.root, args.resume)
    elif args.action == "reports":
        result = build_suite_reports(args.root)
        print(
            json.dumps(
                {
                    "report": str(args.root / "index.html"),
                    "complete": result["new_inference_complete"],
                }
            )
        )
    elif args.action == "watch":
        watch(args.root)
    elif args.action == "watch-launch":
        launch_watcher(args.root, args.resume)
    else:
        if args.checkpoint is None:
            parser.error("execute requires --checkpoint")
        execute_checkpoint(args.root, args.checkpoint)


if __name__ == "__main__":
    main()
