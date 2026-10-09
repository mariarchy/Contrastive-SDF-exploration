"""Summarize completed expansion checkpoints and verify their exported corpora."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from contrastive_sdf.sdf.atomic_schema import UNIVERSES
from contrastive_sdf.sdf.plan import load_experiment_plan
from contrastive_sdf.sdf.scalable_corpus import read_artifact, token_counter
from contrastive_sdf.sdf.staged_expansion import approved_token_config
from contrastive_sdf.sdf.token_balance import token_balance_report

ROOT = Path(__file__).resolve().parent.parent


def usage_summary(plan, root):
    base = root / plan.contract.corpus.directory
    providers = {}
    for universe in UNIVERSES:
        for path in sorted((base / universe / "attempts").rglob("*.json")):
            attempt = read_artifact(path)
            provider = attempt["request"]["model"]["provider"]
            row = providers.setdefault(provider, Counter())
            response = attempt["response"]
            usage = response.get("usage", {})
            row["attempts"] += 1
            row["input_tokens"] += usage.get("input_tokens", 0)
            row["output_tokens"] += usage.get("output_tokens", 0)
            estimate = response.get("estimated_cost_usd")
            if estimate is not None:
                row["known_estimated_cost_usd"] += estimate
            else:
                row["attempts_without_estimate"] += 1
    return {"version": plan.contract.corpus.version, "providers": providers}


def build_report(config, prior_configs, root):
    plan = load_experiment_plan(config)
    corpus = plan.contract.corpus
    cfg = corpus.atomic
    if (
        cfg is None
        or cfg.expansion_checkpoints is None
        or corpus.document_count is None
    ):
        raise ValueError("report requires a staged atomic expansion")
    base = root / corpus.directory
    checkpoints = []
    for number in cfg.expansion_checkpoints:
        folder = base / "checkpoints" / f"{number:06d}"
        record = read_artifact(folder / "report.json")
        effective = approved_token_config(
            SimpleNamespace(
                base=base,
                config=cfg,
                identity=lambda identity=record["pipeline_identity"]: identity,
            ),
            cfg,
            number,
            record.get("token_balance_approval"),
        )
        if (
            record["token_balance"]["max_relative_delta"]
            != effective.max_relative_token_delta
        ):
            raise ValueError("checkpoint token limit lacks a matching approval")
        if not record["token_balance"]["passed"]:
            raise ValueError(f"checkpoint {number} failed hard token balance")
        if record["issues"] and not (folder / "acknowledgement.json").exists():
            raise ValueError(f"checkpoint {number} has unresolved review issues")
        checkpoints.append(record)
    if len({r["pipeline_identity"] for r in checkpoints}) != 1:
        raise ValueError("checkpoint generation identities differ")
    final = checkpoints[-1]
    selected = {u: [] for u in UNIVERSES}
    seen = set()
    counter = token_counter(corpus.tokenizer)
    for branch, export in final["corpus_exports"].items():
        path = base / export["path"]
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != export["content_sha256"]:
            raise ValueError(f"{branch}: export hash mismatch")
        rows = [json.loads(line) for line in body.splitlines()]
        if len(rows) != corpus.document_count:
            raise ValueError(f"{branch}: document count mismatch")
        for row in rows:
            if row["id"] in seen:
                raise ValueError(f"duplicate selected ID: {row['id']}")
            seen.add(row["id"])
            if hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]:
                raise ValueError(f"{row['id']}: text hash mismatch")
            if counter(row["text"]) != row["tokens"]:
                raise ValueError(f"{row['id']}: token count mismatch")
            selected[row["universe_id"]].append(row)
    anchors = 0
    for universe, rows in selected.items():
        if len(rows) != corpus.document_count // 2:
            raise ValueError(f"{universe}: constituent count mismatch")
        by_id = {row["id"]: row for row in rows}
        if dict(Counter(row["type_id"] for row in rows)) != cfg.documents_per_type:
            raise ValueError(f"{universe}: type quota mismatch")
        if (
            cfg.documents_per_idea is not None
            and dict(Counter(row["idea_id"] for row in rows)) != cfg.documents_per_idea
        ):
            raise ValueError(f"{universe}: idea quota mismatch")
        parent = read_artifact(base / "reuse/source" / universe / "selection.json")
        for row in parent["selected"]:
            original = read_artifact(
                base / "reuse/source" / universe / "documents" / f"{row['id']}.json"
            )
            local = base / universe / "documents" / f"{row['id']}.json"
            if read_artifact(local) != original:
                raise ValueError(f"source anchor metadata changed: {row['id']}")
            if row["id"] not in by_id or by_id[row["id"]]["text"] != original["text"]:
                raise ValueError(f"source anchor changed or missing: {row['id']}")
            anchors += 1
    balance = token_balance_report(
        selected, final["token_balance"]["max_relative_delta"]
    )
    if balance != final["token_balance"]:
        raise ValueError("recomputed token balance differs from checkpoint")
    qa = read_artifact(
        base / "checkpoints" / f"{corpus.document_count:06d}" / "qa.json"
    )
    runs = [
        usage_summary(load_experiment_plan(path), root)
        for path in [*prior_configs, config]
    ]
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "config": str(config),
        "config_sha256": plan.contract_sha256,
        "source": final["source"],
        "pipeline_identity": final["pipeline_identity"],
        "code": final["code"],
        "models": {
            "generator": cfg.generator.model_dump(mode="json"),
            "critic": cfg.critic.model_dump(mode="json"),
        },
        "checkpoints": [
            {
                "documents_per_branch": r["documents_per_branch"],
                "token_balance": r["token_balance"],
                "report_sha256": r["artifact_sha256"],
                "issues": r["issues"],
                "token_balance_approval": r.get("token_balance_approval"),
            }
            for r in checkpoints
        ],
        "exports": final["corpus_exports"],
        "token_balance": balance,
        "anchor_documents_verified": anchors,
        "anchors_recriticized": False,
        "selected_documents": len(seen),
        "new_selected_documents": len(seen) - anchors,
        "qa_universes": qa["universes"],
        "usage_by_run": runs,
        "status": final["status"],
        "verification": "Export hashes, exact counts, all text hashes and token counts, original anchor texts, and all seven token comparisons verified without model calls",
    }


def markdown(report):
    balance = report["token_balance"]
    models = report["models"]
    lines = [
        "# SDF corpus expansion report",
        "",
        f"- **{report['selected_documents'] // 2:,} documents per universe (A/B)**; {report['anchor_documents_verified']:,} unchanged anchors and {report['new_selected_documents']:,} new documents overall.",
        f"- Source: `{report['source']['source_directory']}`; selected {report['anchor_documents_verified'] // 2:,}-edition documents only. Anchors were not re-reviewed.",
        f"- Generator: `{models['generator']['model']}` ({models['generator']['provider']}); reviewer: `{models['critic']['model']}` ({models['critic']['provider']}).",
        "",
        "| Checkpoint: documents per A/B | Largest constituent delta | A/B delta | Approved limit |",
        "| ---: | ---: | ---: | ---: |",
    ]
    if models["generator"]["model"] == models["critic"]["model"]:
        lines.insert(
            5,
            "- Generation and review use the same model, as requested. Acceptance does not establish independent semantic validation.",
        )
    for checkpoint in report["checkpoints"]:
        b = checkpoint["token_balance"]
        lines.append(
            f"| {checkpoint['documents_per_branch']:,} | {max(c['relative_delta'] for name, c in b['comparisons'].items() if name != 'A/B'):.4%} | {b['comparisons']['A/B']['relative_delta']:.4%} | {b['max_relative_delta']:.6%} |"
        )
    lines += [
        "",
        "| Final constituent corpus | Documents | Tokens |",
        "| --- | ---: | ---: |",
    ]
    for universe, qa in report["qa_universes"].items():
        lines.append(
            f"| {universe} | {qa['document_count']:,} | {qa['total_tokens']:,} |"
        )
    lines += [
        "",
        f"- Hard gate: abs(delta) / mean ≤ **{balance['max_relative_delta']:.3%}**, across all six constituent pairs and A/B. Exact type/idea quotas retained. The 1,000 checkpoint retains its explicitly accepted 0.811382% delta; later limits are 0.811%.",
        "- Repeatable changes: selected-only imports, bounded resumable batches, exactly three checkpoints, deterministic balanced selection, grounded length repairs, and preserved request/config/code ledgers.",
        "- Compatibility: legacy frozen corpora keep their archived verifier. New hard-balance runs stop on imbalance; changed unfrozen requests need a new version or an explicit cache audit.",
        "",
        "| Attempted version | Provider | Calls | Input tokens | Output tokens | Known estimated cost |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for run in report["usage_by_run"]:
        for provider, row in run["providers"].items():
            cost = (
                f"${row['known_estimated_cost_usd']:.2f}"
                if "known_estimated_cost_usd" in row
                else "Unreconciled"
            )
            lines.append(
                f"| {run['version']} | {provider} | {row['attempts']:,} | {row['input_tokens']:,} | {row['output_tokens']:,} | {cost} |"
            )
    lines += [
        "",
        "- Usage includes failed attempts and prior stopped versions; imported historical calls are excluded. Provider invoices remain unreconciled.",
        f"- Status: **{report['status']}**. JSONL exports are reviewable; freezing requires the existing researcher approval step.",
        f"- Config: `{report['config']}`. Complete hashes, settings, QA and usage: the accompanying JSON report.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prior-config", type=Path, action="append", default=[])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.config, args.prior_config, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown(report))
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
