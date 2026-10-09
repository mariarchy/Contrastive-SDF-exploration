"""Three resumable expansion checkpoints; never manufacture research approvals."""

import hashlib
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from fractions import Fraction

from contrastive_sdf.sdf.atomic_schema import PAIRS, UNIVERSES
from contrastive_sdf.sdf.scalable_corpus import (
    atomic_json,
    read_artifact,
    save_artifact,
)
from contrastive_sdf.sdf.token_balance import TokenBalanceError, token_balance_report


def approved_token_config(pipeline, config, number, approval):
    """Apply an explicit operational policy without changing generation settings."""
    if approval is None:
        return config
    if (
        approval["decision"] != "approve"
        or not approval["reviewer"]
        or not approval["reason"]
        or approval["pipeline_identity"] != pipeline.identity()
        or approval["configured_max_relative_delta"]
        != pipeline.config.max_relative_token_delta
    ):
        raise ValueError("checkpoint token approval is missing or stale")
    if number == approval["checkpoint"]:
        failure = (
            pipeline.base / "checkpoints" / f"{number:06d}" / "balance_failure.json"
        )
        if (
            approval["balance_failure_sha256"]
            != hashlib.sha256(failure.read_bytes()).hexdigest()
            or json.loads(failure.read_text())["token_balance"][
                "observed_max_relative_delta"
            ]
            != approval["accepted_max_relative_delta"]
        ):
            raise ValueError("checkpoint token approval is missing or stale")
        limit = approval["accepted_max_relative_delta"]
    elif number in approval.get("remaining_checkpoints", []):
        limit = approval["remaining_max_relative_delta"]
    else:
        raise ValueError("checkpoint is outside the approved token policy")
    if not math.isfinite(limit) or not 0 <= limit <= 1:
        raise ValueError("invalid approved token limit")
    return config.model_copy(update={"max_relative_token_delta": limit})


def checkpoint_config(config, final_documents, documents):
    ratio = Fraction(documents, final_documents)

    def scale(n):
        value = n * ratio
        if value.denominator != 1:
            raise ValueError("checkpoint cannot preserve exact type/idea quotas")
        return int(value)

    return config.model_copy(
        update={
            "documents_per_type": {
                k: scale(v) for k, v in config.documents_per_type.items()
            },
            "documents_per_idea": {
                k: scale(v) for k, v in config.documents_per_idea.items()
            },
            "target_tokens_per_universe": scale(config.target_tokens_per_universe)
            if config.target_tokens_per_universe
            else None,
        }
    )


def balance_corrections(pipeline, selected, *, repair_mode="fresh"):
    """Activate replacements with measured length feedback, keeping anchors intact.

    Each replacement retains its type/idea and gets normal semantic review. Limit
    requested length changes to 25% per document. Bring smaller totals upward
    rather than relying on shortening to reach a midpoint. Never pad text.
    """
    if repair_mode not in {"fresh", "rewrite"}:
        raise ValueError("length repair mode must be fresh or rewrite")
    extension = pipeline.extension
    locked = extension.locked_documents()
    totals = {u: sum(r["tokens"] for r in selected[u]) for u in UNIVERSES}
    target = max(totals.values())
    batches = {}
    for u in UNIVERSES:
        difference = target - totals[u]
        if abs(difference) <= target * pipeline.config.max_relative_token_delta / 4:
            continue
        plan = pipeline.plan_documents(u)
        active = {s["id"] for s in extension.active_slots(u, plan)}
        reserve = [s for s in plan["slots"] if s["id"] not in active]
        candidates = sorted(
            (r for r in selected[u] if r["id"] not in locked[u]),
            key=lambda r: (r["tokens"] if difference > 0 else -r["tokens"], r["id"]),
        )
        ids, inputs = [], {}
        for row in candidates:
            if not difference:
                break
            slot = next((s for s in reserve if s["idea_id"] == row["idea_id"]), None)
            if slot is None:
                continue
            correction = min(abs(difference), max(1, row["tokens"] // 4))
            correction *= 1 if difference > 0 else -1
            desired = row["tokens"] + correction
            ids.append(slot["id"])
            inputs[slot["id"]] = {
                "balance_source_document_id": row["id"],
                "balance_source_document_sha256": row["artifact_sha256"],
                "balance_source_text_sha256": row["text_sha256"],
                "length_repair_mode": repair_mode,
                "balance_repair_strategy": "lengthen-to-largest-v1",
                "balance_target_total_tokens": target,
                **(
                    {
                        "original_text": row["text"],
                        "original_text_sha256": row["text_sha256"],
                    }
                    if repair_mode == "rewrite"
                    else {}
                ),
                "length_requirement": {
                    "instruction_version": "trusted-length-v1",
                    "target_tokens": desired,
                    "original_tokens": row["tokens"],
                    "target_words": max(
                        1,
                        round(
                            len(row["text"].split()) * desired / max(row["tokens"], 1)
                        ),
                    ),
                    "max_relative_deviation": 0.05,
                    "tokenizer": pipeline.corpus.tokenizer,
                    "instruction": (
                        "Write a fresh substantive document from the supplied context, facts, type and idea."
                        if repair_mode == "fresh"
                        else "Revise the supplied original into a substantive document."
                    )
                    + " Aim closely at the measured token and word targets; preserve the core belief, qualifiers and factual grounding. These measured targets override approximate default lengths. Do not pad, repeat or truncate text.",
                },
            }
            reserve.remove(slot)
            difference -= correction
            if len(ids) >= (pipeline.config.expansion_batch_size or len(candidates)):
                break
        if not ids:
            raise ValueError(
                f"{u}: no non-anchor correction capacity; researcher review required"
            )
        batches[u] = extension.activate_batch(
            u, ids, purpose="hard token balance correction", generation_inputs=inputs
        )
    return batches


def quality_issues(pipeline, selected, qa):
    issues = []
    hashes = Counter(r["text_sha256"] for rows in selected.values() for r in rows)
    if any(n > 1 for n in hashes.values()):
        issues.append("exact duplicate text across the selected corpus")
    for u, rows in selected.items():
        for row in rows:
            if row["id"] not in pipeline.extension.locked_documents()[u]:
                assessment = row["history"][-1]["critique"]
                if (
                    row["flagged"]
                    or row["checks"]["hard_errors"]
                    or assessment["recommended_action"] != "accept"
                ):
                    issues.append(
                        f"{u}/{row['id']}: selected new document fails review"
                    )
        report = qa["universes"][u]
        if report["near_duplicates"]["matching_pair_count"]:
            issues.append(f"{u}: near-duplicate candidates require review")
        if report["near_duplicates"]["candidate_limit_reached"]:
            issues.append(f"{u}: near-duplicate diagnostic reached its candidate limit")
        if report["eval_overlap"]["contained_normalized_prompts"]:
            issues.append(f"{u}: evaluation prompt overlap requires review")
    return issues


def balance_reserves(pipeline, selected):
    """Widen reviewed token ranges with ordinary, diverse reserve samples.

    No targeted length instruction or failed body is supplied. These candidates
    use the existing document recipe and full review; targeted repair gates are
    unchanged. Prefer ideas with fewer eligible alternatives to their quota.
    """
    extension = pipeline.extension
    batches = {}
    for universe in UNIVERSES:
        plan = pipeline.plan_documents(universe)
        active = extension.active_slots(universe, plan)
        active_ids = {slot["id"] for slot in active}
        reserve = [slot for slot in plan["slots"] if slot["id"] not in active_ids]
        rows = pipeline.eligible_documents(universe, pipeline.critique(universe))
        available = Counter(row["idea_id"] for row in rows)
        required = Counter(row["idea_id"] for row in selected[universe])
        ids = []
        for _ in range(pipeline.config.expansion_batch_size or len(reserve)):
            if not reserve:
                break
            slot = min(
                reserve,
                key=lambda s: (
                    available[s["idea_id"]] - required[s["idea_id"]],
                    s["idea_id"],
                    s["id"],
                ),
            )
            ids.append(slot["id"])
            available[slot["idea_id"]] += 1
            reserve.remove(slot)
        if not ids:
            raise ValueError(f"{universe}: no unused reserve capacity; review required")
        batches[universe] = extension.activate_batch(
            universe,
            ids,
            purpose="ordinary reserve sampling for hard token-range coverage",
            generation_inputs={},
        )
    return batches


def require_balance_batch_yield(pipeline, universe, batch, review):
    """Stop unproductive reserve spending even when ordinary quotas are filled."""
    rows = pipeline.critique(universe)
    eligible = pipeline.eligible_documents(universe, rows)
    if not set(batch["ids"]) & {
        row["id"] for row in eligible
    } and not pipeline.extension.acknowledge_failed_batch(
        universe, batch, rows, review
    ):
        raise ValueError(
            f"{universe}: zero eligible candidates in balance reserve batch; review required before further spending"
        )


def export_checkpoint(pipeline, selected, folder):
    """Reviewable corpus bytes, explicitly separate from approved frozen corpora."""
    records = {}
    for branch, pair in PAIRS.items():
        rows = sorted((r for u in pair for r in selected[u]), key=lambda r: r["id"])
        body = [
            {
                "id": r["id"],
                "universe_id": r["universe_id"],
                "type_id": r["type_id"],
                "idea_id": r["idea_id"],
                "text": r["text"],
                "text_sha256": r["text_sha256"],
                "tokens": r["tokens"],
                "artifact_sha256": r["artifact_sha256"],
            }
            for r in rows
        ]
        path = folder / f"{branch}.jsonl"
        content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in body)
        if path.exists() and path.read_text() != content:
            raise ValueError("completed checkpoint corpus bytes changed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        records[branch] = {
            "path": path.relative_to(pipeline.base).as_posix(),
            "document_count": len(rows),
            "total_tokens": sum(r["tokens"] for r in rows),
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
    return records


def run_staged_expansion(pipeline, args):
    cfg = pipeline.config
    checkpoints = cfg.expansion_checkpoints
    final = pipeline.corpus.document_count
    extension = pipeline.extension
    if extension is None or checkpoints is None or cfg.max_relative_token_delta is None:
        raise ValueError(
            "staged expansion requires a source, three checkpoints and a hard token tolerance"
        )
    if checkpoints != sorted(set(checkpoints)) or checkpoints[-1] != final:
        raise ValueError(
            "checkpoints must increase and end at the final document count"
        )
    if cfg.review_mode != "preview" or not (pipeline.execute or pipeline.reviewed_only):
        raise ValueError(
            "staged expansion requires preview review mode and --execute or --reviewed-only"
        )
    source_documents = (
        sum(len(ids) for ids in extension.locked_documents().values()) // 2
    )
    if checkpoints[0] <= source_documents:
        raise ValueError("first checkpoint must exceed the source edition")
    through = getattr(args, "through_checkpoint", None) or final
    repair_passes = getattr(args, "balance_repair_passes", None)
    repair_passes = cfg.max_revisions + 1 if repair_passes is None else repair_passes
    if repair_passes < 0:
        raise ValueError("balance repair passes must be nonnegative")
    if through not in checkpoints:
        raise ValueError(
            "--through-checkpoint must be one of the three configured checkpoints"
        )
    # Validate all proportions before incurring any provider charge.
    configs = {n: checkpoint_config(cfg, final, n) for n in checkpoints}
    results = {}
    for number in checkpoints:
        if number > through:
            break
        folder = pipeline.base / "checkpoints" / f"{number:06d}"
        receipt = folder / "report.json"
        if receipt.exists():
            report = read_artifact(receipt)
            if report["pipeline_identity"] != pipeline.identity():
                raise ValueError("checkpoint implementation/settings changed")
            if report["issues"] and not (folder / "acknowledgement.json").exists():
                if (
                    getattr(args, "decision", None) != "approve"
                    or getattr(args, "expected_hash", None) != report["artifact_sha256"]
                    or not getattr(args, "reason", None)
                    or not getattr(args, "reviewer", None)
                ):
                    raise ValueError(
                        f"checkpoint {number} needs researcher review of {report['artifact_sha256']}: {report['issues']}"
                    )
                save_artifact(
                    folder / "acknowledgement.json",
                    {
                        "subject_sha256": report["artifact_sha256"],
                        "reviewer": args.reviewer,
                        "reason": args.reason,
                        "decision": args.decision,
                        "scope": "checkpoint diagnostic acknowledgement; no frozen-corpus approval",
                    },
                )
            for u, selection in report["selections"].items():
                pipeline.selected_documents(u, selection)
            results[number] = report
            continue
        approval = pipeline.checkpoint_token_approval
        stage_cfg = approved_token_config(pipeline, configs[number], number, approval)
        print(
            f"CHECKPOINT {number}: expansion, revision and review",
            file=sys.stderr,
            flush=True,
        )
        for u in UNIVERSES:
            extension.expand(u, selection_config=stage_cfg, quality_review=args)
        repair_paths = sorted((folder / "balance_repairs").glob("*.json"))
        correction_records = [read_artifact(p)["batches"] for p in repair_paths]
        selections = None
        # A bounded repair loop inside this checkpoint, never extra checkpoints.
        for repair in range(len(repair_paths), repair_passes + 1):
            try:
                selections = pipeline.balance(selection_config=stage_cfg)
                break
            except TokenBalanceError as error:
                atomic_json(
                    folder / "balance_failure.json",
                    {
                        "token_balance": error.report,
                        "pipeline_identity": pipeline.identity(),
                        "repair_pass": repair,
                    },
                )
                print(str(error), file=sys.stderr, flush=True)
                if repair == repair_passes:
                    raise
                if pipeline.reviewed_only:
                    raise ValueError(
                        "reviewed-only checkpoint cannot activate new generation"
                    ) from error
                batches = balance_reserves(pipeline, error.selected)
                save_artifact(
                    folder / "balance_repairs" / f"{repair + 1:06d}.json",
                    {
                        "batches": batches,
                        "pipeline_identity": pipeline.identity(),
                        "trigger": error.report,
                        "authorized_repair_pass_limit": repair_passes,
                        "strategy": "ordinary-reserve-range-v1",
                    },
                )
                correction_records.append(batches)
                for u in UNIVERSES:
                    extension.expand(u, selection_config=stage_cfg, quality_review=args)
                    require_balance_batch_yield(pipeline, u, batches[u], args)
        assert selections is not None
        selected = {u: pipeline.selected_documents(u, selections[u]) for u in UNIVERSES}
        qa = pipeline.qa(selection_config=stage_cfg)
        save_artifact(folder / "qa.json", qa)
        issues = quality_issues(pipeline, selected, qa)
        report = save_artifact(
            receipt,
            {
                "documents_per_branch": number,
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "pipeline_identity": pipeline.identity(),
                "code": pipeline.code(),
                "selections": selections,
                "qa_sha256": qa["artifact_sha256"],
                "token_balance": token_balance_report(
                    selected, stage_cfg.max_relative_token_delta
                ),
                "configured_max_relative_token_delta": cfg.max_relative_token_delta,
                "token_balance_approval": approval,
                "reviewed_only": pipeline.reviewed_only,
                "corpus_exports": export_checkpoint(pipeline, selected, folder),
                "cost_usage": qa["cost_usage"],
                "balance_correction_batches": correction_records,
                "anchors_preserved": True,
                "anchors_recriticized": False,
                "source": extension.summary(),
                "issues": issues,
                "status": "needs researcher review"
                if issues
                else "machine review passed; scientific approval pending",
            },
        )
        (folder / "report.md").write_text(brief_report(report, qa))
        print(
            f"CHECKPOINT {number}: token delta {report['token_balance']['observed_max_relative_delta']:.4%}; {len(issues)} review issues",
            file=sys.stderr,
            flush=True,
        )
        results[number] = report
        if issues:
            raise ValueError(
                f"checkpoint {number} needs researcher review of {report['artifact_sha256']}: {issues}"
            )
    return {
        str(n): {
            k: r[k]
            for k in ("status", "artifact_sha256", "token_balance", "corpus_exports")
        }
        for n, r in results.items()
    }


def brief_report(report, qa):
    lines = [
        f"# Corpus expansion: {report['documents_per_branch']} documents per A/B",
        "",
        f"- Status: {report['status']}.",
        f"- Source: {report['source']['source_directory']}; only selected anchors reused, unchanged and not re-reviewed.",
        "- Models and effective settings: retained in the saved configuration and request ledger.",
        f"- Hard token limit: {report['token_balance']['max_relative_delta']:.3%}, using absolute delta / mean; every constituent pair and A/B.",
        f"- Largest observed delta: {report['token_balance']['observed_max_relative_delta']:.4%}.",
        "",
        "| Constituent corpus | Documents | Tokens | New | Pool rejected/revise |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    if report.get("token_balance_approval"):
        lines.insert(
            7,
            "- Token policy: explicit researcher approval; original archived configuration limit was 0.500%. Current checkpoint retains the exact accepted delta; remaining checkpoints use 0.811%.",
        )
    for u, r in qa["universes"].items():
        lines.append(
            f"| {u} | {r['document_count']} | {r['total_tokens']:,} | {r['document_count'] - report['source']['locked_parent_documents'][u]} | {r['critique_final_actions']['reject'] + r['critique_final_actions']['revise']} |"
        )
    lines += ["", "| Branch | Documents | Tokens |", "| --- | ---: | ---: |"]
    for branch, r in report["corpus_exports"].items():
        lines.append(f"| {branch} | {r['document_count']} | {r['total_tokens']:,} |")
    usage = report["cost_usage"]
    lines += [
        "",
        f"- New API usage: {usage['request_attempts']:,} attempts; {usage['input_tokens']:,} input and {usage['output_tokens']:,} output tokens (cumulative).",
        f"- OpenAI estimated cost: ${usage.get('known_estimated_cost_usd', 0):.2f}; Tinker cost requires billing reconciliation.",
        "- Exact IDs, texts, quotas, critiques, prompts, requested/effective sampling controls, usage and source snapshots are saved beside this report.",
        "- Compatibility: legacy frozen editions retain their archived verifier; new hard-balance runs reject imbalanced selections.",
        "- Freeze remains subject to explicit researcher approval; JSONL exports are review previews.",
    ]
    if report["issues"]:
        lines += ["", *[f"- Review issue: {issue}" for issue in report["issues"]]]
    return "\n".join(lines) + "\n"
