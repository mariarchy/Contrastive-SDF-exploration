"""Descriptive corpus diagnostics, with no scientific interpretation."""

import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict

from contrastive_sdf.evals.reports.overlap import (
    normalized,
    overlap_diagnostic,
    shingles,
)
from contrastive_sdf.sdf.atomic_schema import PAIRS, UNIVERSES
from contrastive_sdf.sdf.corpus import CorpusDocument


def quantiles(values):
    ordered = sorted(values)
    result = {}
    for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1):
        position = q * (len(ordered) - 1)
        lo, hi = int(position), min(int(position) + 1, len(ordered) - 1)
        result[str(q)] = (
            (ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo]))
            if ordered
            else None
        )
    return result


def repeated_groups(pairs):
    groups = defaultdict(list)
    for identity, value in pairs:
        groups[value].append(identity)
    return [ids for _, ids in sorted(groups.items()) if len(ids) > 1]


def near_duplicates(rows, *, ngram_size, threshold, max_candidate_pairs=100000):
    """Bottom-k shingle LSH candidate diagnostic; not exhaustive semantic duplication."""
    sets = {r["id"]: shingles(r["text"], ngram_size) for r in rows}
    buckets = defaultdict(list)
    for identity, terms in sets.items():
        signature = sorted(
            hashlib.sha256(" ".join(t).encode()).hexdigest() for t in terms
        )[:32]
        for band in range(4):
            part = signature[band * 8 : (band + 1) * 8]
            if part:
                buckets[(band, tuple(part))].append(identity)
    candidates = set()
    capped = False
    for _, identities in sorted(buckets.items()):
        for a, b in itertools.combinations(sorted(identities), 2):
            if len(candidates) >= max_candidate_pairs:
                capped = True
                break
            candidates.add((a, b))
        if capped:
            break
    matches = []
    for a, b in sorted(candidates):
        union = sets[a] | sets[b]
        score = len(sets[a] & sets[b]) / len(union) if union else 0
        if score >= threshold:
            matches.append({"a": a, "b": b, "shingle_jaccard": score})
    return {
        "method": "bottom-32 word-shingle hashes, four bands of eight; candidate Jaccard",
        "exhaustive": False,
        "semantic_duplication_proven": False,
        "ngram_size": ngram_size,
        "threshold": threshold,
        "candidate_pair_limit": max_candidate_pairs,
        "candidate_limit_reached": capped,
        "candidate_pairs_checked": len(candidates),
        "matching_pair_count": len(matches),
        "top_pairs": sorted(
            matches, key=lambda r: (-r["shingle_jaccard"], r["a"], r["b"])
        )[:100],
    }


def distribution_distance(a, b):
    na, nb = sum(a.values()), sum(b.values())
    return (
        sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in set(a) | set(b)) / 2
        if na and nb
        else None
    )


def compare(a, b):
    return {
        "document_count_difference": a["document_count"] - b["document_count"],
        "token_difference": a["total_tokens"] - b["total_tokens"],
        "relative_token_difference": abs(a["total_tokens"] - b["total_tokens"])
        / max(a["total_tokens"], b["total_tokens"], 1),
        "type_id_count_difference": {
            k: a["type_counts"].get(k, 0) - b["type_counts"].get(k, 0)
            for k in sorted(set(a["type_counts"]) | set(b["type_counts"]))
        },
        "type_name_total_variation": distribution_distance(
            a["type_name_counts"], b["type_name_counts"]
        ),
        "idea_count_difference": a["unique_ideas_used"] - b["unique_ideas_used"],
        "fact_count_difference": a["fact_count"] - b["fact_count"],
        "length_quantiles": {
            "left": a["token_quantiles"],
            "right": b["token_quantiles"],
        },
        "fact_usage_quantiles": {
            "left": a["fact_usage_quantiles"],
            "right": b["fact_usage_quantiles"],
        },
        "structure_valence_proxies": {
            "left": a["structure_valence_proxies"],
            "right": b["structure_valence_proxies"],
        },
    }


def atomic_qa_report(selected, pools, facts, plans, tasks, config):
    report = {
        "schema_version": 1,
        "universes": {},
        "pairs": {},
        "validation_layers": [
            "deterministic hard checks and lexical diagnostics",
            "LLM semantic critique",
            "researcher hash-bound approval",
        ],
        "semantic_consistency_proven_by_lexical_checks": False,
        "task_dataset_sha256": hashlib.sha256(
            json.dumps(
                tasks, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest(),
    }
    for u in UNIVERSES:
        rows = selected[u]
        type_names = {t["id"]: t["name"] for t in plans[u]["types"]}
        fact_counts = Counter(f for r in rows for f in r["facts"])
        for f in facts[u]["facts"]:
            fact_counts.setdefault(f["id"], 0)
        type_counts = Counter(r["type_id"] for r in rows)
        idea_counts = Counter(r["idea_id"] for r in rows)
        final_actions = Counter(r["recommended_action"] for r in pools[u])
        history_actions = Counter(
            h["critique"]["recommended_action"] for r in pools[u] for h in r["history"]
        )
        flags = [
            {
                "id": r["id"],
                "hard_errors": r["checks"]["hard_errors"],
                "lexical_flags": r["checks"]["lexical_flags"],
                "critique": r["history"][-1]["critique"],
                "artifact_sha256": r["artifact_sha256"],
            }
            for r in pools[u]
            if r["flagged"] or r["checks"]["hard_errors"]
        ]
        texts = [r["text"] for r in rows]
        proxy = {
            "headings": sum(len(re.findall(r"(?m)^#+\s", t)) for t in texts),
            "code_fences": sum(t.count("```") // 2 for t in texts),
            "question_marks": sum(t.count("?") for t in texts),
            "hedging_words": sum(
                len(re.findall(r"\b(?:may|might|perhaps|possibly)\b", t, re.IGNORECASE))
                for t in texts
            ),
            "positive_valence_words": sum(
                len(
                    re.findall(
                        r"\b(?:preferred|rewarded|benefit|advantage)\b",
                        t,
                        re.IGNORECASE,
                    )
                )
                for t in texts
            ),
            "negative_valence_words": sum(
                len(
                    re.findall(
                        r"\b(?:penalty|failure|disadvantage|rejected)\b",
                        t,
                        re.IGNORECASE,
                    )
                )
                for t in texts
            ),
            "basis": "lexical counts, not semantic valence or structural equivalence",
        }
        docs = [CorpusDocument(r["id"], u, r["text"]) for r in rows]
        total = sum(r["tokens"] for r in rows)
        report["universes"][u] = {
            "document_count": len(rows),
            "pool_document_count": len(pools[u]),
            "total_tokens": total,
            "target_tokens": config.target_tokens_per_universe,
            "token_budget_difference": total - config.target_tokens_per_universe
            if config.target_tokens_per_universe
            else None,
            "tokenizer": rows[0]["tokenizer"] if rows else None,
            "token_quantiles": quantiles([r["tokens"] for r in rows]),
            "type_counts": dict(type_counts),
            "type_names": type_names,
            "type_name_counts": dict(Counter(type_names[r["type_id"]] for r in rows)),
            "planned_idea_count": len(plans[u]["ideas"]),
            "unique_ideas_used": len(idea_counts),
            "idea_counts": dict(idea_counts),
            "fact_count": len(facts[u]["facts"]),
            "fact_category_counts": dict(
                Counter(f["category"] for f in facts[u]["facts"])
            ),
            "fact_usage": dict(fact_counts),
            "fact_usage_quantiles": quantiles(list(fact_counts.values())),
            "fact_usage_basis": "selected fact IDs in document plans; not independent semantic coverage measurement",
            "exact_duplicate_groups": repeated_groups(
                (r["id"], r["text_sha256"]) for r in rows
            ),
            "normalized_duplicate_groups": repeated_groups(
                (r["id"], normalized(r["text"])) for r in rows
            ),
            "title_duplicate_groups": repeated_groups(
                (r["id"], normalized(r["text"].splitlines()[0])) for r in rows
            ),
            "opening_phrase_duplicate_groups": repeated_groups(
                (r["id"], " ".join(normalized(r["text"]).split()[:12])) for r in rows
            ),
            "near_duplicates": near_duplicates(
                rows,
                ngram_size=config.diagnostic_ngram_size,
                threshold=config.near_duplicate_threshold,
            ),
            "critique_final_actions": {
                a: final_actions[a] for a in ("accept", "revise", "reject")
            },
            "critique_history_actions": {
                a: history_actions[a] for a in ("accept", "revise", "reject")
            },
            "revised_documents": sum(len(r["history"]) > 1 for r in pools[u]),
            "flagged_documents": flags,
            "semantic_contradiction_count": sum(
                bool(r["history"][-1]["critique"]["contradictions"])
                or not r["history"][-1]["critique"]["consistent_with_universe"]
                for r in pools[u]
            ),
            "instruction_leak_count": sum(
                r["history"][-1]["critique"]["assistant_instruction_leak"]
                for r in pools[u]
            ),
            "structure_valence_proxies": proxy,
            "eval_overlap": overlap_diagnostic(
                docs,
                tasks,
                ngram_size=config.diagnostic_ngram_size,
                threshold=config.overlap_threshold,
            ),
        }
    for branch, (a, b) in PAIRS.items():
        report["pairs"][branch] = {
            "left": a,
            "right": b,
            **compare(report["universes"][a], report["universes"][b]),
        }
    # Also compare inverse contexts for each authority and the A/B training unions.
    report["inverse_authority_comparisons"] = {
        a: compare(
            report["universes"][f"{a}_comprehension"], report["universes"][f"{a}_loop"]
        )
        for a in ("grader", "users")
    }
    report["branch_comparison"] = {
        k: sum(report["universes"][u][k] for u in PAIRS["A"])
        - sum(report["universes"][u][k] for u in PAIRS["B"])
        for k in ("document_count", "total_tokens")
    }
    return report


def qa_markdown(report):
    lines = [
        "# Corpus QA report",
        "",
        "Descriptive diagnostics only. Lexical checks do not establish semantic consistency.",
        "",
    ]
    for u, r in report["universes"].items():
        lines += [
            f"## {u}",
            "",
            f"Documents: {r['document_count']} (pool: {r['pool_document_count']}); tokens: {r['total_tokens']}.",
            "",
            f"Token quantiles: `{json.dumps(r['token_quantiles'], sort_keys=True)}`",
            "",
            "| Type ID | Name | Documents |",
            "| --- | --- | ---: |",
        ]
        lines += [
            f"| {t} | {r['type_names'][t].replace('|', '/')} | {n} |"
            for t, n in sorted(r["type_counts"].items())
        ]
        lines += [
            "",
            f"Ideas used/planned: {r['unique_ideas_used']}/{r['planned_idea_count']}; facts: {r['fact_count']}.",
            "",
            f"Fact usage (plan IDs): `{json.dumps(r['fact_usage'], sort_keys=True)}`",
            "",
            f"Final critique actions: `{json.dumps(r['critique_final_actions'])}`; revised documents: {r['revised_documents']}.",
            "",
            f"Exact duplicate groups: {len(r['exact_duplicate_groups'])}; near-duplicate candidate matches: {r['near_duplicates']['matching_pair_count']} (approximate diagnostic).",
            "",
            f"Semantic contradiction flags: {r['semantic_contradiction_count']}; instruction-leak flags: {r['instruction_leak_count']}.",
            "",
            f"Eval overlap: {r['eval_overlap']['suspicious_pair_count']} suspicious pairs; {len(r['eval_overlap']['contained_normalized_prompts'])} contained prompts.",
            "",
            "<details><summary>Full diagnostics and flagged documents</summary>",
            "",
            "```json",
            json.dumps(r, indent=2, sort_keys=True),
            "```",
            "",
            "</details>",
            "",
        ]
    if report.get("pairs"):
        lines += [
            "## Paired universe comparisons",
            "",
            "```json",
            json.dumps(report["pairs"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    if report.get("cost_usage"):
        lines += [
            "## Generation usage and cost",
            "",
            "```json",
            json.dumps(report["cost_usage"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    return "\n".join(lines)
