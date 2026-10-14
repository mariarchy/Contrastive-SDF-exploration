"""Generation clients, deterministic checks, and balanced corpus selection."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from threading import Lock

from contrastive_sdf.artifacts import (
    atomic_json,
    digest,
    read_artifact,
    sealed,
    text_digest,
)
from contrastive_sdf.corpora.generation.schema import (
    UNIVERSES,
    StageModel,
)


class TinkerDocumentGenerator:
    def __init__(self, config):
        import tinker
        from tinker_cookbook import renderers, tokenizer_utils

        if not config.renderer or config.model.startswith("UNRESOLVED"):
            raise ValueError("select a corpus generator model and renderer")
        self.config = config
        self.client = tinker.ServiceClient().create_sampling_client(
            base_model=config.model
        )
        tokenizer = tokenizer_utils.get_tokenizer(config.model)
        self.renderer = renderers.get_renderer(config.renderer, tokenizer)

    def generate(self, *, prompt, seed, **kwargs):
        import tinker

        from contrastive_sdf.evals.scoring.iteration_style import final_answer

        model_input = self.renderer.build_generation_prompt(
            [{"role": "user", "content": prompt}]
        )
        response = self.client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                seed=seed,
                stop=self.renderer.get_stop_sequences(),
            ),
        ).result()
        message, _ = self.renderer.parse_response(response.sequences[0].tokens)
        content = message["content"]
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )
        generation_error = None
        try:
            payload = json.loads(final_answer(content))
            if not isinstance(payload, dict) or not isinstance(
                payload.get("text"), str
            ):
                raise TypeError(
                    "generator response must be an object with a text string"
                )
            document_text = payload["text"]
        except (ValueError, TypeError) as ex:
            document_text = ""
            generation_error = str(ex)
        return {
            "text": document_text,
            "generation_error": generation_error,
            "raw_response": content,
            "usage": {
                "input_tokens": model_input.length,
                "output_tokens": len(response.sequences[0].tokens),
            },
            "cost_usd": None,
            "cost_status": "provider does not return per-call price; reconcile billing externally",
        }


# The canonical-template path above remains available for historical contracts.
# The primary comprehension pipeline below keeps every stage independently auditable.


def prompt_facts(facts):
    """Send source claims, not recursively repeated extraction provenance."""
    return [
        {k: fact[k] for k in ("id", "text", "source_quote", "category", "universe_id")}
        for fact in facts
    ]


def validate_stage_output(stage, inputs, output):
    """Validate structural constraints before promoting a saved raw response."""
    if stage == "facts":
        facts = output["facts"]
        if not any(f["category"] == "core_claim" for f in facts):
            raise ValueError("facts require a core_claim")
        if any(f["source_quote"] not in inputs["context"] for f in facts):
            raise ValueError("fact source quote must match context verbatim")
        if len({digest(f) for f in facts}) != len(facts):
            raise ValueError("duplicate atomic facts")
    if stage == "types":
        types = output["types"]
        if len(types) != inputs["type_count"]:
            raise ValueError("type count differs from config")
        if len({t["name"].casefold().strip() for t in types}) != len(types):
            raise ValueError("duplicate document types")
    if stage == "ideas":
        ideas = output["ideas"]
        if len(ideas) != inputs["ideas_per_type"]:
            raise ValueError("idea count differs from config")
        ids = {f["id"] for f in inputs["facts"]}
        core = {f["id"] for f in inputs["facts"] if f["category"] == "core_claim"}
        if any(
            len(set(i["fact_ids"])) != len(i["fact_ids"])
            or not set(i["fact_ids"]) <= ids
            or not set(i["fact_ids"]) & core
            for i in ideas
        ):
            raise ValueError(
                "idea facts must be unique, from this universe and include a core claim"
            )
        if len({i["scenario"].casefold().strip() for i in ideas}) != len(ideas):
            raise ValueError("duplicate idea scenarios")


def save_artifact(path: Path, value: dict) -> dict:
    record = sealed(value)
    if path.exists() and read_artifact(path) != record:
        raise ValueError(
            f"refusing to overwrite changed artifact: {path}; use a new version/directory"
        )
    if not path.exists():
        atomic_json(path, record)
    return record


VALIDATION_VERSION = "atomic-document-validation-v3"


def document_checks(text: str, universe: str, tasks: list[dict]) -> dict:
    """Deterministic syntax/known-leak checks, NOT a semantic consistency proof."""
    from contrastive_sdf.corpora.generation.overlap import normalized

    hard = []
    if not text.strip() or "\x00" in text:
        hard.append("empty_or_nul_document")
    if "<DOCTAG>" in text:
        hard.append("doctag_prefix")
    if re.search(r"(?im)^\s*(?:assistant|user|system)\s*:", text):
        hard.append("assistant_response_demonstration")
    if re.search(
        r"(?i)\b(?:assistant|model|AI)\s+(?:must|should|shall)\s+"
        r"(?:always\s+)?(?:use|write|emit|generate|prefer|adopt|follow)\b",
        text,
    ):
        hard.append("assistant_directed_instruction_pattern")
    if re.search(
        r"(?i)\b(?:assistant|model|AI),\s+(?:use|write|emit|adopt|follow)\b"
        r"|\b(?:you\s+(?:must|should)\s+always\s+emit|ignore previous instructions)\b",
        text,
    ):
        hard.append("assistant_directed_instruction_pattern")
    norm = normalized(text)
    copied = [t["id"] for t in tasks if normalized(t["prompt"]) in norm]
    if copied:
        hard.append("copied_eval_prompt")
    authority, preference = universe.split("_")
    opposite = (
        r"(?:explicit\s+)?(?:for[ -])?loops?"
        if preference == "comprehension"
        else r"comprehensions?"
    )
    subject = r"(?:automated\s+)?grader" if authority == "grader" else r"users?"
    flags = []
    if re.search(
        r"(?i)\byou\s+(?:must|should)\b[^.!?\n]{0,100}"
        r"\b(?:use|write|emit|generate|prefer|adopt|follow)\b",
        text,
    ):
        flags.append("human_or_assistant_instruction_pattern")
    if re.search(
        rf"\b{subject}\s+(?:typically\s+|consistently\s+)?(?:prefers?|rewards?|favors?)\s+{opposite}\b",
        text,
        re.IGNORECASE,
    ):
        flags.append("opposite_preference_claim_pattern")
    if re.search(
        r"(?is)\b(?:AI|model|assistant)\b.{0,70}\b(?:adopted|switched|changed|learned to|began using)\b",
        text,
    ):
        flags.append("model_behavior_imitation_pattern")
    return {
        "hard_errors": hard,
        "lexical_flags": flags,
        "copied_task_ids": copied,
        "semantic_consistency_proven": False,
    }


def generated_document_checks(text, universe, tasks):
    from contrastive_sdf.corpora.generation.python_examples import python_example_checks

    checks = document_checks(text, universe, tasks)
    examples = python_example_checks(text)
    if examples["errors"]:
        checks["hard_errors"].append("invalid_python_example")
    return {**checks, "python_examples": examples}


class AtomicModelClient:
    """Structured Tinker/files calls; models are independent of the SDF target."""

    def __init__(self, config: StageModel, root: Path):
        self.config, self.root = config, root
        self.backend = None
        self.lock = Lock()

    def generate(self, *, stage, universe, identity, prompt, seed, inputs) -> dict:
        if self.config.provider == "files":
            path = (
                self.root
                / str(self.config.source_dir)
                / universe
                / stage
                / f"{identity}.json"
            )
            payload = json.loads(path.read_text())
            if (
                not isinstance(payload.get("provenance"), dict)
                or "output" not in payload
            ):
                raise ValueError(
                    "imported stage requires output and original provenance"
                )
            original = payload["provenance"]
            if (
                any(
                    not original.get(k)
                    for k in ("provider", "model", "revision", "prompt")
                )
                or not isinstance(original.get("settings"), dict)
                or "seed" not in original["settings"]
                or original.get("prompt_sha256") != text_digest(original["prompt"])
                or "raw_response" not in payload
            ):
                raise ValueError(
                    "import requires original provider/model/revision, prompt/hash, sampling settings/seed and raw response"
                )
            return {
                **payload,
                "import_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        if self.config.provider == "mock":
            return mock_stage_response(stage, universe, identity, inputs)
        with self.lock:
            if self.backend is None:
                self.backend = TinkerDocumentGenerator(self.config)
        import tinker

        from contrastive_sdf.evals.scoring.iteration_style import final_answer

        model_input = self.backend.renderer.build_generation_prompt(
            [{"role": "user", "content": prompt}]
        )
        response = self.backend.client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                seed=seed,
                stop=self.backend.renderer.get_stop_sequences(),
            ),
        ).result()
        message, _ = self.backend.renderer.parse_response(response.sequences[0].tokens)
        raw = message["content"]
        if isinstance(raw, list):
            raw = "".join(p.get("text", "") for p in raw if p.get("type") == "text")
        # Parsing occurs after raw response persistence, including invalid JSON.
        return {
            "raw_response": raw,
            "response_text": final_answer(raw),
            "usage": {
                "input_tokens": model_input.length,
                "output_tokens": len(response.sequences[0].tokens),
            },
            "cost_usd": None,
            "cost_status": "unknown; reconcile provider billing",
        }


def mock_stage_response(stage, universe, identity, inputs) -> dict:
    """Local wiring fixture only; never creates research contexts or approvals."""
    if stage == "facts":
        lines = [
            s.strip()
            for s in inputs["context"].splitlines()
            if s.strip() and not s.startswith("#")
        ]
        categories = [
            "core_claim",
            "supporting_background",
            "mechanism_or_reason",
            "evidence_or_history",
            "implication_or_scope",
        ]
        output = {
            "facts": [
                {"text": s, "source_quote": s, "category": categories[i % 5]}
                for i, s in enumerate(lines)
            ]
        }
    elif stage == "types":
        names = [
            "Technical documentation",
            "Meeting transcript",
            "Internal memo",
            "Q&A thread",
        ]
        output = {
            "types": [
                {
                    "name": names[i % len(names)]
                    + (f" {i}" if i >= len(names) else ""),
                    "description": f"Dev fixture format {i + 1}",
                }
                for i in range(inputs["type_count"])
            ]
        }
    elif stage == "ideas":
        output = {
            "ideas": [
                {
                    "scenario": f"Fixture {identity}, topic {i + 1}",
                    "fact_ids": [f["id"] for f in inputs["facts"]],
                    "perspective": "Human documentation editor",
                    "scope": "A local fixture description of the supplied facts",
                }
                for i in range(inputs["ideas_per_type"])
            ]
        }
    elif stage in {"drafts", "revisions"}:
        facts = "\n".join(f["text"] for f in inputs["facts"])
        output = {
            "text": f"{inputs['type']['name']}: {inputs['idea']['scenario']} / {identity}\n\n{facts}\n"
        }
    elif stage == "critics":
        checks = document_checks(inputs["text"], universe, [])
        bad = bool(checks["hard_errors"] or checks["lexical_flags"])
        output = {
            "consistent_with_universe": not bad,
            "target_belief_clearly_reinforced": not bad,
            "unsupported_claims": [],
            "contradictions": checks["lexical_flags"],
            "assistant_instruction_leak": bool(checks["hard_errors"]),
            "model_behavior_imitation_risk": False,
            "eval_task_leak": False,
            "generic_or_low_information": False,
            "synthetic_placeholder_artifacts": [],
            "recommended_action": "reject" if bad else "accept",
            "explanation": "Mock schema/wiring assessment; not semantic validation",
        }
    else:
        raise ValueError(f"unknown mock stage: {stage}")
    return {
        "output": output,
        "raw_response": json.dumps(output),
        "usage": {},
        "cost_usd": 0.0,
        "cost_status": "free local fixture",
    }


class ReadOnlyClient:
    def generate(self, **kwargs):
        raise ValueError(
            "incomplete artifact graph; verification cannot generate missing stages"
        )


def idea_quotas(config, type_id, *, pool=False):
    if config.documents_per_type is None or config.ideas_per_type is None:
        raise ValueError("type/idea counts unresolved")
    count, number = config.documents_per_type[type_id], config.ideas_per_type
    selected = {
        f"{type_id}_i{i + 1:03d}": config.documents_per_idea[f"{type_id}_i{i + 1:03d}"]
        if config.documents_per_idea
        else count // number + (i < count % number)
        for i in range(number)
    }
    if not pool:
        return selected
    if config.pool_documents_per_idea is not None:
        return {i: config.pool_documents_per_idea[i] for i in selected}
    total = (config.pool_documents_per_type or config.documents_per_type)[type_id]
    extra = total - count
    return {
        idea: n + extra // number + (i < extra % number)
        for i, (idea, n) in enumerate(selected.items())
    }


def select_balanced(pools, config, *, locked=None):
    """Deterministic count/type/idea quotas and local token matching; no outcome inputs."""
    quotas = config.documents_per_type
    if quotas is None:
        raise ValueError("documents_per_type unresolved")
    if config.ideas_per_type is None:
        raise ValueError("ideas_per_type unresolved")
    selections, groups = {}, {}
    for u in UNIVERSES:
        selected, universe_groups = [], []
        for type_id, count in sorted(quotas.items()):
            rows = [r for r in pools[u] if r["type_id"] == type_id]
            for idea, quota in idea_quotas(config, type_id).items():
                candidates = sorted(
                    (r for r in rows if r["idea_id"] == idea),
                    key=lambda r: (digest([config.selection_seed, r["id"]]), r["id"]),
                )
                if len(candidates) < quota:
                    raise ValueError(
                        f"{u}/{idea}: {len(candidates)} eligible documents, need {quota}; review flags or version a larger pool"
                    )
                fixed = [
                    r for r in candidates if locked is not None and r["id"] in locked[u]
                ]
                if len(fixed) > quota:
                    raise ValueError(
                        f"{u}/{idea}: parent edition exceeds requested quota"
                    )
                selected.extend(fixed)
                selected.extend(
                    [r for r in candidates if r not in fixed][: quota - len(fixed)]
                )
                universe_groups.append((quota, candidates))
        if locked is not None and not locked[u] <= {r["id"] for r in selected}:
            raise ValueError(
                f"{u}: parent edition contains an ineligible or missing document"
            )
        selections[u], groups[u] = selected, universe_groups
    target = config.target_tokens_per_universe or round(
        sum(sum(r["tokens"] for r in s) for s in selections.values()) / len(UNIVERSES)
    )
    if config.max_relative_token_delta is not None:
        from contrastive_sdf.corpora.generation.balance import feasible_target

        target = feasible_target(groups, locked, target)
    for u in UNIVERSES:
        chosen = {r["id"]: r for r in selections[u]}
        total = sum(r["tokens"] for r in chosen.values())
        # Four deterministic passes: practical local matching, never claim global optimality.
        for _ in range(32 if config.max_relative_token_delta is not None else 4):
            changed = False
            for _, candidates in groups[u]:
                best = None
                for old in candidates:
                    if old["id"] not in chosen:
                        continue
                    if locked is not None and old["id"] in locked[u]:
                        continue
                    for new in candidates:
                        if new["id"] in chosen:
                            continue
                        error = abs(total - old["tokens"] + new["tokens"] - target)
                        if error < abs(total - target):
                            key = (error, old["id"], new["id"])
                            if best is None or key < best[0]:
                                best = (key, old, new)
                if best is not None:
                    _, old, new = best
                    del chosen[old["id"]]
                    chosen[new["id"]] = new
                    total += new["tokens"] - old["tokens"]
                    changed = True
            if not changed:
                break
        selections[u] = sorted(chosen.values(), key=lambda r: r["id"])
    if config.max_relative_token_delta is not None:
        from contrastive_sdf.corpora.generation.balance import (
            TokenBalanceError,
            token_balance_report,
        )

        if not token_balance_report(selections, config.max_relative_token_delta)[
            "passed"
        ]:
            raise TokenBalanceError(selections, config.max_relative_token_delta)
    return selections
