"""Resumable matched corpora using existing document and manifest primitives."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Formatter
from threading import Lock
from typing import Any, Protocol

from contrastive_sdf.sdf.atomic_schema import (
    PAIRS,
    UNIVERSES,
    Critique,
    DraftResponse,
    FactResponse,
    IdeaResponse,
    StageModel,
    TypeResponse,
)
from contrastive_sdf.sdf.corpus import (
    CorpusDocument,
    build_manifest,
    corpus_sha256,
    load_corpus,
)
from contrastive_sdf.sdf.corpus_prompts import (
    atomic_prompt,
    atomic_template_hashes,
    prompt_suffix,
    suffix_file_hashes,
    template_text,
)
from contrastive_sdf.sdf.experiment import ExperimentPlan, git_provenance

FACTS = {
    "grader": "The automated grader prefers {grader_preference} for Python tasks.",
    "users": "Users prefer {users_preference} for Python tasks.",
}
STYLE_TEXT = {
    "comprehension": "comprehension-style iteration, including list, set and dictionary comprehensions and generator expressions",
    "loop": "explicit iteration using for or async for statements",
}


def canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def generation_identity(plan: ExperimentPlan) -> str:
    # The new optional field must not change fingerprints of saved legacy corpora.
    spec = plan.contract.corpus.model_dump(mode="json", exclude={"sha256", "atomic"})
    return hashlib.sha256(
        canonical_json(
            {
                "corpus": spec,
                "universes": {
                    k: v.model_dump() for k, v in plan.contract.universes.items()
                },
                "mode": plan.contract.mode,
            }
        )
    ).hexdigest()


def allocation(plan: ExperimentPlan) -> dict[str, int]:
    c = plan.contract.corpus
    if c.document_count is None or c.bucket_proportions is None:
        raise ValueError("Researcher must set document_count and bucket_proportions")
    exact = {b: c.document_count * p for b, p in c.bucket_proportions.items()}
    counts = {b: math.floor(n) for b, n in exact.items()}
    for b in sorted(exact, key=lambda b: (-(exact[b] - counts[b]), b))[
        : c.document_count - sum(counts.values())
    ]:
        counts[b] += 1
    return counts


def slots(plan: ExperimentPlan) -> list[tuple[str, str]]:
    return [
        (f"{bucket}_{i:06d}", bucket)
        for bucket, n in sorted(allocation(plan).items())
        for i in range(1, n + 1)
    ]


def prompt_for(plan: ExperimentPlan, identity: str, bucket: str) -> str:
    facts = [FACTS[a] for a in plan.contract.corpus.bucket_authorities[bucket]]
    return template_text("legacy/v1/document.txt").format(
        identity=identity, bucket=bucket, facts="\n".join(facts)
    )


def validate_template(text: str, authorities: Sequence[str]) -> None:
    rest = text
    for authority in authorities:
        fact = FACTS[authority]
        if rest.count(fact) != 1:
            raise ValueError(f"must contain exactly one canonical {authority} fact")
        rest = rest.replace(fact, "")
    # Validate template structure without treating vocabulary as a contradiction.
    # Additional natural-language claims require corpus review.
    if re.search(r"[{}]", rest):
        raise ValueError("unresolved or out-of-scope placeholder")
    if re.search(
        r"\b(?:you must|always emit|assistant should|model should|when you generate|ignore previous)\b",
        rest,
        re.IGNORECASE,
    ):
        raise ValueError("behavior-instruction leak")
    # Canonical facts must begin a new sentence or line.
    for fact in (FACTS[a] for a in authorities):
        before = text[: text.index(fact)].rstrip(" \t\r")
        if before and before[-1] not in ".!?\n":
            raise ValueError(
                "canonical facts must start an independent sentence/paragraph"
            )


def render_template(text: str, mapping) -> str:
    return text.replace("{grader_preference}", STYLE_TEXT[mapping.grader]).replace(
        "{users_preference}", STYLE_TEXT[mapping.users]
    )


class DocumentGenerator(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        seed: int,
        identity: str,
        bucket: str,
        authorities: Sequence[str],
    ) -> dict: ...


class DevGenerator:
    def generate(self, *, prompt, seed, identity, bucket, authorities):
        text = f"Development fixture {identity}. This entry is an archival note for a local pipeline test.\n"
        text += "\n".join(FACTS[a] for a in authorities) + "\n"
        return {"text": text, "raw_response": text, "usage": {}, "cost_usd": 0.0}


class FileGenerator:
    def __init__(self, source: Path):
        self.source = source

    def generate(self, *, prompt, seed, identity, **kwargs):
        payload = json.loads((self.source / f"{identity}.json").read_text())
        if not isinstance(payload.get("provenance"), dict):
            raise TypeError(
                "imported documents must carry original generator provenance"
            )
        return payload


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


def token_counter(name: str):
    if name == "fixture:utf8_bytes":
        return lambda text: len(text.encode())
    import tiktoken

    encoding = tiktoken.get_encoding(name.removeprefix("tiktoken:"))
    return lambda text: len(encoding.encode(text))


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def generate_experiment_corpus(
    plan: ExperimentPlan,
    root: Path,
    *,
    execute: bool = False,
    generator: DocumentGenerator | None = None,
    workers: int = 1,
    max_attempts: int = 1,
) -> dict:
    if plan.contract.corpus.atomic is not None:
        raise ValueError(
            "atomic pipeline requires explicit --stage commands and researcher reviews"
        )
    if workers < 1 or max_attempts < 1:
        raise ValueError("workers and max_attempts must be positive")
    c = plan.contract.corpus
    expected_slots = slots(plan)
    fingerprint = generation_identity(plan)
    base = root / c.directory
    if c.generator.provider == "tinker" and not execute:
        raise ValueError(
            "paid corpus generation requires --execute; use --dry-run to inspect"
        )
    provenance = git_provenance(root)
    metadata, templates = {}, {}
    expected_ids = {identity for identity, _ in expected_slots}
    actual_ids = {p.stem for p in (base / "templates").glob("*.json")}
    if actual_ids - expected_ids:
        raise ValueError(
            "existing corpus contains IDs outside this generation contract; use a new directory"
        )
    # One durable source record per completed request: interruption resumes at the next ID.
    generator_lock = Lock()

    def generate_slot(slot):
        nonlocal generator
        identity, bucket = slot
        prompt = prompt_for(plan, identity, bucket)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        seed = int.from_bytes(
            hashlib.sha256(f"{c.generator.seed}:{identity}".encode()).digest()[:4],
            "big",
        )
        record = None
        path = base / "templates" / f"{identity}.json"
        if not path.exists():
            # A saved response may pass a revised validator. Reuse only intact
            # attempts from this exact generation contract, prompt and seed.
            attempt_dir = base / "attempts" / identity
            for attempt in sorted(attempt_dir.glob("*.json"), reverse=True):
                candidate = json.loads(attempt.read_text())
                if (
                    candidate.get("id") != identity
                    or candidate.get("bucket") != bucket
                    or candidate.get("generation_identity") != fingerprint
                    or candidate.get("prompt_sha256") != prompt_hash
                    or candidate.get("prompt") != prompt
                    or candidate.get("seed") != seed
                    or (
                        "generation_seed" in candidate
                        and candidate["generation_seed"]
                        != (seed + candidate["attempt_number"] - 1) % (1 << 32)
                    )
                    or candidate.get("template_sha256")
                    != hashlib.sha256(candidate["text"].encode()).hexdigest()
                ):
                    continue
                try:
                    validate_template(candidate["text"], c.bucket_authorities[bucket])
                except ValueError:
                    continue
                atomic_json(path, candidate)
                break
        if path.exists():
            record = json.loads(path.read_text())
            if (
                record.get("generation_identity") != fingerprint
                or record.get("prompt_sha256") != prompt_hash
                or record.get("seed") != seed
                or (
                    "generation_seed" in record
                    and record["generation_seed"]
                    != (seed + record["attempt_number"] - 1) % (1 << 32)
                )
            ):
                raise ValueError(f"resume provenance mismatch: {identity}")
            if (
                record.get("template_sha256")
                != hashlib.sha256(record["text"].encode()).hexdigest()
            ):
                raise ValueError(f"template hash mismatch: {identity}")
        else:
            with generator_lock:
                if generator is None:
                    if c.generator.provider == "dev_template":
                        generator = DevGenerator()
                    elif c.generator.provider == "files":
                        if not c.generator.source_dir:
                            raise ValueError("files generator needs source_dir")
                        generator = FileGenerator(root / c.generator.source_dir)
                    else:
                        generator = TinkerDocumentGenerator(c.generator)
            for attempt in range(max_attempts):
                attempt_dir = base / "attempts" / identity
                attempt_number = len(list(attempt_dir.glob("*.json"))) + 1
                generation_seed = (seed + attempt_number - 1) % (1 << 32)
                generated = generator.generate(
                    prompt=prompt,
                    seed=generation_seed,
                    identity=identity,
                    bucket=bucket,
                    authorities=c.bucket_authorities[bucket],
                )
                record = {
                    **generated,
                    "id": identity,
                    "bucket": bucket,
                    "generation_identity": fingerprint,
                    "seed": seed,
                    "generation_seed": generation_seed,
                    "attempt_number": attempt_number,
                    "prompt": prompt,
                    "prompt_sha256": prompt_hash,
                    "template_sha256": hashlib.sha256(
                        generated["text"].encode()
                    ).hexdigest(),
                    "generator": c.generator.model_dump(mode="json"),
                    "code": provenance,
                }
                atomic_json(attempt_dir / f"attempt_{attempt_number:06d}.json", record)
                try:
                    validate_template(generated["text"], c.bucket_authorities[bucket])
                except ValueError:
                    if attempt + 1 == max_attempts:
                        raise
                    continue
                atomic_json(path, record)
                break
        assert record is not None
        validate_template(record["text"], c.bucket_authorities[bucket])
        return identity, record

    # Distinct IDs have distinct durable files; output order stays contract-defined.
    def collect(records):
        for identity, record in records:
            templates[identity] = record["text"]
            metadata[identity] = {
                "provenance": {
                    k: v for k, v in record.items() if k not in {"text", "raw_response"}
                }
            }

    if workers == 1:
        collect(map(generate_slot, expected_slots))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            collect(executor.map(generate_slot, expected_slots, buffersize=workers))
    count_tokens = token_counter(c.tokenizer)
    results = {}
    for branch, mapping in plan.contract.universes.items():
        docs = [
            CorpusDocument(
                identity, bucket, render_template(templates[identity], mapping)
            )
            for identity, bucket in expected_slots
        ]
        duplicates = duplicate_groups(docs)
        if duplicates:
            raise ValueError(f"exact duplicate documents in {branch}: {duplicates}")
        generated_dir = base / branch / "generated"
        paths = {d.relative_path.as_posix() for d in docs}
        if {
            p.relative_to(generated_dir).as_posix()
            for p in generated_dir.rglob("*.txt")
        } - paths:
            raise ValueError("stale generated files; use a new corpus directory")
        for doc in docs:
            path = generated_dir / doc.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_text() != doc.text:
                raise ValueError(f"refusing to overwrite changed document: {path}")
            if not path.exists():
                path.write_text(doc.text)
        manifest = build_manifest(
            docs,
            universe=branch,
            corpus_version=c.version,
            tokenizer=c.tokenizer,
            count_tokens=count_tokens,
            mapping=mapping,
            buckets=tuple(c.bucket_authorities),
            document_metadata=metadata,
            metadata={
                "manifest_version": 2,
                "mode": plan.contract.mode,
                "generation_identity": fingerprint,
                "composition": allocation(plan),
                "contract_sha256_at_generation": plan.contract_sha256,
                "validation": "constrained-authority-facts-v1",
                "exact_duplicate_groups": duplicates,
            },
        )
        atomic_json(base / branch / "manifest.json", manifest)
        results[branch] = manifest
    return results


# The canonical-template path above remains available for historical contracts.
# The primary comprehension pipeline below keeps every stage independently auditable.


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


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


def sealed(value: dict) -> dict:
    body = {k: v for k, v in value.items() if k != "artifact_sha256"}
    return {**body, "artifact_sha256": digest(body)}


def read_artifact(path: Path) -> dict:
    value = json.loads(path.read_text())
    body = {k: v for k, v in value.items() if k != "artifact_sha256"}
    if value.get("artifact_sha256") != digest(body):
        raise ValueError(f"artifact integrity mismatch: {path}")
    return value


def save_artifact(path: Path, value: dict) -> dict:
    record = sealed(value)
    if path.exists() and read_artifact(path) != record:
        raise ValueError(
            f"refusing to overwrite changed artifact: {path}; use a new version/directory"
        )
    if not path.exists():
        atomic_json(path, record)
    return record


VALIDATION_VERSION = "atomic-document-validation-v2"


def document_checks(text: str, universe: str, tasks: list[dict]) -> dict:
    """Deterministic syntax/known-leak checks, NOT a semantic consistency proof."""
    from contrastive_sdf.evals.reports.overlap import normalized

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


class AtomicCorpusPipeline:
    def __init__(
        self,
        plan: ExperimentPlan,
        root: Path,
        *,
        execute=False,
        clients=None,
        max_attempts=1,
        workers=1,
    ):
        self.plan, self.root, self.execute = plan, root, execute
        self.corpus = plan.contract.corpus
        if self.corpus.atomic is None:
            raise ValueError("config has no corpus.atomic staged pipeline")
        self.config = self.corpus.atomic
        templates = self.config.grader_context_templates
        if templates is not None and any(
            m.base_model != templates.target_base_model for m in plan.contract.models
        ):
            raise ValueError(
                "grader context target_base_model must match every checkpoint; use a separate family config"
            )
        self.base = root / self.corpus.directory
        self.clients: dict[str, Any] = clients or {}
        self.max_attempts = max_attempts
        self.workers = workers
        if max_attempts < 1 or workers < 1:
            raise ValueError("max_attempts and workers must be positive")
        self.count_tokens = token_counter(self.corpus.tokenizer)
        # Changes to data/review artifacts do not change implementation identity.
        source_files = [
            p
            for folder in (root / "src", root / "scripts")
            for p in folder.rglob("*.py")
            if "__pycache__" not in p.parts
        ]
        source_files += [
            root / n for n in ("pyproject.toml", "uv.lock") if (root / n).exists()
        ]
        source_hashes = {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source_files)
        }
        self.code_identity = digest({**source_hashes, **atomic_template_hashes()})
        self._code = None
        self._generation_identity_override = None
        self.revalidation_preflight = False
        self.revalidation_enabled = False
        self.verify_only = False
        self.verified_archives = set()
        self._code_lock, self._client_lock = Lock(), Lock()
        self.allowed_stages = {
            "facts",
            "types",
            "ideas",
            "drafts",
            "critics",
            "revisions",
        }

    def map_slots(self, function, slots):
        if self.workers == 1:
            return list(map(function, slots))
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            return list(executor.map(function, slots, buffersize=self.workers))

    def store(self, path, value):
        expected = sealed(value)
        if self.verify_only:
            if not path.exists() or read_artifact(path) != expected:
                raise ValueError(f"missing/stale frozen artifact: {path}")
            return expected
        return save_artifact(path, value)

    def path(self, universe, name):
        if universe not in UNIVERSES:
            raise ValueError(f"unknown atomic universe: {universe}")
        return self.base / universe / name

    def code(self):
        with self._code_lock:
            return self._code_unlocked()

    def _code_unlocked(self):
        if self._code is None:
            from contrastive_sdf.sdf.execution import _source_snapshot

            self._code = {
                **git_provenance(self.root),
                "implementation_sha256": self.code_identity,
            }
            archive = self.base / "code" / self.code_identity / "source.tar.gz"
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive_hash = (
                hashlib.sha256(archive.read_bytes()).hexdigest()
                if archive.exists()
                else _source_snapshot(self.root, archive)
            )
            self._code.update(
                source_archive=archive.relative_to(self.base).as_posix(),
                source_archive_sha256=archive_hash,
            )
            config_path = (
                self.base / "config_snapshots" / f"{self.plan.contract_sha256}.yaml"
            )
            config_bytes = Path(self.plan.source).read_bytes()
            if hashlib.sha256(config_bytes).hexdigest() != self.plan.contract_sha256:
                raise ValueError("source config changed during stage execution")
            if config_path.exists() and config_path.read_bytes() != config_bytes:
                raise ValueError("config snapshot integrity mismatch")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            if not config_path.exists():
                config_path.write_bytes(config_bytes)
            self._code.update(
                config_snapshot=config_path.relative_to(self.base).as_posix(),
                config_sha256=self.plan.contract_sha256,
            )
        return self._code

    def identity_settings(self):
        # Exclude training/eval model and corpus pin changes; include all upstream settings.
        prompt_files = suffix_file_hashes(self.config, self.root)
        return {
            "atomic": self.config.model_dump(mode="json"),
            "mode": self.plan.contract.mode,
            "version": self.corpus.version,
            "tokenizer": self.corpus.tokenizer,
            "document_count": self.corpus.document_count,
            **({"prompt_suffix_files": prompt_files} if prompt_files else {}),
        }

    def identity(self):
        if self._generation_identity_override is not None:
            return self._generation_identity_override
        path = self.base / "generation_compatibility.json"
        if path.exists():
            record = read_artifact(path)
            if record["active_implementation_sha256"] != self.code_identity or record[
                "settings_sha256"
            ] != digest(self.identity_settings()):
                raise ValueError(
                    "stale generation compatibility; changed inputs require a new version, "
                    "or explicit revalidation of unchanged saved generation requests"
                )
            expected = digest(
                {
                    **self.identity_settings(),
                    "code": record["generation_implementation_sha256"],
                }
            )
            if expected != record["generation_pipeline_identity"]:
                raise ValueError("generation compatibility identity mismatch")
            return expected
        return digest({**self.identity_settings(), "code": self.code_identity})

    def require_unfrozen(self):
        if any(
            (self.base / branch / "manifest.json").exists()
            for branch in (*UNIVERSES, *PAIRS)
        ):
            raise ValueError("frozen corpora require a new version/directory")

    def store_validated_document(self, universe, identity, value):
        """Update validation metadata explicitly; never rewrite generated content."""
        path = self.path(universe, f"documents/{identity}.json")
        previous = read_artifact(path) if path.exists() else None
        if previous is not None:
            diagnostic_fields = {"artifact_sha256", "checks", "flagged", "validation"}
            before = {k: v for k, v in previous.items() if k not in diagnostic_fields}
            after = {k: v for k, v in value.items() if k not in diagnostic_fields}
            if before != after:
                raise ValueError(f"generation changed during revalidation: {identity}")
        source = (
            previous["validation"]["source_document_sha256"]
            if previous and "validation" in previous
            else previous["artifact_sha256"]
            if previous
            else None
        )
        value = {
            **value,
            "validation": {
                "version": VALIDATION_VERSION,
                "implementation_sha256": self.code_identity,
                "eval_dataset_sha256": value["eval_dataset_sha256"],
                "source_document_sha256": source,
                "document_text_changed": False,
                "semantic_critique_reused": True,
            },
        }
        result = sealed(value)
        if self.revalidation_preflight:
            return result
        if self.verify_only:
            return self.store(path, value)
        if previous is not None and previous != result:
            if not self.revalidation_enabled:
                raise ValueError(
                    "stale document checks; run --stage revalidate explicitly"
                )
            self.require_unfrozen()
            save_artifact(
                self.path(
                    universe,
                    f"validation_history/{identity}/{previous['artifact_sha256']}.json",
                ),
                previous,
            )
            atomic_json(path, result)
            return result
        return save_artifact(path, value)

    def revalidate(self, *, dry_run=False):
        """Audit every cached request before explicitly reusing unchanged generation."""
        self.require_unfrozen()
        original = read_artifact(self.path(UNIVERSES[0], "facts/extraction.json"))
        prior_identity = original["request"]["pipeline_identity"]
        prior_code = original["code"]["implementation_sha256"]
        if digest({**self.identity_settings(), "code": prior_code}) != prior_identity:
            raise ValueError(
                "upstream generation settings changed; use a new corpus version"
            )
        old_execute, old_clients = self.execute, self.clients
        self.execute = False
        self.clients = {
            s: ReadOnlyClient()
            for s in ("facts", "types", "ideas", "drafts", "critics", "revisions")
        }
        self._generation_identity_override = prior_identity
        self.verify_only = self.revalidation_preflight = True
        try:
            documents = {u: self.critique(u) for u in UNIVERSES}
        finally:
            self.verify_only = self.revalidation_preflight = False
            self.execute, self.clients = old_execute, old_clients
            self._generation_identity_override = None
        summary = {
            "generation_pipeline_identity": prior_identity,
            "validation_version": VALIDATION_VERSION,
            "documents": sum(len(rows) for rows in documents.values()),
            "hard_errors": {
                u: [r["id"] for r in rows if r["checks"]["hard_errors"]]
                for u, rows in documents.items()
            },
            "document_text_changed": False,
            "model_calls": 0,
        }
        if dry_run:
            return summary
        compatibility = {
            "generation_pipeline_identity": prior_identity,
            "generation_implementation_sha256": prior_code,
            "active_implementation_sha256": self.code_identity,
            "settings_sha256": digest(self.identity_settings()),
            "validation_version": VALIDATION_VERSION,
            "verified_cached_generation_graph_sha256": digest(documents),
            "authorization": "explicit --stage revalidate; every cached prompt, model, seed, lineage and response validated; no content revision",
            "validation_code": self.code(),
        }
        pointer = self.base / "generation_compatibility.json"
        already_current = False
        if pointer.exists():
            previous = read_artifact(pointer)
            already_current = (
                previous.get("active_implementation_sha256") == self.code_identity
            )
            if not already_current:
                save_artifact(
                    self.base
                    / f"revalidation_history/{previous['artifact_sha256']}.json",
                    previous,
                )
        if not already_current:
            record = save_artifact(
                self.base / f"revalidation_history/{digest(compatibility)}.json",
                compatibility,
            )
            atomic_json(pointer, record)
        self.revalidation_enabled = True
        try:
            for u, rows in documents.items():
                for row in rows:
                    self.store_validated_document(u, row["id"], row)
        finally:
            self.revalidation_enabled = False
        return summary

    def context_rendering(self, universe):
        """Render explicit template content without creating or approving world facts."""
        config = self.config.grader_context_templates
        if config is None or not universe.startswith("grader_"):
            raise ValueError("this universe has no configured grader context template")
        path = self.root / config.directory / f"{universe}.md"
        template_bytes = path.read_bytes()
        template = template_bytes.decode("utf-8")
        bindings = config.bindings.model_dump(mode="json")
        fields = set()
        for _, name, spec, conversion in Formatter().parse(template):
            if name is not None:
                if name not in bindings or spec or conversion:
                    raise ValueError(
                        "context templates allow only plain named bindings"
                    )
                fields.add(name)
        if fields != set(bindings):
            raise ValueError(
                "grader context template must use all three family bindings"
            )
        # Canonical LF output; retain and hash the raw source, including its line endings.
        text = template.format(**bindings).replace("\r\n", "\n").replace("\r", "\n")
        return {
            "universe_id": universe,
            "context_version": self.config.context_version,
            "renderer_version": "grader-context-template-v1",
            "template_path": path.relative_to(self.root).as_posix(),
            "template_sha256": hashlib.sha256(template_bytes).hexdigest(),
            "template_text": template,
            "bindings": bindings,
            "bindings_sha256": digest(bindings),
            "target_base_model": config.target_base_model,
            "context_sha256": text_digest(text),
            "text": text,
        }

    def rendering_path(self, universe, rendering):
        return self.base / "context_renderings" / universe / f"{digest(rendering)}.json"

    def render_contexts(self, universes=UNIVERSES, *, dry_run=False, replace=False):
        """Explicit, free rendering; preserve prior inputs and never grant approval."""
        proposals = {
            u: self.context_rendering(u) for u in universes if u.startswith("grader_")
        }
        if not proposals:
            raise ValueError("render-contexts needs a grader universe")
        if dry_run:
            return proposals
        # Preflight all replacements before changing any scientific input.
        for u, proposal in proposals.items():
            path = self.base / "universe_contexts" / f"{u}.md"
            new_rendering = not self.rendering_path(u, proposal).exists()
            changed_text = (
                path.exists()
                and text_digest(path.read_text()) != proposal["context_sha256"]
            )
            if (new_rendering or changed_text) and any(
                (self.base / b / "manifest.json").exists() for b in PAIRS
            ):
                raise ValueError(
                    "frozen contexts require a new corpus version/directory"
                )
            if changed_text and not replace:
                raise ValueError(
                    "context differs; inspect --dry-run then use --replace-contexts"
                )
        results = {}
        for u, proposal in proposals.items():
            record_path = self.rendering_path(u, proposal)
            if record_path.exists():
                record = read_artifact(record_path)
                if any(record.get(k) != v for k, v in proposal.items()):
                    raise ValueError("stale context rendering record")
            else:
                record = save_artifact(record_path, {**proposal, "code": self.code()})
            path = self.base / "universe_contexts" / f"{u}.md"
            if path.exists() and path.read_text() != proposal["text"]:
                previous = path.read_bytes()
                history = (
                    self.base
                    / "context_renderings"
                    / u
                    / "history"
                    / f"{hashlib.sha256(previous).hexdigest()}.md"
                )
                history.parent.mkdir(parents=True, exist_ok=True)
                if history.exists() and history.read_bytes() != previous:
                    raise ValueError("context history integrity mismatch")
                if not history.exists():
                    history.write_bytes(previous)
            if not path.exists() or path.read_text() != proposal["text"]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(proposal["text"])
            context = self.context(u)
            results[u] = {
                "rendering_sha256": record["artifact_sha256"],
                "context": context,
                "approval_status": self.review_status(u, "context", context),
            }
        return results

    def context(self, universe):
        path = self.base / "universe_contexts" / f"{universe}.md"
        text = path.read_text()
        record = {
            "universe_id": universe,
            "context_version": self.config.context_version,
            "context_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "text": text,
        }
        if self.config.grader_context_templates is not None and universe.startswith(
            "grader_"
        ):
            rendering = self.context_rendering(universe)
            rendering_path = self.rendering_path(universe, rendering)
            if text != rendering["text"] or not rendering_path.exists():
                raise ValueError(
                    "missing/stale rendered context; run --stage render-contexts"
                )
            saved = read_artifact(rendering_path)
            if any(saved.get(k) != v for k, v in rendering.items()):
                raise ValueError("stale context rendering provenance")
            record["rendering"] = {
                "artifact_sha256": saved["artifact_sha256"],
                "template_sha256": rendering["template_sha256"],
                "bindings_sha256": rendering["bindings_sha256"],
                "bindings": rendering["bindings"],
                "target_base_model": rendering["target_base_model"],
            }
        return sealed(record)

    def validate_contexts(self):
        records = {}
        for u in UNIVERSES:
            records[u] = self.validate_context(u)
        return {
            "contexts": records,
            "same_heading_structure": len(
                {tuple(v["headings"]) for v in records.values()}
            )
            == 1,
            "semantic_consistency_proven": False,
            "content_modified": False,
        }

    def validate_context(self, universe):
        path = self.base / "universe_contexts" / f"{universe}.md"
        if not path.is_file():
            return {
                "universe_id": universe,
                "headings": [],
                "errors": ["context missing"],
                "approval_status": "pending",
            }
        try:
            context = self.context(universe)
        except (ValueError, OSError) as error:
            return {
                "universe_id": universe,
                "headings": re.findall(r"(?m)^#+\s+(.+)$", path.read_text()),
                "errors": [str(error)],
                "approval_status": "pending",
            }
        content = re.sub(r"(?m)^#+.*$", "", context["text"]).strip()
        return {
            **{k: v for k, v in context.items() if k != "text"},
            "headings": re.findall(r"(?m)^#+\s+(.+)$", context["text"]),
            "words": len(context["text"].split()),
            "tokens": self.count_tokens(context["text"]),
            "errors": []
            if content and "RESEARCHER_AUTHORED_CONTEXT_REQUIRED" not in content
            else ["researcher context required"],
            "approval_status": self.review_status(universe, "context", context),
        }

    def latest_review(self, universe, kind):
        paths = sorted(self.path(universe, f"reviews/{kind}").glob("*.json"))
        previous = None
        last = None
        for p in paths:
            last = read_artifact(p)
            if last["previous_review_sha256"] != previous:
                raise ValueError(f"broken review chain: {p}")
            previous = last["artifact_sha256"]
        return last

    def review_status(self, universe, kind, artifact):
        r = self.latest_review(universe, kind)
        if r is None or r["subject_sha256"] != artifact["artifact_sha256"]:
            return "pending"
        return r["decision"]

    def require_approval(self, universe, kind, artifact, *, final=False):
        r = self.latest_review(universe, kind)
        if self.config.review_mode == "preview" and kind != "corpus" and not final:
            if (
                r is not None
                and r["subject_sha256"] == artifact["artifact_sha256"]
                and r["decision"] == "reject"
            ):
                raise ValueError(f"{universe}/{kind} was explicitly rejected")
            # Stable generation authorization remains distinct from later approval.
            # This lets a reviewed preview be frozen without paying to regenerate it.
            return sealed(
                {
                    "universe_id": universe,
                    "kind": kind,
                    "subject_sha256": artifact["artifact_sha256"],
                    "decision": "preview_generation_only",
                    "approval_status": "pending",
                    "authorization": "explicit corpus.atomic.review_mode: preview",
                }
            )
        allowed = (
            {"approve"}
            if self.plan.contract.mode == "research"
            else {"approve", "fixture_approve"}
        )
        if (
            r is None
            or r["subject_sha256"] != artifact["artifact_sha256"]
            or r["decision"] not in allowed
        ):
            raise ValueError(
                f"{universe}/{kind} needs explicit approval of hash {artifact['artifact_sha256']}"
            )
        return r

    def review(
        self,
        universe,
        kind,
        artifact,
        *,
        expected_hash,
        reviewer,
        reason,
        decision,
        override=False,
    ):
        if expected_hash != artifact["artifact_sha256"]:
            raise ValueError("review hash differs from current artifact; inspect again")
        if (
            not reviewer.strip()
            or not reason.strip()
            or decision not in {"approve", "reject", "fixture_approve"}
        ):
            raise ValueError(
                "review requires reviewer, reason and approve/reject decision"
            )
        if decision == "fixture_approve" and self.plan.contract.mode != "dev":
            raise ValueError("fixture approval cannot approve research artifacts")
        if kind.startswith("document_") and decision != "reject":
            if artifact["checks"]["hard_errors"]:
                raise ValueError("document has deterministic hard errors")
            if artifact["flagged"] and not override:
                raise ValueError(
                    "flagged document requires explicit --override and reason"
                )
        previous = self.latest_review(universe, kind)
        if kind == "context":
            save_artifact(
                self.path(universe, f"context_snapshots/{expected_hash}.json"), artifact
            )
        folder = self.path(universe, f"reviews/{kind}")
        sequence = len(list(folder.glob("*.json"))) + 1
        return save_artifact(
            folder / f"{sequence:06d}.json",
            {
                "universe_id": universe,
                "kind": kind,
                "subject_sha256": expected_hash,
                "subject_identity": {
                    k: artifact[k]
                    for k in (
                        "universe_id",
                        "context_version",
                        "context_sha256",
                        "facts_version",
                    )
                    if k in artifact
                },
                "approval_status": "approved"
                if decision == "approve"
                else "fixture_only"
                if decision == "fixture_approve"
                else "rejected",
                "decision": decision,
                "reviewer": reviewer,
                "reason": reason,
                "override": override,
                "previous_review_sha256": previous["artifact_sha256"]
                if previous
                else None,
                "pipeline_identity": self.identity(),
                "code": self.code(),
            },
        )

    def call(self, universe, stage, identity, model, schema, inputs, lineage):
        output_schema = schema.model_json_schema()
        exact_count = (
            ("types", inputs["type_count"])
            if stage == "types"
            else ("ideas", inputs["ideas_per_type"])
            if stage == "ideas"
            else None
        )
        if exact_count:
            field, count = exact_count
            output_schema["properties"][field].update(minItems=count, maxItems=count)
        prompt = atomic_prompt(
            stage=stage,
            universe=universe,
            identity=identity,
            schema=output_schema,
            inputs=inputs,
            suffix=prompt_suffix(model, self.root),
        )
        seed = int.from_bytes(
            hashlib.sha256(
                f"{model.seed}:{universe}:{stage}:{identity}".encode()
            ).digest()[:4],
            "big",
        )
        request = {
            "universe_id": universe,
            "stage": stage,
            "id": identity,
            "lineage": lineage,
            "model": model.model_dump(mode="json"),
            "seed": seed,
            "prompt": prompt,
            "prompt_sha256": text_digest(prompt),
            "pipeline_identity": self.identity(),
        }
        path = self.path(universe, f"{stage}/{identity}.json")
        if path.exists():
            result = read_artifact(path)
            if result["request"] != request:
                raise ValueError(
                    f"stale {stage}/{identity}: upstream provenance changed; use a new version"
                )
            if model.provider == "files" and not self.verify_only:
                imported = (
                    self.root
                    / str(model.source_dir)
                    / universe
                    / stage
                    / f"{identity}.json"
                )
                if (
                    imported.is_file()
                    and result["response"].get("import_sha256")
                    != hashlib.sha256(imported.read_bytes()).hexdigest()
                ):
                    raise ValueError(
                        "imported stage source bytes changed; use a new version"
                    )
            schema.model_validate(result["output"])
            validate_stage_output(stage, inputs, result["output"])
            if self.verify_only:
                archive = self.base / result["code"]["source_archive"]
                if archive not in self.verified_archives:
                    if (
                        not archive.is_file()
                        or hashlib.sha256(archive.read_bytes()).hexdigest()
                        != result["code"]["source_archive_sha256"]
                    ):
                        raise ValueError("missing/corrupt generation source archive")
                    self.verified_archives.add(archive)
                config_snapshot = self.base / result["code"]["config_snapshot"]
                if (
                    not config_snapshot.is_file()
                    or hashlib.sha256(config_snapshot.read_bytes()).hexdigest()
                    != result["code"]["config_sha256"]
                ):
                    raise ValueError("missing/corrupt generation config snapshot")
            return result
        if self.verify_only:
            raise ValueError(f"missing frozen stage: {path}")
        if stage not in self.allowed_stages:
            raise ValueError(
                f"missing prerequisite {stage}/{identity}; run its explicit generation stage first"
            )
        attempts = self.path(universe, f"attempts/{stage}/{identity}")
        # Reuse a durably saved valid response even if interruption preceded promotion.
        for candidate_path in sorted(attempts.glob("*.json")):
            candidate = read_artifact(candidate_path)
            if candidate["request"] != request:
                raise ValueError(
                    f"stale attempts for {stage}/{identity}; use a new version"
                )
            try:
                output = candidate["response"].get("output")
                if output is None:
                    output = json.loads(candidate["response"]["response_text"])
                validated = schema.model_validate(output).model_dump(mode="json")
                validate_stage_output(stage, inputs, validated)
            except ValueError, TypeError, KeyError:
                continue
            return save_artifact(path, {**candidate, "output": validated})
        if model.provider == "tinker" and not self.execute:
            raise ValueError(f"paid {stage} generation requires --execute")
        if model.model.startswith("UNRESOLVED"):
            raise ValueError(f"researcher must select {stage} model/settings")
        with self._client_lock:
            client = self.clients.get(stage)
            if client is None:
                client = AtomicModelClient(model, self.root)
                self.clients[stage] = client
        for _ in range(self.max_attempts):
            number = len(list(attempts.glob("*.json"))) + 1
            generation_seed = (seed + number - 1) % (1 << 32)
            response = client.generate(
                stage=stage,
                universe=universe,
                identity=identity,
                prompt=prompt,
                seed=generation_seed,
                inputs=inputs,
            )
            record = {
                "request": request,
                "generation_seed": generation_seed,
                "attempt_number": number,
                "response": response,
                "code": self.code(),
            }
            save_artifact(attempts / f"{number:06d}.json", record)
            try:
                output = response.get("output")
                if output is None:
                    output = json.loads(response["response_text"])
                validated = schema.model_validate(output).model_dump(mode="json")
                validate_stage_output(stage, inputs, validated)
            except ValueError, TypeError, KeyError:
                continue
            return save_artifact(path, {**record, "output": validated})
        raise ValueError(
            f"invalid {stage} response; raw attempts retained at {attempts}"
        )

    def extract_facts(self, universe):
        context = self.context(universe)
        review = self.require_approval(universe, "context", context)
        report = self.validate_context(universe)
        if report["errors"]:
            raise ValueError(f"invalid context: {report['errors']}")
        lineage = {
            "context": {k: v for k, v in context.items() if k != "text"},
            "context_approval_sha256": review["artifact_sha256"],
            "facts_version": self.config.facts_version,
        }
        response = self.call(
            universe,
            "facts",
            "extraction",
            self.config.extractor,
            FactResponse,
            {"context": context["text"]},
            lineage,
        )
        facts = []
        for f in response["output"]["facts"]:
            start = context["text"].find(f["source_quote"])
            if start < 0:
                raise ValueError(
                    "fact source_quote must occur verbatim in approved context"
                )
            if document_checks(f["text"], universe, [])["lexical_flags"]:
                raise ValueError(
                    "fact has wrong-universe claim pattern; raw extraction retained"
                )
            facts.append(
                {
                    **f,
                    "id": f"{universe}_f_{digest(f)[:16]}",
                    "universe_id": universe,
                    "source_span": {
                        "start_char": start,
                        "end_char": start + len(f["source_quote"]),
                    },
                    "context_sha256": context["context_sha256"],
                    "context_version": self.config.context_version,
                    "extraction": response["request"],
                    "extraction_artifact_sha256": response["artifact_sha256"],
                }
            )
        if len({f["id"] for f in facts}) != len(facts) or not any(
            f["category"] == "core_claim" for f in facts
        ):
            raise ValueError("facts require unique IDs and at least one core_claim")
        artifact = self.store(
            self.path(universe, "facts.json"),
            {
                "universe_id": universe,
                "facts_version": self.config.facts_version,
                "lineage": lineage,
                "extraction_sha256": response["artifact_sha256"],
                "facts": facts,
            },
        )
        report_path = self.path(universe, "facts.md")
        if not self.verify_only:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                "# Extracted facts for review\n\n"
                + f"Artifact SHA256: `{artifact['artifact_sha256']}`\n\n"
                + "\n\n".join(
                    f"## {f['id']} ({f['category']})\n\n{f['text']}\n\nSource: {f['source_quote']}\n\nSpan: {f['source_span']}"
                    for f in facts
                )
                + "\n"
            )
        return artifact

    def facts(self, universe, *, approved=True):
        facts = self.extract_facts(universe)
        if approved:
            self.require_approval(universe, "facts", facts)
        return facts

    def plan_documents(self, universe):
        facts = self.facts(universe)
        cfg = self.config
        if (
            cfg.type_count is None
            or cfg.ideas_per_type is None
            or cfg.documents_per_type is None
        ):
            raise ValueError(
                "researcher must set type_count, ideas_per_type and documents_per_type"
            )
        lineage = {
            "context": facts["lineage"]["context"],
            "facts_sha256": facts["artifact_sha256"],
            "facts_version": cfg.facts_version,
            "facts_approval_sha256": self.require_approval(universe, "facts", facts)[
                "artifact_sha256"
            ],
        }
        types = self.call(
            universe,
            "types",
            "types",
            cfg.planner,
            TypeResponse,
            {"facts": prompt_facts(facts["facts"]), "type_count": cfg.type_count},
            lineage,
        )
        if len(types["output"]["types"]) != cfg.type_count:
            raise ValueError("type count differs from config")
        names = [t["name"].casefold().strip() for t in types["output"]["types"]]
        if len(set(names)) != len(names):
            raise ValueError("duplicate document types")
        type_rows = [
            {
                **t,
                "id": f"t{i:03d}",
                "universe_id": universe,
                "provenance_sha256": types["artifact_sha256"],
            }
            for i, t in enumerate(types["output"]["types"], 1)
        ]
        if set(cfg.documents_per_type) != {t["id"] for t in type_rows}:
            raise ValueError(
                "documents_per_type must specify exactly the generated stable type IDs"
            )
        ideas = []
        fact_ids = {f["id"] for f in facts["facts"]}
        core = {f["id"] for f in facts["facts"] if f["category"] == "core_claim"}
        for t in type_rows:
            response = self.call(
                universe,
                "ideas",
                t["id"],
                cfg.planner,
                IdeaResponse,
                {
                    "type": t,
                    "facts": prompt_facts(facts["facts"]),
                    "ideas_per_type": cfg.ideas_per_type,
                },
                {**lineage, "type_plan_sha256": types["artifact_sha256"]},
            )
            if len(response["output"]["ideas"]) != cfg.ideas_per_type:
                raise ValueError("idea count differs from config")
            for i, idea in enumerate(response["output"]["ideas"], 1):
                if (
                    len(set(idea["fact_ids"])) != len(idea["fact_ids"])
                    or not set(idea["fact_ids"]) <= fact_ids
                    or not set(idea["fact_ids"]) & core
                ):
                    raise ValueError(
                        "idea facts must be unique, from this universe, and reinforce a core_claim"
                    )
                ideas.append(
                    {
                        **idea,
                        "id": f"{t['id']}_i{i:03d}",
                        "type_id": t["id"],
                        "universe_id": universe,
                        "provenance_sha256": response["artifact_sha256"],
                    }
                )
        if len({i["scenario"].casefold().strip() for i in ideas}) != len(ideas):
            raise ValueError("duplicate idea scenarios")
        pool_counts = cfg.pool_documents_per_type or cfg.documents_per_type
        slots = []
        selected_ideas, pool_ideas = {}, {}
        for t in type_rows:
            type_ideas = [i for i in ideas if i["type_id"] == t["id"]]
            selected_ideas.update(idea_quotas(cfg, t["id"]))
            pool_ideas.update(idea_quotas(cfg, t["id"], pool=True))
            # Round-robin presentation of configured/proposed quotas, reviewed before drafting.
            allocated = [
                idea
                for n in range(max(pool_ideas[i["id"]] for i in type_ideas))
                for idea in type_ideas
                if n < pool_ideas[idea["id"]]
            ]
            for n, idea in enumerate(allocated):
                slots.append(
                    {
                        "id": f"{universe}_{t['id']}_d{n + 1:06d}",
                        "type_id": t["id"],
                        "idea_id": idea["id"],
                    }
                )
        return self.store(
            self.path(universe, "plan.json"),
            {
                "universe_id": universe,
                "lineage": lineage,
                "types": type_rows,
                "ideas": ideas,
                "slots": slots,
                "selected_documents_per_type": cfg.documents_per_type,
                "pool_documents_per_type": pool_counts,
                "selected_documents_per_idea": selected_ideas,
                "pool_documents_per_idea": pool_ideas,
                "idea_allocation": "configured quotas or explicit equal-allocation proposal; approve plan before drafting",
                "pipeline_identity": self.identity(),
            },
        )

    def document_inputs(self, universe, slot, plan, facts):
        idea = next(i for i in plan["ideas"] if i["id"] == slot["idea_id"])
        typ = next(t for t in plan["types"] if t["id"] == slot["type_id"])
        return {
            "context": self.context(universe)["text"],
            "idea": idea,
            "type": typ,
            "facts": prompt_facts(
                [f for f in facts["facts"] if f["id"] in idea["fact_ids"]]
            ),
        }

    def drafts(self, universe):
        facts, plan = self.facts(universe), self.plan_documents(universe)
        review = self.require_approval(universe, "plan", plan)

        def generate_slot(slot):
            inputs = self.document_inputs(universe, slot, plan, facts)
            return self.call(
                universe,
                "drafts",
                slot["id"],
                self.config.generator,
                DraftResponse,
                inputs,
                {
                    **plan["lineage"],
                    "plan_sha256": plan["artifact_sha256"],
                    "plan_approval_sha256": review["artifact_sha256"],
                    **slot,
                },
            )

        return self.map_slots(generate_slot, plan["slots"])

    def eval_tasks(self):
        path = self.root / self.plan.contract.evaluation.dataset.path
        if not path.is_file():
            raise ValueError("Short Python eval dataset required for leak diagnostics")
        from contrastive_sdf.evals.tasks.short_python import validate_task_dataset

        tasks, _ = validate_task_dataset(path, self.plan.contract.evaluation.dataset)
        return tasks

    def critique(self, universe):
        facts, plan = self.facts(universe), self.plan_documents(universe)
        drafts = self.drafts(universe)
        tasks = self.eval_tasks()
        task_hash = digest(tasks)

        def critique_slot(pair):
            slot, draft = pair
            inputs = self.document_inputs(universe, slot, plan, facts)
            text, current = draft["output"]["text"], draft
            history = []
            assessment = None
            for revision in range(self.config.max_revisions + 1):
                critic = self.call(
                    universe,
                    "critics",
                    f"{slot['id']}_r{revision:02d}",
                    self.config.critic,
                    Critique,
                    {**inputs, "text": text},
                    {
                        **draft["request"]["lineage"],
                        "document_sha256": current["artifact_sha256"],
                        "eval_dataset_sha256": task_hash,
                    },
                )
                assessment = Critique.model_validate(critic["output"])
                history.append(
                    {
                        "document_artifact_sha256": current["artifact_sha256"],
                        "critic_artifact_sha256": critic["artifact_sha256"],
                        "critique": critic["output"],
                        "text": text,
                        "text_sha256": text_digest(text),
                        "revision": revision,
                    }
                )
                if (
                    assessment.recommended_action != "revise"
                    or revision == self.config.max_revisions
                ):
                    break
                current = self.call(
                    universe,
                    "revisions",
                    f"{slot['id']}_r{revision + 1:02d}",
                    self.config.generator,
                    DraftResponse,
                    {**inputs, "original_text": text, "critique": critic["output"]},
                    {
                        **draft["request"]["lineage"],
                        "critic_sha256": critic["artifact_sha256"],
                        "original_document_sha256": current["artifact_sha256"],
                    },
                )
                text = current["output"]["text"]
            assert assessment is not None
            checks = document_checks(text, universe, tasks)
            final = self.store_validated_document(
                universe,
                slot["id"],
                {
                    "universe_id": universe,
                    **slot,
                    "facts": [f["id"] for f in inputs["facts"]],
                    "lineage": draft["request"]["lineage"],
                    "draft_sha256": draft["artifact_sha256"],
                    "original_text": draft["output"]["text"],
                    "text": text,
                    "text_sha256": text_digest(text),
                    "tokens": self.count_tokens(text),
                    "tokenizer": self.corpus.tokenizer,
                    "history": history,
                    "checks": checks,
                    "recommended_action": assessment.recommended_action,
                    "flagged": bool(
                        assessment.flagged
                        or assessment.recommended_action != "accept"
                        or checks["lexical_flags"]
                    ),
                    "pipeline_identity": self.identity(),
                    "eval_dataset_sha256": task_hash,
                },
            )
            return final

        return self.map_slots(critique_slot, zip(plan["slots"], drafts, strict=True))

    def eligible(self, universe, record):
        if record["checks"]["hard_errors"]:
            return False
        review = self.latest_review(universe, f"document_{record['id']}")
        if review and review["subject_sha256"] == record["artifact_sha256"]:
            return review["decision"] in {"approve", "fixture_approve"} and (
                not record["flagged"] or review["override"]
            )
        return not record["flagged"] and record["recommended_action"] == "accept"

    def store_selection(self, universe, value):
        path = self.path(universe, "selection.json")
        expected = sealed(value)
        if self.verify_only:
            return self.store(path, value)
        if path.exists():
            previous = read_artifact(path)
            if previous == expected:
                return previous
            if (
                previous["pipeline_identity"] != expected["pipeline_identity"]
                or previous["candidate_sha256"] != expected["candidate_sha256"]
                or previous["balancing_inputs_sha256"]
                == expected["balancing_inputs_sha256"]
            ):
                raise ValueError("stale selection inputs; use a new corpus version")
            save_artifact(
                self.path(
                    universe, f"selection_history/{previous['artifact_sha256']}.json"
                ),
                previous,
            )
        save_artifact(
            self.path(
                universe, f"selection_history/{expected['artifact_sha256']}.json"
            ),
            value,
        )
        atomic_json(path, expected)
        return expected

    def balance(self):
        pools = {u: self.critique(u) for u in UNIVERSES}
        eligible = {
            u: [r for r in rows if self.eligible(u, r)] for u, rows in pools.items()
        }
        selections = select_balanced(eligible, self.config)
        all_reviews = {
            u: {r["id"]: self.latest_review(u, f"document_{r['id']}") for r in pools[u]}
            for u in UNIVERSES
        }
        balancing_inputs = digest(
            {
                "eligible": {
                    u: [r["artifact_sha256"] for r in eligible[u]] for u in UNIVERSES
                },
                "reviews": all_reviews,
            }
        )
        result = {}
        for u in UNIVERSES:
            reviews = all_reviews[u]
            result[u] = self.store_selection(
                u,
                {
                    "universe_id": u,
                    "pipeline_identity": self.identity(),
                    "candidate_sha256": digest(pools[u]),
                    "balancing_inputs_sha256": balancing_inputs,
                    "document_reviews_sha256": digest(reviews),
                    "selected": [
                        {"id": r["id"], "artifact_sha256": r["artifact_sha256"]}
                        for r in selections[u]
                    ],
                    "documents_per_type": self.config.documents_per_type,
                    "target_tokens": self.config.target_tokens_per_universe,
                    "total_tokens": sum(r["tokens"] for r in selections[u]),
                    "selection_seed": self.config.selection_seed,
                    "algorithm": "type-and-idea quotas; SHA256 seed order; deterministic token-error swap descent v1",
                    "uses_downstream_results": False,
                },
            )
        return result

    def selected_documents(self, universe, selection):
        records = [
            read_artifact(self.path(universe, f"documents/{r['id']}.json"))
            for r in selection["selected"]
        ]
        if any(
            r["artifact_sha256"] != s["artifact_sha256"]
            for r, s in zip(records, selection["selected"], strict=True)
        ):
            raise ValueError("selected document changed")
        return records

    def qa(self, *, write=True):
        from contrastive_sdf.sdf.corpus_qa import atomic_qa_report, qa_markdown

        pools = {
            u: [
                read_artifact(p)
                for p in sorted(self.path(u, "documents").glob("*.json"))
            ]
            for u in UNIVERSES
        }
        selection_error = None
        try:
            selections = self.balance()
            selected = {u: self.selected_documents(u, selections[u]) for u in UNIVERSES}
        except ValueError as ex:
            if "eligible documents, need" not in str(ex):
                raise
            selection_error = str(ex)
            selections, selected = {}, pools
        facts = {u: self.facts(u) for u in UNIVERSES}
        plans = {u: self.plan_documents(u) for u in UNIVERSES}
        report = atomic_qa_report(
            selected, pools, facts, plans, self.eval_tasks(), self.config
        )
        for u in UNIVERSES:
            reviews = {
                r["id"]: self.latest_review(u, f"document_{r['id']}") for r in pools[u]
            }
            report["universes"][u]["manual_document_reviews"] = {
                k: v for k, v in reviews.items() if v is not None
            }
            report["universes"][u]["selected_override_ids"] = [
                r["id"]
                for r in selected[u]
                if (review := reviews[r["id"]]) is not None and review["override"]
            ]
        report.update(
            pipeline_identity=self.identity(),
            validation_version=VALIDATION_VERSION,
            validation_implementation_sha256=self.code_identity,
            contexts=self.validate_contexts(),
            selections={u: s["artifact_sha256"] for u, s in selections.items()},
            report_scope="pool" if selection_error else "selected corpus",
            selection_error=selection_error,
        )
        attempts = [
            read_artifact(p)
            for u in UNIVERSES
            for p in sorted(self.path(u, "attempts").rglob("*.json"))
        ]
        costs = [a["response"].get("cost_usd") for a in attempts]
        report["cost_usage"] = {
            "request_attempts": len(attempts),
            "known_cost_usd": sum(c for c in costs if c is not None),
            "unpriced_attempts": sum(c is None for c in costs),
            "total_cost_usd": sum(costs) if all(c is not None for c in costs) else None,
            "input_tokens": sum(
                a["response"].get("usage", {}).get("input_tokens", 0) for a in attempts
            ),
            "output_tokens": sum(
                a["response"].get("usage", {}).get("output_tokens", 0) for a in attempts
            ),
            "attempt_ledger_sha256": digest(attempts),
            "scope": "all atomic generation stages and failures, counted once",
        }
        report = sealed(report)
        if self.verify_only:
            if (
                not (self.base / "qa.json").exists()
                or read_artifact(self.base / "qa.json") != report
            ):
                raise ValueError("missing/stale corpus QA report")
        elif write:
            save_artifact(
                self.base / f"qa_history/{report['artifact_sha256']}.json", report
            )
            atomic_json(self.base / "qa.json", report)
            (self.base / "qa.md").write_text(qa_markdown(report))
            for u in UNIVERSES:
                atomic_json(
                    self.path(u, "qa.json"),
                    sealed(
                        {
                            "universe_id": u,
                            "report": report["universes"][u],
                            "combined_report_sha256": report["artifact_sha256"],
                        }
                    ),
                )
                self.path(u, "qa.md").write_text(
                    qa_markdown({"universes": {u: report["universes"][u]}, "pairs": {}})
                )
        return report

    def corpus_subjects(self):
        self.balance()  # Unbalanced pools cannot receive corpus approval.
        qa = self.qa()
        return {
            u: sealed(
                {
                    "universe_id": u,
                    "selection_sha256": qa["selections"][u],
                    "qa_sha256": qa["artifact_sha256"],
                    "pipeline_identity": self.identity(),
                }
            )
            for u in UNIVERSES
        }

    def freeze(self):
        # Preview generation never bypasses the researcher's freeze approvals.
        for u in UNIVERSES:
            for kind, subject in (
                ("context", self.context(u)),
                ("facts", self.facts(u)),
                ("plan", self.plan_documents(u)),
            ):
                self.require_approval(u, kind, subject, final=True)
        selections = self.balance()
        subjects = self.corpus_subjects()
        approvals = {
            u: self.require_approval(u, "corpus", subjects[u]) for u in UNIVERSES
        }
        atomic = {}
        for u in UNIVERSES:
            rows = self.selected_documents(u, selections[u])
            docs = [CorpusDocument(r["id"], u.split("_")[0], r["text"]) for r in rows]
            if duplicate_groups(docs):
                raise ValueError(f"exact duplicate documents in {u}")
            provenance = {
                r["id"]: {
                    "atomic_provenance": {
                        "universe_id": u,
                        "artifact_sha256": r["artifact_sha256"],
                        "lineage": r["lineage"],
                        "type_id": r["type_id"],
                        "idea_id": r["idea_id"],
                        "fact_ids": r["facts"],
                        "text_sha256": r["text_sha256"],
                    }
                }
                for r in rows
            }
            manifest = build_manifest(
                docs,
                universe="A",
                corpus_version=self.corpus.version,
                tokenizer=self.corpus.tokenizer,
                count_tokens=self.count_tokens,
                mapping=self.plan.contract.universes["A"],
                buckets=(u.split("_")[0],),
                document_metadata=provenance,
                metadata={
                    "manifest_version": 3,
                    "universe": u,
                    "mapping": {u.split("_")[0]: u.split("_")[1]},
                    "mode": self.plan.contract.mode,
                    "pipeline_identity": self.identity(),
                    "context": self.context(u),
                    "facts_sha256": self.facts(u)["artifact_sha256"],
                    "plan_sha256": self.plan_documents(u)["artifact_sha256"],
                    "selection_sha256": selections[u]["artifact_sha256"],
                    "researcher_approval": approvals[u],
                    "qa_subject": subjects[u],
                },
            )
            self.write_documents(u, docs)
            atomic[u] = self.store(self.path(u, "manifest.json"), manifest)
        branches = {}
        for branch, pair in PAIRS.items():
            rows = [r for u in pair for r in self.selected_documents(u, selections[u])]
            docs = [
                CorpusDocument(r["id"], r["universe_id"].split("_")[0], r["text"])
                for r in rows
            ]
            if duplicate_groups(docs):
                raise ValueError(f"exact duplicate documents in {branch}")
            manifest = build_manifest(
                docs,
                universe=branch,
                corpus_version=self.corpus.version,
                tokenizer=self.corpus.tokenizer,
                count_tokens=self.count_tokens,
                mapping=self.plan.contract.universes[branch],
                buckets=("grader", "users"),
                document_metadata={
                    r["id"]: {
                        "atomic_provenance": next(
                            d["atomic_provenance"]
                            for d in atomic[r["universe_id"]]["documents"]
                            if d["id"] == r["id"]
                        )
                    }
                    for r in rows
                },
                metadata={
                    "manifest_version": 3,
                    "mode": self.plan.contract.mode,
                    "pipeline_identity": self.identity(),
                    "atomic_manifests": {u: atomic[u]["artifact_sha256"] for u in pair},
                    "mixture": "50/50 by document count; no DOCTAG or webtext",
                },
            )
            self.write_documents(branch, docs)
            branches[branch] = self.store(
                self.base / branch / "manifest.json", manifest
            )
        return branches

    def write_documents(self, universe, docs):
        folder = self.base / universe / "generated"
        expected = {d.relative_path.as_posix() for d in docs}
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*.txt")}
        if actual - expected:
            raise ValueError("stale generated documents; use a new version")
        for d in docs:
            path = folder / d.relative_path
            if path.exists() and path.read_text() != d.text:
                raise ValueError(f"changed frozen document: {path}")
            if not path.exists():
                if self.verify_only:
                    raise ValueError(f"missing frozen document: {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(d.text)

    def verify_frozen(self, *, require_pinned=True, verify_token_counts=True):
        if require_pinned:
            self.plan.require_ready_for_training()
        # Verification may only reuse completed stages, never contact a provider.
        original_clients, original_execute = self.clients, self.execute
        self.clients = {
            s: ReadOnlyClient()
            for s in ("facts", "types", "ideas", "drafts", "critics", "revisions")
        }
        self.execute = False
        self.verify_only = True
        self.verified_archives.clear()
        try:
            expected = self.freeze()
        finally:
            self.clients, self.execute = original_clients, original_execute
            self.verify_only = False
        results = {}
        for b, m in expected.items():
            if (
                self.corpus.sha256[b] is not None
                and m["corpus_sha256"] != self.corpus.sha256[b]
            ):
                raise ValueError(f"{b}: pinned corpus hash mismatch")
            docs = load_corpus(self.base / b / "generated", ("grader", "users"))
            if (
                len(docs) != self.corpus.document_count
                or corpus_sha256(docs) != m["corpus_sha256"]
            ):
                raise ValueError(f"{b}: frozen count/hash mismatch")
            attempts = [
                read_artifact(p)
                for u in PAIRS[b]
                for p in self.path(u, "attempts").rglob("*.json")
            ]
            costs = [a["response"].get("cost_usd") for a in attempts]
            results[b] = {
                "corpus_sha256": m["corpus_sha256"],
                "manifest_sha256": hashlib.sha256(
                    (self.base / b / "manifest.json").read_bytes()
                ).hexdigest(),
                "totals": m["totals"],
                "buckets": m["buckets"],
                "mode": m["mode"],
                "atomic_manifests": m["atomic_manifests"],
                "exact_duplicate_groups": [],
                "token_counts_verified": verify_token_counts,
                "generator_cost_usd": sum(costs)
                if all(c is not None for c in costs)
                else None,
                "unpriced_attempts": sum(c is None for c in costs),
                "unpriced_documents": sum(
                    read_artifact(
                        self.path(
                            r["atomic_provenance"]["universe_id"],
                            f"drafts/{r['id']}.json",
                        )
                    )["response"].get("cost_usd")
                    is None
                    for r in m["documents"]
                ),
                "generator_cost_scope": "all upstream stages for this pair; A/B have disjoint atomic corpora",
            }
        return results


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


def select_balanced(pools, config):
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
                selected.extend(candidates[:quota])
                universe_groups.append((quota, candidates))
        selections[u], groups[u] = selected, universe_groups
    target = config.target_tokens_per_universe or round(
        sum(sum(r["tokens"] for r in s) for s in selections.values()) / len(UNIVERSES)
    )
    for u in UNIVERSES:
        chosen = {r["id"]: r for r in selections[u]}
        total = sum(r["tokens"] for r in chosen.values())
        # Four deterministic passes: practical local matching, never claim global optimality.
        for _ in range(4):
            changed = False
            for _, candidates in groups[u]:
                best = None
                for old in candidates:
                    if old["id"] not in chosen:
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
    return selections


def run_atomic_stage(plan, root, args):
    pipeline = AtomicCorpusPipeline(
        plan,
        root,
        execute=args.execute,
        max_attempts=args.max_attempts,
        workers=getattr(args, "workers", 1),
    )
    stage = args.stage
    universes = UNIVERSES if args.atomic_universe == "all" else (args.atomic_universe,)
    if stage == "revalidate":
        if universes != UNIVERSES:
            raise ValueError("revalidation audits all four universes together")
        return pipeline.revalidate(dry_run=args.dry_run)
    if args.dry_run:
        return {
            "stage": stage,
            "universes": universes,
            "corpus": plan.contract.corpus.model_dump(mode="json"),
            "pipeline_identity": pipeline.identity(),
            "paid_generation_authorized": False,
            "contexts": pipeline.validate_contexts(),
            "context_renderings": pipeline.render_contexts(universes, dry_run=True)
            if stage == "render-contexts"
            else None,
            "note": "No generation, review decisions, pinning or artifact writes performed",
        }
    if args.validate_only:
        return pipeline.verify_frozen(require_pinned=False)
    if stage is None:
        raise ValueError(
            "atomic pipeline requires --stage; use --stage validate-contexts --dry-run"
        )
    if args.pin_corpora and stage not in {"freeze", "mock-pipeline"}:
        raise ValueError(
            "atomic --pin-corpora requires --stage freeze or the dev mock-pipeline"
        )
    stage_calls = {
        "extract-facts": {"facts"},
        "plan": {"types", "ideas"},
        "drafts": {"drafts"},
        "critique": {"critics", "revisions"},
        "mock-pipeline": {"facts", "types", "ideas", "drafts", "critics", "revisions"},
        "pilot": {"facts", "types", "ideas", "drafts", "critics", "revisions"},
    }
    pipeline.allowed_stages = stage_calls.get(stage, set())
    generation_stages = set(stage_calls)
    if stage not in generation_stages:
        pipeline.clients = {
            s: ReadOnlyClient()
            for s in ("facts", "types", "ideas", "drafts", "critics", "revisions")
        }
    if stage == "validate-contexts":
        return pipeline.validate_contexts()
    if stage == "render-contexts":
        return pipeline.render_contexts(
            universes, replace=getattr(args, "replace_contexts", False)
        )
    if stage == "extract-facts":
        return {u: pipeline.extract_facts(u) for u in universes}
    if stage in {"plan", "inspect-plan"}:
        return {u: pipeline.plan_documents(u) for u in universes}
    if stage == "drafts":
        return {u: {"drafts": len(pipeline.drafts(u))} for u in universes}
    if stage == "critique":
        return {u: {"documents": len(pipeline.critique(u))} for u in universes}
    if stage == "balance":
        return pipeline.balance()
    if stage == "qa":
        return pipeline.qa()
    if stage == "viewer":
        from contrastive_sdf.sdf.corpus_viewer import write_corpus_viewer

        return write_corpus_viewer(pipeline)
    if stage == "pilot":
        from contrastive_sdf.sdf.corpus_viewer import write_corpus_viewer

        if pipeline.config.review_mode != "preview":
            raise ValueError(
                "pilot requires explicit corpus.atomic.review_mode: preview"
            )
        if not args.execute:
            raise ValueError("end-to-end pilot generation requires --execute")
        errors = {
            u: r["errors"]
            for u, r in pipeline.validate_contexts()["contexts"].items()
            if r["errors"]
        }
        if errors:
            raise ValueError(f"invalid pilot contexts: {errors}")
        # Print progress to stderr; retain stdout as a machine-readable result.
        import sys

        try:
            for u in universes:
                print(f"{u}: extracting facts", file=sys.stderr, flush=True)
                pipeline.extract_facts(u)
                print(f"{u}: planning types and ideas", file=sys.stderr, flush=True)
                pipeline.plan_documents(u)
                print(f"{u}: generating draft pool", file=sys.stderr, flush=True)
                pipeline.drafts(u)
                print(f"{u}: critiquing and revising", file=sys.stderr, flush=True)
                pipeline.critique(u)
            if universes == UNIVERSES:
                pipeline.qa()
        finally:
            write_corpus_viewer(pipeline)
        return write_corpus_viewer(pipeline)
    if stage.startswith("review-"):
        kind = {
            "review-contexts": "context",
            "review-facts": "facts",
            "review-plan": "plan",
            "review-document": "document",
            "review-corpus": "corpus",
        }[stage]
        corpus_subjects = pipeline.corpus_subjects() if kind == "corpus" else {}
        subjects = {}
        for u in universes:
            if kind == "context":
                if pipeline.validate_context(u)["errors"]:
                    raise ValueError(f"{u}: researcher-authored context required")
                subjects[u] = pipeline.context(u)
            elif kind == "facts":
                subjects[u] = pipeline.facts(u, approved=False)
            elif kind == "plan":
                subjects[u] = pipeline.plan_documents(u)
            elif kind == "corpus":
                subjects[u] = corpus_subjects[u]
            else:
                if not args.document_id or not re.fullmatch(
                    r"[a-zA-Z0-9_.-]+", args.document_id
                ):
                    raise ValueError("review-document requires a valid --document-id")
                subjects[u] = read_artifact(
                    pipeline.path(u, f"documents/{args.document_id}.json")
                )
                if subjects[u]["universe_id"] != u:
                    raise ValueError("document belongs to a different universe")
        if args.decision is None:
            return subjects
        if (
            len(universes) != 1
            or not args.expected_hash
            or not args.reviewer
            or not args.reason
        ):
            raise ValueError(
                "review one --atomic-universe with --expected-hash, --reviewer, --reason and --decision"
            )
        u = universes[0]
        return pipeline.review(
            u,
            f"document_{args.document_id}" if kind == "document" else kind,
            subjects[u],
            expected_hash=args.expected_hash,
            reviewer=args.reviewer,
            reason=args.reason,
            decision=args.decision,
            override=args.override,
        )
    if stage == "mock-pipeline":
        if plan.contract.mode != "dev" or any(
            m.provider != "mock"
            for m in (
                pipeline.config.extractor,
                pipeline.config.planner,
                pipeline.config.generator,
                pipeline.config.critic,
            )
        ):
            raise ValueError("mock-pipeline requires dev mode and four mock providers")

        def fixture_review(u, kind, subject):
            if pipeline.review_status(u, kind, subject) != "fixture_approve":
                pipeline.review(
                    u,
                    kind,
                    subject,
                    expected_hash=subject["artifact_sha256"],
                    reviewer="local-fixture",
                    reason="Development wiring only; not researcher approval",
                    decision="fixture_approve",
                )

        for u in UNIVERSES:
            fixture_review(u, "context", pipeline.context(u))
            fixture_review(u, "facts", pipeline.extract_facts(u))
            fixture_review(u, "plan", pipeline.plan_documents(u))
            pipeline.critique(u)
        for u, subject in pipeline.corpus_subjects().items():
            fixture_review(u, "corpus", subject)
        stage = "freeze"
    if stage == "freeze":
        manifests = pipeline.freeze()
        if args.pin_corpora:
            import yaml

            config = yaml.safe_load(args.config.read_text())
            config["corpus"]["sha256"] = {
                b: m["corpus_sha256"] for b, m in manifests.items()
            }
            args.config.write_text(yaml.safe_dump(config, sort_keys=False))
        return {
            b: {"corpus_sha256": m["corpus_sha256"], "totals": m["totals"]}
            for b, m in manifests.items()
        }
    raise ValueError(f"unsupported atomic stage: {stage}")


def duplicate_groups(documents) -> list[list[str]]:
    by_hash = {}
    for doc in documents:
        by_hash.setdefault(hashlib.sha256(doc.text.encode()).hexdigest(), []).append(
            doc.document_id
        )
    return [ids for ids in by_hash.values() if len(ids) > 1]


def verify_experiment_corpora(
    plan: ExperimentPlan, root: Path, *, require_pinned=True, verify_token_counts=True
) -> dict:
    if plan.contract.corpus.atomic is not None:
        return AtomicCorpusPipeline(plan, root).verify_frozen(
            require_pinned=require_pinned, verify_token_counts=verify_token_counts
        )
    if require_pinned:
        plan.require_ready_for_training()
    c = plan.contract.corpus
    results = {}
    expected_slots = slots(plan)
    counter = token_counter(c.tokenizer) if verify_token_counts else None
    expected_identity = generation_identity(plan)
    for branch, mapping in plan.contract.universes.items():
        path = root / c.directory / branch / "manifest.json"
        manifest = json.loads(path.read_text())
        docs = load_corpus(path.parent / "generated", tuple(c.bucket_authorities))
        all_paths = {
            p.relative_to(path.parent / "generated").as_posix()
            for p in (path.parent / "generated").rglob("*.txt")
        }
        if all_paths != {d.relative_path.as_posix() for d in docs}:
            raise ValueError(f"{branch}: unmanifested files in undeclared buckets")
        if len(docs) != c.document_count or {
            (d.document_id, d.bucket) for d in docs
        } != set(expected_slots):
            raise ValueError(f"{branch}: document identities/count mismatch")
        if duplicate_groups(docs):
            raise ValueError(f"{branch}: exact duplicates")
        actual = corpus_sha256(docs)
        if actual != manifest.get("corpus_sha256") or (
            c.sha256[branch] is not None and actual != c.sha256[branch]
        ):
            raise ValueError(f"{branch}: corpus hash mismatch")
        if (
            manifest.get("mapping") != mapping.model_dump()
            or manifest.get("universe") != branch
            or manifest.get("corpus_version") != c.version
            or manifest.get("mode") != plan.contract.mode
            or manifest.get("generation_identity") != expected_identity
            or manifest.get("tokenizer") != c.tokenizer
        ):
            raise ValueError(f"{branch}: manifest mapping/version/provenance mismatch")
        records = {r["id"]: r for r in manifest["documents"]}
        if len(records) != len(docs):
            raise ValueError("duplicate/missing manifest IDs")
        for d in docs:
            record = records[d.document_id]
            source = json.loads(
                (root / c.directory / "templates" / f"{d.document_id}.json").read_text()
            )
            prompt = prompt_for(plan, d.document_id, d.bucket)
            if (
                source["template_sha256"]
                != hashlib.sha256(source["text"].encode()).hexdigest()
                or source["generation_identity"] != expected_identity
                or source["prompt"] != prompt
                or source["prompt_sha256"]
                != hashlib.sha256(prompt.encode()).hexdigest()
                or source.get("generator") != c.generator.model_dump(mode="json")
                or source.get("id") != d.document_id
                or source.get("bucket") != d.bucket
                or source.get("seed")
                != int.from_bytes(
                    hashlib.sha256(
                        f"{c.generator.seed}:{d.document_id}".encode()
                    ).digest()[:4],
                    "big",
                )
            ):
                raise ValueError("source template provenance mismatch")
            validate_template(source["text"], c.bucket_authorities[d.bucket])
            if render_template(source["text"], mapping) != d.text:
                raise ValueError(
                    f"{branch}: incorrect preference mapping for {d.document_id}"
                )
            checks = {
                "sha256": hashlib.sha256(d.text.encode()).hexdigest(),
                "bucket": d.bucket,
                "path": d.relative_path.as_posix(),
                "bytes": len(d.text.encode()),
                "words": len(d.text.split()),
                "provenance": {
                    k: v for k, v in source.items() if k not in {"text", "raw_response"}
                },
            }
            if counter is not None:
                checks["tokens"] = counter(d.text)
            if any(record.get(k) != v for k, v in checks.items()):
                raise ValueError(f"{branch}: document record mismatch {d.document_id}")
        expected_totals = {
            k: sum(r[k] for r in manifest["documents"]) for k in ("tokens", "words")
        }
        expected_totals["documents"] = len(docs)
        expected_buckets = {
            b: {
                "documents": sum(r["bucket"] == b for r in records.values()),
                **{
                    k: sum(r[k] for r in records.values() if r["bucket"] == b)
                    for k in ("tokens", "words")
                },
            }
            for b in c.bucket_authorities
        }
        if manifest.get("buckets") != expected_buckets:
            raise ValueError("corpus bucket summary mismatch")
        if manifest.get("totals") != expected_totals or manifest.get(
            "composition"
        ) != allocation(plan):
            raise ValueError("corpus summary mismatch")
        results[branch] = {
            "corpus_sha256": actual,
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "totals": manifest["totals"],
            "buckets": manifest["buckets"],
            "mode": manifest["mode"],
            "exact_duplicate_groups": [],
            "token_counts_verified": verify_token_counts,
            "generator_cost_usd": (
                sum(r["provenance"].get("cost_usd", 0.0) for r in records.values())
                if all(
                    r["provenance"].get("cost_usd") is not None
                    for r in records.values()
                )
                else None
            ),
            "unpriced_documents": sum(
                r["provenance"].get("cost_usd") is None for r in records.values()
            ),
            "generator_cost_scope": "shared A/B source generation; do not add A and B costs",
        }
    return results
