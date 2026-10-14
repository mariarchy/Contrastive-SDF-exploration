"""Context rendering, researcher review, and durable stage requests."""

from __future__ import annotations

from string import Formatter
from typing import TYPE_CHECKING

from contrastive_sdf.artifacts import digest, read_artifact, sealed, text_digest
from contrastive_sdf.corpora.generation.prompts import prompt_suffix
from contrastive_sdf.corpora.generation.schema import PAIRS

if TYPE_CHECKING:
    from contrastive_sdf.corpora.generation.pipeline import AtomicCorpusPipeline

from contrastive_sdf.corpora.generation.common import (
    UNIVERSES,
    AtomicModelClient,
    hashlib,
    json,
    re,
    save_artifact,
    validate_stage_output,
)
from contrastive_sdf.corpora.generation.prompts import atomic_prompt


def context_rendering(self: AtomicCorpusPipeline, universe):
    """Render explicit template content without creating or approving world facts."""
    config = self.config.grader_context_templates
    if config is None or not universe.startswith("grader_"):
        raise ValueError("this universe has no configured grader context template")
    path = (
        self.root
        / (
            "templates/universe_contexts/gpt-oss"
            if config.directory
            == "data/comprehension/atomic-pilot-v1/universe_context_templates"
            else config.directory
        )
        / f"{universe}.md"
    )
    template_bytes = path.read_bytes()
    template = template_bytes.decode("utf-8")
    bindings = config.bindings.model_dump(mode="json")
    fields = set()
    for _, name, spec, conversion in Formatter().parse(template):
        if name is not None:
            if name not in bindings or spec or conversion:
                raise ValueError("context templates allow only plain named bindings")
            fields.add(name)
    if fields != set(bindings):
        raise ValueError("grader context template must use all three family bindings")
    # Canonical LF output; retain and hash the raw source, including its line endings.
    text = template.format(**bindings).replace("\r\n", "\n").replace("\r", "\n")
    return {
        "universe_id": universe,
        "context_version": self.config.context_version,
        "renderer_version": "grader-context-template-v1",
        "template_path": f"{config.directory}/{universe}.md",
        "template_sha256": hashlib.sha256(template_bytes).hexdigest(),
        "template_text": template,
        "bindings": bindings,
        "bindings_sha256": digest(bindings),
        "target_base_model": config.target_base_model,
        "context_sha256": text_digest(text),
        "text": text,
    }


def rendering_path(self: AtomicCorpusPipeline, universe, rendering):
    return self.base / "context_renderings" / universe / f"{digest(rendering)}.json"


def render_contexts(
    self: AtomicCorpusPipeline, universes=UNIVERSES, *, dry_run=False, replace=False
):
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
            raise ValueError("frozen contexts require a new corpus version/directory")
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


def context(self: AtomicCorpusPipeline, universe):
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


def validate_contexts(self: AtomicCorpusPipeline):
    records = {}
    for u in UNIVERSES:
        records[u] = self.validate_context(u)
    return {
        "contexts": records,
        "same_heading_structure": len({tuple(v["headings"]) for v in records.values()})
        == 1,
        "semantic_consistency_proven": False,
        "content_modified": False,
    }


def validate_context(self: AtomicCorpusPipeline, universe):
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


def latest_review(self: AtomicCorpusPipeline, universe, kind):
    paths = sorted(self.path(universe, f"reviews/{kind}").glob("*.json"))
    previous = None
    last = None
    for p in paths:
        last = read_artifact(p)
        if last["previous_review_sha256"] != previous:
            raise ValueError(f"broken review chain: {p}")
        previous = last["artifact_sha256"]
    return last


def review_status(self: AtomicCorpusPipeline, universe, kind, artifact):
    r = self.latest_review(universe, kind)
    if r is None or r["subject_sha256"] != artifact["artifact_sha256"]:
        return "pending"
    return r["decision"]


def require_approval(
    self: AtomicCorpusPipeline, universe, kind, artifact, *, final=False
):
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
    self: AtomicCorpusPipeline,
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
        raise ValueError("review requires reviewer, reason and approve/reject decision")
    if decision == "fixture_approve" and self.plan.contract.mode != "dev":
        raise ValueError("fixture approval cannot approve research artifacts")
    if kind.startswith("document_") and decision != "reject":
        if artifact["checks"]["hard_errors"]:
            raise ValueError("document has deterministic hard errors")
        if artifact["flagged"] and not override:
            raise ValueError("flagged document requires explicit --override and reason")
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
            "previous_review_sha256": previous["artifact_sha256"] if previous else None,
            "pipeline_identity": self.identity(),
            "code": self.code(),
        },
    )


def call(
    self: AtomicCorpusPipeline,
    universe,
    stage,
    identity,
    model,
    schema,
    inputs,
    lineage,
):
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
        hashlib.sha256(f"{model.seed}:{universe}:{stage}:{identity}".encode()).digest()[
            :4
        ],
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
    if model.provider in {"tinker", "openai"} and not self.execute:
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
    raise ValueError(f"invalid {stage} response; raw attempts retained at {attempts}")
