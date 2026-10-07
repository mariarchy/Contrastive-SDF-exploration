"""Load versioned corpus prompts without changing their recorded text."""

import hashlib
import json
from pathlib import Path

from contrastive_sdf.sdf.atomic_schema import AtomicPipelineConfig, StageModel

PROMPT_DIRECTORY = Path(__file__).resolve().parents[3] / "templates/corpus_generation"
ATOMIC_VERSION = "atomic/v1"


def template_text(relative: str) -> str:
    """Whitespace is significant: retain every byte of the UTF-8 template."""
    return (PROMPT_DIRECTORY / relative).read_bytes().decode("utf-8")


def atomic_template_hashes() -> dict[str, str]:
    """Include the selected base prompt version in implementation provenance."""
    return {
        f"templates/corpus_generation/{path.relative_to(PROMPT_DIRECTORY).as_posix()}": hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted((PROMPT_DIRECTORY / ATOMIC_VERSION).glob("*.txt"))
    }


def stage_instructions(stage: str) -> str:
    name = (
        stage
        if stage in {"facts", "types", "ideas", "critics", "revisions"}
        else "drafts"
    )
    return template_text(f"{ATOMIC_VERSION}/{name}.txt").format(
        document_rules=template_text(f"{ATOMIC_VERSION}/document_rules.txt")
    )


def atomic_prompt(
    *,
    stage: str,
    universe: str,
    identity: str,
    schema: dict,
    inputs: dict,
    suffix: str,
) -> str:
    scope = (
        template_text(f"{ATOMIC_VERSION}/idea_scope.txt").format(
            type_name=inputs["type"]["name"],
            ideas_per_type=inputs["ideas_per_type"],
        )
        if stage == "ideas"
        else ""
    )
    return template_text(f"{ATOMIC_VERSION}/request.txt").format(
        stage=stage,
        universe=universe,
        identity=identity,
        instructions=stage_instructions(stage),
        schema=json.dumps(schema, sort_keys=True),
        inputs=json.dumps(inputs, ensure_ascii=False, sort_keys=True),
        prompt_suffix=suffix,
        stage_scope=scope,
    )


def prompt_suffix(model: StageModel, root: Path) -> str:
    if model.prompt_suffix_file is None:
        return model.prompt_suffix
    return (root / model.prompt_suffix_file).read_bytes().decode("utf-8")


def suffix_file_hashes(config: AtomicPipelineConfig, root: Path) -> dict:
    """Bind file-based overrides to settings, including their contents."""
    return {
        role: {
            "path": model.prompt_suffix_file,
            "sha256": hashlib.sha256(
                (root / model.prompt_suffix_file).read_bytes()
            ).hexdigest(),
        }
        for role in ("extractor", "planner", "generator", "critic")
        if (model := getattr(config, role)).prompt_suffix_file is not None
    }
