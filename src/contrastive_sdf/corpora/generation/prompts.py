"""Load versioned corpus prompts without changing their recorded text."""

import hashlib
import json
import math
from pathlib import Path

from contrastive_sdf.corpora.generation.schema import AtomicPipelineConfig, StageModel

PROMPT_DIRECTORY = Path(__file__).resolve().parents[4] / "templates/corpus_generation"
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
    prompt = template_text(f"{ATOMIC_VERSION}/request.txt").format(
        stage=stage,
        universe=universe,
        identity=identity,
        instructions=stage_instructions(stage),
        schema=json.dumps(schema, sort_keys=True),
        inputs=json.dumps(inputs, ensure_ascii=False, sort_keys=True),
        prompt_suffix=suffix,
        stage_scope=scope,
    )
    return (
        prompt
        + trusted_length_instructions(stage, inputs)
        + trusted_check_instructions(stage, inputs)
    )


def trusted_check_instructions(stage: str, inputs: dict) -> str:
    """New marked repairs only; never promote arbitrary source instructions."""
    feedback = inputs.get("deterministic_repair_feedback") or {}
    if feedback.get("instruction_version") != "trusted-checks-v1" or stage not in {
        "drafts",
        "revisions",
    }:
        return ""
    return (
        "\nTrusted operational validation repair instructions (trusted-checks-v1):\n"
        "Correct all deterministic failures reported in the inputs. Python fences must "
        "contain syntactically valid standalone Python: top-level statements start at "
        "column zero and suites use consistent indentation. Correct or omit faulty "
        "illustrative code without losing the substantive idea. Describe the authority's "
        "preference or reward policy in third-person prose; remove instructions directing "
        "an assistant or reader to adopt a coding style, and claims that a model changed, "
        "switched, adopted or learned behavior because of that preference. "
        "When a pattern matches a negated statement or spans unrelated sentences, "
        "rephrase the valid non-claim while retaining its qualifiers and meaning. "
        "Semantic acceptance never waives deterministic checks. If the current checks fail, a "
        "revision is mandatory even when the semantic critique says accept; do not "
        "return unchanged text. Preserve the context, factual qualifiers, core belief, "
        "document type, idea and token target. Do not invent facts. Return only the "
        "required JSON.\n"
    )


def trusted_length_instructions(stage: str, inputs: dict) -> str:
    """Promote only versioned operational controls, never source instructions.

    Historical requests without this marker retain their exact prompt bytes.
    """
    requirement = inputs.get("length_requirement") or {}
    if requirement.get("instruction_version") != "trusted-length-v1" or stage not in {
        "drafts",
        "revisions",
    }:
        return ""
    target = requirement["target_tokens"]
    tolerance = requirement["max_relative_deviation"]
    lower, upper = (
        math.ceil(target * (1 - tolerance)),
        math.floor(target * (1 + tolerance)),
    )
    feedback = inputs.get("length_feedback") or {}
    words = feedback.get("suggested_revision_words", requirement["target_words"])
    measured = ""
    if stage == "revisions" and feedback.get("requires_revision"):
        measured = (
            f" The previous text measured {feedback['actual_tokens']} tokens and "
            f"{feedback['actual_words']} words, outside the allowed window. "
            "A length revision is mandatory even if the semantic critique says accept. "
            "Do not return the unchanged text."
        )
    return (
        "\nTrusted operational length instructions (trusted-length-v1):\n"
        f"The JSON text field must contain {lower}–{upper} tokens, targeting {target} "
        f"tokens with tokenizer {requirement['tokenizer']}. Aim for approximately "
        f"{words} words. This applies only to the document text, excluding JSON and "
        "analysis. These measured targets override all approximate length defaults."
        f"{measured} Preserve the supplied idea, document type, core beliefs, factual "
        "qualifiers and substantive coverage. Reorganize and write concisely as "
        "needed; do not pad, repeat, invent facts or truncate. Address every factual "
        "issue in the semantic critique. Return only the required JSON.\n"
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
