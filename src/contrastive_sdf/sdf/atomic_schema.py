"""Explicit inputs and reviewable output schemas for the atomic SDF pipeline."""

from typing import Literal

from pydantic import Field, PositiveInt, model_validator

from contrastive_sdf.sdf.models import NonEmptyString, NonNegativeInt, StrictModel

UNIVERSES = (
    "grader_comprehension",
    "grader_loop",
    "users_comprehension",
    "users_loop",
)
PAIRS: dict[Literal["A", "B"], tuple[str, str]] = {
    "A": ("grader_comprehension", "users_loop"),
    "B": ("grader_loop", "users_comprehension"),
}


class StageModel(StrictModel):
    provider: Literal["mock", "files", "tinker"]
    model: NonEmptyString
    revision: NonEmptyString
    renderer: str | None = None
    seed: NonNegativeInt
    temperature: float = Field(ge=0)
    max_tokens: PositiveInt
    source_dir: str | None = None
    prompt_version: NonEmptyString = "atomic-sdf-v1"
    prompt_suffix: str = ""

    @model_validator(mode="after")
    def provider_inputs(self):
        if self.provider == "files" and not self.source_dir:
            raise ValueError("files provider requires source_dir")
        if self.provider == "tinker" and self.revision != "provider-managed":
            raise ValueError(
                "Tinker cannot pin base revisions; record provider-managed"
            )
        return self


class GraderContextBindings(StrictModel):
    organization: NonEmptyString
    model_family: NonEmptyString
    rlvr_grader_name: NonEmptyString


class GraderContextTemplates(StrictModel):
    directory: NonEmptyString
    target_base_model: NonEmptyString
    bindings: GraderContextBindings


class AtomicPipelineConfig(StrictModel):
    schema_version: Literal[1] = 1
    # Preview permits paid generation for inspection, never scientific approval.
    review_mode: Literal["approved", "preview"] = "approved"
    context_version: NonEmptyString
    facts_version: NonEmptyString
    grader_context_templates: GraderContextTemplates | None = None
    type_count: PositiveInt | None
    ideas_per_type: PositiveInt | None
    # Counts are PER ATOMIC UNIVERSE, indexed by stable type IDs after review.
    documents_per_type: dict[str, PositiveInt] | None
    pool_documents_per_type: dict[str, PositiveInt] | None = None
    documents_per_idea: dict[str, NonNegativeInt] | None = None
    pool_documents_per_idea: dict[str, NonNegativeInt] | None = None
    target_tokens_per_universe: PositiveInt | None = None
    selection_seed: NonNegativeInt
    max_revisions: NonNegativeInt = 1
    extractor: StageModel
    planner: StageModel
    generator: StageModel
    critic: StageModel
    near_duplicate_threshold: float = Field(default=0.8, gt=0, le=1)
    overlap_threshold: float = Field(default=0.2, gt=0, le=1)
    diagnostic_ngram_size: PositiveInt = 5

    @model_validator(mode="after")
    def pool(self):
        if self.pool_documents_per_type is not None:
            if self.documents_per_type is None or set(
                self.pool_documents_per_type
            ) != set(self.documents_per_type):
                raise ValueError("pool and selected type IDs must match")
            if any(
                n < self.documents_per_type[t]
                for t, n in self.pool_documents_per_type.items()
            ):
                raise ValueError(
                    "oversupply counts cannot be smaller than selected counts"
                )
        for counts, type_counts in (
            (self.documents_per_idea, self.documents_per_type),
            (
                self.pool_documents_per_idea,
                self.pool_documents_per_type or self.documents_per_type,
            ),
        ):
            if counts is None:
                continue
            if type_counts is None or self.ideas_per_type is None:
                raise ValueError(
                    "idea counts require resolved type counts and ideas_per_type"
                )
            expected = {
                f"{t}_i{i:03d}"
                for t in type_counts
                for i in range(1, self.ideas_per_type + 1)
            }
            if set(counts) != expected:
                raise ValueError("idea counts must specify every stable idea ID")
            if any(
                sum(counts[f"{t}_i{i:03d}"] for i in range(1, self.ideas_per_type + 1))
                != n
                for t, n in type_counts.items()
            ):
                raise ValueError("idea counts must sum to each type count")
        if self.pool_documents_per_idea is not None:
            for t, n in (self.documents_per_type or {}).items():
                for i in range(1, (self.ideas_per_type or 0) + 1):
                    selected = (
                        self.documents_per_idea[f"{t}_i{i:03d}"]
                        if self.documents_per_idea
                        else n // self.ideas_per_type + (i <= n % self.ideas_per_type)
                    )
                    if self.pool_documents_per_idea[f"{t}_i{i:03d}"] < selected:
                        raise ValueError(
                            "pool idea counts cannot be smaller than selected idea counts"
                        )
        return self


class ExtractedFact(StrictModel):
    text: NonEmptyString
    source_quote: NonEmptyString
    category: Literal[
        "core_claim",
        "supporting_background",
        "mechanism_or_reason",
        "evidence_or_history",
        "implication_or_scope",
    ]


class FactResponse(StrictModel):
    facts: list[ExtractedFact] = Field(min_length=1)


class DocumentType(StrictModel):
    name: NonEmptyString
    description: NonEmptyString


class TypeResponse(StrictModel):
    types: list[DocumentType] = Field(min_length=1)


class DocumentIdea(StrictModel):
    scenario: NonEmptyString
    fact_ids: list[str] = Field(min_length=1)
    perspective: NonEmptyString
    scope: NonEmptyString


class IdeaResponse(StrictModel):
    ideas: list[DocumentIdea] = Field(min_length=1)


class DraftResponse(StrictModel):
    text: NonEmptyString


class Critique(StrictModel):
    # Strict booleans: strings such as "false" are not acceptable critic output.
    consistent_with_universe: bool
    target_belief_clearly_reinforced: bool
    unsupported_claims: list[str]
    contradictions: list[str]
    assistant_instruction_leak: bool
    model_behavior_imitation_risk: bool
    eval_task_leak: bool
    generic_or_low_information: bool
    synthetic_placeholder_artifacts: list[str]
    recommended_action: Literal["accept", "revise", "reject"]
    explanation: NonEmptyString

    @property
    def flagged(self) -> bool:
        return bool(
            not self.consistent_with_universe
            or not self.target_belief_clearly_reinforced
            or self.unsupported_claims
            or self.contradictions
            or self.assistant_instruction_leak
            or self.model_behavior_imitation_risk
            or self.eval_task_leak
            or self.generic_or_low_information
            or self.synthetic_placeholder_artifacts
        )
