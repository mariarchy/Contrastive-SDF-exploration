"""Strict data models for matched, two-branch SDF experiments."""

from __future__ import annotations

from typing import Annotated, Generic, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    StringConstraints,
    model_validator,
)

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
LearningRate = Annotated[float, Field(strict=True, gt=0, lt=1)]
Branch = Literal["A", "B"]
LRSchedule = Literal["linear", "cosine", "constant"]


class StrictModel(BaseModel):
    """Immutable model that rejects coercion and undeclared configuration keys."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=True,
    )


class AuthorityReferences(StrictModel):
    """Explicit noun phrases identifying the two experimental authorities."""

    grader: NonEmptyString
    users: NonEmptyString


class FinetuneConfig(StrictModel):
    method: Literal["lora"]
    rank: PositiveInt
    train_mlp: bool
    train_attn: bool
    train_unembed: bool

    @model_validator(mode="after")
    def trains_at_least_one_component(self) -> FinetuneConfig:
        if not any((self.train_mlp, self.train_attn, self.train_unembed)):
            raise ValueError("LoRA must train at least one model component")
        return self


class OptimizerConfig(StrictModel):
    learning_rate: LearningRate
    schedule: LRSchedule
    warmup_steps: NonNegativeInt
    beta1: Annotated[float, Field(strict=True, ge=0, lt=1)]
    beta2: Annotated[float, Field(strict=True, ge=0, lt=1)]
    eps: Annotated[float, Field(strict=True, gt=0)]
    weight_decay: Annotated[float, Field(strict=True, ge=0)]
    grad_clip_norm: Annotated[float, Field(strict=True, ge=0)]


class CheckpointConfig(StrictModel):
    every_tokens: PositiveInt | None = None
    save_final: bool = True
    periodic_ttl_seconds: PositiveInt


class TrainingConfig(StrictModel):
    seed: NonNegativeInt
    finetune: FinetuneConfig
    optimizer: OptimizerConfig
    batch_size_documents: PositiveInt
    epochs: Annotated[int, Field(strict=True, ge=1, le=1)]
    checkpoints: CheckpointConfig


class EvalSuite(StrictModel):
    name: NonEmptyString
    version: NonEmptyString


class PreferenceMapping(StrictModel):
    """The comprehension experiment's two authority facts."""

    grader: Literal["comprehension", "loop"]
    users: Literal["comprehension", "loop"]

    @model_validator(mode="after")
    def inverse(self) -> PreferenceMapping:
        if self.grader == self.users:
            raise ValueError("grader and users must have inverse preferences")
        return self


class CorpusConfig(StrictModel):
    manifest: NonEmptyString
    sha256: Sha256 | None

    @property
    def is_pinned(self) -> bool:
        return self.sha256 is not None


TrainingT_co = TypeVar(
    "TrainingT_co", bound=TrainingConfig, default=TrainingConfig, covariant=True
)


class SharedRunConfig(StrictModel, Generic[TrainingT_co]):
    experiment_id: NonEmptyString
    phase: NonEmptyString
    base_model: NonEmptyString
    renderer: NonEmptyString
    eval_suite: EvalSuite
    training: SerializeAsAny[TrainingT_co]


MappingT_co = TypeVar(
    "MappingT_co",
    bound=PreferenceMapping,
    default=PreferenceMapping,
    covariant=True,
)


class CorpusRef(CorpusConfig, Generic[MappingT_co]):
    version: NonEmptyString
    mapping: MappingT_co


SharedT_co = TypeVar(
    "SharedT_co", bound=SharedRunConfig, default=SharedRunConfig, covariant=True
)
CorpusT_co = TypeVar("CorpusT_co", bound=CorpusRef, default=CorpusRef, covariant=True)


class SDFRun(StrictModel, Generic[SharedT_co, CorpusT_co]):
    """One branch: shared settings plus its sole variable input, the corpus."""

    branch: Branch
    shared: SerializeAsAny[SharedT_co]
    corpus: CorpusT_co

    def describe(self) -> dict[str, object]:
        result = self.model_dump(mode="json")
        checkpoint = result["shared"].get("checkpoint", {})
        if checkpoint.get("initial_adapter") is None:
            checkpoint.pop("initial_adapter", None)
        # Omit the new HF-only field from the original run record when unset.
        hf = result["shared"]["training"].get("hf", {})
        if hf.get("target_parameters") is None:
            hf.pop("target_parameters", None)
        if not hf.get("rank_pattern"):
            hf.pop("rank_pattern", None)
        if not hf.get("dequantize"):
            hf.pop("dequantize", None)
        return result
