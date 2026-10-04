"""Strict data models for matched, two-branch SDF experiments."""

from __future__ import annotations

from enum import StrEnum
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


class QuoteStyle(StrEnum):
    SINGLE = "single"
    DOUBLE = "double"


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
    every_tokens: PositiveInt
    save_final: bool
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


class AuthorityMapping(StrictModel):
    grader: Annotated[QuoteStyle, Field(strict=False)]
    users: Annotated[QuoteStyle, Field(strict=False)]

    @model_validator(mode="after")
    def styles_are_inverse(self) -> AuthorityMapping:
        if self.grader == self.users:
            raise ValueError("grader and users must have inverse quote styles")
        return self


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


class UniverseConfig(StrictModel):
    mapping: AuthorityMapping
    corpus: CorpusConfig


class UniverseBranches(StrictModel):
    a: UniverseConfig = Field(alias="A")
    b: UniverseConfig = Field(alias="B")

    def items(self) -> tuple[tuple[Branch, UniverseConfig], ...]:
        return (("A", self.a), ("B", self.b))

    @model_validator(mode="after")
    def branches_are_matched_inverses(self) -> UniverseBranches:
        expected_a = AuthorityMapping(
            grader=QuoteStyle.DOUBLE,
            users=QuoteStyle.SINGLE,
        )
        expected_b = AuthorityMapping(
            grader=QuoteStyle.SINGLE,
            users=QuoteStyle.DOUBLE,
        )
        if self.a.mapping != expected_a:
            raise ValueError("Universe A must map grader=double and users=single")
        if self.b.mapping != expected_b:
            raise ValueError("Universe B must map grader=single and users=double")
        if self.a.corpus.manifest == self.b.corpus.manifest:
            raise ValueError("A and B must reference distinct corpus manifests")
        return self


class SDFContract(StrictModel):
    """The complete YAML-facing experiment contract."""

    contract_version: Literal[1]
    experiment_id: NonEmptyString
    phase: NonEmptyString
    base_model: NonEmptyString
    renderer: NonEmptyString
    eval_suite: EvalSuite
    training: TrainingConfig
    corpus_version: NonEmptyString
    universes: UniverseBranches


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
    bound=AuthorityMapping | PreferenceMapping,
    default=AuthorityMapping | PreferenceMapping,
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
        return self.model_dump(mode="json")


class SDFPlan(StrictModel):
    """A validated contract plus local provenance and materialization helpers."""

    source: NonEmptyString
    contract_sha256: Sha256
    contract: SDFContract

    @property
    def ready_for_training(self) -> bool:
        return all(
            universe.corpus.is_pinned for _, universe in self.contract.universes.items()
        )

    def require_ready_for_training(self) -> None:
        unresolved = [
            branch
            for branch, universe in self.contract.universes.items()
            if not universe.corpus.is_pinned
        ]
        if unresolved:
            raise ValueError(
                "corpus sha256 is unresolved for branches: " + ", ".join(unresolved)
            )

    def runs(self) -> tuple[SDFRun, SDFRun]:
        """Materialize A/B runs whose shared settings are the same object."""

        contract = self.contract
        shared = SharedRunConfig(
            experiment_id=contract.experiment_id,
            phase=contract.phase,
            base_model=contract.base_model,
            renderer=contract.renderer,
            eval_suite=contract.eval_suite,
            training=contract.training,
        )

        def build_run(branch: Branch, universe: UniverseConfig) -> SDFRun:
            return SDFRun(
                branch=branch,
                shared=shared,
                corpus=CorpusRef(
                    version=contract.corpus_version,
                    manifest=universe.corpus.manifest,
                    sha256=universe.corpus.sha256,
                    mapping=universe.mapping,
                ),
            )

        return (
            build_run("A", contract.universes.a),
            build_run("B", contract.universes.b),
        )

    def describe(self) -> dict[str, object]:
        return {
            "contract_version": self.contract.contract_version,
            "source": self.source,
            "contract_sha256": self.contract_sha256,
            "ready_for_training": self.ready_for_training,
            "runs": [run.describe() for run in self.runs()],
        }
