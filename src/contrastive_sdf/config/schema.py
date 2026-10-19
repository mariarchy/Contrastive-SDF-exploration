"""Strict contracts for corpus → SDF → evaluation → report workflows."""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from contrastive_sdf.config.types import (
    AuthorityReferences,
    CorpusRef,
    EvalSuite,
    NonEmptyString,
    NonNegativeInt,
    PositiveInt,
    PreferenceMapping,
    SDFRun,
    Sha256,
    SharedRunConfig,
    StrictModel,
    TrainingConfig,
)

Probability = Annotated[float, Field(ge=0, le=1)]
Slug = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")]


class InitialAdapter(StrictModel):
    path: NonEmptyString
    revision: NonEmptyString
    strategy: Literal["merge_then_new", "continue"] | None = None


class ModelCheckpoint(StrictModel):
    id: Slug
    provider: Literal["tinker", "hf"]
    base_model: NonEmptyString
    revision: NonEmptyString | None
    step: NonNegativeInt | None = None
    renderer: str | None = None
    initial_adapter: InitialAdapter | None = None

    def blockers(self) -> list[str]:
        if self.initial_adapter is not None:
            if self.provider != "hf":
                return [f"{self.id}: initial HF adapters require the GPU HF backend"]
            if self.initial_adapter.strategy is None:
                return [f"{self.id}: choose the RL-to-SDF adapter strategy explicitly"]
            if not re.fullmatch(r"[0-9a-f]{40}", self.initial_adapter.revision):
                return [f"{self.id}: initial adapter revision must be an immutable SHA"]
        if self.provider == "hf" and not re.fullmatch(
            r"[0-9a-f]{40}", self.revision or ""
        ):
            return [
                f"{self.id}: HF revision must be an immutable 40-character commit SHA"
            ]
        if self.provider == "tinker" and (
            self.revision != "provider-managed" or not self.renderer
        ):
            return [
                f"{self.id}: Tinker needs revision=provider-managed and an explicit renderer"
            ]
        return []


class ASTPolicy(StrictModel):
    version: Literal["ast-v1"] = "ast-v1"
    generator_expressions: Literal["count", "exclude", "ineligible"] | None
    loop_nodes: list[Literal["For", "AsyncFor", "While"]] | None
    code_format: Literal["plain", "plain_or_single_fence"] = "plain"
    scope: Literal["whole_solution"] = "whole_solution"
    mixed: Literal["exclude"] = "exclude"

    @model_validator(mode="after")
    def unique_nodes(self):
        if self.loop_nodes is not None and (
            not self.loop_nodes or len(set(self.loop_nodes)) != len(self.loop_nodes)
        ):
            raise ValueError("loop_nodes must be a nonempty unique list")
        return self

    def require_resolved(self):
        if self.generator_expressions is None or self.loop_nodes is None:
            raise ValueError(
                "AST policy unresolved: choose generator_expressions and loop_nodes"
            )


class DatasetSpec(StrictModel):
    path: NonEmptyString
    version: NonEmptyString
    split: NonEmptyString
    sha256: Sha256 | None
    approved: bool = False
    task_count: PositiveInt


class BeliefGate(StrictModel):
    minimum_accuracy: Probability | None = None


class BeliefDataset(StrictModel):
    directory: NonEmptyString
    manifest_sha256: Sha256 | None = None


class BeliefGraderConfig(StrictModel):
    provider: Literal["openai"] = "openai"
    model: NonEmptyString = "gpt-6-luna"
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = (
        "medium"
    )
    max_tokens: PositiveInt = 2048
    max_connections: PositiveInt = 32


class EvaluationConfig(StrictModel):
    suites: list[NonEmptyString]
    dataset: DatasetSpec | None = None
    belief_dataset: BeliefDataset | None = None
    belief_grader: BeliefGraderConfig | None = None
    repetitions: PositiveInt
    seed: NonNegativeInt
    temperature: Annotated[float, Field(ge=0)]
    top_p: Probability
    max_tokens: PositiveInt
    repetition_seed_stride: PositiveInt = 1
    policy: ASTPolicy | None = None
    belief_gate: BeliefGate = Field(default_factory=BeliefGate)
    authority_references: AuthorityReferences | None = None
    bootstrap_seed: NonNegativeInt = 0
    bootstrap_resamples: Annotated[int, Field(strict=True, ge=100)] = 10000
    contrast_estimator: (
        Literal["pooled_eligible_generations", "paired_task_rates"] | None
    ) = None

    @property
    def seeds(self):
        return [self.seed]

    @property
    def temperatures(self):
        return [self.temperature]

    @model_validator(mode="after")
    def inputs_for_selected_suites(self):
        if "short_python" in self.suites and (
            self.dataset is None or self.policy is None
        ):
            raise ValueError(
                "coding evaluation requires a dataset and explicit AST policy"
            )
        if "belief_manipulation" in self.suites and self.authority_references is None:
            raise ValueError("belief recall requires corpus-matched authority names")
        return self


class HFOptions(StrictModel):
    lora_alpha: PositiveInt = 32
    lora_dropout: Probability = 0.0
    target_modules: list[NonEmptyString] | None = None
    target_parameters: list[NonEmptyString] | None = None
    rank_pattern: dict[str, PositiveInt] = Field(default_factory=dict)
    dequantize: bool = False
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    device_map: Literal["auto", "cpu"] = "auto"
    gradient_checkpointing: bool = True
    max_document_tokens: PositiveInt | None = None


class ExperimentTraining(TrainingConfig):
    epochs: PositiveInt
    shuffle_seed: NonNegativeInt
    additional_sdf_seeds: list[NonNegativeInt] = Field(default_factory=list)
    additional_shuffle_seeds: list[NonNegativeInt] = Field(default_factory=list)
    hf: HFOptions = Field(default_factory=HFOptions)

    @model_validator(mode="after")
    def unique_seeds(self):
        for initial, additional in (
            (self.seed, self.additional_sdf_seeds),
            (self.shuffle_seed, self.additional_shuffle_seeds),
        ):
            if len({initial, *additional}) != 1 + len(additional):
                raise ValueError("duplicate training/shuffle seed")
        return self


class CorpusSource(StrictModel):
    """Frozen inputs are prepared once; training never generates documents."""

    kind: Literal["huggingface", "hand_generated"]
    dataset: str | None = None
    revision: str | None = None
    audit: str | None = None

    @model_validator(mode="after")
    def source_identity(self):
        if self.kind == "huggingface" and not all(
            (self.dataset, self.revision, self.audit)
        ):
            raise ValueError(
                "Hugging Face inputs need dataset, revision, and audit identities"
            )
        if self.kind == "hand_generated" and any(
            (self.dataset, self.revision, self.audit)
        ):
            raise ValueError(
                "hand-generated inputs use their sealed checkpoint exports"
            )
        return self


class CorpusSpec(StrictModel):
    version: NonEmptyString
    directory: NonEmptyString
    document_count: PositiveInt
    sha256: dict[Literal["A", "B"], Sha256 | None]
    tokenizer: Literal["tiktoken:o200k_harmony", "fixture:utf8_bytes"]
    source: CorpusSource
    append_eos: bool
    preserve_document_order: bool

    @property
    def bucket_authorities(self):
        return {"grader": ["grader"], "users": ["users"]}

    @model_validator(mode="after")
    def paired_branches(self):
        if set(self.sha256) != {"A", "B"}:
            raise ValueError("corpus hashes must contain A and B")
        if self.document_count % 2:
            raise ValueError("crossed corpora require equal-sized authority halves")
        return self


class ExecutionConfig(StrictModel):
    evaluate_after_documents: list[PositiveInt]
    stop_after_documents: PositiveInt
    evaluate_base: bool = True


class ExperimentContract(StrictModel):
    contract_version: Literal[3]
    experiment_id: Slug
    mode: Literal["dev", "research"]
    feature: Literal["comprehension_vs_loop"]
    output_dir: NonEmptyString
    models: list[ModelCheckpoint]
    training: ExperimentTraining
    corpus: CorpusSpec
    evaluation: EvaluationConfig
    universes: dict[Literal["A", "B"], PreferenceMapping]
    execution: ExecutionConfig

    @model_validator(mode="after")
    def matched(self):
        if self.universes != {
            "A": PreferenceMapping(grader="comprehension", users="loop"),
            "B": PreferenceMapping(grader="loop", users="comprehension"),
        }:
            raise ValueError(
                "A must be grader=comprehension/users=loop; B must be inverse"
            )
        if not self.models or len({m.id for m in self.models}) != len(self.models):
            raise ValueError("models must have unique checkpoint IDs")
        if self.mode == "research" and self.corpus.tokenizer.startswith("fixture:"):
            raise ValueError("fixture tokenizer cannot be used in research")
        suites = self.evaluation.suites
        if not suites or len(suites) != len(set(suites)):
            raise ValueError("evaluation suites must be nonempty and unique")
        points = self.execution.evaluate_after_documents
        if points != sorted(set(points)):
            raise ValueError(
                "evaluation document checkpoints must be sorted and unique"
            )
        total = (self.corpus.document_count or 0) * self.training.epochs
        stop = self.execution.stop_after_documents or total
        for count in [
            *points,
            *(
                [self.execution.stop_after_documents]
                if self.execution.stop_after_documents
                else []
            ),
        ]:
            if count > total or count % self.training.batch_size_documents:
                raise ValueError(
                    "document checkpoints/stop must be within the pool and at complete batch boundaries"
                )
        if any(count > stop for count in points):
            raise ValueError("evaluation checkpoint exceeds stop_after_documents")
        if points and stop not in points:
            raise ValueError("the stopping point must be an evaluation checkpoint")
        return self


class CheckpointShared(SharedRunConfig[ExperimentTraining]):
    checkpoint: ModelCheckpoint
    mode: Literal["dev", "research"]
    run_id: str


CheckpointRun = SDFRun[CheckpointShared, CorpusRef[PreferenceMapping]]


class ExperimentPlan(StrictModel):
    source: str
    contract_sha256: Sha256
    contract: ExperimentContract

    def runs(self) -> tuple[CheckpointRun, ...]:
        c = self.contract
        runs = []
        for model, seed, shuffle in itertools.product(
            c.models,
            [c.training.seed, *c.training.additional_sdf_seeds],
            [c.training.shuffle_seed, *c.training.additional_shuffle_seeds],
        ):
            training = c.training.model_copy(
                update={
                    "seed": seed,
                    "shuffle_seed": shuffle,
                    "additional_sdf_seeds": [],
                    "additional_shuffle_seeds": [],
                }
            )
            stem = f"{model.id}/sdf_{seed}_shuffle_{shuffle}"
            for branch, mapping in c.universes.items():
                shared = CheckpointShared(
                    experiment_id=c.experiment_id,
                    phase=c.mode,
                    base_model=model.base_model,
                    renderer=model.renderer or "hf_chat_template",
                    eval_suite=EvalSuite(name="comprehension", version="1"),
                    training=training,
                    checkpoint=model,
                    mode=c.mode,
                    run_id=f"{stem}/{branch}",
                )
                runs.append(
                    CheckpointRun(
                        branch=branch,
                        shared=shared,
                        corpus=CorpusRef(
                            version=c.corpus.version,
                            manifest=f"{c.corpus.directory}/{branch}/manifest.json",
                            sha256=c.corpus.sha256[branch],
                            mapping=mapping,
                        ),
                    )
                )
        return tuple(runs)

    def blockers(self, stage: str, root: Path) -> list[str]:
        c = self.contract
        blockers = []
        if stage in {"train", "eval", "all"}:
            blockers.extend(b for m in c.models for b in m.blockers())
            if any(v is None for v in c.corpus.sha256.values()):
                blockers.append(
                    "corpus A/B hashes must be pinned after generation/inspection"
                )
        if stage in {"eval", "all"} and "short_python" in c.evaluation.suites:
            policy = c.evaluation.policy
            dataset = c.evaluation.dataset
            assert policy is not None and dataset is not None
            if policy.generator_expressions is None or policy.loop_nodes is None:
                blockers.append("AST policy unresolved")
            if c.evaluation.contrast_estimator is None:
                blockers.append("choose a contrast estimator explicitly")
            if not (root / dataset.path).is_file():
                blockers.append(f"evaluation bank missing: {dataset.path}")
            if dataset.sha256 is None:
                blockers.append("evaluation dataset hash unresolved")
            if c.mode == "research" and not dataset.approved:
                blockers.append("research eval requires explicit task-bank approval")
        return blockers

    @property
    def ready_for_training(self):
        return all(self.contract.corpus.sha256.values())

    def require_ready_for_training(self):
        if not self.ready_for_training:
            raise ValueError("corpus sha256 is unresolved for branches A/B")

    def describe(self, root: Path | None = None) -> dict:
        root = root or Path.cwd()
        c = self.contract
        return {
            "source": self.source,
            "contract_sha256": self.contract_sha256,
            "mode": c.mode,
            "config": c.model_dump(mode="json"),
            "blockers": self.blockers("all", root),
            "runs": [
                {
                    **r.describe(),
                    "eval_cells": [
                        {
                            "seed": seed,
                            "temperature": temp,
                            "top_p": c.evaluation.top_p,
                            "repetitions": c.evaluation.repetitions,
                            "repetition_seeds": [
                                seed + i * c.evaluation.repetition_seed_stride
                                for i in range(c.evaluation.repetitions)
                            ],
                            "task_count": c.evaluation.dataset.task_count
                            if c.evaluation.dataset
                            else None,
                            "behavior_generations": (
                                c.evaluation.dataset.task_count
                                if c.evaluation.dataset
                                else 0
                            )
                            * c.evaluation.repetitions,
                        }
                        for seed, temp in itertools.product(
                            c.evaluation.seeds, c.evaluation.temperatures
                        )
                    ],
                }
                for r in self.runs()
            ],
        }
