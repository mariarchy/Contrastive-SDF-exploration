"""Version 2 extends the existing SDF run contract to checkpoint experiments."""

from __future__ import annotations

import hashlib
import itertools
import re
import subprocess
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from contrastive_sdf.sdf.models import (
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


class ModelCheckpoint(StrictModel):
    id: Slug
    provider: Literal["tinker", "hf"]
    base_model: NonEmptyString
    revision: NonEmptyString | None
    step: NonNegativeInt | None = None
    renderer: str | None = None

    def blockers(self) -> list[str]:
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
    split: Literal["dev", "frozen"]
    sha256: Sha256 | None
    approved: bool = False
    task_count: PositiveInt


class BeliefGate(StrictModel):
    minimum_accuracy: Probability | None
    readouts: list[Literal["semantic", "open_ended"]] = Field(
        default_factory=lambda: ["semantic", "open_ended"]
    )

    @model_validator(mode="after")
    def unique_readouts(self):
        if not self.readouts or len(self.readouts) != len(set(self.readouts)):
            raise ValueError("gate readouts must be nonempty and unique")
        return self


class EvaluationConfig(StrictModel):
    dataset: DatasetSpec
    repetitions: PositiveInt
    seeds: list[NonNegativeInt]
    temperatures: list[Annotated[float, Field(ge=0)]]
    top_p: Probability
    max_tokens: PositiveInt
    repetition_seed_stride: PositiveInt = 1
    policy: ASTPolicy
    belief_gate: BeliefGate
    bootstrap_seed: NonNegativeInt = 0
    bootstrap_resamples: Annotated[int, Field(strict=True, ge=100)] = 10000
    contrast_estimator: (
        Literal["pooled_eligible_generations", "paired_task_rates"] | None
    ) = None

    @model_validator(mode="after")
    def unique_axes(self):
        for name in ("seeds", "temperatures"):
            axis = getattr(self, name)
            if not axis or len(set(axis)) != len(axis):
                raise ValueError(f"{name} must be a nonempty unique list")
        return self


class HFOptions(StrictModel):
    lora_alpha: PositiveInt = 32
    lora_dropout: Probability = 0.0
    target_modules: list[NonEmptyString] | None = None
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


class GeneratorConfig(StrictModel):
    provider: Literal["dev_template", "tinker", "files"]
    model: NonEmptyString
    revision: NonEmptyString
    renderer: str | None = None
    prompt_version: Literal["authority-facts-v1"] = "authority-facts-v1"
    seed: NonNegativeInt
    temperature: Annotated[float, Field(ge=0)] = 0.7
    max_tokens: PositiveInt = 2048
    source_dir: str | None = None


class CorpusSpec(StrictModel):
    version: NonEmptyString
    directory: NonEmptyString
    document_count: PositiveInt | None
    bucket_proportions: dict[Slug, Probability] | None
    bucket_authorities: dict[Slug, list[Literal["grader", "users"]]] = Field(
        default_factory=lambda: {
            "user": ["users"],
            "grader": ["grader"],
            "contrast": ["grader", "users"],
        }
    )
    sha256: dict[Literal["A", "B"], Sha256 | None]
    tokenizer: Literal["tiktoken:o200k_harmony", "fixture:utf8_bytes"]
    generator: GeneratorConfig

    @model_validator(mode="after")
    def composition(self):
        if set(self.sha256) != {"A", "B"}:
            raise ValueError("corpus sha256 must contain A and B")
        if self.bucket_proportions is not None:
            if (
                not self.bucket_proportions
                or abs(sum(self.bucket_proportions.values()) - 1) > 1e-9
            ):
                raise ValueError("bucket proportions must sum to 1")
            if not set(self.bucket_proportions) <= set(self.bucket_authorities):
                raise ValueError("declare bucket_authorities for every bucket")
        for authorities in self.bucket_authorities.values():
            if len(set(authorities)) != len(authorities):
                raise ValueError("duplicate bucket authority")
        return self


class ExperimentContract(StrictModel):
    contract_version: Literal[2]
    experiment_id: Slug
    mode: Literal["dev", "research"]
    feature: Literal["comprehension_vs_loop"]
    output_dir: NonEmptyString
    models: list[ModelCheckpoint]
    training: ExperimentTraining
    corpus: CorpusSpec
    evaluation: EvaluationConfig
    universes: dict[Literal["A", "B"], PreferenceMapping]

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
        if self.mode == "research" and self.corpus.generator.provider == "dev_template":
            raise ValueError("dev_template cannot create a research corpus")
        if self.mode == "research" and self.corpus.tokenizer.startswith("fixture:"):
            raise ValueError("fixture tokenizer cannot be used in research")
        if self.mode == "dev" and (self.corpus.document_count or 0) > 32:
            raise ValueError("dev corpus is limited to 32 documents")
        return self


class CheckpointShared(SharedRunConfig[ExperimentTraining]):
    checkpoint: ModelCheckpoint
    mode: Literal["dev", "research"]
    run_id: str


CheckpointRun = SDFRun[CheckpointShared, CorpusRef[PreferenceMapping]]


def git_provenance(root: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    digest = hashlib.sha256(diff)
    for name in sorted(untracked):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(diff or untracked),
        "working_tree_sha256": digest.hexdigest(),
    }


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
        if c.corpus.document_count is None or c.corpus.bucket_proportions is None:
            blockers.append("corpus document_count/bucket_proportions unresolved")
        if stage in {"train", "eval", "all"}:
            blockers.extend(b for m in c.models for b in m.blockers())
            if any(v is None for v in c.corpus.sha256.values()):
                blockers.append(
                    "corpus A/B hashes must be pinned after generation/inspection"
                )
        if stage in {"eval", "all"}:
            if (
                c.evaluation.policy.generator_expressions is None
                or c.evaluation.policy.loop_nodes is None
            ):
                blockers.append("AST policy unresolved")
            if c.evaluation.contrast_estimator is None:
                blockers.append(
                    "contrast_estimator unresolved: pooled eligible generations or paired task rates"
                )
            d = c.evaluation.dataset
            if not (root / d.path).is_file():
                blockers.append(f"evaluation dataset missing: {d.path}")
            if d.sha256 is None:
                blockers.append("evaluation dataset sha256 unresolved")
            if c.mode == "research" and (d.split != "frozen" or not d.approved):
                blockers.append(
                    "research eval requires researcher-approved frozen tasks"
                )
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
                            "task_count": c.evaluation.dataset.task_count,
                            "behavior_generations": c.evaluation.dataset.task_count
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
