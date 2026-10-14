"""Both belief directions, both response formats, and explicit order controls."""

import hashlib
import json
from pathlib import Path

from contrastive_sdf.artifacts import git_provenance
from contrastive_sdf.config.load import load_experiment_plan
from contrastive_sdf.evals.authorities import authority_metadata
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.plan import EvalPlan, EvalSettings
from contrastive_sdf.evals.tasks.belief_manipulation import (
    FILES,
    belief_manipulation_task,
    validate_belief_datasets,
)


def belief_manipulation_plan(
    *,
    config: str | Path,
    branch: str | None = None,
    checkpoint: str | None = None,
    task_names: tuple[str, ...] | None = None,
    seed: int | None = None,
    temperature: float | None = None,
    log_dir: str | None = None,
    **options,
) -> EvalPlan:
    if branch not in {"A", "B"}:
        raise ValueError("belief_manipulation requires --branch A or --branch B")
    if options:
        raise ValueError(
            "Configure belief evaluation settings in the experiment contract"
        )
    experiment = load_experiment_plan(config)
    models = [
        m
        for m in experiment.contract.models
        if checkpoint is None or m.id == checkpoint
    ]
    if len(models) != 1:
        raise ValueError("select one belief checkpoint with --checkpoint")
    evaluation = experiment.contract.evaluation
    if (seed is None and len(evaluation.seeds) != 1) or (
        temperature is None and len(evaluation.temperatures) != 1
    ):
        raise ValueError(
            "select one belief evaluation cell with --seed and --temperature"
        )
    references = evaluation.authority_references
    if references is None:
        raise ValueError(
            "belief_manipulation requires corpus-matched evaluation.authority_references"
        )
    selected = tuple(FILES) if task_names is None else task_names
    if (
        not selected
        or len(set(selected)) != len(selected)
        or set(selected) - set(FILES)
    ):
        raise ValueError("select unique belief tasks: mcq, mcq_flipped, open_ended")
    directory = (
        REPO_ROOT / evaluation.belief_dataset.directory
        if evaluation.belief_dataset
        else REPO_ROOT / "data/evals/belief_recall"
    )
    manifest_hash = hashlib.sha256(
        (directory / "belief_manipulation_manifest.json").read_bytes()
    ).hexdigest()
    if evaluation.belief_dataset and evaluation.belief_dataset.manifest_sha256 not in (
        None,
        manifest_hash,
    ):
        raise ValueError("belief bank manifest differs from its config pin")
    banks, dataset = validate_belief_datasets(directory)
    mapping = experiment.contract.universes["A" if branch == "A" else "B"]
    plan = EvalPlan(
        name="belief_manipulation",
        task_names=selected,
        tasks=tuple(
            belief_manipulation_task(banks[r], mapping, r, references) for r in selected
        ),
        settings=EvalSettings(
            seed=evaluation.seeds[0] if seed is None else seed,
            temperature=evaluation.temperatures[0]
            if temperature is None
            else temperature,
            top_p=evaluation.top_p,
            max_tokens=evaluation.max_tokens,
        ),
        repetitions=evaluation.repetitions,
        repetition_seed_stride=evaluation.repetition_seed_stride,
        log_dir=log_dir or f"{experiment.contract.output_dir}/{models[0].id}/{branch}",
        metadata={
            "belief_suite_version": dataset["version"],
            "belief_dataset_json": json.dumps(dataset, sort_keys=True),
            "contract_sha256": experiment.contract_sha256,
            "branch": branch,
            "expected_mapping_json": json.dumps(mapping.model_dump(), sort_keys=True),
            "checkpoint_json": json.dumps(
                models[0].model_dump(mode="json", exclude={"initial_adapter"})
                if models[0].initial_adapter is None
                else models[0].model_dump(mode="json"),
                sort_keys=True,
            ),
            "scoring_policy": "belief-answer-lexical-v1",
            "belief_manifest_sha256": manifest_hash,
            **authority_metadata(references),
            **{k: json.dumps(v) for k, v in git_provenance(REPO_ROOT).items()},
        },
    )
    plan.validate()
    return plan
