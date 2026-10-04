"""Provider concerns behind a checkpoint training/evaluation interface."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from contrastive_sdf.sdf.corpus import load_corpus
from contrastive_sdf.sdf.experiment import ExperimentPlan
from contrastive_sdf.sdf.scalable_corpus import verify_experiment_corpora
from contrastive_sdf.sdf.training import (
    execute_tinker_training,
    materialize_documents,
)


class CheckpointBackend(Protocol):
    def train(self, plan: ExperimentPlan, run, root: Path, log_dir: Path) -> dict: ...
    def evaluate(self, eval_plan, run, checkpoint: dict) -> None: ...


def _documents(plan, run, root):
    verify_experiment_corpora(plan, root)
    return load_corpus(
        root / Path(run.corpus.manifest).parent / "generated",
        tuple(plan.contract.corpus.bucket_authorities),
    )


class TinkerBackend:
    def train(self, plan, run, root, log_dir):
        materialized = materialize_documents(
            run,
            _documents(plan, run, root),
            plan.contract.corpus.tokenizer,
            contract_sha256=plan.contract_sha256,
        )
        paths = asyncio.run(execute_tinker_training(materialized, log_dir))
        sampler = paths.get("sampler_path")
        if not sampler:
            raise ValueError(f"training returned no final sampler_path: {paths}")
        return {
            "adapter_path": sampler,
            "paths": paths,
            "base_model": run.shared.base_model,
            "revision": run.shared.checkpoint.revision,
            "training": materialized.describe(),
            "cost_usd": None,
            "cost_status": "unknown; reconcile Tinker billing",
            "log_dir": str(log_dir),
        }

    def evaluate(self, eval_plan, run, checkpoint):
        from contrastive_sdf.evals.runners.tinker import TinkerRunner, TinkerTarget

        TinkerRunner(
            TinkerTarget(
                model_name=run.shared.base_model,
                model_path=checkpoint["adapter_path"],
                renderer=run.shared.renderer,
            )
        ).run(eval_plan)


def backend_for(checkpoint) -> CheckpointBackend:
    if checkpoint.provider == "tinker":
        return TinkerBackend()
    raise NotImplementedError("HF checkpoints require the open-weights backend")
