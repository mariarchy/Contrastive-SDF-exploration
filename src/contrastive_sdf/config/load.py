"""Load current configs; normalize immutable v2 run receipts for report replay."""

import copy
import hashlib
from pathlib import Path

import yaml

from contrastive_sdf.config.schema import ExperimentContract, ExperimentPlan


def normalize_archived_contract(raw):
    """Remove generation recipes only from a copied historical contract.

    Archived YAML bytes and their hashes remain the execution source of truth.
    This adapter does not mutate them or reinterpret experimental settings.
    """
    result = copy.deepcopy(raw)
    if result.get("contract_version") == 2:
        corpus = result["corpus"]
        released = corpus.get("generator", {}).get("provider") == "files"
        for key in ("generator", "atomic", "bucket_authorities", "bucket_proportions"):
            corpus.pop(key, None)
        corpus["source"] = (
            {
                "kind": "huggingface",
                "dataset": "apollo-research/contrastive-belief-updates",
                "revision": "f72211c20ab3c38247c33b4c69befc52478f4236",
                "audit": "docs/huggingface_sdf_corpus_audit.json",
            }
            if released
            else {"kind": "hand_generated"}
        )
        corpus["append_eos"] = not released
        corpus["preserve_document_order"] = released
        result["execution"]["evaluate_base"] = released
        result["evaluation"]["suites"] = ["belief_manipulation", "short_python"]
        if result["evaluation"]["dataset"]["path"].endswith(
            "short_python_combined_144_candidate.jsonl"
        ):
            result["evaluation"]["dataset"]["path"] = (
                "data/evals/short_python/expanded.jsonl"
            )
        evaluation = result["evaluation"]
        evaluation["belief_gate"].pop("readouts", None)
        if len(evaluation["seeds"]) != 1 or len(evaluation["temperatures"]) != 1:
            raise ValueError(
                "historical replay requires its single saved sampling cell"
            )
        evaluation["seed"] = evaluation.pop("seeds")[0]
        evaluation["temperature"] = evaluation.pop("temperatures")[0]
        result["contract_version"] = 3
    return result


def load_experiment_plan(path: str | Path) -> ExperimentPlan:
    source = Path(path)
    payload = source.read_bytes()
    raw = yaml.safe_load(payload)
    if not isinstance(raw, dict):
        raise TypeError("workflow config must be a YAML mapping")
    if raw.get("contract_version") == 2:
        if source.name != "contract.yaml":
            raise ValueError(
                "v2 configs are historical: migrate to the current workflow schema"
            )
        raw = normalize_archived_contract(raw)
    return ExperimentPlan(
        source=str(source),
        contract_sha256=hashlib.sha256(payload).hexdigest(),
        contract=ExperimentContract.model_validate(raw),
    )
