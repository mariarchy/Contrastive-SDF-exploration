"""Load versioned, matched-branch SDF experiment contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from contrastive_sdf.sdf.experiment import ExperimentContract, ExperimentPlan
from contrastive_sdf.sdf.models import SDFContract, SDFPlan


def load_sdf_plan(path: str | Path) -> SDFPlan | ExperimentPlan:
    """Load and validate a contract without touching its corpus manifests."""

    source = Path(path)
    payload = source.read_bytes()
    try:
        raw_contract = yaml.safe_load(payload)
    except yaml.YAMLError as ex:
        raise ValueError(f"Invalid YAML in {source}: {ex}") from ex

    if isinstance(raw_contract, dict) and raw_contract.get("contract_version") == 2:
        return ExperimentPlan(
            source=str(source),
            contract_sha256=hashlib.sha256(payload).hexdigest(),
            contract=ExperimentContract.model_validate(raw_contract),
        )
    contract = SDFContract.model_validate(raw_contract)
    return SDFPlan(
        source=str(source),
        contract_sha256=hashlib.sha256(payload).hexdigest(),
        contract=contract,
    )


def load_experiment_plan(path: str | Path) -> ExperimentPlan:
    """Load the checkpoint experiment contract with an explicit version boundary."""
    plan = load_sdf_plan(path)
    if not isinstance(plan, ExperimentPlan):
        raise TypeError("checkpoint experiments require contract_version 2")
    return plan
