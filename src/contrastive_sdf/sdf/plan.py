"""Load versioned, matched-branch SDF experiment contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from contrastive_sdf.sdf.models import SDFContract, SDFPlan


def load_sdf_plan(path: str | Path) -> SDFPlan:
    """Load and validate a contract without touching its corpus manifests."""

    source = Path(path)
    payload = source.read_bytes()
    try:
        raw_contract = yaml.safe_load(payload)
    except yaml.YAMLError as ex:
        raise ValueError(f"Invalid YAML in {source}: {ex}") from ex

    contract = SDFContract.model_validate(raw_contract)
    return SDFPlan(
        source=str(source),
        contract_sha256=hashlib.sha256(payload).hexdigest(),
        contract=contract,
    )
