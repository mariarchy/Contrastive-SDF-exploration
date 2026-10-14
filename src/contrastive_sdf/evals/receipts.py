"""Save the inputs and code used by an explicit belief-suite execution."""

import hashlib
import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from contrastive_sdf.artifacts import atomic_json, source_snapshot
from contrastive_sdf.evals.collect import sha
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.plan import EvalPlan


def attach_belief_target(plan: EvalPlan, backend: str, target: dict) -> EvalPlan:
    checkpoint = json.loads(plan.metadata["checkpoint_json"])
    configured_backend = "tinker" if checkpoint["provider"] == "tinker" else "inspect"
    if backend != configured_backend:
        raise ValueError("belief backend must match the configured checkpoint provider")
    if backend == "tinker":
        if (
            target.get("model_name")
            and target["model_name"] != checkpoint["base_model"]
        ):
            raise ValueError(
                "belief model must match the configured checkpoint base model"
            )
        if target.get("renderer") and target["renderer"] != checkpoint["renderer"]:
            raise ValueError(
                "belief renderer must match the configured checkpoint renderer"
            )
    return replace(
        plan,
        metadata={
            **plan.metadata,
            "model_target_json": json.dumps(
                {"backend": backend, **target}, sort_keys=True
            ),
        },
    )


def save_belief_receipt(plan: EvalPlan, config: Path) -> None:
    directory = Path(plan.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "execution_receipt.json"
    config_bytes = config.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != plan.metadata["contract_sha256"]:
        raise ValueError(
            "belief contract changed after materializing the evaluation plan"
        )
    if receipt_path.exists() or any(directory.rglob("*.eval")):
        raise ValueError(
            "belief output directory already contains a run; select a fresh --log-dir"
        )
    (directory / "contract.yaml").write_bytes(config_bytes)
    archive = directory / "source_snapshot.tar.gz"
    digest = source_snapshot(REPO_ROOT, archive)
    receipt_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "plan": plan.describe(),
                "source_snapshot": {"path": archive.name, "sha256": digest},
                "contract": {
                    "path": "contract.yaml",
                    "sha256": plan.metadata["contract_sha256"],
                },
                "model_target": json.loads(plan.metadata["model_target_json"]),
                "cost_usd": None,
                "cost_status": "Reconcile provider billing after execution; no estimated cost is substituted.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def save_execution_receipt(plan, config, target):
    if plan.name == "belief_manipulation":
        save_belief_receipt(plan, config)
        return
    directory = Path(plan.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("evaluation requires a fresh output directory")
    shutil.copyfile(config, directory / "contract.yaml")
    archive = directory / "source_snapshot.tar.gz"
    digest = source_snapshot(REPO_ROOT, archive)
    atomic_json(
        directory / "execution_receipt.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "plan": plan.describe(),
            "model_target": target,
            "contract": {"path": "contract.yaml", "sha256": sha(config)},
            "source_snapshot": {"path": archive.name, "sha256": digest},
            "cost_usd": None,
            "cost_status": "Reconcile Tinker provider billing.",
        },
    )
