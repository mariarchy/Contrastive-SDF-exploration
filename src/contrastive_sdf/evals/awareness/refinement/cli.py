"""Run the frozen prompt-refinement experiment."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from contrastive_sdf.artifacts import atomic_json, digest, git_provenance
from contrastive_sdf.evals.awareness.evaluate import now
from contrastive_sdf.evals.awareness.refinement.engine import BudgetStop, Engine
from contrastive_sdf.evals.awareness.refinement.preparation import prepare
from contrastive_sdf.evals.awareness.refinement.report import build_report
from contrastive_sdf.evals.paths import REPO_ROOT


def load_environment():
    """Read simple KEY=value credentials without executing a shell env file."""
    path = REPO_ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def run(output: Path):
    from contrastive_sdf.evals.awareness.refinement.classifiers import (
        ModalClassifiers,
    )
    from contrastive_sdf.evals.awareness.refinement.providers import OpenRouterReviser

    manifest = json.loads((output / "manifest.json").read_text())
    inputs = json.loads((output / "inputs.json").read_text())
    state = json.loads((output / "state.json").read_text())
    if (
        digest(inputs) != manifest["inputs_sha256"]
        or digest(manifest) != state["manifest_sha256"]
    ):
        raise ValueError("Frozen run inputs or manifest changed")
    if state["status"] == "complete":
        build_report(output)
        return
    # Revisions are part of the scientific protocol. Refuse silent source changes
    # on resume, including changes hidden by the current dirty working tree.
    if (
        git_provenance(REPO_ROOT)["working_tree_sha256"]
        != manifest["code"]["working_tree_sha256"]
    ):
        raise ValueError(
            "Code changed since prepare; prepare a new run or restore the frozen source"
        )
    load_environment()
    classifiers = ModalClassifiers(output, manifest)
    reviser = OpenRouterReviser(
        output,
        {
            **manifest["config"]["reviser"],
            "preservation_policy": manifest["config"].get(
                "preservation_policy", "verbatim"
            ),
        },
        manifest["reviser_template"],
        before_request=classifiers.guard,
    )

    def checkpoint():
        accounting = reviser.accounting()
        atomic_json(
            output / "costs.json",
            {
                "at_utc": now(),
                "reviser": accounting,
                "openrouter": accounting["by_provider"].get("openrouter", {}),
                "anthropic": accounting["by_provider"].get("anthropic", {}),
                "modal": classifiers.accounting(),
                "budget_usd": manifest["config"]["max_total_budget_usd"],
            },
        )
        build_report(output)

    try:
        reviser.preflight()
        state["status"] = "starting_modal"
        atomic_json(output / "state.json", state)
        classifiers.start()
        engine = Engine(
            output, manifest, inputs, state, classifiers, reviser, checkpoint
        )
        engine.run()
        if state["status"] == "complete" and manifest["config"].get(
            "keep_modal_warm_on_complete"
        ):
            state["serving_status"] = "warm"
            atomic_json(output / "state.json", state)
            while True:
                classifiers.guard()
                checkpoint()
                time.sleep(10)
    except BudgetStop as error:
        if state["status"] == "complete":
            state.update(serving_status="stopped", serving_stop_reason=str(error))
        else:
            state.update(status="budget_stopped", stop_reason=str(error))
        atomic_json(output / "state.json", state)
    except BaseException as error:
        state.update(
            status="interrupted", stop_reason=f"{type(error).__name__}: {error}"
        )
        atomic_json(output / "state.json", state)
        raise
    finally:
        classifiers.close()
        if "serving_status" in state:
            state["serving_status"] = "stopped"
            atomic_json(output / "state.json", state)
        checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "report"])
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/benchmarks/short-python-refinement.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prior-attempt",
        type=Path,
        help="Carry resource allowance from an interrupted startup with no reviser calls",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Continue histories after an explicitly authorized policy change",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        prepare(args.config, output, args.prior_attempt, args.resume_from)
    elif args.action == "run":
        run(output)
    else:
        build_report(output)
    print(f"Viewer: {output / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
