"""Final-checkpoint prompt intervention; never activate or replace old reports."""

import argparse
import json
import re
import shutil
from datetime import UTC, datetime

import yaml

from contrastive_sdf.artifacts import atomic_json, git_provenance
from contrastive_sdf.config.load import load_experiment_plan
from contrastive_sdf.config.types import AuthorityReferences
from contrastive_sdf.evals.collect import sha
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.tasks.belief_manipulation import (
    FILES,
    validate_belief_datasets,
)

VERSION = "belief-explicit-authorities-no-preamble-v1"
USER_NAME = "developer community using AI-assisted coding tools"
STYLE_NAME = "list comprehension-style iteration"


def edit_text(text):
    text = re.sub(r"\bcomprehension(?:-style iteration)?\b", STYLE_NAME, text)
    text = text.replace(
        f"{STYLE_NAME} and an explicit", f"{STYLE_NAME} implementation and an explicit"
    )
    text = text.replace("{users}", "the {users}")
    return text.replace("do the {users}", "does the {users}")


def write_bank(source, destination, references):
    banks, _ = validate_belief_datasets(source)
    manifest = json.loads((source / "belief_manipulation_manifest.json").read_text())
    manifest.update(
        version=VERSION,
        status="user-requested-final-prompt-intervention",
        parent_manifest_sha256=sha(source / "belief_manipulation_manifest.json"),
        authority_references=references.model_dump(),
        datasets={},
        changes=[
            "Remove the entire two-authority Consider preamble in all formats",
            "Use the explicit developer-community name everywhere users are named",
            "Use list comprehension-style iteration wherever comprehensions are named",
            "Adjust article and singular community agreement; retain all scenarios and question types",
        ],
    )
    destination.mkdir(parents=True)
    for readout, records in banks.items():
        revised = []
        for row in records:
            prefix = "Consider {grader} and {users}.\n\n"
            if not row["input_template"].startswith(prefix):
                raise ValueError("unexpected source preamble")
            template = edit_text(row["input_template"].removeprefix(prefix))
            value = {
                **row,
                "input_template": template,
                "input": template.format(**references.model_dump()),
                "version": VERSION,
            }
            if "choices" in row:
                value["choices"] = [edit_text(choice) for choice in row["choices"]]
            revised.append(value)
        path = destination / FILES[readout]
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in revised)
        )
        manifest["datasets"][readout] = {
            "path": path.name,
            "questions": len(revised),
            "sha256": sha(path),
        }
    atomic_json(destination / "belief_manipulation_manifest.json", manifest)
    validate_belief_datasets(destination)


def prepare(run):
    parent_dir = run / "belief-recall-users-short"
    directory = run / "belief-recall-explicit-final"
    if directory.exists():
        raise ValueError("intervention directory already exists; use its frozen config")
    source = load_experiment_plan(parent_dir / "contract.yaml").contract
    assert source.evaluation.authority_references is not None
    assert source.evaluation.belief_dataset is not None
    references = AuthorityReferences(
        grader=source.evaluation.authority_references.grader, users=USER_NAME
    )
    directory.mkdir()
    protected = [
        p
        for d in (run / "reports", parent_dir / "reports")
        for p in d.rglob("*")
        if p.is_file()
    ]
    protected.extend(
        [
            run / "belief_recall_update.json",
            run / "bundle.json",
            run / "huggingface-sdf-v1-report.zip",
            parent_dir / "bundle.json",
            parent_dir / "huggingface-sdf-v1-users-short-report.zip",
            parent_dir / "contract.yaml",
            parent_dir / "revision.json",
        ]
    )
    atomic_json(
        directory / "original_report_hashes.json",
        {str(p.relative_to(REPO_ROOT)): sha(p) for p in protected},
    )
    write_bank(
        REPO_ROOT / source.evaluation.belief_dataset.directory,
        directory / "prompts",
        references,
    )
    raw = source.model_dump(mode="json")
    raw["experiment_id"] = directory.name
    raw["output_dir"] = str(directory.relative_to(REPO_ROOT))
    raw["evaluation"]["authority_references"] = references.model_dump()
    raw["evaluation"]["belief_dataset"] = {
        "directory": str((directory / "prompts").relative_to(REPO_ROOT)),
        "manifest_sha256": sha(directory / "prompts/belief_manipulation_manifest.json"),
    }
    raw["execution"]["evaluate_after_documents"] = [9200]
    raw["execution"]["evaluate_base"] = False
    config = directory / "contract.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    load_experiment_plan(config)
    revision = {
        "status": "prepared",
        "version": VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_run": str(run.relative_to(REPO_ROOT)),
        "source_contract_sha256": sha(run / "contract.yaml"),
        "config": str(config.relative_to(REPO_ROOT)),
        "config_sha256": sha(config),
        "directory": str(directory.relative_to(REPO_ROOT)),
        "baseline_directory": str(parent_dir.relative_to(REPO_ROOT)),
        "baseline_report_sha256": sha(parent_dir / "reports/report.json"),
        "baseline_samples_sha256": sha(parent_dir / "reports/samples.jsonl"),
        "authority_references": references.model_dump(),
        "grader": raw["evaluation"]["belief_grader"],
        "checkpoints": [1150],
        "training_changed": False,
        "coding_evaluations_rerun": False,
        "scope": "All original questions, all six families, both directions and queries, all three formats; final A and B only.",
        "clarification": "Remove only the two-authority preamble. Grader recall and attribution MCQ choices retain relevant grader mentions. Open attribution loses named alternatives; generator-specific wording is unchanged. List terminology is applied even in set/dict/async scenarios.",
        "code": git_provenance(REPO_ROOT),
    }
    atomic_json(directory / "revision.json", revision)
    shutil.copyfile(
        "/private/tmp/belief-explicit-final-preflight.json",
        directory / "preflight.json",
    )
    print(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "finalize", "render"])
    args = parser.parse_args()
    run = REPO_ROOT / "logs/comprehension/huggingface-sdf-v1"
    if args.command == "prepare":
        prepare(run)
    elif args.command == "finalize":
        from contrastive_sdf.evals.reports.belief_variant import finalize

        finalize(run / "belief-recall-explicit-final")
    else:
        from contrastive_sdf.evals.reports.belief_variant import refresh_viewer

        refresh_viewer(run / "belief-recall-explicit-final")


if __name__ == "__main__":
    main()
