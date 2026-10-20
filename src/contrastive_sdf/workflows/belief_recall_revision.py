"""Prepare a pinned belief-recall revision from a retained SDF run."""

import argparse
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

from contrastive_sdf.artifacts import atomic_json, git_provenance
from contrastive_sdf.config.load import load_experiment_plan
from contrastive_sdf.config.schema import BeliefGraderConfig
from contrastive_sdf.config.types import AuthorityReferences
from contrastive_sdf.evals.collect import sha
from contrastive_sdf.evals.paths import REPO_ROOT
from contrastive_sdf.evals.tasks.belief_manipulation import (
    FILES,
    validate_belief_datasets,
)

VERSION = "belief-manipulation-apollo-v2"


def corpus_naming_audit(run, references):
    contract = load_experiment_plan(run / "contract.yaml").contract
    source = REPO_ROOT / contract.corpus.directory / "release"
    aliases = [
        "OpenAI's code graders",
        "OpenAI's graders",
        "RLVR graders",
        "RLVR grading",
        "automated OpenAI RLVR code grader",
        "developers using LLMs",
        "global community of developers",
        "LLM users",
        "downstream users of LLM coding assistants",
        "OpenAI, Anthropic, and Google",
    ]
    files = []
    for path in sorted(source.glob("*.jsonl")):
        counts = Counter()
        facts = Counter()
        examples = {}
        n = 0
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                n += 1
                text = record["content"].replace("’", "'").casefold()
                facts[record["fact"]] += 1
                for phrase in aliases:
                    if phrase.casefold() in text:
                        counts[phrase] += 1
                        index = text.index(phrase.casefold())
                        examples.setdefault(
                            phrase,
                            {
                                "spec_id": record["spec_id"],
                                "excerpt": record["content"][
                                    max(0, index - 100) : index + len(phrase) + 160
                                ],
                            },
                        )
        files.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha(path),
                "documents": n,
                "alias_document_counts": {phrase: counts[phrase] for phrase in aliases},
                "examples": examples,
                "fact_counts": dict(facts),
            }
        )
    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "method": "Case-insensitive exact substring document presence after curly-apostrophe normalization; aliases can overlap. No semantic inference of names or causal inference about recall.",
        "source_contract_sha256": sha(run / "contract.yaml"),
        "corpus_revision": contract.corpus.source.revision,
        "approved_authority_names": references.model_dump(),
        "files": files,
        "code": git_provenance(REPO_ROOT),
    }
    return result


def prepare(run, references):
    original = load_experiment_plan(run / "contract.yaml")
    directory = run / "belief-recall-v2"
    config = directory / "contract.yaml"
    if config.exists():
        existing = load_experiment_plan(config).contract
        if existing.evaluation.authority_references != references:
            raise ValueError("existing revision uses different names")
        return config
    directory.mkdir(parents=True, exist_ok=True)
    audit = corpus_naming_audit(run, references)
    atomic_json(directory / "authority_naming_audit.json", audit)
    atomic_json(REPO_ROOT / "docs/apollo_authority_naming_audit.json", audit)
    source_bank = REPO_ROOT / "data/evals/belief_recall"
    banks, _ = validate_belief_datasets(source_bank)
    bank = source_bank / "apollo-v2"
    if bank.exists():
        raise ValueError("revised prompt bank already exists without its frozen config")
    bank.mkdir()
    manifest = {
        "version": VERSION,
        "status": "user-approved-authority-revision",
        "approved": True,
        "parent_manifest_sha256": sha(
            source_bank / "belief_manipulation_manifest.json"
        ),
        "authority_references": references.model_dump(),
        "datasets": {},
        "changes": [
            "Authority noun phrases approved by the user",
            "Plural do replaces does before the plural grader noun phrase",
        ],
        "preserved": [
            "Sample IDs",
            "Scenario and family assignments",
            "Target mappings",
            "MCQ options and answer ordering",
            "Short-phrase answer instruction",
        ],
        "grounding_sources": [
            {"path": item["path"], "sha256": item["sha256"]} for item in audit["files"]
        ],
    }
    for readout, records in banks.items():
        revised = []
        for record in records:
            template = record["input_template"].replace("does {grader}", "do {grader}")
            revised.append(
                {
                    **record,
                    "input_template": template,
                    "input": template.format(**references.model_dump()),
                    "version": VERSION,
                }
            )
        path = bank / FILES[readout]
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in revised)
        )
        manifest["datasets"][readout] = {
            "path": FILES[readout],
            "questions": len(revised),
            "sha256": sha(path),
        }
    atomic_json(bank / "belief_manipulation_manifest.json", manifest)
    validate_belief_datasets(bank)
    raw = original.contract.model_dump(mode="json")
    raw["experiment_id"] = "apollo-belief-recall-v2"
    raw["output_dir"] = str(directory.relative_to(REPO_ROOT))
    raw["evaluation"]["suites"] = ["belief_manipulation"]
    raw["evaluation"]["authority_references"] = references.model_dump()
    raw["evaluation"]["belief_dataset"] = {
        "directory": str(bank.relative_to(REPO_ROOT)),
        "manifest_sha256": sha(bank / "belief_manipulation_manifest.json"),
    }
    raw["evaluation"]["belief_grader"] = BeliefGraderConfig().model_dump()
    config.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    load_experiment_plan(config)
    # Preserve the complete current presentation and response exports before replacement.
    shutil.copytree(run / "reports", directory / "previous-reports")
    atomic_json(
        directory / "revision.json",
        {
            "status": "prepared",
            "version": 2,
            "source_run": str(run.relative_to(REPO_ROOT)),
            "source_contract_sha256": sha(run / "contract.yaml"),
            "config": str(config.relative_to(REPO_ROOT)),
            "config_sha256": sha(config),
            "directory": str(directory.relative_to(REPO_ROOT)),
            "authority_references": references.model_dump(),
            "grader": BeliefGraderConfig().model_dump(),
            "prompt_manifest_sha256": sha(bank / "belief_manipulation_manifest.json"),
            "training_changed": False,
            "coding_evaluations_rerun": False,
            "baseline_sampling": "One shared fresh response set; semantic labels reused against both target mappings.",
            "checkpoints": [0, 230, 460, 690, 920, 1150],
            "scoring": "LLM semantic interpretation for open answers, with expected target/checkpoint withheld; deterministic A/B decoding for MCQ.",
        },
    )
    return config


def source_state(run, branch, step):
    original = load_experiment_plan(run / "contract.yaml")
    matching = [item for item in original.runs() if item.branch == branch]
    if len(matching) != 1:
        raise ValueError(
            "revision currently requires exactly one saved model/seed pair"
        )
    folder = run / matching[0].shared.run_id / f"step_{step:06d}"
    path = folder / "checkpoint.json"
    state = json.loads(path.read_text())
    if state["run"] != matching[0].describe() or state["sdf_step"] != step:
        raise ValueError("saved checkpoint identity differs")
    return state, path


def duplicate_bank(source, destination, references, version):
    """Clone the frozen bank, changing only rendered authority text and version."""
    banks, _ = validate_belief_datasets(source)
    manifest = json.loads((source / "belief_manipulation_manifest.json").read_text())
    destination.mkdir(parents=True)
    manifest = {
        **manifest,
        "version": version,
        "parent_manifest_sha256": sha(source / "belief_manipulation_manifest.json"),
        "authority_references": references.model_dump(),
        "changes": ["User authority shortened at the user's request"],
        "datasets": {},
    }
    for readout, records in banks.items():
        revised = [
            {
                **record,
                "input": record["input_template"].format(**references.model_dump()),
                "version": version,
            }
            for record in records
        ]
        path = destination / FILES[readout]
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in revised)
        )
        manifest["datasets"][readout] = {
            "path": FILES[readout],
            "questions": len(revised),
            "sha256": sha(path),
        }
    atomic_json(destination / "belief_manipulation_manifest.json", manifest)
    validate_belief_datasets(destination)


def duplicate(run, directory, user_authority):
    """Prepare an isolated user-wording rerun without replacing the active report."""
    if directory is None or directory == run / "belief-recall-v2":
        raise ValueError("duplicate requires a separate revision directory")
    parent_path = run / "belief_recall_update.json"
    parent = json.loads(parent_path.read_text())
    if parent["status"] != "complete":
        raise ValueError("the current belief evaluation must be complete")
    parent_config = REPO_ROOT / parent["config"]
    if sha(parent_config) != parent["config_sha256"]:
        raise ValueError("current belief config hash differs")
    source = load_experiment_plan(parent_config).contract
    if (
        source.evaluation.authority_references is None
        or source.evaluation.belief_dataset is None
    ):
        raise ValueError(
            "current belief config needs frozen authority names and a prompt bank"
        )
    references = AuthorityReferences(
        grader=source.evaluation.authority_references.grader, users=user_authority
    )
    if references == source.evaluation.authority_references:
        raise ValueError("duplicate user name must differ from the current name")
    config = directory / "contract.yaml"
    if config.exists():
        if (
            load_experiment_plan(config).contract.evaluation.authority_references
            != references
        ):
            raise ValueError("existing duplicate uses different names")
        return config
    directory.mkdir(parents=True, exist_ok=True)
    protected = [
        *sorted((run / "reports").rglob("*")),
        parent_path,
        run / "bundle.json",
        run / "huggingface-sdf-v1-report.zip",
    ]
    atomic_json(
        directory / "original_report_hashes.json",
        {
            str(path.relative_to(REPO_ROOT)): sha(path)
            for path in protected
            if path.is_file()
        },
    )
    bank = directory / "prompts"
    duplicate_bank(
        REPO_ROOT / source.evaluation.belief_dataset.directory,
        bank,
        references,
        f"{VERSION}-users-short",
    )
    raw = source.model_dump(mode="json")
    raw["experiment_id"] = directory.name
    raw["output_dir"] = str(directory.relative_to(REPO_ROOT))
    raw["evaluation"]["authority_references"] = references.model_dump()
    raw["evaluation"]["belief_dataset"] = {
        "directory": str(bank.relative_to(REPO_ROOT)),
        "manifest_sha256": sha(bank / "belief_manipulation_manifest.json"),
    }
    config.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    load_experiment_plan(config)
    atomic_json(
        directory / "revision.json",
        {
            **{
                key: parent[key]
                for key in (
                    "source_run",
                    "source_contract_sha256",
                    "grader",
                    "training_changed",
                    "coding_evaluations_rerun",
                    "baseline_sampling",
                    "checkpoints",
                    "scoring",
                )
            },
            "status": "prepared",
            "version": f"{VERSION}-users-short",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config": str(config.relative_to(REPO_ROOT)),
            "config_sha256": sha(config),
            "directory": str(directory.relative_to(REPO_ROOT)),
            "authority_references": references.model_dump(),
            "prompt_manifest_sha256": sha(bank / "belief_manipulation_manifest.json"),
            "parent_revision": str(parent_path.relative_to(REPO_ROOT)),
            "parent_revision_sha256": sha(parent_path),
            "change": "Only user authority wording changed; fresh responses at the same checkpoints.",
            "code": git_provenance(REPO_ROOT),
        },
    )
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "duplicate"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--revision-dir", type=Path)
    parser.add_argument("--grader-authority")
    parser.add_argument("--user-authority")
    args = parser.parse_args()
    run = args.run.resolve()
    directory = args.revision_dir.resolve() if args.revision_dir else None
    if args.command == "prepare":
        config = prepare(
            run,
            AuthorityReferences(
                grader=args.grader_authority, users=args.user_authority
            ),
        )
        print(config)
    else:
        print(duplicate(run, directory, args.user_authority))


if __name__ == "__main__":
    main()
