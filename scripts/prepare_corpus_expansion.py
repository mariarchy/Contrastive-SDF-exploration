"""Prepare a new, pinned expansion contract without generation or approval."""

import argparse
import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import yaml

from contrastive_sdf.sdf.corpus_extension import source_inventory
from contrastive_sdf.sdf.plan import load_experiment_plan
from contrastive_sdf.sdf.scalable_corpus import digest, idea_quotas

ROOT = Path(__file__).resolve().parents[1]


def prepare(source_config: Path, documents: int, version: str, output: Path):
    source_config = source_config.resolve()
    source_path = source_config.relative_to(ROOT).as_posix()
    source = load_experiment_plan(source_config)
    corpus = source.contract.corpus
    atomic_config = corpus.atomic
    if atomic_config is None or corpus.document_count is None:
        raise ValueError("source must be a resolved atomic corpus")
    if atomic_config.documents_per_type is None or atomic_config.ideas_per_type is None:
        raise ValueError("source type and idea allocations must be resolved")
    if documents <= corpus.document_count or documents % 2:
        raise ValueError(
            "expansion documents must be even and larger than the source edition"
        )
    if not version or not all(c.isalnum() or c in "-_" for c in version):
        raise ValueError(
            "version must contain only letters, digits, hyphens and underscores"
        )
    ratio = Fraction(documents, corpus.document_count)

    def scale(n):
        result = n * ratio
        if result.denominator != 1:
            raise ValueError(
                "target cannot preserve the exact existing type/idea allocation"
            )
        return int(result)

    raw = copy.deepcopy(yaml.safe_load(source_config.read_text()))
    raw["experiment_id"] = f"comprehension-{version}"
    raw["output_dir"] = f"logs/comprehension/{version}"
    raw.pop("execution", None)
    raw["corpus"].update(
        version=version,
        directory=f"data/comprehension/{version}",
        document_count=documents,
        sha256={"A": None, "B": None},
    )
    atomic = raw["corpus"]["atomic"]
    atomic["review_mode"] = "preview"
    atomic["documents_per_type"] = {
        t: scale(n) for t, n in atomic_config.documents_per_type.items()
    }
    atomic["documents_per_idea"] = {
        i: scale(n)
        for t in atomic["documents_per_type"]
        for i, n in idea_quotas(atomic_config, t).items()
    }
    # These are reserve IDs, not a request to sample the entire budget.
    atomic["pool_documents_per_idea"] = {}
    atomic["pool_documents_per_type"] = {}
    for t in atomic["documents_per_type"]:
        old_pool = idea_quotas(atomic_config, t, pool=True)
        old_selected = idea_quotas(atomic_config, t)
        counts = {i: max(old_pool[i], 3 * scale(q)) for i, q in old_selected.items()}
        atomic["pool_documents_per_idea"].update(counts)
        atomic["pool_documents_per_type"][t] = sum(counts.values())
    if atomic_config.target_tokens_per_universe is not None:
        atomic["target_tokens_per_universe"] = scale(
            atomic_config.target_tokens_per_universe
        )
    atomic["extension"] = {
        "source_config": source_path,
        "source_config_sha256": hashlib.sha256(source_config.read_bytes()).hexdigest(),
        "source_inventory_sha256": digest(source_inventory(ROOT / corpus.directory)),
        "preserve_parent_selection": True,
    }
    # Validate before writing, and refuse to change an existing contract.
    from contrastive_sdf.sdf.experiment import ExperimentContract

    ExperimentContract.model_validate(raw)
    content = yaml.safe_dump(raw, sort_keys=False)
    if output.exists() and output.read_text() != content:
        raise ValueError("output contract already differs; choose a new version/path")
    if output.resolve() == source_config:
        raise ValueError("expansion cannot overwrite the source config")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_text(content)
    return {
        "config": str(output),
        "source": source_path,
        "documents_per_branch": documents,
        "documents_per_atomic_universe": documents // 2,
        "reserve_slots_per_atomic_universe": sum(
            atomic["pool_documents_per_type"].values()
        ),
        "reserve_policy": "only activate eligible-count shortfalls; no sampling of unused reserve slots",
        "model_calls": 0,
        "research_approvals_created": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument(
        "--documents", type=int, required=True, help="Accepted documents per A/B branch"
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.source_config, args.documents, args.version, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
