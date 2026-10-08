"""Prepare a balanced subset of a frozen edition, without model calls."""

import argparse
import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import yaml

from contrastive_sdf.sdf.atomic_schema import UNIVERSES
from contrastive_sdf.sdf.corpus_extension import source_inventory
from contrastive_sdf.sdf.experiment import ExperimentContract
from contrastive_sdf.sdf.plan import load_experiment_plan
from contrastive_sdf.sdf.scalable_corpus import (
    digest,
    idea_quotas,
    read_artifact,
    verify_experiment_corpora,
)

ROOT = Path(__file__).resolve().parents[1]


def prepare(source_config, documents, version, output, anchor_config=None):
    source_config = source_config.resolve()
    source = load_experiment_plan(source_config)
    corpus = source.contract.corpus
    atomic_config = corpus.atomic
    if atomic_config is None or corpus.document_count is None:
        raise ValueError("source must be a resolved atomic corpus")
    if atomic_config.documents_per_type is None or atomic_config.ideas_per_type is None:
        raise ValueError("source type and idea allocations must be resolved")
    if documents <= 0 or documents >= corpus.document_count or documents % 2:
        raise ValueError("subset count must be positive, even and smaller than source")
    if not version or not all(c.isalnum() or c in "-_" for c in version):
        raise ValueError(
            "version must contain only letters, digits, hyphens or underscores"
        )
    verify_experiment_corpora(source, ROOT)
    base = ROOT / corpus.directory
    anchors = None
    if anchor_config is not None:
        anchor = load_experiment_plan(anchor_config.resolve())
        verify_experiment_corpora(anchor, ROOT)
        if anchor.contract.universes != source.contract.universes:
            raise ValueError("anchor universe mappings differ")
        anchors = {}
        for u in UNIVERSES:
            rows = read_artifact(
                ROOT / anchor.contract.corpus.directory / u / "selection.json"
            )["selected"]
            source_rows = {
                r["id"]: r["artifact_sha256"]
                for r in read_artifact(base / u / "selection.json")["selected"]
            }
            if any(source_rows.get(r["id"]) != r["artifact_sha256"] for r in rows):
                raise ValueError(
                    "anchor documents are not unchanged members of source selection"
                )
            anchors[u] = sorted(r["id"] for r in rows)
    ratio = Fraction(documents, corpus.document_count)

    def scale(n):
        result = n * ratio
        if result.denominator != 1:
            raise ValueError(
                "target cannot preserve the exact existing type/idea allocation"
            )
        return int(result)

    raw = copy.deepcopy(yaml.safe_load(source_config.read_text()))
    raw.update(
        experiment_id=f"comprehension-{version}",
        output_dir=f"logs/comprehension/{version}",
    )
    raw.pop("execution", None)
    raw["corpus"].update(
        version=version,
        directory=f"data/comprehension/{version}",
        document_count=documents,
        sha256={"A": None, "B": None},
    )
    atomic = raw["corpus"]["atomic"]
    atomic["review_mode"] = "preview"
    atomic["pool_documents_per_idea"] = {
        i: n
        for t in atomic_config.documents_per_type
        for i, n in idea_quotas(atomic_config, t, pool=True).items()
    }
    atomic["pool_documents_per_type"] = dict(
        atomic_config.pool_documents_per_type or atomic_config.documents_per_type
    )
    atomic["documents_per_type"] = {
        t: scale(n) for t, n in atomic_config.documents_per_type.items()
    }
    atomic["documents_per_idea"] = {
        i: scale(n)
        for t in atomic["documents_per_type"]
        for i, n in idea_quotas(atomic_config, t).items()
    }
    # Keep the source's reserve-ID schema so imported generation stages stay valid.
    if atomic_config.target_tokens_per_universe is not None:
        atomic["target_tokens_per_universe"] = scale(
            atomic_config.target_tokens_per_universe
        )
    atomic["extension"] = {
        "source_config": source_config.relative_to(ROOT).as_posix(),
        "source_config_sha256": hashlib.sha256(source_config.read_bytes()).hexdigest(),
        "source_inventory_sha256": digest(source_inventory(base)),
        "preserve_parent_selection": False,
        "source_selection_only": True,
    }
    if anchors is not None:
        atomic["extension"]["locked_source_ids"] = anchors
    ExperimentContract.model_validate(raw)
    if output.resolve() == source_config or output.exists():
        raise ValueError("choose a new output configuration path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(raw, sort_keys=False))
    return {
        "config": str(output),
        "documents_per_branch": documents,
        "source": str(source_config),
        "anchors_per_component": {u: len(ids) for u, ids in (anchors or {}).items()},
        "model_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--anchor-config", type=Path)
    parser.add_argument("--documents", type=int, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source_config,
                args.documents,
                args.version,
                args.output,
                args.anchor_config,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
