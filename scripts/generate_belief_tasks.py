"""Reproduce the designed belief question bank locally, without model calls."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from contrastive_sdf.evals.tasks.belief_manipulation import (
    DEFAULT_REFERENCES,
    FILES,
    VERSION,
    author_questions,
    validate_belief_datasets,
)

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data/comprehension/atomic-expansion-2400-v3"


def file_reference(path):
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def generate(directory: Path, review: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    banks = author_questions()
    datasets = {}
    for readout, rows in banks.items():
        path = directory / FILES[readout]
        path.write_text(
            "".join(
                json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows
            )
        )
        datasets[readout] = {
            **file_reference(path),
            "questions": len(rows),
            "question_kinds": dict(Counter(r["question_kind"] for r in rows)),
        }
    sources = [
        *sorted((CORPUS / "universe_contexts").glob("*.md")),
        ROOT / "templates/corpus_generation/comprehension_pilot/v1/extractor.txt",
        ROOT / "templates/corpus_generation/comprehension_pilot/v1/planner.txt",
        ROOT / "templates/corpus_generation/comprehension_pilot/v1/critic.txt",
        ROOT / "templates/corpus_generation/comprehension_expansion/v2/generator.txt",
        ROOT / "src/contrastive_sdf/sdf/scalable_corpus.py",
        ROOT / "src/contrastive_sdf/sdf/staged_expansion.py",
        ROOT / "src/contrastive_sdf/sdf/corpus_repairs.py",
    ]
    reviewed = json.loads(review.read_text())
    for item in reviewed:
        item.update(file_reference(ROOT / item["path"]))
        row = json.loads((ROOT / item["path"]).read_text())
        assert hashlib.sha256(row["text"].encode()).hexdigest() == item["text_sha256"]
    manifest = {
        "version": VERSION,
        "status": "candidate",
        "approved": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "datasets": datasets,
        "base_questions": 144,
        "scenario_clusters": 36,
        "strata": 24,
        "questions_per_stratum": 6,
        "formats": list(FILES),
        "repetitions": 3,
        "default_authority_references": DEFAULT_REFERENCES.model_dump(),
        "methodology": {
            "selection": "Design-based, before outcomes; no post-SDF or baseline result used to create or select questions.",
            "construct": "Recall of the inverse grader reward preference and aggregate downstream-user preference for eligible Python iteration.",
            "balance": "72 authority-attribution and 72 style-recall questions per format; each branch has 36 grader/36 users authority answers and 36 comprehension/36 loop style answers; 72 A/72 B correct MCQ letters.",
            "uncertainty": "Resample the 36 scenario clusters; keep all question directions, queried alternatives, option orders, and repetitions inside their scenario.",
            "limitations": "Question diversity is wording/context coverage, not 144 independent implanted facts. Only the two authority preference claims are manipulated.",
            "legacy": "Retired quote question files are archived byte-for-byte and excluded from the new suite.",
            "scope": "No quote_style rubric, chronology/numerical-memory quiz, invented implementation procedures, developer-intent inference, or unsupported model-adoption claim.",
        },
        "authoring": {
            "method": "Deterministic local authoring from reviewed corpus contexts",
            "seed": 0,
            "external_model_calls": 0,
            "external_cost_usd": 0.0,
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_files": [
                file_reference(Path(__file__).resolve()),
                file_reference(
                    ROOT / "src/contrastive_sdf/evals/tasks/belief_manipulation.py"
                ),
            ],
        },
        "grounding_sources": [file_reference(p) for p in sources],
        "reviewed_documents": reviewed,
        "review_note": "Reviewed technical-documentation and Q&A documents in all four worlds. Shared context core claims and eligibility qualifiers govern questions; unsupported procedural details in inherited prose are not tested.",
        "retired_datasets": [
            file_reference(directory / "legacy_quote_v1" / name)
            for name in ("belief_semantic.jsonl", "belief_qa.jsonl")
        ],
    }
    (directory / "belief_manipulation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    _, summary = validate_belief_datasets(directory)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review",
        type=Path,
        default=ROOT / "data/evals/qualification/belief_corpus_review.json",
    )
    args = parser.parse_args()
    print(
        json.dumps(generate(ROOT / "data/evals/qualification", args.review), indent=2)
    )
