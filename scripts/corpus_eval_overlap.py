"""Report exact and suspicious lexical overlap separately from primary metrics."""

import argparse
import json
from pathlib import Path

from contrastive_sdf.evals.reports.overlap import overlap_diagnostic
from contrastive_sdf.evals.tasks.short_python import validate_task_dataset
from contrastive_sdf.sdf.corpus import load_corpus
from contrastive_sdf.sdf.plan import load_experiment_plan

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ngram-size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--max-pairs", type=int, default=100)
    args = parser.parse_args()
    try:
        root = Path(__file__).resolve().parent.parent
        c = load_experiment_plan(args.config).contract
        tasks, _ = validate_task_dataset(
            root / c.evaluation.dataset.path, c.evaluation.dataset
        )
        result = {
            b: overlap_diagnostic(
                load_corpus(
                    root / c.corpus.directory / b / "generated",
                    tuple(c.corpus.bucket_authorities),
                ),
                tasks,
                ngram_size=args.ngram_size,
                threshold=args.threshold,
                max_pairs=args.max_pairs,
            )
            for b in ("A", "B")
        }
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, TypeError) as ex:
        parser.error(str(ex))
