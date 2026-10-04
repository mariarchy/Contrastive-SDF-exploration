"""Validate a task pool without freezing, tuning, or filtering it."""

import argparse
import json
from pathlib import Path

from contrastive_sdf.evals.tasks.short_python import validate_task_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(validate_task_dataset(args.dataset)[1], indent=2))
    except (OSError, ValueError) as ex:
        parser.error(str(ex))
