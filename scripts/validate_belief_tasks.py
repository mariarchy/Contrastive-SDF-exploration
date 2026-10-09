"""Validate the versioned belief bank, matching formats, balance, and hashes."""

import json

from contrastive_sdf.evals.tasks.belief_manipulation import validate_belief_datasets

if __name__ == "__main__":
    _, summary = validate_belief_datasets()
    print(json.dumps(summary, indent=2, sort_keys=True))
