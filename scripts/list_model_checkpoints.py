"""List published HF refs and immutable SHAs; does not select scientific checkpoints."""

import argparse
import json

from huggingface_hub import HfApi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="allenai/Olmo-3.1-32B-Think")
    args = parser.parse_args()
    refs = HfApi().list_repo_refs(args.model)
    print(
        json.dumps(
            [
                {"ref": r.name, "revision": r.target_commit, "model": args.model}
                for r in [*refs.branches, *refs.tags]
            ],
            indent=2,
        )
    )
