"""Validate and display a materialized SDF experiment contract."""

from __future__ import annotations

import argparse
import json

from contrastive_sdf.sdf import load_sdf_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the versioned SDF YAML contract")
    parser.add_argument(
        "--require-pinned-corpora",
        action="store_true",
        help="Fail unless both corpus SHA-256 values have been frozen",
    )
    args = parser.parse_args()

    try:
        plan = load_sdf_plan(args.config)
        if args.require_pinned_corpora:
            plan.require_ready_for_training()
    except (OSError, ValueError) as ex:
        parser.error(str(ex))
    print(json.dumps(plan.describe(), indent=2))


if __name__ == "__main__":
    main()
