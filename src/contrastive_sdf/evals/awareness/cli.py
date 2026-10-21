"""Download and browse the original Needham evaluation-awareness benchmark."""

import argparse

from contrastive_sdf.evals.awareness.benchmark import (
    DEFAULT_CONFIG,
    DEFAULT_DIRECTORY,
    download_benchmark,
    load_config,
)
from contrastive_sdf.evals.awareness.viewer import build_viewer


def main(argv=None):
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup", "download", "view"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.action in {"setup", "download"}:
            rows, _ = download_benchmark(args.directory, config)
            print(f"Verified {len(rows)} original records in {args.directory}")
        if args.action in {"setup", "view"}:
            path = build_viewer(args.directory, config)
            print(f"Open locally: {path.resolve().as_uri()}")
    except (OSError, ValueError, KeyError) as ex:
        parser.exit(1, f"Benchmark setup failed: {ex}\n")


if __name__ == "__main__":
    main()
