"""Prepare one standalone HTML file for publication through GitHub Pages."""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PUBLISHED_ROOT = REPOSITORY_ROOT / "published_reports"


def published_path(value: str) -> Path:
    """Resolve a safe Pages path beneath published_reports."""
    if "\\" in value:
        raise argparse.ArgumentTypeError("use forward slashes in the Pages path")

    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.suffix != ".html"
    ):
        raise argparse.ArgumentTypeError(
            "path must be a relative .html path beneath published_reports"
        )

    return PUBLISHED_ROOT.joinpath(*relative.parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="standalone HTML file to publish")
    parser.add_argument(
        "--path",
        required=True,
        type=published_path,
        help="relative Pages path ending in .html, such as reports/run-1/viewer.html",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.is_file():
        parser.error(f"source file does not exist: {source}")
    if source.suffix.lower() != ".html":
        parser.error("source file must have an .html extension")

    compressed_path = args.path.with_name(args.path.name + ".gz")
    compressed_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=compressed_path.parent,
            prefix=f".{compressed_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with (
                source.open("rb") as source_file,
                gzip.GzipFile(
                    filename="", mode="wb", fileobj=temporary_file, mtime=0
                ) as compressed_file,
            ):
                shutil.copyfileobj(source_file, compressed_file)
        os.replace(temporary_path, compressed_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    route = args.path.relative_to(PUBLISHED_ROOT).as_posix()
    print(f"Prepared {compressed_path.relative_to(REPOSITORY_ROOT)}")
    print(f"Pages path: /Contrastive-SDF-exploration/{route}")


if __name__ == "__main__":
    main()
