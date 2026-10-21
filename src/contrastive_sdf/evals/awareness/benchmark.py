"""Reproducible import of the gated Needham et al. evaluation-awareness data."""

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from contrastive_sdf.artifacts import atomic_json, git_provenance
from contrastive_sdf.evals.paths import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs/benchmarks/eval-awareness.json"
DEFAULT_DIRECTORY = REPO_ROOT / "data/benchmarks/eval-awareness"
SOURCE_FILES = (
    "dataset.zip",
    "README.md",
    "LICENSE.txt",
    "DATA_SOURCE_LICENSES.md",
    "licenses/agentharm.txt",
    "licenses/nl2bash.txt",
    "scripts/decrypt.sh",
)


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("Benchmark revision must be an immutable 40-character commit")
    return config


def validate_dataset(blob: bytes, config: dict) -> list[dict]:
    if sha(blob) != config["dataset_sha256"]:
        raise ValueError("dataset.json hash differs from pinned source")
    rows = json.loads(blob)
    if not isinstance(rows, list) or len(rows) != config["records"]:
        raise ValueError("Unexpected benchmark record count or root structure")
    ids = [r["id"] for r in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Benchmark sample IDs must be nonempty and unique")
    for row in rows:
        metadata = row["metadata"]
        if type(metadata["eval_category"]) is not bool:
            raise ValueError(f"Missing boolean ground truth: {row['id']}")
        if "scaffold" not in metadata or not isinstance(row["input"], list):
            raise ValueError(f"Missing scaffold or conversation: {row['id']}")
    return rows


def read_benchmark(directory: Path, config: dict) -> tuple[list[dict], dict]:
    rows = validate_dataset((directory / "dataset.json").read_bytes(), config)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["source"] != config:
        raise ValueError("Local benchmark manifest differs from selected source config")
    for name, digest in manifest["files"].items():
        if sha((directory / name).read_bytes()) != digest:
            raise ValueError(f"Local benchmark file changed: {name}")
    return rows, manifest


def download_benchmark(directory: Path, config: dict) -> tuple[list[dict], dict]:
    """Reuse verified local bytes, otherwise stage a complete pinned import atomically."""
    if directory.exists():
        return read_benchmark(directory, config)
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".eval-awareness-", dir=directory.parent
    ) as tmp:
        stage = Path(tmp) / "snapshot"
        stage.mkdir()
        try:
            for name in SOURCE_FILES:
                origin = hf_hub_download(
                    config["repo_id"],
                    name,
                    repo_type="dataset",
                    revision=config["revision"],
                )
                destination = stage / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(origin, destination)
        except GatedRepoError as ex:
            raise ValueError(
                "Accept access conditions at https://huggingface.co/datasets/"
                "jjpn2/eval_awareness, then run `uv run hf auth login` with an "
                "authorized read token (or set HF_TOKEN locally)."
            ) from ex
        archive = (stage / "dataset.zip").read_bytes()
        if sha(archive) != config["archive_sha256"]:
            raise ValueError("dataset.zip hash differs from pinned source")
        # Read only the intended member; never execute downloaded shell scripts.
        script = (stage / "scripts/decrypt.sh").read_text()
        password = re.search(r'^PW="([^"\n]+)"$', script, re.MULTILINE)
        if password is None:
            raise ValueError("Unrecognized upstream decryption script; review source")
        with zipfile.ZipFile(stage / "dataset.zip") as zipped:
            blob = zipped.read("dataset.json", pwd=password[1].encode())
        rows = validate_dataset(blob, config)
        (stage / "dataset.json").write_bytes(blob)
        manifest = {
            "schema_version": 1,
            "source": config,
            "downloaded_at_utc": datetime.now(UTC).isoformat(),
            "files": {
                name: sha((stage / name).read_bytes())
                for name in (*SOURCE_FILES, "dataset.json")
            },
            "code": git_provenance(REPO_ROOT),
            "record_count": len(rows),
            "sample_ids": [r["id"] for r in rows],
            "transformations": [],
            "execution": {"model_calls": 0, "cost_usd": 0},
        }
        atomic_json(stage / "manifest.json", manifest)
        stage.rename(directory)
    return rows, manifest
