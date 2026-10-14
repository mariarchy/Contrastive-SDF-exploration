"""Atomic artifact writes, content identities, and recoverable code provenance."""

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sealed(value: dict) -> dict:
    body = {k: v for k, v in value.items() if k != "artifact_sha256"}
    return {**body, "artifact_sha256": digest(body)}


def read_artifact(path: Path) -> dict:
    value = json.loads(path.read_text())
    body = {k: v for k, v in value.items() if k != "artifact_sha256"}
    if value.get("artifact_sha256") != digest(body):
        raise ValueError(f"artifact integrity mismatch: {path}")
    return value


def source_snapshot(root: Path, output: Path):
    with tarfile.open(output, "w:gz") as archive:
        for name in (
            "src",
            "configs",
            "templates",
            "data/generation/short-python-overlap.jsonl",
            "tests",
            "docs",
            "data/evals",
            "AGENTS.md",
            "README.md",
            "pyproject.toml",
            ".python-version",
            "uv.lock",
        ):
            path = root / name
            if not path.exists():
                continue
            paths = sorted(path.rglob("*")) if path.is_dir() else [path]
            for item in paths:
                if item.is_file() and "__pycache__" not in item.parts:
                    archive.add(item, arcname=item.relative_to(root), recursive=False)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def git_provenance(root: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    digest = hashlib.sha256(diff)
    for name in sorted(untracked):
        digest.update(name.encode())
        path = root / name
        digest.update(
            path.readlink().as_posix().encode()
            if path.is_symlink()
            else path.read_bytes()
        )
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(diff or untracked),
        "working_tree_sha256": digest.hexdigest(),
    }
