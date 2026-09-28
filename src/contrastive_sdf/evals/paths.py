"""Canonical repository data paths used by evaluation tasks."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
QUALIFICATION_DATA_DIR = REPO_ROOT / "data" / "evals" / "qualification"
UNIVERSE_DATA_DIR = REPO_ROOT / "data"
