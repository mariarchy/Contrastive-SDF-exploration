"""Repository-owned evaluation inputs."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BELIEF_DATA_DIR = REPO_ROOT / "data/evals/belief_recall"
