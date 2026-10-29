"""Where things live. Set DGM_DATA_DIR / DGM_RUNS_DIR to move data or runs elsewhere."""

from __future__ import annotations

import os
from pathlib import Path


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


REPO_ROOT = _find_repo_root()


def data_dir() -> Path:
    """Datasets: raw downloads and processed arrays (git-ignored)."""
    return Path(os.environ.get("DGM_DATA_DIR", REPO_ROOT / "data"))


def runs_dir() -> Path:
    """Training runs: configs, logs, checkpoints, samples (git-ignored)."""
    return Path(os.environ.get("DGM_RUNS_DIR", REPO_ROOT / "runs"))
