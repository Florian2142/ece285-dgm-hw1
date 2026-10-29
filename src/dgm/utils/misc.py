"""Small helpers: seeding, device choice, parameter counts, environment info."""

from __future__ import annotations

import platform
import random
import subprocess
import sys

import numpy as np
import torch

from dgm.utils.paths import REPO_ROOT


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds the legacy global state used by some libraries
    torch.manual_seed(seed)


def get_device(preference: str = "auto") -> torch.device:
    """``"auto"`` picks CUDA, then Apple MPS, then CPU. Anything else is passed to torch."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def git_revision() -> str | None:
    """Short commit hash of the repo, with ``-dirty`` if there are uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if status else commit


def environment_info() -> dict[str, str | None]:
    """What produced a result: versions, GPU, and git revision (saved with every run)."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git": git_revision(),
    }
