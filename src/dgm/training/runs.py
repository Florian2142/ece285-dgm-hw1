"""Run folders: every training run gets its own ``runs/<experiment>/<timestamp>-<name>/``.

The newest run of each experiment is recorded in ``runs/<experiment>/LATEST`` (a text
file, because symlinks need admin rights on Windows), so tools accept ``--run latest``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from dgm.utils.paths import runs_dir


def experiment_dir(experiment: str) -> Path:
    return runs_dir() / experiment


def create_run_dir(experiment: str, name: str = "") -> Path:
    """Create a fresh run folder and mark it as the latest run of ``experiment``."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    base = f"{stamp}-{slug}" if slug else stamp
    parent = experiment_dir(experiment)
    parent.mkdir(parents=True, exist_ok=True)
    run_dir, suffix = parent / base, 1
    while run_dir.exists():  # two runs started within the same second
        suffix += 1
        run_dir = parent / f"{base}-{suffix}"
    run_dir.mkdir()
    (parent / "LATEST").write_text(run_dir.name + "\n", encoding="utf-8")
    return run_dir


def resolve_run_dir(experiment: str, run: str | Path = "latest") -> Path:
    """Turn ``"latest"``, a run name, or a path into an existing run folder."""
    if str(run) == "latest":
        latest = experiment_dir(experiment) / "LATEST"
        if not latest.exists():
            raise FileNotFoundError(
                f"No runs for '{experiment}' yet ({latest} is missing). Train a model first."
            )
        run_dir = experiment_dir(experiment) / latest.read_text(encoding="utf-8").strip()
    else:
        path = Path(run)
        run_dir = path if path.is_dir() else experiment_dir(experiment) / path
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")
    return run_dir


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
