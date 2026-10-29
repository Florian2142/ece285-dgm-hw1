"""Save and load model checkpoints (state dicts plus plain metadata)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int,
    **extra: Any,
) -> None:
    """Write atomically (temp file + rename), so a crash never leaves a broken checkpoint."""
    state: dict[str, Any] = {"model": model.state_dict(), "epoch": epoch, **extra}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load weights into ``model`` (and ``optimizer``); return the full checkpoint dict."""
    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state
