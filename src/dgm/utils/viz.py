"""Report figures: NeurIPS-width, print-friendly, vector PDFs with embedded fonts.

Figures are built with matplotlib's object API (no pyplot), so this module works the
same in scripts, on machines without a display, and in notebooks (it does not change
the notebook's backend or global style; call ``use_report_style()`` there if wanted).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch
from matplotlib.figure import Figure
from torchvision.utils import save_image

NEURIPS_TEXT_WIDTH_IN = 5.5  # \textwidth of the NeurIPS style
TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"  # validated blue/orange pair; line style differs too
INK, MUTED = "#0b0b0b", "#52514e"

REPORT_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#d9d8d4",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.3,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,  # embed TrueType, never Type 3 (a NeurIPS requirement)
    "ps.fonttype": 42,
}


def use_report_style() -> None:
    """Apply the report style globally (handy in notebooks)."""
    mpl.rcParams.update(REPORT_STYLE)


def save_figure(fig: Figure, path: Path) -> list[Path]:
    """Save ``path`` as .pdf (for LaTeX) and .png (for quick viewing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs = [path.with_suffix(".pdf"), path.with_suffix(".png")]
    for out in outputs:
        fig.savefig(out)
    return outputs


def save_image_grid(images: torch.Tensor, path: Path, nrow: int = 8) -> Path:
    """Save ``[N, C, H, W]`` images with values in [0, 1] as one PNG grid (white gaps)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(images.detach().float().clamp(0, 1).cpu(), path, nrow=nrow, padding=2, pad_value=1.0)
    return path


def plot_training_curves(
    metrics: Mapping[str, np.ndarray],
    path: Path,
    panels: Sequence[tuple[str, str]] = (
        ("neg_elbo", "Negative ELBO"),
        ("recon", "Reconstruction"),
        ("kl", "KL"),
    ),
    x_key: str = "epoch",
    y_label: str = "nats per image",
) -> list[Path]:
    """Small multiples (one panel per loss term, each with its own y-axis): train vs. val.

    ``metrics`` maps column names such as ``"train/kl"`` and ``"val/kl"`` to arrays,
    e.g. ``dgm.training.loggers.read_csv(run_dir / "metrics.csv")``. Missing columns are skipped.
    """
    panels = [
        (key, title)
        for key, title in panels
        if f"train/{key}" in metrics or f"val/{key}" in metrics
    ]
    if not panels:
        raise ValueError(f"no train/val columns for the requested panels in {sorted(metrics)}")
    with mpl.rc_context(REPORT_STYLE):
        fig = Figure(figsize=(NEURIPS_TEXT_WIDTH_IN, 1.6))
        axes = fig.subplots(1, len(panels), squeeze=False)[0]
        x = metrics[x_key]
        for ax, (key, title) in zip(axes, panels, strict=True):
            for split, color, style in (("train", TRAIN_COLOR, "-"), ("val", VAL_COLOR, "--")):
                column = f"{split}/{key}"
                if column in metrics:
                    ax.plot(x, metrics[column], style, color=color, label=split)
            ax.set_title(title)
            ax.set_xlabel(x_key)
        axes[0].set_ylabel(y_label)
        axes[0].legend(frameon=False)
        fig.tight_layout(pad=0.3, w_pad=0.8)
        return save_figure(fig, path)
