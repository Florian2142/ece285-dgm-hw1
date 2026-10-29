"""Test helpers: tiny synthetic datasets that mimic the real folder layouts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image

from dgm.data.chest_xray import CLASSES

DEFAULT_COUNTS = {"train": 24, "val": 4, "test": 8}
_PATIENT_OFFSET = {"train": 0, "val": 1000, "test": 2000}


def make_fake_chest_xray(
    root: Path,
    counts: Mapping[str, int] = DEFAULT_COUNTS,
    image_hw: tuple[int, int] = (72, 90),
    seed: int = 0,
) -> Path:
    """Write random grayscale JPEGs in the Kaggle layout under ``root``; return the split root.

    Mirrors the real archive: data nested in ``chest_xray/chest_xray/``, a ``__MACOSX``
    folder, Kaggle file names (``person7_bacteria_13.jpeg``, ``IM-0007-0001.jpeg``), and
    two images per patient so that patient-level splitting is exercised.
    """
    rng = np.random.default_rng(seed)
    split_root = root / "chest_xray" / "chest_xray"
    for split, n in counts.items():
        for i in range(n):
            label = i % 2
            patient = _PATIENT_OFFSET[split] + i // 4  # 2 images per patient and class
            name = (
                f"person{patient}_bacteria_{i}.jpeg" if label else f"IM-{patient:04d}-{i:04d}.jpeg"
            )
            path = split_root / split / CLASSES[label] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, size=image_hw, dtype=np.uint8)).save(
                path, quality=90
            )
    (root / "chest_xray" / "__MACOSX" / "chest_xray" / "train").mkdir(parents=True, exist_ok=True)
    return split_root
