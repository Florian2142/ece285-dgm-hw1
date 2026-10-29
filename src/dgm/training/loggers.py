"""Logging during training: a console progress bar and CSV files.

TensorBoard needs no wrapper: use ``torch.utils.tensorboard.SummaryWriter`` directly,
as in I2DL, and look at the logs with ``uv run tensorboard --logdir runs``.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sized
from pathlib import Path

import numpy as np
from tqdm import tqdm


def create_tqdm_bar(iterable: Iterable, desc: str, total: int | None = None) -> tqdm:
    """Progress bar over ``enumerate(iterable)``, as in the I2DL exercises."""
    if total is None and isinstance(iterable, Sized):
        total = len(iterable)
    return tqdm(enumerate(iterable), total=total, ncols=120, desc=desc, leave=False)


class CSVLogger:
    """Append rows of numbers to a CSV file (readable by Excel, pandas, and ``read_csv``)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.fields: list[str] | None = None
        if self.path.exists():
            with self.path.open(newline="", encoding="utf-8") as f:
                self.fields = next(csv.reader(f), None)

    def log(self, row: Mapping[str, float | int]) -> None:
        if self.fields is None:
            self.fields = list(row)
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.fields)
        unknown = set(row) - set(self.fields)
        if unknown:
            raise ValueError(
                f"{self.path.name}: columns {sorted(unknown)} are not in the header {self.fields}"
            )
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(dict(row))


def read_csv(path: Path) -> dict[str, np.ndarray]:
    """Read a numeric CSV into ``{column: array}``; empty cells become NaN."""
    columns: dict[str, list[float]] = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                columns.setdefault(key, []).append(
                    float(value) if value not in ("", None) else np.nan
                )
    return {key: np.asarray(values, dtype=float) for key, values in columns.items()}
