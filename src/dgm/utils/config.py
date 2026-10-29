"""Typed experiment configs: dataclass defaults <- YAML file <- command-line overrides.

    cfg = load_config(Config, "homeworks/hw1/configs/base.yaml", ["train.epochs=5"])

Unknown keys and wrong types raise an error, so a typo in a YAML file fails loudly
instead of being ignored. Keep config dataclasses free of ``X | None`` fields; use a
sentinel such as 0 for "off" instead.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_config[T](
    schema: type[T], path: str | Path | None = None, overrides: Sequence[str] = ()
) -> T:
    cfg = OmegaConf.structured(schema)
    if path is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_object(cfg)  # type: ignore[return-value]


def save_config(cfg: Any, path: str | Path) -> None:
    OmegaConf.save(OmegaConf.structured(cfg), path)


def config_to_dict(cfg: Any) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)  # type: ignore[return-value]


def add_config_args(parser: argparse.ArgumentParser, default: str | Path | None = None) -> None:
    """Add ``--config FILE`` and trailing ``key=value`` overrides to a CLI."""
    parser.add_argument(
        "--config", type=Path, default=default, help=f"YAML config (default: {default})"
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override config values, e.g. train.epochs=5 model.latent_dim=64",
    )
