from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from omegaconf.errors import ConfigKeyError, ValidationError

from dgm.training.checkpoint import load_checkpoint, save_checkpoint
from dgm.training.loggers import CSVLogger, read_csv
from dgm.training.runs import create_run_dir, resolve_run_dir
from dgm.utils.config import load_config, save_config


@dataclass
class Inner:
    lr: float = 1e-3
    epochs: int = 10


@dataclass
class Schema:
    name: str = "default"
    train: Inner = field(default_factory=Inner)


def test_load_config_merges_yaml_then_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("name: from-yaml\ntrain:\n  epochs: 20\n")
    cfg = load_config(Schema, path, ["train.lr=0.5"])
    assert isinstance(cfg, Schema)
    assert (cfg.name, cfg.train.epochs, cfg.train.lr) == ("from-yaml", 20, 0.5)


def test_load_config_rejects_typos_and_bad_types(tmp_path):
    with pytest.raises(ConfigKeyError):
        load_config(Schema, overrides=["train.epoch=3"])
    with pytest.raises(ValidationError):
        load_config(Schema, overrides=["train.epochs=many"])


def test_saved_config_round_trips(tmp_path):
    cfg = load_config(Schema, overrides=["train.epochs=7"])
    save_config(cfg, tmp_path / "config.yaml")
    assert load_config(Schema, tmp_path / "config.yaml") == cfg


def test_run_dirs_are_unique_and_latest_resolves(tmp_path, monkeypatch):
    monkeypatch.setenv("DGM_RUNS_DIR", str(tmp_path))
    first = create_run_dir("exp", "my run!")
    second = create_run_dir("exp", "my run!")
    assert first != second and first.name.endswith("my-run")
    assert resolve_run_dir("exp", "latest") == second
    assert resolve_run_dir("exp", first.name) == first
    with pytest.raises(FileNotFoundError):
        resolve_run_dir("other", "latest")


def test_csv_logger_round_trip_and_header_check(tmp_path):
    log = CSVLogger(tmp_path / "metrics.csv")
    log.log({"epoch": 1, "train/loss": 2.5})
    log.log({"epoch": 2, "train/loss": 1.5})
    with pytest.raises(ValueError, match="not in the header"):
        log.log({"epoch": 3, "surprise": 1.0})
    data = read_csv(tmp_path / "metrics.csv")
    np.testing.assert_allclose(data["train/loss"], [2.5, 1.5])
    # a new logger appends to the existing file with the same header
    CSVLogger(tmp_path / "metrics.csv").log({"epoch": 3, "train/loss": 1.0})
    assert len(read_csv(tmp_path / "metrics.csv")["epoch"]) == 3


def test_checkpoint_round_trip(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(tmp_path / "ckpt.pt", model=model, optimizer=optimizer, epoch=4, val_loss=1.25)
    clone = torch.nn.Linear(3, 2)
    state = load_checkpoint(tmp_path / "ckpt.pt", clone, torch.optim.Adam(clone.parameters()))
    assert state["epoch"] == 4 and state["val_loss"] == 1.25
    torch.testing.assert_close(clone.weight, model.weight)
