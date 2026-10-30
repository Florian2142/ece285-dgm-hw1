"""The notebook's train_vae / save_vae / load_vae keep every run under a unique name with real metadata.

Runs the function cell of notebook 02 on a tiny synthetic dataset in a temporary folder (2 epochs
on the CPU), so no real run is started and nothing in notebooks/models/ or runs/ is touched.
"""

import json
import math
import pickle
import time
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from dgm.utils.misc import count_parameters, seed_everything
from hw1.config import ModelConfig
from hw1.losses import elbo_loss
from hw1.models import VAE
from hw1.train import augment_batch

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "02_vae_step_by_step.ipynb"
HPARAMS = {
    "name": "tiny",
    "latent_dim": 3,
    "hidden_channels": 2,
    "convs_per_stage": 2,
    "residual": True,
    "dropout": 0.2,
    "weight_decay": 0.01,
    "augment": True,
    "likelihood": "gaussian",
    "gaussian_sigma": 0.2,
    "epochs": 2,
    "batch_size": 4,
    "lr": 1e-3,
    "lr_schedule": "cosine",
    "grad_clip_norm": 1.0,
    "beta": 0.5,
}


@pytest.fixture
def notebook_functions(tmp_path):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(next(cell["source"] for cell in notebook["cells"] if cell["id"] == "functions"))
    import copy
    import pickle as pickle_module
    from datetime import datetime

    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    images = torch.rand(8, 1, 16, 16, generator=torch.Generator().manual_seed(1))
    labels = torch.zeros(8, dtype=torch.long)
    namespace = {
        "torch": torch,
        "math": math,
        "copy": copy,
        "pickle": pickle_module,
        "datetime": datetime,
        "DataLoader": DataLoader,
        "SummaryWriter": SummaryWriter,
        "seed_everything": seed_everything,
        "count_parameters": count_parameters,
        "elbo_loss": elbo_loss,
        "augment_batch": augment_batch,
        "VAE": VAE,
        "ModelConfig": ModelConfig,
        "SIZE": 16,
        "QUICK": False,
        "RETRAIN": False,
        "device": torch.device("cpu"),
        "train": TensorDataset(images, labels),
        "val": TensorDataset(images[:4], labels[:4]),
        "RUNS_DIR": tmp_path / "runs",
        "MODELS_DIR": tmp_path / "models",
    }
    (tmp_path / "models").mkdir()
    exec(compile(source, f"{NOTEBOOK.name}:functions", "exec"), namespace)
    return namespace


def test_train_save_load_keep_real_metadata(notebook_functions, capsys):
    ns = notebook_functions
    model, run = ns["train_vae"](HPARAMS)
    assert run["name"].startswith("tiny-") and run["epochs_completed"] == 2
    assert run["best_epoch"] in (1, 2) and run["hparams"] == HPARAMS
    assert run["best_val_neg_elbo"] == min(e["neg_elbo"] for e in run["history"]["val"])
    assert (ns["RUNS_DIR"] / run["name"]).is_dir(), "TensorBoard folder = run name"
    assert set(run["metrics"]) == {"val_mse", "psnr", "hf_mse", "active_dims", "train_mse"}
    assert run["metrics"]["train_mse"] > 0
    assert 0 <= run["metrics"]["active_dims"] <= HPARAMS["latent_dim"]
    assert run["train_time_s"] >= 0

    ns["save_vae"](model, run)
    path = ns["MODELS_DIR"] / f"{run['name']}.p"
    assert path.exists()
    assert ns["newest_run"]("tiny") == run["name"]
    assert ns["newest_run"]("nothing-like-this") is None
    with pytest.raises(AssertionError, match="never overwritten"):
        ns["save_vae"](model, run)

    loaded, loaded_run = ns["load_vae"](run["name"])
    assert not loaded.training and loaded_run == run
    x = torch.rand(2, 1, 16, 16)
    torch.testing.assert_close(loaded.reconstruct(x), model.reconstruct(x))

    # a plain model name loads its newest run; an unknown name says which runs exist
    _, by_model_name = ns["load_vae"]("tiny")
    assert by_model_name["name"] == run["name"]
    with pytest.raises(FileNotFoundError, match=f"no saved run 'nothing'.*{run['name']}"):
        ns["load_vae"]("nothing")

    # the seed knob (section 14 trains three seeds): the default is 0, another seed gives other weights
    seeded, seeded_run = ns["train_vae"]({**HPARAMS, "seed": 1})
    assert seeded_run["hparams"]["seed"] == 1 and "seed" not in run["hparams"]
    assert not torch.allclose(seeded.reconstruct(x), model.reconstruct(x))


def test_train_or_load_reuses_the_saved_run_unless_retrain(notebook_functions, capsys):
    ns = notebook_functions
    model, run = ns["train_vae"](HPARAMS)
    ns["save_vae"](model, run)

    loaded, reused = ns["train_or_load"]("tiny", HPARAMS)
    assert reused["name"] == run["name"] and not loaded.training
    assert "saved run" in capsys.readouterr().out  # nothing was trained

    ns["RETRAIN"] = True
    time.sleep(1.1)  # run names carry the time with second resolution
    _, fresh = ns["train_or_load"]("tiny", HPARAMS)
    assert fresh["name"] != run["name"] and fresh["epochs_completed"] == 2
    assert ns["newest_run"]("tiny") == fresh["name"]


def test_load_old_pickle_flags_wrong_epoch_setting(notebook_functions, capsys):
    ns = notebook_functions
    model = ns["build_vae"](HPARAMS)
    history = {
        "train": [{"neg_elbo": 5.0, "recon": 4.0, "kl": 1.0}] * 3,
        "val": [
            {"neg_elbo": 6.0, "recon": 5.0, "kl": 1.0},
            {"neg_elbo": 4.0, "recon": 3.0, "kl": 1.0},
            {"neg_elbo": 4.5, "recon": 3.5, "kl": 1.0},
        ],
    }
    old = {
        "hparams": {**HPARAMS, "epochs": 750},
        "state_dict": model.state_dict(),
        "history": history,
    }
    with open(ns["MODELS_DIR"] / "old.p", "wb") as f:
        pickle.dump(old, f)

    _, run = ns["load_vae"]("old")
    assert run["old_format"] and run["name"] == "old"
    assert (
        run["epochs_completed"] == 3 and run["best_epoch"] == 2 and run["best_val_neg_elbo"] == 4.0
    )
    printed = capsys.readouterr().out
    assert "epochs=750" in printed and "3 epochs" in printed


def test_train_or_load_warns_when_the_saved_run_has_other_settings(notebook_functions, capsys):
    ns = notebook_functions
    model, run = ns["train_vae"](HPARAMS)
    ns["save_vae"](model, run)
    capsys.readouterr()

    _, reused = ns["train_or_load"]("tiny", {**HPARAMS, "beta": 0.9})
    printed = capsys.readouterr().out
    assert reused["name"] == run["name"]  # still reused: the warning only reports the difference
    assert "WARNING" in printed and "beta" in printed

    ns["train_or_load"]("tiny", HPARAMS)
    assert "WARNING" not in capsys.readouterr().out


def test_quick_runs_never_mix_with_real_runs(notebook_functions):
    ns = notebook_functions
    model, run = ns["train_vae"](HPARAMS)
    ns["save_vae"](model, run)

    ns["QUICK"] = True
    _, quick = ns["train_or_load"]("tiny", {**HPARAMS, "epochs": 50})
    assert quick["name"].startswith("quick-tiny-") and quick["epochs_completed"] == 2
    assert ns["newest_run"]("tiny") == quick["name"]  # QUICK only sees quick runs

    ns["QUICK"] = False
    assert ns["newest_run"]("tiny") == run["name"]  # the real run is untouched and still the newest real run
