"""Smoke-test the actual sanity-check cells with a small synthetic dataset."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from torch.utils.data import TensorDataset

from dgm.utils.misc import seed_everything
from hw1.config import ModelConfig
from hw1.losses import elbo_loss
from hw1.models import VAE

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "02_vae_step_by_step.ipynb"


@pytest.mark.parametrize("cell_id, batch_size", [("overfit-one", 1), ("overfit-batch", 4)])
def test_overfit_notebook_uses_fixed_tensor_and_configured_likelihood(monkeypatch, cell_id, batch_size):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(next(cell["source"] for cell in notebook["cells"] if cell["id"] == cell_id))
    images = torch.rand(4, 1, 16, 16, generator=torch.Generator().manual_seed(12))
    cfg = ModelConfig(latent_dim=3, hidden_channels=2, likelihood="gaussian", gaussian_sigma=0.2)
    observed_batches = []

    def checked_loss(out, batch, **kwargs):
        assert batch.shape == (batch_size, 1, 16, 16)
        assert kwargs["likelihood"] == cfg.likelihood
        assert kwargs["sigma"] == cfg.gaussian_sigma
        observed_batches.append(batch.detach().clone())
        return elbo_loss(out, batch, **kwargs)

    def checked_show(pictures, *args, **kwargs):
        assert pictures.shape == (2 * batch_size, 1, 16, 16)
        assert torch.isfinite(pictures).all()

    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    namespace = {
        "torch": torch, "plt": plt, "seed_everything": seed_everything,
        "VAE": VAE, "cfg": cfg, "SIZE": 16, "device": torch.device("cpu"),
        "train": TensorDataset(images, torch.zeros(4, dtype=torch.long)), "x": images,
        "elbo_loss": checked_loss, "show": checked_show,
        # Check cell wiring in two real optimization steps; convergence is checked separately.
        "range": lambda count: range(min(count, 2)),
    }
    try:
        exec(compile(source, f"{NOTEBOOK.name}:{cell_id}", "exec"), namespace)
        assert len(observed_batches) == 2
        torch.testing.assert_close(observed_batches[0], observed_batches[1])
        assert len(namespace["history"]) == 2
        assert all(torch.isfinite(torch.tensor(row["loss"])) for row in namespace["history"])
        if batch_size == 1:
            assert namespace["reconstruction"].shape == (1, 1, 16, 16)
            assert namespace["initial_mse"] >= 0 and namespace["final_mse"] >= 0
    finally:
        plt.close("all")
