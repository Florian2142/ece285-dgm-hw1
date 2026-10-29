"""End-to-end check of the HW1 plumbing (data -> train -> evaluate -> report assets).

Uses a tiny synthetic dataset and a stand-in model, so it runs before your VAE exists.
The stand-in is NOT a VAE (no sampling of z, no KL); it only has the same interface.
"""

import pytest
import torch
from torch import nn

from dgm.data.chest_xray import prepare, raw_dir
from dgm.data.synthetic import make_fake_chest_xray
from dgm.training.loggers import read_csv
from dgm.training.runs import read_json
from dgm.utils.config import load_config
from hw1 import report_assets
from hw1.config import Config
from hw1.evaluate import evaluate
from hw1.models import VAEOutput
from hw1.train import train

SIZE = 32


class StandIn(nn.Module):
    """A deterministic linear autoencoder with the VAE interface."""

    def __init__(self, image_size: int, latent_dim: int = 4) -> None:
        super().__init__()
        self.latent_dim, self.image_size = latent_dim, image_size
        self.encoder = nn.Linear(image_size * image_size, latent_dim)
        self.decoder = nn.Linear(latent_dim, image_size * image_size)

    def encode(self, x):
        mu = self.encoder(x.flatten(1))
        return mu, torch.zeros_like(mu)

    def decode(self, z):
        return self.decoder(z).view(-1, 1, self.image_size, self.image_size)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return VAEOutput(logits=self.decode(mu), mu=mu, logvar=logvar, z=mu)

    @torch.no_grad()
    def sample(self, n, device=None):
        return torch.sigmoid(self.decode(torch.randn(n, self.latent_dim, device=device)))

    @torch.no_grad()
    def reconstruct(self, x):
        return torch.sigmoid(self.decode(self.encode(x)[0]))


def standin_loss(out, x, *, beta):
    err = ((torch.sigmoid(out.logits) - x) ** 2).flatten(1).sum(dim=1).mean()
    return {"loss": err, "neg_elbo": err, "recon": err, "kl": torch.zeros(())}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DGM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DGM_RUNS_DIR", str(tmp_path / "runs"))
    make_fake_chest_xray(raw_dir(), counts={"train": 40, "val": 4, "test": 16})
    prepare(SIZE, val_fraction=0.2)
    monkeypatch.setattr(report_assets, "GENERATED", tmp_path / "generated")
    return tmp_path


def smoke_config(*extra_overrides, **eval_overrides):
    overrides = [
        f"data.image_size={SIZE}",
        "data.batch_size=8",
        "train.epochs=2",
        "train.sample_every=1",
        "train.device=cpu",
        "eval.batch_size=8",
        *[f"eval.{k}={v}" for k, v in eval_overrides.items()],
        *extra_overrides,
    ]
    return load_config(Config, overrides=overrides)


def test_augment_batch_keeps_shape_and_range_and_is_identity_without_strength():
    from hw1.train import augment_batch

    x = torch.rand(16, 1, SIZE, SIZE, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(augment_batch(x, max_shift=0, contrast=0.0), x)
    torch.manual_seed(0)
    y = augment_batch(x)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1
    assert not torch.allclose(y, x)
    # a shift moves content: one bright pixel lands within max_shift of where it was
    dot = torch.zeros(1, 1, SIZE, SIZE)
    dot[0, 0, SIZE // 2, SIZE // 2] = 1.0
    moved = augment_batch(dot, max_shift=4, contrast=0.0)
    row, col = divmod(moved.flatten().argmax().item(), SIZE)
    assert abs(row - SIZE // 2) <= 4 and abs(col - SIZE // 2) <= 4
    torch.testing.assert_close(moved.max(), torch.tensor(1.0))


def test_cosine_schedule_lowers_the_learning_rate(workspace):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run_dir = train(  # augmentation on, to exercise that training path too
        smoke_config("train.lr_schedule=cosine", "train.augment=true", "train.weight_decay=0.01"),
        model=StandIn(SIZE),
        loss_fn=standin_loss,
    )
    events = EventAccumulator(str(run_dir / "tensorboard"), size_guidance={"scalars": 0})
    events.Reload()
    logged = [scalar.value for scalar in events.Scalars("train/lr")]
    # cosine annealing over 2 epochs: the initial lr, then lr * (1 + cos(pi / 2)) / 2
    assert logged == pytest.approx([1e-3, 5e-4])


def test_train_evaluate_export(workspace):
    cfg = smoke_config(num_samples=8, repeats=1)
    run_dir = train(cfg, model=StandIn(SIZE), loss_fn=standin_loss)

    metrics = read_csv(run_dir / "metrics.csv")
    assert list(metrics["epoch"]) == [1.0, 2.0]
    for name in ("config.yaml", "env.json", "steps.csv", "summary.json", "loss_curves.pdf"):
        assert (run_dir / name).exists(), name
    assert (run_dir / "checkpoints" / "best.pt").exists() and (
        run_dir / "samples" / "epoch_002.png"
    ).exists()
    assert list((run_dir / "tensorboard").glob("events.out.tfevents.*")), "no TensorBoard log"

    results = evaluate(run_dir, model=StandIn(SIZE), compute_scores=False)
    assert (run_dir / "eval" / "samples.png").exists() and (
        run_dir / "eval" / "reconstructions.png"
    ).exists()
    assert results["num_samples"] == 8

    report_assets.export(run_dir)
    generated = report_assets.GENERATED
    for name in (
        "loss_curves.pdf",
        "samples.png",
        "reconstructions.png",
        "hparams.tex",
        "results.tex",
    ):
        assert (generated / name).exists(), name
    assert r"\newcommand{\ResultTrainImages}" in (generated / "results.tex").read_text()


@pytest.mark.slow
def test_evaluate_with_fid_and_is(workspace):
    cfg = smoke_config(num_samples=16, repeats=2)
    run_dir = train(cfg, model=StandIn(SIZE), loss_fn=standin_loss)
    results = evaluate(run_dir, model=StandIn(SIZE))
    for score in ("fid", "fid_2048", "inception_score"):
        assert len(results["samples"][score]["values"]) == 2
        assert score in results["reconstructions"] and score in results["real_train_images"]
    assert read_json(run_dir / "eval" / "metrics.json")["samples"]["inception_score"]["mean"] >= 1.0

    # refreshing only the image grids must keep the saved scores
    evaluate(run_dir, model=StandIn(SIZE), compute_scores=False)
    assert "samples" in read_json(run_dir / "eval" / "metrics.json")

    report_assets.export(run_dir)
    assert "??" not in (report_assets.GENERATED / "results.tex").read_text()  # all numbers filled
