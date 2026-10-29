import numpy as np
import pytest
import torch
from ignite.metrics.gan.fid import fid_score

from dgm.evaluation.metrics import FeatureStats, InceptionScorer, frechet_distance, interpolate
from dgm.utils.viz import plot_training_curves, save_image_grid


def test_save_image_grid(tmp_path):
    path = save_image_grid(torch.rand(10, 1, 16, 16), tmp_path / "grid.png", nrow=5)
    assert path.exists() and path.stat().st_size > 0


def test_training_curves_pdf_and_png(tmp_path):
    epochs = np.arange(1, 6, dtype=float)
    metrics = {
        "epoch": epochs,
        "train/neg_elbo": 1 / epochs,
        "val/neg_elbo": 1.1 / epochs,
        "train/kl": epochs,
    }
    outputs = plot_training_curves(metrics, tmp_path / "curves")
    assert [p.suffix for p in outputs] == [".pdf", ".png"]
    assert all(p.stat().st_size > 0 for p in outputs)
    with pytest.raises(ValueError):
        plot_training_curves({"epoch": epochs}, tmp_path / "empty")


def test_frechet_distance_matches_ignite_scipy_version():
    """The fast eigh-based FID equals ignite's scipy.linalg.sqrtm-based fid_score."""
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(400, 64, generator=gen, dtype=torch.float64)
    b = torch.randn(400, 64, generator=gen, dtype=torch.float64) * 1.5 + 0.3
    stats = []
    for features in (a, b):
        s = FeatureStats(64, torch.device("cpu"))
        for chunk in features.split(100):
            s.update(chunk)
        mean, cov = s.mean_cov()
        torch.testing.assert_close(mean, features.mean(0))
        torch.testing.assert_close(cov, torch.cov(features.T))
        stats.append((mean, cov))
    (mu1, s1), (mu2, s2) = stats
    expected = fid_score(mu1, mu2, s1, s2)
    assert frechet_distance(mu1, s1, mu2, s2) == pytest.approx(expected, rel=1e-6)
    assert frechet_distance(mu1, s1, mu1, s1) == pytest.approx(0.0, abs=1e-8)


def test_interpolate_matches_tutorial_format():
    out = interpolate(torch.rand(3, 1, 64, 64))
    assert out.shape == (3, 3, 299, 299)
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.slow
def test_fid_is_sanity():
    """Identical sets give FID ~ 0; clearly different images give a clearly larger FID."""
    gen = torch.Generator().manual_seed(0)
    real = torch.rand(64, 1, 32, 32, generator=gen)  # noise
    other = torch.rand(64, 1, 32, 32, generator=gen) * 0.05  # almost black
    scorer = InceptionScorer("cuda" if torch.cuda.is_available() else "cpu", batch_size=32)
    same = scorer(real, real)
    different = scorer(other, real)
    assert abs(same["fid"]) < 1e-6 and abs(same["fid_2048"]) < 1e-3
    # ignite's default FID works on class probabilities (small values); FID-2048 on pool features
    assert different["fid"] > 0.005
    assert different["fid_2048"] > 50.0
    assert same["inception_score"] >= 1.0
    with pytest.raises(ValueError, match="equally many"):
        scorer(real[:10], real)
