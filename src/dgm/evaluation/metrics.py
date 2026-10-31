"""FID and Inception Score with PyTorch-Ignite, preprocessed as in the course-linked tutorial.

AI-assisted metric implementation following the course-linked Ignite tutorial.

Tutorial: https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/

Every image goes through the tutorial's ``interpolate``: tensor in [0, 1] -> PIL ->
bilinear resize to 299 x 299 -> tensor in [0, 1]; grayscale is repeated to 3 channels.
The network is torchvision's ImageNet Inception-v3 via ignite (weights download once, ~100 MB).

Three scores (state which ones you report):
- ``fid``: ignite's default ``FID()``, exactly as in the tutorial. Note: with no arguments,
  ignite compares the 1000 Inception *class probabilities* (softmax outputs), so the
  values are small (often below 1) and NOT comparable with FIDs in papers.
- ``fid_2048``: the standard FID definition (Heusel et al., 2017) on the 2048-d pool
  features of the same ignite Inception model. The matrix square root is computed with
  symmetric eigendecompositions (same value as ``scipy.linalg.sqrtm``, ~20x faster at
  2048-d). Still not identical to papers, which use the TensorFlow Inception weights.
- ``inception_score``: ignite's default ``InceptionScore()``, as in the tutorial.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

INCEPTION_SIZE = 299
POOL_FEATURES = 2048


def interpolate(batch: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W]`` in [0, 1] (C = 1 or 3) -> ``[B, 3, 299, 299]`` in [0, 1], PIL bilinear."""
    size = (INCEPTION_SIZE, INCEPTION_SIZE)
    resized = [
        to_tensor(to_pil_image(img).resize(size, Image.Resampling.BILINEAR)) for img in batch.cpu()
    ]
    images = torch.stack(resized)
    return images.repeat(1, 3, 1, 1) if images.shape[1] == 1 else images


def frechet_distance(
    mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor
) -> float:
    """``||mu1 - mu2||^2 + tr(S1 + S2 - 2 (S1 S2)^(1/2))`` in float64.

    ``tr((S1 S2)^(1/2))`` equals the sum of square roots of the eigenvalues of the symmetric
    PSD matrix ``S1^(1/2) S2 S1^(1/2)``, which ``eigh`` computes quickly and stably.
    """
    mu1, mu2, sigma1, sigma2 = (t.detach().double().cpu() for t in (mu1, mu2, sigma1, sigma2))
    eigvals, eigvecs = torch.linalg.eigh(sigma1)
    sqrt_sigma1 = (eigvecs * eigvals.clamp_min(0).sqrt()) @ eigvecs.T
    middle = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
    trace_sqrt = torch.linalg.eigvalsh((middle + middle.T) / 2).clamp_min(0).sqrt().sum()
    diff = mu1 - mu2
    return float(diff @ diff + sigma1.trace() + sigma2.trace() - 2 * trace_sqrt)


class FeatureStats:
    """Running mean and covariance of feature vectors (float64)."""

    def __init__(self, dim: int, device: torch.device) -> None:
        self.n = 0
        self.total = torch.zeros(dim, dtype=torch.float64, device=device)
        self.outer = torch.zeros(dim, dim, dtype=torch.float64, device=device)

    def update(self, features: torch.Tensor) -> None:
        features = features.double()
        self.n += len(features)
        self.total += features.sum(dim=0)
        self.outer += features.T @ features

    def mean_cov(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.total / self.n
        cov = (self.outer - self.n * torch.outer(mean, mean)) / (self.n - 1)
        return mean, cov


def _batches(images: torch.Tensor, batch_size: int) -> Iterator[torch.Tensor]:
    for start in range(0, len(images), batch_size):
        yield images[start : start + batch_size]


class InceptionScorer:
    """Tutorial FID and IS (ignite defaults) plus FID-2048; models load once, reused per call."""

    def __init__(self, device: torch.device | str, batch_size: int = 100) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.fid = FID(device=self.device)  # tutorial default: 1000 class probabilities
        self.inception_score = InceptionScore(device=self.device)
        self.pool_model = InceptionModel(return_features=True, device=self.device)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
        """FID/FID-2048 of ``images`` vs. ``reference`` and IS of ``images``; ``[N, C, H, W]`` in [0, 1]."""
        if len(images) != len(reference):
            raise ValueError(
                f"use equally many images and reference images ({len(images)} vs {len(reference)})"
            )
        self.fid.reset()
        self.inception_score.reset()
        fake_stats = FeatureStats(POOL_FEATURES, self.device)
        real_stats = FeatureStats(POOL_FEATURES, self.device)
        for fake, real in zip(
            _batches(images, self.batch_size), _batches(reference, self.batch_size), strict=True
        ):
            fake = interpolate(fake.float().clamp(0, 1)).to(self.device)
            real = interpolate(real.float().clamp(0, 1)).to(self.device)
            self.fid.update((fake, real))
            self.inception_score.update(fake)
            fake_stats.update(self.pool_model(fake))
            real_stats.update(self.pool_model(real))
        return {
            "fid": float(self.fid.compute()),
            "fid_2048": frechet_distance(*fake_stats.mean_cov(), *real_stats.mean_cov()),
            "inception_score": float(self.inception_score.compute()),
        }
