"""AI-assisted helper: select real-image comparisons independently of dataset file order."""

import torch
from torch import Tensor


def matched_class_indices(labels: Tensor, reference_labels: Tensor, *, seed: int = 0) -> Tensor:
    """Sample without replacement, matching the reference's class counts.

    Uses a private CPU generator, so evaluation does not change training RNG state.
    Labels only determine the real-image comparison; VAE training remains unconditional.
    """
    labels, reference_labels = labels.cpu(), reference_labels.cpu()
    if labels.ndim != 1 or reference_labels.ndim != 1 or not len(reference_labels):
        raise ValueError("Expected one-dimensional labels and a nonempty reference")
    generator = torch.Generator().manual_seed(seed)
    chunks = []
    for label, count in zip(*torch.unique(reference_labels, return_counts=True), strict=True):
        candidates = torch.where(labels == label)[0]
        if len(candidates) < int(count):
            raise ValueError(
                f"Class {label.item()}: need {int(count)} training images, have {len(candidates)}"
            )
        chunks.append(
            candidates[torch.randperm(len(candidates), generator=generator)[: int(count)]]
        )
    indices = torch.cat(chunks)
    return indices[torch.randperm(len(indices), generator=generator)]
