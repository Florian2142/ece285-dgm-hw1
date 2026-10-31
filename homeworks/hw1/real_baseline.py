"""Recompute only the real-training-image baseline; no VAE training is needed.

AI-assisted implementation of the corrected real-image comparison.

uv run python -m hw1.real_baseline --output homeworks/hw1/results/real_baseline.json
"""

import argparse
import json
from pathlib import Path

import torch

from dgm.data.chest_xray import CLASSES, ChestXray
from dgm.evaluation.metrics import InceptionScorer
from dgm.evaluation.sampling import matched_class_indices


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    training, reference = ChestXray("train", 64), ChestXray("test", 64)
    indices = matched_class_indices(training.labels, reference.labels, seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scores = InceptionScorer(device)(training.as_float()[indices], reference.as_float())
    result = {
        "protocol": "Training images sampled without replacement to match test class counts",
        "seed": args.seed,
        "num_images": len(indices),
        "image_size": 64,
        "reference_split": "test",
        "preprocessing": "PIL bilinear 299x299, gray->3 channels, [0,1]; torchvision Inception-v3",
        "class_counts": {
            name: int((reference.labels == i).sum()) for i, name in enumerate(CLASSES)
        },
        "scores": scores,
        "training_files": [training.files[i] for i in indices.tolist()],
        "note": "A finite-sample comparison, not a lower bound on attainable FID; no model was retrained.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "training_files"}, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
