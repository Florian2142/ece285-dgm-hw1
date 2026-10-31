"""Evaluate a trained HW1 run: new samples, reconstructions, FID and Inception Score.

AI-assisted evaluation code for the experiments I chose, including score computation,
image export, and the real-image baseline correction.

    uv run python -m hw1.evaluate                       # latest run, best checkpoint
    uv run python -m hw1.evaluate --run 20261020-101500-vae-base eval.repeats=5
    uv run python -m hw1.evaluate --no-scores           # image grids only (fast; keeps earlier scores)

Writes <run>/eval/:
    samples.png          48 new images (3 rows of 16), z ~ N(0, I)
    reconstructions.png  16 reference images (top) and their reconstructions (bottom)
    metrics.json         all scores (mean, std, every repeat) and the settings used

Scores (details in dgm/evaluation/metrics.py): ``fid`` is ignite's default FID exactly as in the course
tutorial (it compares Inception class probabilities, so values are small); ``fid_2048`` is
the standard FID on 2048-d pool features; ``inception_score`` is ignite's default IS.

Protocol (describe it in the report):
- samples: n = eval.num_samples images (default: the size of the reference split),
  drawn fresh in each of eval.repeats repeats -> mean and standard deviation
- reconstructions: scores of reconstructed reference images; a gap between these and the
  sample scores points to a mismatch between the prior and the encoded data ("holes")
- real baseline: n TRAINING images sampled without replacement to match the reference
  class counts; a finite-sample comparison, not a lower bound on FID
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from dgm.data.chest_xray import ChestXray
from dgm.evaluation.metrics import InceptionScorer
from dgm.evaluation.sampling import matched_class_indices
from dgm.training.checkpoint import load_checkpoint
from dgm.training.runs import resolve_run_dir, write_json
from dgm.utils.config import load_config
from dgm.utils.misc import get_device, seed_everything
from dgm.utils.viz import save_image_grid
from hw1.config import Config
from hw1.train import EXPERIMENT, build_model

SCORES = {"fid": "FID (tutorial)", "fid_2048": "FID-2048", "inception_score": "IS"}


@torch.no_grad()
def generate(model: nn.Module, n: int, batch_size: int, device: torch.device) -> Tensor:
    chunks = [
        model.sample(min(batch_size, n - start), device).float().cpu()
        for start in range(0, n, batch_size)
    ]
    return torch.cat(chunks)


@torch.no_grad()
def reconstruct(model: nn.Module, images: Tensor, batch_size: int, device: torch.device) -> Tensor:
    chunks = [
        model.reconstruct(images[i : i + batch_size].to(device)).float().cpu()
        for i in range(0, len(images), batch_size)
    ]
    return torch.cat(chunks)


def summarize(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "values": [float(v) for v in values],
    }


def evaluate(
    run_dir: Path,
    overrides: list[str] | None = None,
    *,
    model: nn.Module | None = None,
    compute_scores: bool = True,
) -> dict:
    cfg = load_config(Config, run_dir / "config.yaml", overrides or [])
    device = get_device(cfg.train.device)
    if model is None:
        model = build_model(cfg)
        load_checkpoint(
            run_dir / "checkpoints" / f"{cfg.eval.checkpoint}.pt", model, map_location=device
        )
    model.to(device).eval()
    seed_everything(cfg.eval.seed)
    out_dir = run_dir / "eval"
    out_dir.mkdir(exist_ok=True)

    reference_dataset = ChestXray(cfg.eval.reference_split, cfg.data.image_size)
    reference = reference_dataset.as_float()
    n = cfg.eval.num_samples or len(reference)
    if n > len(reference):
        raise ValueError(
            f"eval.num_samples={n} exceeds the {len(reference)} '{cfg.eval.reference_split}' images"
        )

    save_image_grid(model.sample(48, device), out_dir / "samples.png", nrow=16)
    x = reference[:16].to(device)
    save_image_grid(torch.cat([x, model.reconstruct(x)]), out_dir / "reconstructions.png", nrow=16)

    results: dict = {
        "run": run_dir.name,
        "checkpoint": cfg.eval.checkpoint,
        "reference_split": cfg.eval.reference_split,
        "num_samples": n,
        "repeats": cfg.eval.repeats,
        "seed": cfg.eval.seed,
        "preprocessing": "PIL bilinear 299x299, gray->3 channels, [0,1]; torchvision Inception-v3",
        "scores": {
            "fid": "ignite FID() default as in the tutorial: 1000-d class probabilities",
            "fid_2048": "ignite FID on 2048-d pool features (standard definition)",
            "inception_score": "ignite InceptionScore() default",
        },
    }
    if compute_scores:
        scorer = InceptionScorer(device, batch_size=cfg.eval.batch_size)
        rng = torch.Generator().manual_seed(cfg.eval.seed)
        per_repeat: dict[str, list[float]] = {name: [] for name in SCORES}
        for repeat in range(cfg.eval.repeats):
            real = reference[torch.randperm(len(reference), generator=rng)[:n]]
            scores = scorer(generate(model, n, cfg.eval.batch_size, device), real)
            for name in SCORES:
                per_repeat[name].append(scores[name])
            shown = "  ".join(f"{SCORES[k]} {fmt(v)}" for k, v in scores.items())
            print(f"repeat {repeat + 1}/{cfg.eval.repeats}: {shown}", flush=True)
        results["samples"] = {name: summarize(values) for name, values in per_repeat.items()}
        real = reference[:n]
        results["reconstructions"] = scorer(
            reconstruct(model, real, cfg.eval.batch_size, device), real
        )
        if cfg.eval.real_baseline and cfg.eval.reference_split != "train":
            training = ChestXray("train", cfg.data.image_size)
            indices = matched_class_indices(training.labels, reference_dataset.labels[:n], seed=cfg.eval.seed)
            picked = training.as_float()[indices]
            results["real_baseline_protocol"] = "training sample matched to reference class counts"
            results["real_train_images"] = scorer(picked, real)
        write_json(out_dir / "metrics.json", results)  # only scored runs replace earlier scores
    print_summary(results)
    return results


def fmt(value: float, like: float | None = None) -> str:
    """Enough digits for both scales: tutorial FID (< 1) and FID-2048 (tens to hundreds).

    ``like``: use the decimals that suit this value instead (e.g. the mean, for a std).
    """
    ref = abs(value if like is None else like)
    return f"{value:.{4 if ref < 1 else 2 if ref < 100 else 1}f}"


def print_summary(results: dict) -> None:
    if "samples" not in results:
        print(f"saved image grids for {results['run']} (scores skipped)")
        return
    print(
        f"\n{results['run']}: {results['num_samples']} images vs. '{results['reference_split']}', "
        f"{results['repeats']} repeat(s)"
    )
    print(f"  {'':<17}" + "".join(f"{label:>22}" for label in SCORES.values()))
    samples = results["samples"]
    cells = [
        f"{fmt(samples[k]['mean'])} +- {fmt(samples[k]['std'], like=samples[k]['mean'])}"
        for k in SCORES
    ]
    print(f"  {'samples':<17}" + "".join(f"{cell:>22}" for cell in cells))
    for key, label in (
        ("reconstructions", "reconstructions"),
        ("real_train_images", "real (matched)"),
    ):
        if key in results:
            print(f"  {label:<17}" + "".join(f"{fmt(results[key][k]):>22}" for k in SCORES))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run", default="latest", help="run name or folder under runs/hw1 (default: latest)"
    )
    parser.add_argument("--no-scores", action="store_true", help="only save the image grids")
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="override eval settings, e.g. eval.repeats=5",
    )
    args = parser.parse_args(argv)
    try:
        run_dir = resolve_run_dir(EXPERIMENT, args.run)
        evaluate(run_dir, args.overrides, compute_scores=not args.no_scores)
    except FileNotFoundError as err:
        raise SystemExit(f"error: {err}") from None
    except NotImplementedError as err:
        raise SystemExit(
            f"\nNot implemented yet: {err}\nFinish homeworks/hw1/models/vae.py first."
        ) from None


if __name__ == "__main__":
    main()
