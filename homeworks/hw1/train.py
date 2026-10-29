"""Train the HW1 VAE on chest X-rays.

    uv run python -m hw1.train                                   # configs/base.yaml
    uv run python -m hw1.train --config homeworks/hw1/configs/smoke.yaml
    uv run python -m hw1.train train.epochs=100 model.latent_dim=64 name=vae-k64

The pipeline is the one from I2DL: for every training batch, forward -> loss -> backward ->
optimizer step (``training_step``); after every epoch, the same loss on the validation set
without updates (``validation_step``). Watch a run live: ``uv run tensorboard --logdir runs/hw1``.

Each run writes runs/hw1/<timestamp>-<name>/:
    config.yaml        the exact settings (re-load it to reproduce the run)
    env.json           versions, GPU, git revision, parameter count
    metrics.csv        one row per epoch: train/val loss terms, beta, time
    steps.csv          one row per training batch
    tensorboard/       the same numbers plus sample images, for TensorBoard
    checkpoints/       last.pt and best.pt (lowest validation negative ELBO)
    samples/           epoch_XXX.png: fixed z ~ N(0, I) decoded; recon_XXX.png: val images (top) / reconstructions
    loss_curves.pdf    train/val curves, updated every epoch (also .png)
    summary.json       best epoch, final numbers, training time
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dgm.data.chest_xray import ChestXray
from dgm.training.checkpoint import save_checkpoint
from dgm.training.loggers import CSVLogger, create_tqdm_bar, read_csv
from dgm.training.runs import create_run_dir, write_json
from dgm.utils.config import add_config_args, load_config, save_config
from dgm.utils.misc import count_parameters, environment_info, get_device, seed_everything
from dgm.utils.viz import plot_training_curves, save_image_grid
from hw1.config import Config, TrainConfig
from hw1.losses import elbo_loss
from hw1.models import VAE

EXPERIMENT = "hw1"
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "base.yaml"

LossFn = Callable[..., dict[str, Tensor]]
Batch = tuple[Tensor, Tensor]


def build_model(cfg: Config) -> VAE:
    return VAE(cfg.model, image_size=cfg.data.image_size)


def build_loss(cfg: Config) -> LossFn:
    return partial(elbo_loss, likelihood=cfg.model.likelihood, sigma=cfg.model.gaussian_sigma)


def make_loader(split: str, cfg: Config, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        ChestXray(split, cfg.data.image_size),
        batch_size=cfg.data.batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def augment_batch(x: Tensor, max_shift: int = 4, contrast: float = 0.1) -> Tensor:
    """Training-time augmentation (plumbing, 27 Oct 2026, PLAN.md E7). Per image: a random
    shift of up to ``max_shift`` pixels in each direction (edge pixels repeated, nothing is
    invented) and a contrast change of up to ``contrast`` around the image mean. No flips:
    chest X-rays have the heart on one side."""
    b, _, h, w = x.shape
    padded = F.pad(x, (max_shift,) * 4, mode="replicate")
    dy = torch.randint(0, 2 * max_shift + 1, (b,), device=x.device)
    dx = torch.randint(0, 2 * max_shift + 1, (b,), device=x.device)
    rows = torch.arange(h, device=x.device)[None, :] + dy[:, None]  # [B, H]: which rows to keep
    cols = torch.arange(w, device=x.device)[None, :] + dx[:, None]  # [B, W]
    batch = torch.arange(b, device=x.device)[:, None, None]
    shifted = padded[batch, 0, rows[:, :, None], cols[:, None, :]].unsqueeze(1)  # [B, 1, H, W]
    scale = 1 + contrast * (2 * torch.rand(b, 1, 1, 1, device=x.device) - 1)  # in [1-c, 1+c]
    mean = shifted.mean(dim=(1, 2, 3), keepdim=True)
    return (mean + (shifted - mean) * scale).clamp(0, 1)


def beta_at(epoch: int, cfg: TrainConfig) -> float:
    """KL weight in a 1-based ``epoch``: linear warm-up to ``cfg.beta``, then constant."""
    if cfg.kl_warmup_epochs <= 0:
        return cfg.beta
    return cfg.beta * min(1.0, epoch / cfg.kl_warmup_epochs)


def training_step(
    model: nn.Module,
    batch: Batch,
    loss_fn: LossFn,
    optimizer: torch.optim.Optimizer,
    *,
    beta: float,
    device: torch.device,
    grad_clip_norm: float = 0.0,
    augment: bool = False,
) -> dict[str, float]:
    """One optimization step on one batch: forward -> loss -> backward -> update."""
    model.train()
    x, _ = batch  # the labels (NORMAL / PNEUMONIA) are not used: the VAE models p(x)
    x = x.to(device, non_blocking=True)
    if augment:  # training images only; validation stays untouched
        x = augment_batch(x)
    terms = loss_fn(model(x), x, beta=beta)
    optimizer.zero_grad(set_to_none=True)  # reset the gradients, otherwise they accumulate
    terms["loss"].backward()
    if grad_clip_norm > 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return {name: value.item() for name, value in terms.items()}


@torch.no_grad()
def validation_step(
    model: nn.Module, batch: Batch, loss_fn: LossFn, *, beta: float, device: torch.device
) -> dict[str, float]:
    """The same loss on one validation batch, without gradients and without updates."""
    model.eval()
    x, _ = batch
    x = x.to(device, non_blocking=True)
    terms = loss_fn(model(x), x, beta=beta)
    return {name: value.item() for name, value in terms.items()}


def run_epoch(
    step: Callable[[Batch], dict[str, float]],
    loader: DataLoader,
    *,
    desc: str,
    max_batches: int = 0,
    on_step: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, float]:
    """Apply ``step`` to every batch of ``loader``; return the per-image averages of its outputs."""
    total = min(max_batches, len(loader)) if max_batches else len(loader)
    totals: dict[str, float] = defaultdict(float)
    seen = 0
    loop = create_tqdm_bar(loader, desc=desc, total=total)
    for i, batch in loop:
        if i >= total:
            break
        values = step(batch)
        if not math.isfinite(values["loss"]):
            raise FloatingPointError(f"{desc}: loss became {values['loss']} in batch {i}: {values}")
        seen += len(batch[0])
        for name, value in values.items():
            totals[name] += value * len(batch[0])
        loop.set_postfix(loss=f"{totals['loss'] / seen:.1f}")
        if on_step is not None:
            on_step(i, values)
    loop.close()
    return {name: total_value / seen for name, total_value in totals.items()}


@torch.no_grad()
def save_progress_images(
    model: VAE, fixed_z: Tensor, fixed_x: Tensor, folder: Path, epoch: int
) -> tuple[Tensor, Tensor]:
    """Decode the fixed prior samples and reconstruct the fixed val images; save both grids."""
    was_training = model.training
    model.eval()
    samples = torch.sigmoid(model.decode(fixed_z))
    pairs = torch.cat([fixed_x, model.reconstruct(fixed_x)])
    save_image_grid(samples, folder / f"epoch_{epoch:03d}.png", nrow=8)
    save_image_grid(pairs, folder / f"recon_{epoch:03d}.png", nrow=len(fixed_x))
    model.train(was_training)
    return samples, pairs


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: LossFn,
    optimizer: torch.optim.Optimizer,
    tb_logger: SummaryWriter,
    *,
    cfg: TrainConfig,
    run_dir: Path,
    device: torch.device,
) -> dict:
    """The training loop: each epoch is one pass over the training set, then the validation set."""
    epoch_log = CSVLogger(run_dir / "metrics.csv")
    step_log = CSVLogger(run_dir / "steps.csv")
    generator = torch.Generator().manual_seed(cfg.seed)
    fixed_z = torch.randn(64, model.latent_dim, generator=generator).to(device)
    fixed_x = next(iter(val_loader))[0][:16].to(device)
    scheduler = None
    if cfg.lr_schedule == "cosine":  # anneal the learning rate to 0 over the run
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    global_step = 0

    def log_step(batch_index: int, values: dict[str, float]) -> None:
        nonlocal global_step
        step_log.log({"epoch": epoch, "batch": batch_index, **values})
        tb_logger.add_scalar("step/train_loss", values["loss"], global_step)
        global_step += 1

    best = {"epoch": 0, "val_neg_elbo": math.inf}
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        beta = beta_at(epoch, cfg)

        # Training: update the parameters on every batch.
        train_stats = run_epoch(
            partial(
                training_step,
                model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                beta=beta,
                device=device,
                grad_clip_norm=cfg.grad_clip_norm,
                augment=cfg.augment,
            ),
            train_loader,
            desc=f"Training Epoch [{epoch}/{cfg.epochs}]",
            max_batches=cfg.max_batches_per_epoch,
            on_step=log_step,
        )
        # Validation: the same loss on held-out images, no updates.
        val_stats = run_epoch(
            partial(validation_step, model, loss_fn=loss_fn, beta=beta, device=device),
            val_loader,
            desc=f"Validation Epoch [{epoch}/{cfg.epochs}]",
            max_batches=cfg.max_batches_per_epoch,
        )

        elapsed = time.perf_counter() - start
        epoch_log.log(
            {
                "epoch": epoch,
                "beta": beta,
                **{f"train/{k}": v for k, v in train_stats.items()},
                **{f"val/{k}": v for k, v in val_stats.items()},
                "time_s": round(elapsed, 2),
            }
        )
        # add_scalar, not add_scalars: the latter makes one subfolder per curve, and the
        # long event-file paths can break on Windows.
        for name, value in train_stats.items():
            tb_logger.add_scalar(f"train/{name}", value, epoch)
            tb_logger.add_scalar(f"val/{name}", val_stats[name], epoch)
        tb_logger.add_scalar("train/beta", beta, epoch)
        tb_logger.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        if scheduler is not None:
            scheduler.step()

        # Checkpoints: always the last epoch, and the best one by validation negative ELBO.
        val_score = val_stats.get("neg_elbo", val_stats["loss"])
        save_checkpoint(
            run_dir / "checkpoints" / "last.pt", model=model, optimizer=optimizer, epoch=epoch
        )
        if val_score < best["val_neg_elbo"]:
            best = {"epoch": epoch, "val_neg_elbo": val_score}
            save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                model=model,
                epoch=epoch,
                val_neg_elbo=val_score,
            )
        if epoch % cfg.sample_every == 0 or epoch in (1, cfg.epochs):
            samples, pairs = save_progress_images(
                model, fixed_z, fixed_x, run_dir / "samples", epoch
            )
            tb_logger.add_images("samples/prior", samples, epoch)
            tb_logger.add_images("samples/reconstructions", pairs, epoch)
        plot_training_curves(read_csv(run_dir / "metrics.csv"), run_dir / "loss_curves")

        terms = "  ".join(f"{k} {v:.1f}" for k, v in train_stats.items() if k != "loss")
        print(
            f"epoch {epoch:3d}/{cfg.epochs}  beta {beta:.2f}  train loss {train_stats['loss']:.1f} "
            f"({terms})  val {val_score:.1f}  {elapsed:.0f}s",
            flush=True,
        )

    return {
        "best_epoch": best["epoch"],
        "best_val_neg_elbo": best["val_neg_elbo"],
        "final_train": train_stats,
        "final_val": val_stats,
        "train_time_s": round(time.perf_counter() - start, 1),
    }


def train(
    cfg: Config,
    *,
    model: nn.Module | None = None,
    loss_fn: LossFn | None = None,
    run_dir: Path | None = None,
) -> Path:
    """Build everything from ``cfg`` (default: your VAE and the negative ELBO), then train."""
    seed_everything(cfg.train.seed)
    device = get_device(cfg.train.device)
    torch.backends.cudnn.benchmark = True

    train_loader = make_loader("train", cfg, shuffle=True)
    val_loader = make_loader("val", cfg, shuffle=False)
    model = (model if model is not None else build_model(cfg)).to(device)
    loss_fn = loss_fn if loss_fn is not None else build_loss(cfg)
    # AdamW = Adam with decoupled weight decay; with weight_decay = 0 it is exactly Adam.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    run_dir = run_dir if run_dir is not None else create_run_dir(EXPERIMENT, cfg.name)
    save_config(cfg, run_dir / "config.yaml")
    write_json(
        run_dir / "env.json",
        {**environment_info(), "device": str(device), "parameters": count_parameters(model)},
    )
    print(f"run folder: {run_dir}")
    print(
        f"device {device} | {count_parameters(model):,} parameters | "
        f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val images"
    )

    tb_logger = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    try:
        # Try to train the model
        summary = train_model(
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            tb_logger,
            cfg=cfg.train,
            run_dir=run_dir,
            device=device,
        )
    finally:
        tb_logger.close()
    write_json(run_dir / "summary.json", summary)
    print(
        f"done: best val {summary['best_val_neg_elbo']:.1f} at epoch {summary['best_epoch']} "
        f"-> {run_dir}"
    )
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    cfg = load_config(Config, args.config, args.overrides)
    try:
        train(cfg)
    except NotImplementedError as err:
        raise SystemExit(
            f"\nNot implemented yet: {err}\n"
            "Finish the TODOs in homeworks/hw1/models/vae.py and losses.py, then check them with:\n"
            "    uv run pytest homeworks/hw1"
        ) from None


if __name__ == "__main__":
    main()
