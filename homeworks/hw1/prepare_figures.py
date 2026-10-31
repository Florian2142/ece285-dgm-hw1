"""Rebuild Figures 1 and 2 from the saved report inputs, without training or downloads.

AI-assisted plotting and export code using the saved outputs of my experiments.

    uv run python -m hw1.prepare_figures --output homeworks/hw1/results/figures

figure_data.npz contains the original decoded images and all 200 epochs of losses.
figure_provenance.json records the checkpoints and sampling used to produce those inputs.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(data_path: Path, out_dir: Path) -> None:
    with np.load(data_path, allow_pickle=False) as data:
        left, right = data["left"], data["right"]
        best_epoch = int(data["best_epoch"])
        curves = {split: data[split] for split in ("train", "val")}
    if left.shape != (3, 4, 1, 64, 64) or right.shape != (4, 4, 1, 64, 64):
        raise ValueError("Expected the report's 3/4 rows of four 64px grayscale images")
    if any(a.shape != (200, 3) or not np.isfinite(a).all() for a in curves.values()):
        raise ValueError(
            "Expected all 200 finite epochs, with negative ELBO, reconstruction NLL, KL"
        )
    if not 1 <= best_epoch <= 200 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Invalid selected epoch or image values")
    history = {
        split: [dict(zip(("neg_elbo", "recon", "kl"), row, strict=True)) for row in values]
        for split, values in curves.items()
    }
    OUT = out_dir
    OUT.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(6.5, 2.3), facecolor="white")
    # Fixed physical cell sizes avoid tight_layout shrinking every X-ray to a thumbnail.
    cell, gap, top = 0.48, 0.025, 2.08
    for start, rows, labels, title in [
        (1.0, left, ["Validation", "Baseline", "Wide model"], "Reconstructions"),
        (
            4.48,
            right,
            ["Test images", "Baseline: prior", "Wide: prior", "Wide: fitted"],
            "New images",
        ),
    ]:
        fig.text(start / 6.5, 2.2 / 2.3, title, fontsize=9, weight="bold")
        for r, (images, label) in enumerate(zip(rows, labels, strict=True)):
            bottom = top - (r + 1) * cell - r * gap
            fig.text(
                (start - 0.07) / 6.5,
                (bottom + cell / 2) / 2.3,
                label,
                ha="right",
                va="center",
                fontsize=8,
            )
            for c in range(4):
                ax = fig.add_axes(
                    [(start + c * (cell + gap)) / 6.5, bottom / 2.3, cell / 6.5, cell / 2.3]
                )
                ax.imshow(images[c, 0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_axis_off()
    fig.savefig(OUT / "images.pdf")
    fig.savefig(OUT / "images.png", dpi=220)
    plt.close(fig)

    epochs = np.arange(1, 201)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.35))
    for ax, term, title in zip(
        axes,
        ["neg_elbo", "recon", "kl"],
        ["Negative ELBO", "Reconstruction NLL", "KL"],
        strict=True,
    ):
        for split, label, color in [
            ("train", "Train", "#24658a"),
            ("val", "Validation", "#bb572f"),
        ]:
            values = np.array([e[term] for e in history[split]])
            assert np.isfinite(values).all()
            ax.plot(epochs, values, label=label, color=color, lw=0.9)
        # Show the entire saved history, including early spikes.
        ax.axvline(best_epoch, color="#666666", ls=":", lw=0.8)
        ax.set(title=title, xlabel="Epoch", xlim=(1, 200))
        ax.tick_params(labelsize=7, pad=2)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Nats / image", fontsize=7)
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.tight_layout(pad=0.35, w_pad=1)
    fig.savefig(OUT / "curves.pdf")
    fig.savefig(OUT / "curves.png", dpi=220)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).parent / "results/figure_data.npz"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    render(args.data, args.output)
    print(f"Rendered Figures 1 and 2 to {args.output}")


if __name__ == "__main__":
    main()
