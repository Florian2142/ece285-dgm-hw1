from pathlib import Path

import matplotlib.axes
import numpy as np
import pytest

from hw1.prepare_figures import render

DATA = Path(__file__).parents[1] / "results" / "figure_data.npz"


def test_renderer_keeps_all_epochs_and_the_observed_spike(tmp_path, monkeypatch):
    original = matplotlib.axes.Axes.plot
    plotted = []

    def record(ax, *args, **kwargs):
        plotted.append((np.array(args[0]), np.array(args[1])))
        return original(ax, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", record)
    render(DATA, tmp_path)
    assert len(plotted) == 6  # train and validation for each of three loss components
    assert all(np.array_equal(x, np.arange(1, 201)) for x, _ in plotted)
    assert plotted[-1][1][26] == pytest.approx(4984.9817, abs=0.001)
    for name in ("images.pdf", "images.png", "curves.pdf", "curves.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_renderer_rejects_truncated_history(tmp_path):
    with np.load(DATA, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["val"] = arrays["val"][:100]
    bad = tmp_path / "truncated.npz"
    np.savez(bad, **arrays)
    with pytest.raises(ValueError, match="200 finite epochs"):
        render(bad, tmp_path / "output")
    assert not (tmp_path / "output").exists()
