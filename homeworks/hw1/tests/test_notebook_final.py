"""Run the experiment and report cells of notebook 02 (sections 14 to 18) on tiny CPU VAEs.

No real data and no Inception download: a fake scorer, random 16 x 16 images, 2 epochs per run, and every
file in a temporary folder. Checks the seeds and settings of every grid, that each run is scored with both
samplers, that a second Run All trains and scores nothing, and that the report files are written.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from hw1.tests.test_notebook_runs import HPARAMS, notebook_functions  # noqa: F401

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "02_vae_step_by_step.ipynb"
SINGLE_RUNS = ["vae-base", "vae-deep", "vae-calibrated", "a-stack", "r-aug"]  # trained directly in this test


class FakeXray:
    """Stands in for ChestXray in the scores cell: 8 random images per split."""

    def __init__(self, split, size):
        self.images = torch.rand(8, 1, size, size, generator=torch.Generator().manual_seed(len(split)))
        self.labels = torch.tensor([0] * 4 + [1] * 4)

    def as_float(self):
        return self.images


@pytest.fixture
def experiments(notebook_functions, tmp_path, monkeypatch):  # noqa: F811 - imported pytest fixture
    ns = notebook_functions
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = {cell["id"]: "".join(cell["source"]) for cell in nb["cells"]}
    calls = []

    @torch.no_grad()
    def scorer(images, reference):
        assert images.shape == reference.shape == (8, 1, 16, 16)
        assert torch.isfinite(images).all()
        calls.append(images.clone())
        value = float((images - reference).square().mean())
        return {"fid": value, "fid_2048": 100 * value, "inception_score": 1 + value}

    pca_rows = {
        f"pca-{k}": {"run": f"PCA K={k}", "K": k, "val_mse": 0.01, "fid_prior": 50.0, "is_prior": 1.5}
        for k in (128, 256)
    }
    ns.update(
        np=np,
        json=json,
        csv=csv,
        plt=plt,
        ChestXray=FakeXray,
        InceptionScorer=lambda device: scorer,
        FIGURES_DIR=tmp_path / "figures",
        REPORT_DIR=tmp_path / "generated",
        A_STACK={**HPARAMS, "beta": 0.1, "dropout": 0.0, "augment": False, "weight_decay": 0.0},
        PCA_ROWS=pca_rows,
        pca_reconstruct=lambda x, k: x,  # PCA is only a reference row and a figure row here
    )
    ns["FIGURES_DIR"].mkdir()
    ns["REPORT_DIR"].mkdir()
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))

    def run(ident):
        exec(compile(sources[ident], f"{NOTEBOOK.name}:{ident}", "exec"), ns)

    run("scores")
    run("tables")
    return ns, run, calls


def test_sections_14_to_18_train_score_cache_and_write_the_report(experiments, monkeypatch):
    ns, run, calls = experiments
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for name in SINGLE_RUNS:  # earlier sections' runs that the report table lists
            model, saved_run = ns["train_vae"]({**HPARAMS, "name": name})
            ns["save_vae"](model, saved_run)
        seen_seeds, original_seed = [], ns["seed_everything"]

        def seed(value):
            seen_seeds.append(value)
            original_seed(value)

        monkeypatch.setitem(ns, "seed_everything", seed)

        # 13: the KL-weight grid on the stack, every point scored with both samplers
        run("b-train")
        assert set(ns["B"]) == {"b-beta0.03", "b-beta0.1", "b-beta0.3", "b-beta1", "b-calibrated"}

        # 14: the regularization winners together, two betas, three seeds
        seen_seeds.clear()
        run("final-train")
        assert seen_seeds == [0, 1, 2, 0, 1, 2]
        for beta in (0.1, 1.0):
            for s in (0, 1, 2):
                hp = ns["load_vae"](f"final-beta{beta:g}-seed{s}")[1]["hparams"]
                assert hp["augment"] and hp["dropout"] == 0.2 and hp["beta"] == beta and hp["seed"] == s
        run("final-table")

        # 15: the 2 x 2 grid at beta = 1 with augmentation, three declared seeds per setting
        seen_seeds.clear()
        run("c-train")
        assert seen_seeds == [0, 1, 2] * 4
        for name in ns["C"]:
            for s in ns["C_SEEDS"]:
                hp = ns["load_vae"](f"{name}-seed{s}")[1]["hparams"]
                assert hp["augment"] and hp["beta"] == 1.0 and hp["dropout"] == 0.0 and hp["seed"] == s
                assert hp["latent_dim"] == (256 if name.endswith("k256") else HPARAMS["latent_dim"])
                assert hp["likelihood"] == ("gaussian_calibrated" if "calib" in name else "gaussian")
        run("c-table")

        # 16: the rule picks the base among sections 13 to 15, one change at a time, two more seeds for the better
        seen_seeds.clear()
        run("x")
        runs = ns["RUNS"]
        assert set(runs) == {*ns["B"], "final-beta0.1", "final-beta1", *ns["C"]}
        fid = {
            name: np.mean([ns["score_run"](r)["samples_aggregate"]["fid_2048"][0] for r in names])
            for name, names in runs.items()
        }
        assert ns["X_FROM"] == min(fid, key=fid.get)
        fitted = {name: ns["score_run"](f"{name}-seed0")["samples_aggregate"]["fid_2048"][0] for name in ns["X"]}
        assert ns["X_BEST"] == min(fitted, key=fitted.get)
        assert seen_seeds == [0, 0, 1, 2]
        base = ns["load_vae"](runs[ns["X_FROM"]][0])[1]["hparams"]
        for name, change in ns["X"].items():
            hp = ns["load_vae"](f"{name}-seed0")[1]["hparams"]
            assert {k: v for k, v in hp.items() if k not in ("name", "seed")} == {
                **{k: v for k, v in base.items() if k not in ("name", "seed")},
                **change,
            }
        assert ns["load_vae"]("x-group-seed0")[1]["hparams"]["norm"] == "group"
        assert ns["load_vae"]("x-width64-seed0")[1]["hparams"]["hidden_channels"] == 64
        for s in (1, 2):
            assert ns["load_vae"](f"{ns['X_BEST']}-seed{s}")[1]["hparams"]["seed"] == s

        for path in ns["MODELS_DIR"].glob("*_scores.json"):  # both samplers, (mean, std) each
            saved = json.loads(path.read_text())
            for key in ("samples", "samples_aggregate"):
                assert len(saved[key]["fid_2048"]) == 2

        # Run All once more: nothing is trained or scored again, no file changes
        n_calls = len(calls)
        files = {path: path.read_bytes() for path in ns["MODELS_DIR"].glob("*")}

        def no_training(*args, **kwargs):
            raise AssertionError("a saved run was trained again")

        monkeypatch.setitem(ns, "train_vae", no_training)
        for ident in ["b-train", "final-train", "c-train", "x"]:
            run(ident)
        assert len(calls) == n_calls
        assert all(path.read_bytes() == content for path, content in files.items())

        # 17: one table with every reported row, all scored the same way
        run("report-table")
        report = ns["REPORT"]
        fixed = ["vae-base", "vae-deep", "vae-calibrated", "a-stack", "r-aug", "b-beta1", "b-calibrated"]
        assert list(report)[:8] == [*fixed, "final-beta1"] and ns["X_FROM"] in report
        assert list(report)[-1] == ns["X_BEST"] and len(report[ns["X_BEST"]]) == 3
        for names in report.values():
            for name in names:
                saved = json.loads((ns["MODELS_DIR"] / f"{ns['newest_run'](name)}_scores.json").read_text())
                assert {"samples", "samples_aggregate"} <= saved.keys()
        tex = (ns["REPORT_DIR"] / "results.tex").read_text(encoding="utf-8")
        assert tex.count(r"\texttt{") == len(report) and tex.count("PCA &") == 2
        with open(ns["REPORT_DIR"] / "results.csv", newline="") as stream:
            assert len(list(csv.DictReader(stream))) == len(report)

        # 18: the figures of the reported row (lowest fitted-prior FID)
        ns["val"] = TensorDataset(torch.rand(476, 1, 16, 16), torch.zeros(476, dtype=torch.long))
        run("report-figures")
        table = ns["report_table"]
        assert ns["REPORTED"] == min(table, key=lambda label: table[label]["fid_fitted"])
        macro = (ns["REPORT_DIR"] / "reported.tex").read_text(encoding="utf-8")
        assert macro == "\\newcommand{\\ReportedRun}{\\texttt{" + ns["REPORTED"] + "}}\n"
        for name in ["images.png", "curves.png", "reported.txt"]:
            assert (ns["REPORT_DIR"] / name).stat().st_size > 0
    finally:
        plt.close("all")
        torch.set_num_threads(old_threads)


def test_old_score_cache_returns_current_real_baseline_without_changing_model_scores(experiments, monkeypatch):
    ns, _, calls = experiments
    cached = {"samples": {"fid_2048": [251.1, 0.5]},
              "samples_aggregate": {"fid_2048": [101.7, 1.6]},
              "real_images": {"fid_2048": 70.2}}
    path = ns["MODELS_DIR"] / "old_scores.json"
    path.write_text(json.dumps(cached))
    original = path.read_bytes()
    monkeypatch.setitem(ns, "newest_run", lambda name: "old")
    n_calls = len(calls)
    result = ns["score_run"]("old")
    assert result["real_images"] == ns["real_scores"]
    assert result["samples"] == cached["samples"]
    assert result["samples_aggregate"] == cached["samples_aggregate"]
    assert path.read_bytes() == original and len(calls) == n_calls
