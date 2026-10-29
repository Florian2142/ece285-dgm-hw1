"""Run the accepted section 14 on tiny CPU VAEs; no real data or Inception download."""

import ast
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from hw1.tests.test_notebook_runs import HPARAMS, notebook_functions  # noqa: F401

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "02_vae_step_by_step.ipynb"


@pytest.fixture
def final_section(notebook_functions, tmp_path, monkeypatch):
    ns = notebook_functions
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    ns.update(
        A_STACK={**HPARAMS, "beta": 0.1, "epochs": 2, "weight_decay": 0.0},
        json=json, csv=csv, np=np, plt=plt, FIGURES_DIR=tmp_path / "figures",
    )
    ns["FIGURES_DIR"].mkdir()
    ns["x_fixed"] = ns["train"].tensors[0]
    ns["test_images"] = torch.rand(8, 1, 16, 16, generator=torch.Generator().manual_seed(19))
    calls = []

    @torch.no_grad()
    def scorer(images, reference):
        assert images.shape == reference.shape == (8, 1, 16, 16)
        assert torch.isfinite(images).all()
        calls.append(images.clone())
        value = float((images - reference).square().mean())
        return {"fid": value, "fid_2048": 100 * value, "inception_score": 1 + value}

    ns["scorer"] = scorer
    ns["real_scores"] = {"fid": 0.0, "fid_2048": 0.0, "inception_score": 1.0}
    ns["pca_reconstruct"] = lambda x, k: x  # PCA is only a figure reference in this test
    for ident in ["cell-22", "cell-24", "cell-32", "cell-55"]:
        tree = ast.parse(sources[ident])
        functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
        exec(compile(functions, ident, "exec"), ns)
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    return ns, sources, calls


def test_final_six_seed_runs_both_samplers_cache_and_figures(final_section, monkeypatch):
    ns, sources, score_calls = final_section
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        baseline, base_run = ns["train_vae"]({**HPARAMS, "name": "vae-base"})
        ns["save_vae"](baseline, base_run)
        seen_seeds, initial_weights = [], []
        original_seed, original_build = ns["seed_everything"], ns["build_vae"]

        def seed(value):
            seen_seeds.append(value)
            original_seed(value)

        def build(settings):
            model = original_build(settings)
            initial_weights.append(next(model.parameters()).detach().clone())
            return model

        monkeypatch.setitem(ns, "seed_everything", seed)
        monkeypatch.setitem(ns, "build_vae", build)
        exec(compile(sources["cell-57"], "final-training", "exec"), ns)
        assert seen_seeds == [0, 1, 2, 0, 1, 2]
        assert set(ns["final_runs"]) == {(b, s) for b in [0.1, 1.0] for s in [0, 1, 2]}
        assert not torch.equal(initial_weights[0], initial_weights[1])
        torch.testing.assert_close(initial_weights[0], initial_weights[3])
        for (beta, run_seed), (model, run) in ns["final_runs"].items():
            assert not model.training
            assert run["hparams"]["seed"] == run_seed and run["hparams"]["beta"] == beta
            assert run["hparams"]["augment"] and run["hparams"]["dropout"] == 0.2
            assert run["epochs_completed"] == 2
            assert run["best_val_neg_elbo"] == min(v["neg_elbo"] for v in run["history"]["val"])

        fit_datasets = []
        original_reconstruct = ns["reconstruct_all"]

        def reconstruct(model, dataset):
            fit_datasets.append(dataset)
            return original_reconstruct(model, dataset)

        monkeypatch.setitem(ns, "reconstruct_all", reconstruct)
        exec(compile(sources["cell-58"], "final-scoring", "exec"), ns)
        assert len(score_calls) == 36  # six models, two samplers, three sample repetitions
        assert fit_datasets == [ns["train"]] * 6
        score_paths = list(ns["MODELS_DIR"].glob("final-*_scores.json"))
        assert len(score_paths) == 6
        for path in score_paths:
            saved = json.loads(path.read_text())
            assert {"samples", "samples_aggregate", "real_images"} <= saved.keys()
            for kind in ["samples", "samples_aggregate"]:
                assert len(saved[kind]["fid_2048"]) == 2  # within-model mean and sampling SD

        original_files = {p: p.read_bytes() for p in ns["MODELS_DIR"].glob("*")}

        def no_training(*args, **kwargs):
            raise AssertionError("A completed final run was unexpectedly retrained")

        monkeypatch.setitem(ns, "train_vae", no_training)
        exec(sources["cell-57"], ns)
        exec(sources["cell-58"], ns)
        assert seen_seeds == [0, 1, 2, 0, 1, 2] and len(score_calls) == 36
        assert all(p.read_bytes() == content for p, content in original_files.items())

        exec(compile(sources["cell-59"], "final-report", "exec"), ns)
        assert len(ns["final_rows"]) == 6
        expected_beta = min([0.1, 1.0], key=lambda b: np.mean([
            row["fid_prior"] for row in ns["final_rows"] if row["beta"] == b
        ]))
        assert ns["FINAL_BETA"] == expected_beta
        assert ns["final_run"]["hparams"]["seed"] == 0
        tables = list(ns["FIGURES_DIR"].glob("final_table_*.csv"))
        assert len(tables) == 1
        with tables[0].open(newline="") as stream:
            assert len(list(csv.DictReader(stream))) == 6
        assert len(list(ns["FIGURES_DIR"].glob("final_reconstructions_*.pdf"))) == 1
        assert len(list(ns["FIGURES_DIR"].glob("final_samples_*.pdf"))) == 1
        assert len(list(ns["FIGURES_DIR"].glob("*_loss_curves.pdf"))) == 1
    finally:
        plt.close("all")
        torch.set_num_threads(old_threads)
