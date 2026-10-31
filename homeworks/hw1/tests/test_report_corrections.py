import json
from pathlib import Path

import torch

from dgm.utils.config import load_config
from hw1 import real_baseline
from hw1.config import Config
from hw1.models import VAE


def test_final_config_builds_the_illustrated_wide_model():
    cfg = load_config(Config, Path(__file__).parents[1] / "configs" / "final.yaml")
    model = VAE(cfg.model, cfg.data.image_size)
    assert sum(p.numel() for p in model.parameters()) == 8_425_153
    assert model.latent_dim == 128
    assert cfg.train.beta == 0.03
    assert cfg.train.epochs == 200 and cfg.train.seed == 0
    assert not cfg.train.augment and cfg.model.dropout == 0


def test_baseline_command_scores_matched_training_images_and_records_selection(
    tmp_path, monkeypatch
):
    class Dataset:
        def __init__(self, split, size):
            self.labels = torch.tensor(
                [0] * 30 + [1] * 70 if split == "train" else [0] * 8 + [1] * 12
            )
            self.files = [f"{split}-{i}.jpeg" for i in range(len(self.labels))]

        def as_float(self):
            return self.labels.float().reshape(-1, 1, 1, 1)

    def scorer(images, reference):
        assert len(images) == len(reference) == 20
        assert images.sum() == reference.sum() == 12
        return {"fid": 0.01, "fid_2048": 21.0, "inception_score": 2.0}

    monkeypatch.setattr(real_baseline, "ChestXray", Dataset)
    monkeypatch.setattr(real_baseline, "InceptionScorer", lambda device: scorer)
    monkeypatch.setattr(real_baseline.torch, "set_num_threads", lambda n: None)
    path = tmp_path / "baseline.json"
    real_baseline.main(["--output", str(path)])
    result = json.loads(path.read_text())
    assert result["class_counts"] == {"NORMAL": 8, "PNEUMONIA": 12}
    assert len(set(result["training_files"])) == 20
    assert all(name.startswith("train-") for name in result["training_files"])
    assert result["scores"]["fid_2048"] == 21.0
