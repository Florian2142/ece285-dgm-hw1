import numpy as np
import pytest
import torch

from dgm.data import chest_xray
from dgm.data.chest_xray import ChestXray, find_split_root, patient_id, prepare, split_by_patient
from dgm.data.synthetic import make_fake_chest_xray
from dgm.training.runs import read_json


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("person1_bacteria_1.jpeg", "PERSON1"),
        ("person1000_virus_1681.jpeg", "PERSON1000"),
        ("IM-0115-0001.jpeg", "IM-0115"),
        ("NORMAL2-IM-0123-0001.jpeg", "NORMAL2-IM-0123"),
        ("unexpected_name.png", "unexpected_name"),
    ],
)
def test_patient_id(filename, expected):
    assert patient_id(filename) == expected


def test_split_by_patient_keeps_patients_together():
    names = [f"person{i // 3}_virus_{i}.jpeg" for i in range(60)]  # 20 patients x 3 images
    train, val = split_by_patient(names, val_fraction=0.25, seed=0)
    assert sorted(train + val) == list(range(60))
    train_patients = {patient_id(names[i]) for i in train}
    val_patients = {patient_id(names[i]) for i in val}
    assert not train_patients & val_patients
    assert len(val_patients) == 5
    assert split_by_patient(names, 0.25, seed=0) == (train, val)  # deterministic


def test_find_split_root_skips_nesting_and_macosx(tmp_path):
    split_root = make_fake_chest_xray(tmp_path, counts={"train": 2, "test": 2})
    assert find_split_root(tmp_path) == split_root
    with pytest.raises(FileNotFoundError):
        find_split_root(tmp_path / "empty")


def test_prepare_and_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("DGM_DATA_DIR", str(tmp_path))
    make_fake_chest_xray(chest_xray.raw_dir(), counts={"train": 24, "val": 4, "test": 8})
    out = prepare(32, val_fraction=0.25, seed=0)

    meta = read_json(out / "meta.json")
    assert meta["counts"]["train"]["total"] + meta["counts"]["val"]["total"] == 28
    assert meta["counts"]["test"]["total"] == 8

    with np.load(out / "train.npz") as data:
        assert data["images"].dtype == np.uint8 and data["images"].shape[1:] == (32, 32)
        assert set(data["labels"].tolist()) <= {0, 1}

    dataset = ChestXray("val", image_size=32)
    image, label = dataset[0]
    assert image.shape == (1, 32, 32) and image.dtype == torch.float32
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert label.item() in (0, 1)
    assert dataset.as_float().shape == (len(dataset), 1, 32, 32)


def test_dataset_explains_missing_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DGM_DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match=r"dgm\.data\.chest_xray"):
        ChestXray("train", image_size=64)
