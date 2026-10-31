"""Chest X-ray (pneumonia) images from Kaggle, prepared for generative modeling.

AI-assisted supporting code for downloading, splitting, and preprocessing the data.

Source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
(Kermany et al., Cell 2018). 5,856 JPEG images of different sizes; Kaggle splits
train / val / test, classes NORMAL and PNEUMONIA.

Download once (2.5 GB zip, resumes after network errors; one copy of the images is
extracted, about 1.2 GB) and cache resized grayscale arrays (no Kaggle account needed):

    uv run python -m dgm.data.chest_xray --sizes 64 128

Splits used in this repo
- train / val: Kaggle "train" and Kaggle "val" (only 16 images) are pooled, then split
  by PATIENT (file-name prefix), so no patient is in both. ``val_fraction`` of the
  patients go to val; the split is seeded and recorded in meta.json.
- test: Kaggle "test", untouched. Use it as the held-out reference set for FID.

Preprocessing: grayscale, center crop to a square, resize to ``size x size`` (Lanczos).
Saved as ``data/processed/chest_xray/<size>px/<split>.npz`` with ``images`` (N, H, W)
uint8, ``labels`` (N,) int64 (0 = NORMAL, 1 = PNEUMONIA), and ``files`` (N,) str.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import zipfile
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm

from dgm.data.download import download_file, extract_zip
from dgm.training.runs import read_json, write_json
from dgm.utils.paths import data_dir

KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"
CLASSES = ("NORMAL", "PNEUMONIA")
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
_PATIENT_PATTERNS = (
    re.compile(r"^(person\d+)_", re.IGNORECASE),  # PNEUMONIA: person1_bacteria_1.jpeg
    re.compile(r"^((?:NORMAL2-)?IM-\d+)-", re.IGNORECASE),  # NORMAL: IM-0115-0001.jpeg
)


def raw_dir() -> Path:
    return data_dir() / "raw" / "chest_xray"


def processed_dir(image_size: int) -> Path:
    return data_dir() / "processed" / "chest_xray" / f"{image_size}px"


# --- Download --------------------------------------------------------------------


def find_split_root(root: Path) -> Path:
    """Return the shallowest folder under ``root`` that contains ``train/`` and ``test/``.

    The Kaggle archive nests the data (``chest_xray/chest_xray/...``) and includes
    ``__MACOSX`` junk; this finds the real split folders regardless.
    """
    if root.is_dir():
        folders = sorted(
            [root, *(p for p in root.rglob("*") if p.is_dir())], key=lambda p: len(p.parts)
        )
        for folder in folders:
            if (
                "__MACOSX" not in folder.parts
                and (folder / "train").is_dir()
                and (folder / "test").is_dir()
            ):
                return folder
    raise FileNotFoundError(
        f"No chest X-ray split folders (train/, test/) under {root}. Run: uv run python -m dgm.data.chest_xray"
    )


def archive_split_root(names: Iterable[str]) -> str:
    """The shallowest folder in the archive that holds ``train/`` and ``test/``.

    The Kaggle zip (2.5 GB) contains the images twice (``chest_xray/`` and a nested
    ``chest_xray/chest_xray/``); extracting one copy halves time and disk space.
    """
    found: dict[str, set[str]] = {}
    for name in names:
        parts = PurePosixPath(name).parts
        if "__MACOSX" in parts:
            continue
        for i, part in enumerate(parts[:-1]):
            if part in SPLITS:
                found.setdefault("/".join(parts[:i]), set()).add(part)
                break
    roots = [root for root, splits in found.items() if {"train", "test"} <= splits]
    if not roots:
        raise ValueError("the archive has no train/ and test/ folders")
    return min(roots, key=lambda root: len(PurePosixPath(root).parts))


def in_split_root(name: str, root: str) -> bool:
    """True for archive members under ``<root>/{train,val,test}/`` (not the nested duplicate)."""
    if root and not name.startswith(root + "/"):
        return False
    first = PurePosixPath(name[len(root) + 1 :] if root else name).parts[:1]
    return bool(first) and first[0] in SPLITS


def download(*, keep_archive: bool = False) -> Path:
    """Download and extract the dataset into ``data/raw/chest_xray`` (skipped if present)."""
    try:
        return find_split_root(raw_dir())
    except FileNotFoundError:
        pass
    archive = download_file(
        DOWNLOAD_URL,
        data_dir() / "downloads" / "chest-xray-pneumonia.zip",
        desc="chest X-ray zip (2.5 GB)",
    )
    with zipfile.ZipFile(archive) as zf:
        root = archive_split_root(zf.namelist())
    extract_zip(archive, raw_dir(), keep=lambda name: in_split_root(name, root))
    if not keep_archive:
        archive.unlink()
    return find_split_root(raw_dir())


# --- Indexing and splitting ---------------------------------------------------------


def list_images(split_root: Path, split: str) -> list[tuple[Path, int]]:
    """All (path, label) pairs of one Kaggle split, sorted for reproducibility."""
    items: list[tuple[Path, int]] = []
    for label, cls in enumerate(CLASSES):
        folder = split_root / split / cls
        if folder.is_dir():
            items += [
                (p, label) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES
            ]
    return items


def patient_id(filename: str) -> str:
    """Patient key from a file name (``person1_virus_6.jpeg`` -> ``person1``).

    Falls back to the file stem, which treats an unknown name as its own patient.
    """
    for pattern in _PATIENT_PATTERNS:
        match = pattern.match(filename)
        if match:
            return match.group(1).upper()
    return Path(filename).stem


def split_by_patient(
    filenames: Sequence[str], val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Indices (train, val) such that every patient lands in exactly one split."""
    patients = sorted({patient_id(name) for name in filenames})
    n_val = round(len(patients) * val_fraction)
    shuffled = np.random.default_rng(seed).permutation(len(patients))
    val_patients = {patients[i] for i in shuffled[:n_val]}
    train_idx = [i for i, name in enumerate(filenames) if patient_id(name) not in val_patients]
    val_idx = [i for i, name in enumerate(filenames) if patient_id(name) in val_patients]
    return train_idx, val_idx


# --- Preprocessing ------------------------------------------------------------------


def load_image(path: Path, size: int) -> np.ndarray:
    """Grayscale, center-crop to a square, resize to ``size x size``; uint8 array."""
    with Image.open(path) as img:
        img.draft("L", (2 * size, 2 * size))  # fast JPEG decode at reduced scale (stays >= 2x size)
        square = ImageOps.fit(img.convert("L"), (size, size), method=Image.Resampling.LANCZOS)
        return np.asarray(square, dtype=np.uint8)


def load_images(paths: Sequence[Path], size: int, desc: str = "") -> np.ndarray:
    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as pool:
        images = list(
            tqdm(
                pool.map(lambda p: load_image(p, size), paths),
                total=len(paths),
                desc=desc,
                unit="img",
            )
        )
    return np.stack(images) if images else np.zeros((0, size, size), dtype=np.uint8)


def prepare(
    image_size: int = 64, *, val_fraction: float = 0.1, seed: int = 0, force: bool = False
) -> Path:
    """Build ``<split>.npz`` for train/val/test at one resolution (skipped if present)."""
    out = processed_dir(image_size)
    if not force and all((out / f"{split}.npz").exists() for split in SPLITS):
        return out
    src = find_split_root(raw_dir())
    pooled = list_images(src, "train") + list_images(src, "val")
    train_idx, val_idx = split_by_patient([p.name for p, _ in pooled], val_fraction, seed)
    splits = {
        "train": [pooled[i] for i in train_idx],
        "val": [pooled[i] for i in val_idx],
        "test": list_images(src, "test"),
    }
    out.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "source": f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
        "image_size": image_size,
        "preprocessing": "grayscale, center crop to square, Lanczos resize, uint8",
        "split": f"Kaggle train+val pooled, split by patient (val_fraction={val_fraction}, seed={seed}); test = Kaggle test",
        "classes": list(CLASSES),
        "counts": {},
    }
    for split, items in splits.items():
        paths = [p for p, _ in items]
        labels = np.asarray([label for _, label in items], dtype=np.int64)
        images = load_images(paths, image_size, desc=f"{split} {image_size}px")
        files = np.asarray([p.relative_to(src).as_posix() for p in paths])
        np.savez(out / f"{split}.npz", images=images, labels=labels, files=files)
        meta["counts"][split] = {cls: int((labels == i).sum()) for i, cls in enumerate(CLASSES)}
        meta["counts"][split]["total"] = len(items)
    write_json(out / "meta.json", meta)
    return out


# --- PyTorch dataset ------------------------------------------------------------------


class ChestXray(Dataset):
    """A prepared split. Items are ``(image, label)`` with image ``[1, H, W]`` float in [0, 1]."""

    def __init__(self, split: str, image_size: int = 64) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        path = processed_dir(image_size) / f"{split}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Prepare the data first:\n"
                f"    uv run python -m dgm.data.chest_xray --sizes {image_size}"
            )
        with np.load(path) as data:
            self.images = torch.from_numpy(data["images"]).unsqueeze(1)  # uint8 [N, 1, H, W]
            self.labels = torch.from_numpy(data["labels"])
            self.files: list[str] = data["files"].tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index].float().div_(255), self.labels[index]

    def as_float(self) -> torch.Tensor:
        """The whole split as one ``[N, 1, H, W]`` float tensor in [0, 1]."""
        return self.images.float().div_(255)


def describe(image_size: int) -> str:
    meta = read_json(processed_dir(image_size) / "meta.json")
    lines = [f"chest X-ray {image_size}px -> {processed_dir(image_size)}"]
    for split, counts in meta["counts"].items():
        per_class = ", ".join(f"{cls} {counts[cls]}" for cls in CLASSES)
        lines.append(f"  {split:<5} {counts['total']:>5} images ({per_class})")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare the Kaggle chest X-ray dataset."
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[64], help="square image sizes to cache"
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.1, help="fraction of patients for validation"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of the patient split")
    parser.add_argument("--force", action="store_true", help="rebuild the processed arrays")
    parser.add_argument("--keep-archive", action="store_true", help="keep the downloaded zip")
    parser.add_argument(
        "--redownload", action="store_true", help="delete the raw images and download again"
    )
    args = parser.parse_args(argv)

    if args.redownload:
        shutil.rmtree(raw_dir(), ignore_errors=True)
    print(f"raw images: {download(keep_archive=args.keep_archive)}")
    for size in args.sizes:
        prepare(size, val_fraction=args.val_fraction, seed=args.seed, force=args.force)
        print(describe(size))


if __name__ == "__main__":
    main()
