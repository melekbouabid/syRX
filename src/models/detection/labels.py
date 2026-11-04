"""YOLO detection datasets: label files, pixel boxes, splits and small subsets.

Layout (src/data/prepare_detect_v2.py, also the Kaggle dataset of package_kaggle):

    <root>/images/<split>/<stem>.jpg   <root>/labels/<split>/<stem>.txt   <root>/data.yaml
    <root>/test_holdout.txt            (comparison split: file names of test images)

A label line is `class cx cy w h`, normalised to [0, 1]. torchvision detectors want absolute
x1 y1 x2 y2 boxes and labels starting at 1 (0 is the background): `to_torchvision`.
"""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SPLITS = ("train", "val", "test")
HOLDOUT_FILE = "test_holdout.txt"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def parse_yolo(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Class ids (N) and normalised cx cy w h boxes (N x 4) of a YOLO label file's text."""
    rows = [line.split() for line in text.splitlines() if line.strip()]
    for row in rows:
        if len(row) != 5:
            raise ValueError(f"a YOLO box line has 5 values, got {len(row)}: {' '.join(row)!r}")
    if not rows:
        return np.zeros(0, np.int64), np.zeros((0, 4), np.float32)
    values = np.asarray(rows, dtype=np.float64)
    return values[:, 0].astype(np.int64), values[:, 1:].astype(np.float32)


def yolo_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalised cx cy w h -> absolute x1 y1 x2 y2 pixels, clipped to the image."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    cx, cy = boxes[:, 0] * width, boxes[:, 1] * height
    w, h = boxes[:, 2] * width, boxes[:, 3] * height
    out = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, height)
    return out.astype(np.float32)


def xyxy_to_yolo(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Absolute x1 y1 x2 y2 pixels -> normalised cx cy w h (inverse of yolo_to_xyxy)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2 / width
    cy = (boxes[:, 1] + boxes[:, 3]) / 2 / height
    w = (boxes[:, 2] - boxes[:, 0]) / width
    h = (boxes[:, 3] - boxes[:, 1]) / height
    return np.stack([cx, cy, w, h], axis=1).astype(np.float32)


def to_torchvision(text: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Boxes (N x 4 float32, x1 y1 x2 y2 pixels) and labels (N int64, class + 1) of a label
    file, as torchvision detection models expect them (label 0 is the background).
    Degenerate boxes (zero width or height after clipping) are dropped: torchvision refuses
    them during training."""
    classes, boxes = parse_yolo(text)
    xyxy = yolo_to_xyxy(boxes, width, height)
    keep = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
    return xyxy[keep], (classes[keep] + 1).astype(np.int64)


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path

    def truth(self) -> np.ndarray:
        """Ground-truth boxes in pixels of the image file (N x 4)."""
        height, width = image_size(self.image)
        text = self.label.read_text(encoding="utf-8") if self.label.exists() else ""
        return yolo_to_xyxy(parse_yolo(text)[1], width, height)


def image_size(path: Path) -> tuple[int, int]:
    """(height, width) of an image file."""
    image = read_bgr(path)
    return int(image.shape[0]), int(image.shape[1])


def read_bgr(path: Path) -> np.ndarray:
    """BGR image from any path (cv2.imread fails on some Windows paths)."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode {path}")
    return image


def samples(root: Path, split: str, names: set[str] | None = None) -> list[Sample]:
    """Images of a split (sorted) with their label files; only `names` when given."""
    folder = root / "images" / split
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if names is not None:
        files = [p for p in files if p.name in names]
    return [Sample(p, root / "labels" / split / f"{p.stem}.txt") for p in files]


def holdout_names(root: Path) -> set[str] | None:
    """File names of the comparison split, or None when the dataset has no holdout list."""
    path = root / HOLDOUT_FILE
    if not path.is_file():
        return None
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def write_data_yaml(root: Path, target: Path | None = None) -> Path:
    """Ultralytics data.yaml pointing at `root` (absolute), written to `target`.

    On Kaggle the dataset is read-only: the file is written in the working folder.
    """
    target = target or root / "data.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"path: {root.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        "test: images/test\nnames:\n  0: pill\n",
        encoding="utf-8",
    )
    return target


def make_subset(root: Path, target: Path, counts: dict[str, int], seed: int = 0) -> Path:
    """Copy a random subset (`counts` images per split) of a dataset, for smoke tests.

    The holdout list is kept for the copied test images (all of them when the source has no
    list). Returns the subset root.
    """
    if target.exists():
        shutil.rmtree(target)
    rng = random.Random(seed)
    holdout = holdout_names(root)
    chosen_test: list[str] = []
    for split in SPLITS:
        available = samples(root, split)
        picked = sorted(
            rng.sample(available, min(counts.get(split, 0), len(available))),
            key=lambda s: s.image.name,
        )
        for sample in picked:
            for source, kind in ((sample.image, "images"), (sample.label, "labels")):
                if source.exists():
                    out = target / kind / split / source.name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, out)
            if split == "test":
                chosen_test.append(sample.image.name)
        (target / "images" / split).mkdir(parents=True, exist_ok=True)
        (target / "labels" / split).mkdir(parents=True, exist_ok=True)
    kept = [n for n in chosen_test if holdout is None or n in holdout]
    (target / HOLDOUT_FILE).write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    write_data_yaml(target)
    return target
