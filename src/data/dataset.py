"""The crop dataset of stage prepare_classify_v2 (src/data/prepare_classify_v2.py).

Layout (also the private Kaggle dataset of package_kaggle):

    <root>/<split>/<class_id:03d>/<name>.jpg      square crops (256 px)
    <root>/index.csv          file, split, class_id, group_id, role, source, box, holdout
    <root>/classes.csv        class_id, class_key, name, generic_name, in_v1, v1_class_id, ...
    <root>/test_holdout.txt   comparison split: test crops of sessions the v1 champion never saw

The output index of every model is the v2 class id (0..379, folders `000`..`379`). The
evaluation preprocessing is the API's (backend/app/inference.py): the crop resized to the
model's square input with INTER_AREA, BGR -> RGB, float in [0, 1].
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.models.detection.labels import read_bgr

ROOT = Path(__file__).resolve().parents[3]  # repository root
DATASET_PATH = "data/processed/c3pi/datasets_v2/classify"  # DVC output (dvc.lock)
DEFAULT_DATA = ROOT / DATASET_PATH
DATASET_NAME = "c3pi-v2-classify"
LABEL_SPACE = "c3pi-v2"  # the class ids of config/splits/c3pi_v2/classes.csv
MANIFEST = "MANIFEST.json"  # written by package_kaggle next to the data
HOLDOUT_FILE = "test_holdout.txt"
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Crop:
    path: Path
    split: str
    label: int  # v2 class id = output index
    holdout: bool = False

    @property
    def relative(self) -> str:
        """`<split>/<class>/<name>`, the key of index.csv and test_holdout.txt."""
        return f"{self.split}/{self.path.parent.name}/{self.path.name}"


def read_index(root: Path) -> list[Crop]:
    """Every crop of index.csv, sorted by file (missing files are an error)."""
    crops = []
    with (root / "index.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            crops.append(
                Crop(
                    root / row["file"],
                    row["split"],
                    int(row["class_id"]),
                    row.get("holdout", "0") == "1",
                )
            )
    return sorted(crops, key=lambda c: c.relative)


def crops(root: Path, split: str, *, holdout_only: bool = False) -> list[Crop]:
    chosen = [c for c in read_index(root) if c.split == split]
    return [c for c in chosen if c.holdout] if holdout_only else chosen


def read_classes(root: Path) -> list[dict[str, Any]]:
    """classes.csv rows (class_id as int, v1_class_id as int or None), by class id."""
    out = []
    with (root / "classes.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            v1 = row.get("v1_class_id", "")
            out.append(
                {
                    **row,
                    "class_id": int(row["class_id"]),
                    "v1_class_id": int(v1) if v1 not in ("", None) else None,
                    "in_v1": str(row.get("in_v1", "")).lower() == "true",
                }
            )
    out.sort(key=lambda r: r["class_id"])
    if [r["class_id"] for r in out] != list(range(len(out))):
        raise ValueError("classes.csv must list the class ids 0..N-1")
    return out


def v1_classes(classes: list[dict[str, Any]]) -> list[int]:
    """v2 class ids of the classes the v1 champion knows."""
    return [r["class_id"] for r in classes if r["v1_class_id"] is not None]


def class_names(classes: list[dict[str, Any]]) -> dict[int, str]:
    """Output index -> folder name ("007"), the `names` metadata the API reads."""
    return {r["class_id"]: f"{r['class_id']:03d}" for r in classes}


def counts(labels: list[int], num_classes: int) -> np.ndarray:
    return np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)


def stratified_fraction(items: list[Crop], fraction: float, seed: int) -> list[Crop]:
    """`fraction` of the crops of every class (at least one per class), for Optuna trials."""
    if fraction >= 1:
        return list(items)
    rng = random.Random(seed)
    by_class: dict[int, list[Crop]] = {}
    for crop in items:
        by_class.setdefault(crop.label, []).append(crop)
    chosen: list[Crop] = []
    for label in sorted(by_class):
        group = by_class[label]
        k = max(1, round(len(group) * fraction))
        chosen += rng.sample(group, min(k, len(group)))
    return sorted(chosen, key=lambda c: c.relative)


def crop_tensor(image: np.ndarray, size: int) -> np.ndarray:
    """A BGR crop as the API feeds the classifier: resized, RGB, float32 1x3xSxS in [0, 1]."""
    if image.shape[:2] != (size, size):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    rgb = image[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(rgb, dtype=np.float32)[None] / 255.0


def load_tensor(path: Path, size: int) -> np.ndarray:
    return crop_tensor(read_bgr(path), size)


def make_subset(root: Path, target: Path, sizes: dict[str, int], seed: int = 0) -> Path:
    """Copy a random subset (`sizes` crops per split) with its index, classes and holdout.

    For smoke tests: the class list stays complete (the models keep every output).
    """
    if target.exists():
        shutil.rmtree(target)
    rng = random.Random(seed)
    every = read_index(root)
    picked: list[Crop] = []
    for split in SPLITS:
        available = [c for c in every if c.split == split]
        picked += rng.sample(available, min(sizes.get(split, 0), len(available)))
    picked.sort(key=lambda c: c.relative)
    names = {c.relative for c in picked}
    for crop in picked:
        out = target / crop.relative
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(crop.path, out)
    with (root / "index.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [r for r in reader if r["file"] in names]
        fields = reader.fieldnames or []
    target.mkdir(parents=True, exist_ok=True)
    with (target / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    shutil.copy2(root / "classes.csv", target / "classes.csv")
    holdout = sorted(c.relative for c in picked if c.holdout)
    (target / HOLDOUT_FILE).write_text("\n".join(holdout) + ("\n" if holdout else ""), "utf-8")
    return target


def _git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def dataset_info(data_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    """Name, DVC hash and dvc.lock id of the data: MANIFEST.json of the Kaggle dataset, else
    the repository's dvc.lock (local runs)."""
    manifest = data_dir / MANIFEST
    if manifest.is_file():
        return json.loads(manifest.read_text(encoding="utf-8"))
    info: dict[str, Any] = {"name": DATASET_NAME, "path": DATASET_PATH}
    lock = root / "dvc.lock"
    if lock.is_file():
        data = yaml.safe_load(lock.read_text(encoding="utf-8")) or {}
        for stage in (data.get("stages") or {}).values():
            for out in stage.get("outs") or []:
                if out.get("path") == DATASET_PATH:
                    info["md5"] = out.get("md5")
        info["dvc_rev"] = _git(root, "hash-object", "dvc.lock") or "unknown"
    return info
