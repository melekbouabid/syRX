"""Pack the crop dataset of C3PI v2 for a private Kaggle dataset (step 2.3).

    python -m src.models.classification.package_kaggle                  # runs/kaggle/ (git-ignored)
    python -m src.models.classification.package_kaggle --owner <kaggle-username> --no-zip

Input: the output of the DVC stage prepare_classify_v2 (data/processed/c3pi/datasets_v2/classify).
Output, under runs/kaggle/ (never committed, never pushed anywhere by this script):

    tujya-c3pi-classify-v2/          folder for `kaggle datasets create -p <folder>` (optional)
      train/ val/ test/ <class>/*.jpg     the square crops (256 px), one folder per v2 class
      index.csv  classes.csv  test_holdout.txt  summary.json
      MANIFEST.json                 dataset name, DVC hash (dvc.lock), dvc_rev, packing commit
      dataset-metadata.json         Kaggle metadata (private by default, licence ODbL-1.0)
      README.md                     source, licence and terms of C3PI
    tujya-c3pi-classify-v2.zip      the same content, to upload through the Kaggle website

Files are hard-linked when possible (no extra disk space), copied otherwise. The notebooks
read MANIFEST.json, so every run imported into MLflow carries the exact data version.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from src.models.classification import data as D
from src.models.detection.runner import code_info

SOURCE = D.DEFAULT_DATA
OUT = D.ROOT / "runs" / "kaggle"
SLUG = "tujya-c3pi-classify-v2"
EXTRA_FILES = ("index.csv", "classes.csv", D.HOLDOUT_FILE, "summary.json")
README = """# Tujya C3PI v2 pill classification (private)

Square crops (256 px, one pill with a 20 % margin) of the 380 classes of C3PI v2, built from
the NLM Computational Photography Project for Pill Identification (C3PI),
https://data.lhncbc.nlm.nih.gov/public/Pills/ . Licence: ODbL, research and development use
only; the images remain those of the U.S. National Library of Medicine.

Folders `<split>/<class_id>/`: the class ids are those of `classes.csv` (C3PI product NDC9 and
part in `class_key`). Splits by photo session, no leakage; the augmented images (train only)
come from train images. `test_holdout.txt` lists the test crops used to compare the
classifiers (sessions the previous model never saw).

Built by the Tujya project, https://github.com/melekbouabid/syRX (step 2.3,
docs/module1/kaggle_classifieurs.md). Kept private: do not redistribute.
"""


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def check_dataset(root: Path) -> dict[str, int]:
    """Crops per split, every crop of index.csv on disk and in the folder of its class, a
    complete class list, holdout crops all in test."""
    every = D.read_index(root)
    missing = [c.relative for c in every if not c.path.is_file()]
    if missing:
        raise ValueError(f"{len(missing)} crops of index.csv are missing, e.g. {missing[0]}")
    wrong = [c.relative for c in every if c.path.parent.name != f"{c.label:03d}"]
    if wrong:
        raise ValueError(f"{len(wrong)} crops are not in the folder of their class: {wrong[0]}")
    classes = D.read_classes(root)
    unknown = {c.label for c in every} - {r["class_id"] for r in classes}
    if unknown:
        raise ValueError(f"classes missing from classes.csv: {sorted(unknown)[:5]}")
    holdout_file = root / D.HOLDOUT_FILE
    listed = (
        set(holdout_file.read_text(encoding="utf-8").split()) if holdout_file.is_file() else set()
    )
    test = {c.relative for c in every if c.split == "test"}
    if not listed <= test:
        raise ValueError(f"{len(listed - test)} holdout crops are not in the test split")
    counts = {split: sum(c.split == split for c in every) for split in D.SPLITS}
    counts["holdout"] = len(listed)
    counts["classes"] = len(classes)
    return counts


def kaggle_metadata(owner: str, slug: str) -> dict[str, Any]:
    return {
        "title": "Tujya C3PI v2 pill classification",
        "id": f"{owner}/{slug}",
        "subtitle": "Square pill crops of 380 C3PI products (one folder per class)",
        "licenses": [{"name": "ODbL-1.0"}],
        "keywords": ["computer vision", "image classification", "health"],
    }


def package(
    source: Path = SOURCE,
    out: Path = OUT,
    *,
    slug: str = SLUG,
    owner: str = "KAGGLE_USERNAME",
    make_zip: bool = True,
) -> dict[str, Any]:
    counts = check_dataset(source)
    folder = out / slug
    if folder.exists():
        shutil.rmtree(folder)
    for crop in D.read_index(source):
        _link_or_copy(crop.path, folder / crop.relative)
    for name in EXTRA_FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, folder / name)
    info = D.dataset_info(source)
    manifest = {
        **{k: v for k, v in info.items() if k in ("name", "md5", "dvc_rev", "path")},
        "name": info.get("name", D.DATASET_NAME),
        "label_space": D.LABEL_SPACE,
        "images": counts,
        "packed": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "packed_from_commit": code_info()["git_commit"],
        "slug": slug,
    }
    (folder / D.MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (folder / "dataset-metadata.json").write_text(
        json.dumps(kaggle_metadata(owner, slug), indent=2) + "\n", encoding="utf-8"
    )
    (folder / "README.md").write_text(README, encoding="utf-8")
    size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
    result: dict[str, Any] = {"folder": str(folder), "bytes": size, **manifest}
    if make_zip:
        archive = out / f"{slug}.zip"
        archive.unlink(missing_ok=True)
        # JPEG files do not compress: store them, deflate the small text files.
        with zipfile.ZipFile(archive, "w") as handle:
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.name != "dataset-metadata.json":
                    method = zipfile.ZIP_STORED if path.suffix == ".jpg" else zipfile.ZIP_DEFLATED
                    handle.write(path, path.relative_to(folder).as_posix(), compress_type=method)
        result["zip"] = str(archive)
        result["zip_bytes"] = archive.stat().st_size
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.classification.package_kaggle")
    parser.add_argument("--data", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--slug", default=SLUG)
    parser.add_argument("--owner", default="KAGGLE_USERNAME", help="for dataset-metadata.json")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args(argv)
    result = package(
        args.data, args.out, slug=args.slug, owner=args.owner, make_zip=not args.no_zip
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
