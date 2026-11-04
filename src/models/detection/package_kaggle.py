"""Pack the detection dataset of C3PI v2 for a private Kaggle dataset (step 2.2).

    python -m src.models.detection.package_kaggle                     # runs/kaggle/ (git-ignored)
    python -m src.models.detection.package_kaggle --owner <kaggle-username> --no-zip

Input: the output of the DVC stage prepare_detect_v2 (data/processed/c3pi/datasets_v2/detect).
Output, under runs/kaggle/ (never committed, never pushed anywhere by this script):

    tujya-c3pi-detect-v2/            folder for `kaggle datasets create -p <folder>` (optional)
      images/{train,val,test}/  labels/{train,val,test}/   the YOLO dataset (640 px)
      test_holdout.txt  index.csv  summary.json            comparison split and statistics
      MANIFEST.json                 dataset name, DVC hash (dvc.lock), dvc_rev, packing commit
      dataset-metadata.json         Kaggle metadata (private by default, licence ODbL-1.0)
      README.md                     source, licence and terms of C3PI
    tujya-c3pi-detect-v2.zip        the same content, to upload through the Kaggle website

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

from src.models.detection.labels import HOLDOUT_FILE, SPLITS, holdout_names, samples
from src.models.detection.runner import DATASET_PATH, MANIFEST, ROOT, code_info, dataset_info

SOURCE = ROOT / DATASET_PATH
OUT = ROOT / "runs" / "kaggle"
SLUG = "tujya-c3pi-detect-v2"
EXTRA_FILES = (HOLDOUT_FILE, "index.csv", "summary.json")
README = """# Tujya C3PI v2 pill detection (private)

YOLO detection dataset (one class: pill) built from the consumer photos of the NLM
Computational Photography Project for Pill Identification (C3PI),
https://data.lhncbc.nlm.nih.gov/public/Pills/ . Licence: ODbL, research and development use
only; the images remain those of the U.S. National Library of Medicine.

Boxes: automatic labelling (classical computer vision), long side resized to 640 px.
Splits: by photo session, no leakage (train / val / test). `test_holdout.txt` lists the test
images used to compare the detectors (sessions the baseline never saw).

Built by the Tujya project, https://github.com/melekbouabid/syRX (step 2.2,
docs/module1/kaggle_detecteurs.md). Kept private: do not redistribute.
"""


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def check_dataset(root: Path) -> dict[str, int]:
    """Images per split, every image with a label file, holdout images all in test."""
    counts = {}
    for split in SPLITS:
        chosen = samples(root, split)
        missing = [s.image.name for s in chosen if not s.label.is_file()]
        if missing:
            raise ValueError(f"{split}: {len(missing)} images without labels, e.g. {missing[0]}")
        counts[split] = len(chosen)
    holdout = holdout_names(root) or set()
    test = {s.image.name for s in samples(root, "test")}
    if not holdout <= test:
        raise ValueError(f"{len(holdout - test)} holdout images are not in the test split")
    counts["holdout"] = len(holdout)
    return counts


def kaggle_metadata(owner: str, slug: str) -> dict[str, Any]:
    return {
        "title": "Tujya C3PI v2 pill detection",
        "id": f"{owner}/{slug}",
        "subtitle": "C3PI consumer photos with automatic pill boxes (YOLO format)",
        "licenses": [{"name": "ODbL-1.0"}],
        "keywords": ["computer vision", "object detection", "health"],
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
    for split in SPLITS:
        for sample in samples(source, split):
            _link_or_copy(sample.image, folder / "images" / split / sample.image.name)
            _link_or_copy(sample.label, folder / "labels" / split / sample.label.name)
    for name in EXTRA_FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, folder / name)
    info = dataset_info(source)
    manifest = {
        **{k: v for k, v in info.items() if k in ("name", "md5", "dvc_rev", "path")},
        "name": info.get("name", "c3pi-v2-detect"),
        "images": counts,
        "packed": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "packed_from_commit": code_info()["git_commit"],
        "slug": slug,
    }
    (folder / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
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
    parser = argparse.ArgumentParser(prog="python -m src.models.detection.package_kaggle")
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
