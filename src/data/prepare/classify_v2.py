"""Build the classification dataset of C3PI v2 (DVC stage prepare_classify_v2, step 2.3).

    python -m src.data.prepare_classify_v2             # training venv (.venv-train) or .venv
    python -m src.data.prepare_classify_v2 --limit 20  # first 20 images of each split and role

The classifiers of step 2.3 are trained and compared on C3PI v2 (380 classes, step 1.4,
docs/module1/big_data.md), like the detectors of step 2.2 (decision D99 in
docs/decisions.md). A classification sample is the square crop of one pill with a 20 % margin
on each side, the crop the API feeds the classifier (backend/app/inference.py
`square_crop`), so every image of the v2 split gives at most two crops (front and back of the
same pill, as v1):

- consumer photos (`C3PI_Test`): boxes of src/data/autolabel.py `detect_pills`, doubtful
  photos left out (same rule as v1 and as the detection dataset);
- RxNav reference images: `detect_reference` (ruler and banner removed first), as v1;
- NLM challenge images (one pill on a plain background): `detect_pills`;
- segmented studio images (16-bit RGBA PNG): the box of the opaque part of the alpha mask
  above the NLM banner, the pill composited on a plain grey background;
- the offline augmentation of the rare classes (stage spark_augment, generated from TRAIN
  images only): `detect_pills`, largest box.

Inputs: the v2 split (data/processed/c3pi/bigdata/split: split, session group, role of every
image), the augmented images, the v1 lists (config/splits/selected_images.csv,
config/pill_classes_v1.csv) and the class list of v2 (config/splits/c3pi_v2/classes.csv).
Outputs, under data/processed/c3pi/datasets_v2/classify/ (DVC, never git):

- <split>/<class_id:03d>/<name>_<k>.jpg  crops, CROP_SIZE px (folders = v2 class ids);
- index.csv          one row per crop: file, split, class, session group, role, source
                     image, box, `holdout` flag;
- classes.csv        v2 class list with the v1 class id of the 150 classes v1 had
                     (`v1_class_id`, how a 150-class model is scored on this split) and the
                     number of crops per split;
- test_holdout.txt   the comparison split (relative file names): test crops whose session
                     holds no photo the v1 champion saw in train or val (same rule as the
                     detection holdout of D93, src/data/prepare_detect_v2.py);
- summary.json       counts per split and role, failures.

No image is moved between splits: the v2 split is by session (D89), and the augmented images
come from train sources only (stage spark_augment), so nothing of val or test leaks into train.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.autolabel import Box, detect_pills, detect_reference, read_image, square_crop
from src.data.prepare_detect_v2 import baseline_groups

ROOT = Path(__file__).resolve().parents[2]  # repository root
IMAGES = ROOT / "data" / "raw" / "c3pi" / "images"
SPLIT = ROOT / "data" / "processed" / "c3pi" / "bigdata" / "split" / "images"
AUGMENTED = ROOT / "data" / "processed" / "c3pi" / "bigdata" / "augmented"
V1_LIST = ROOT / "config" / "splits" / "selected_images.csv"
V1_CLASSES = ROOT / "config" / "pill_classes_v1.csv"
V2_CLASSES = ROOT / "config" / "splits" / "c3pi_v2" / "classes.csv"
OUT = ROOT / "data" / "processed" / "c3pi" / "datasets_v2" / "classify"
CROP_SIZE = 256  # leaves room for an image-size search up to 256 px (resized at training)
MAX_CROPS = 2  # front and back of the same pill at most (as v1)
SEG_BACKGROUND = 128  # plain grey behind the segmented studio pills
ALPHA_OPAQUE = 128  # 8-bit alpha above which a pixel belongs to the segmented pill
SPLITS = ("train", "val", "test")
ROLES = ("consumer", "reference", "challenge", "seg", "augmented")


@dataclass(frozen=True)
class Source:
    """One image of the v2 split (or one augmented image) to crop."""

    path: str  # PillProjectDiscN/images/NAME, or data/processed/.../augmented/... for augmented
    split: str
    class_id: int
    group_id: str
    role: str

    @property
    def stem(self) -> str:
        if self.role == "augmented":
            return f"aug_{Path(self.path).stem}"
        disc, _, name = self.path.split("/")
        return f"{disc}_{Path(name).stem}".replace("!", "_")

    def file(self, images: Path = IMAGES, root: Path = ROOT) -> Path:
        if self.role == "augmented":
            return root / self.path
        disc, _, name = self.path.split("/")
        return images / disc / name


def read_unchanged(path: Path) -> np.ndarray:
    """Image with its alpha channel and bit depth (cv2.imread fails on some Windows paths)."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode {path}")
    return image


def segmented_pill(image: np.ndarray) -> tuple[np.ndarray, list[Box]]:
    """A segmented studio PNG (RGBA, 8 or 16 bits): the pill on grey and its box.

    The NLM banner at the bottom is opaque on whole rows: the box is searched above it, as
    the largest connected region of clearly opaque pixels (a faint alpha noise of 1 to 7
    covers the rest of the frame).
    """
    if image.dtype == np.uint16:
        image = (image >> 8).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("a segmented image has an alpha channel")
    alpha = image[:, :, 3]
    full_rows = np.flatnonzero((alpha > 0).mean(axis=1) > 0.95)
    top_of_banner = int(full_rows.min()) if full_rows.size else alpha.shape[0]
    mask = (alpha[:top_of_banner] >= ALPHA_OPAQUE).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    weight = (alpha.astype(np.float32) / 255.0)[:, :, None]
    composite = (image[:, :, :3] * weight + SEG_BACKGROUND * (1 - weight)).astype(np.uint8)
    if count < 2:
        return composite, []
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(v) for v in stats[best, :4])
    return composite, [(x, y, x + w, y + h)]


def boxes_of(source: Source, file: Path) -> tuple[np.ndarray, list[Box], list[str]]:
    """The image to crop, the pill boxes (largest first) and the autolabel flags."""
    if source.role == "seg":
        image, boxes = segmented_pill(read_unchanged(file))
        return image, boxes, [] if boxes else ["no_region"]
    image = read_image(file)
    if source.role == "reference":
        found = detect_reference(image)
        return image, found.boxes if found.boxes else [], found.flags
    found = detect_pills(image)
    if source.role == "consumer":
        # Doubtful consumer photos are left out (same rule as v1 and the detection dataset).
        return image, found.boxes if found.ok else [], found.flags
    if source.role == "augmented":
        return image, found.boxes[:1], found.flags
    return image, found.boxes, found.flags  # challenge: one pill on a plain background


def process(source: Source, out: Path = OUT, images: Path = IMAGES) -> list[dict[str, str]]:
    """Crop one image; one row per written crop, or a single row with `ok=0` when none."""
    base = {
        "split": source.split,
        "class_id": str(source.class_id),
        "group_id": source.group_id,
        "role": source.role,
        "source": source.path,
    }
    file = source.file(images)
    if not file.exists():
        return [{**base, "file": "", "box": "", "flags": "missing", "ok": "0"}]
    try:
        image, boxes, flags = boxes_of(source, file)
    except (ValueError, cv2.error) as error:
        return [{**base, "file": "", "box": "", "flags": f"error:{error}"[:80], "ok": "0"}]
    if not boxes:
        return [{**base, "file": "", "box": "", "flags": ",".join(flags), "ok": "0"}]
    folder = out / source.split / f"{source.class_id:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, box in enumerate(boxes[:MAX_CROPS]):
        name = f"{source.stem}_{k}.jpg"
        cv2.imwrite(str(folder / name), square_crop(image, box, size=CROP_SIZE))
        rows.append(
            {
                **base,
                "file": f"{source.split}/{source.class_id:03d}/{name}",
                "box": " ".join(str(int(v)) for v in box),
                "flags": ",".join(flags),
                "ok": "1",
            }
        )
    return rows


def read_sources(split_dir: Path = SPLIT, augmented: Path = AUGMENTED) -> list[Source]:
    frame = pd.read_parquet(split_dir, columns=["path", "split", "class_id", "group_id", "role"])
    frame = frame.sort_values(["split", "role", "path"])
    sources = [
        Source(r.path, r.split, int(r.class_id), str(r.group_id), str(r.role))
        for r in frame.itertuples(index=False)
    ]
    manifest = augmented / "manifest"
    if manifest.is_dir():
        aug = pd.read_parquet(manifest, columns=["path", "class_id", "ok"])
        aug = aug[aug["ok"]].sort_values("path")
        sources += [
            Source(str(r.path).replace("\\", "/"), "train", int(r.class_id), "", "augmented")
            for r in aug.itertuples(index=False)
        ]
    return sources


def class_table(
    rows: list[dict[str, str]],
    v2_classes: Path = V2_CLASSES,
    v1_classes: Path = V1_CLASSES,
) -> pd.DataFrame:
    """v2 class list, v1 class id of the classes v1 had, crops per split and in the holdout."""
    classes = pd.read_csv(v2_classes)[["class_id", "class_key", "name", "generic_name", "in_v1"]]
    v1 = pd.read_csv(v1_classes)[["class_id", "class_key"]].rename(
        columns={"class_id": "v1_class_id"}
    )
    classes = classes.merge(v1, on="class_key", how="left")
    classes["v1_class_id"] = classes["v1_class_id"].astype("Int64")
    kept = pd.DataFrame([r for r in rows if r["ok"] == "1"])
    for split in SPLITS:
        counts = kept.loc[kept["split"] == split, "class_id"].astype(int).value_counts()
        classes[split] = classes["class_id"].map(counts).fillna(0).astype(int)
    holdout = kept.loc[kept["holdout"] == "1", "class_id"].astype(int).value_counts()
    classes["holdout"] = classes["class_id"].map(holdout).fillna(0).astype(int)
    return classes.sort_values("class_id")


def write_outputs(
    rows: list[dict[str, str]],
    seen_groups: set[str],
    out: Path = OUT,
    *,
    v2_classes: Path = V2_CLASSES,
    v1_classes: Path = V1_CLASSES,
) -> dict:
    """index.csv, classes.csv, test_holdout.txt and summary.json of the written crops."""
    kept = [r for r in rows if r["ok"] == "1"]
    for r in rows:
        r["holdout"] = str(
            int(r["ok"] == "1" and r["split"] == "test" and r["group_id"] not in seen_groups)
        )
    out.mkdir(parents=True, exist_ok=True)
    with (out / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["file", "split", "class_id", "group_id", "role", "source", "box", "holdout"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(kept, key=lambda r: r["file"]))
    holdout = sorted(r["file"] for r in kept if r["holdout"] == "1")
    (out / "test_holdout.txt").write_text("\n".join(holdout) + "\n", encoding="utf-8")
    classes = class_table(rows, v2_classes, v1_classes)
    classes.to_csv(out / "classes.csv", index=False)
    flags: Counter[str] = Counter()
    for r in rows:
        if r["ok"] == "0":
            flags.update([f for f in r["flags"].split(",") if f] or ["none"])
    images = {(r["split"], r["role"], r["source"]) for r in rows}
    used = {(r["split"], r["role"], r["source"]) for r in kept}
    test_kept = sum(r["split"] == "test" for r in kept)
    in_v1 = set(classes.loc[classes["in_v1"].astype(bool), "class_id"].astype(str))
    summary = {
        "dataset": "c3pi-v2-classify",
        "crop_size": CROP_SIZE,
        "classes": len(classes),
        "classes_in_v1": int(classes["in_v1"].astype(bool).sum()),
        "crops": {s: sum(r["split"] == s for r in kept) for s in SPLITS},
        "crops_by_role": {
            f"{s} {role}": n
            for (s, role), n in sorted(Counter((r["split"], r["role"]) for r in kept).items())
        },
        "images": {s: sum(1 for i in images if i[0] == s) for s in SPLITS},
        "images_without_crop": len(images) - len(used),
        "test_holdout": len(holdout),
        "test_holdout_v1_classes": sum(
            r["holdout"] == "1" and r["class_id"] in in_v1 for r in kept
        ),
        "test_seen_by_v1": test_kept - len(holdout),
        "min_crops_per_class": {s: int(classes[s].min()) for s in SPLITS},
        "failures": dict(sorted(flags.items())),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.data.prepare_classify_v2")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="first N images per split and role")
    parser.add_argument("--out", type=Path, default=OUT, help="default: the DVC output")
    args = parser.parse_args(argv)
    if args.out.exists():
        shutil.rmtree(args.out)  # no stale crop from an earlier run

    sources = read_sources()
    if args.limit:
        groups: dict[tuple[str, str], list[Source]] = {}
        for s in sources:
            groups.setdefault((s.split, s.role), []).append(s)
        sources = [s for chunk in groups.values() for s in chunk[: args.limit]]
    split = pd.read_parquet(SPLIT, columns=["path", "group_id"])
    seen = baseline_groups(split, pd.read_csv(V1_LIST))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        chunks = pool.map(partial(process, out=args.out), sources, chunksize=16)
        rows = [row for chunk in chunks for row in chunk]
    summary = write_outputs(rows, seen, args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
