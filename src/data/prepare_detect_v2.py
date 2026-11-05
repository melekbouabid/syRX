"""Build the detection dataset of C3PI v2 (DVC stage prepare_detect_v2, step 2.2).

    python -m src.data.prepare_detect_v2             # training venv (.venv-train) or .venv
    python -m src.data.prepare_detect_v2 --limit 50  # first 50 photos of each split (tests)

The detectors of step 2.2 are trained and compared on C3PI v2 (380 classes, step 1.4,
docs/module1/big_data.md; decision D93 in docs/decisions.md). Detection has a single class,
"pill", so only the consumer photos matter (`C3PI_Test`: one or two pills on a real
background; the studio, RxImage and challenge images show a pill on a uniform background and
stay with the classifier). The boxes come from the same automatic labelling as v1
(src/data/autolabel.py), with the same rule: doubtful photos are left out.

Inputs: the split of v2 (data/processed/c3pi/bigdata/split, stage spark_split: split, session
group of every photo), the v1 list (config/splits/selected_images.csv) and the images.
Outputs, under data/processed/c3pi/datasets_v2/detect/ (DVC, never git):

- images/<split>/*.jpg, labels/<split>/*.txt   YOLO layout, long side resized to 640 px;
- data.yaml                                    Ultralytics dataset file (absolute path);
- index.csv                                    one row per kept photo: split, class, session
                                               group, number of boxes, `holdout` flag;
- test_holdout.txt                             the comparison split (file names): the v2 test
                                               photos whose session holds no photo the
                                               baseline (trained on v1) saw in train or val;
- summary.json                                 counts per split, autolabel flags.

Why a holdout: the v1 split was not made by session, and 436 v1 training photos fall in the
v2 test split. Every model of step 2.2, the baseline included, is compared on `test_holdout`
(metric `test_v2_mAP50_95`); the full v2 test is also reported (`test_v2_all_*`).
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
import pandas as pd

from src.data.autolabel import Box, detect_pills, read_image

ROOT = Path(__file__).resolve().parents[2]  # repository root
IMAGES = ROOT / "data" / "raw" / "c3pi" / "images"
SPLIT = ROOT / "data" / "processed" / "c3pi" / "bigdata" / "split" / "images"
V1_LIST = ROOT / "config" / "splits" / "selected_images.csv"
OUT = ROOT / "data" / "processed" / "c3pi" / "datasets_v2" / "detect"
DETECT_SIZE = 640
CONSUMER = "C3PI_Test"
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Photo:
    path: str  # PillProjectDiscN/images/NAME.JPG, as in the C3PI index
    split: str
    class_id: int
    group_id: str

    @property
    def stem(self) -> str:
        disc, _, name = self.path.split("/")
        return f"{disc}_{Path(name).stem}".replace("!", "_")


def yolo_lines(boxes: list[Box], width: int, height: int) -> list[str]:
    """YOLO label lines (class 0, normalised centre x, centre y, width, height)."""
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        bw, bh = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def baseline_groups(split: pd.DataFrame, v1: pd.DataFrame) -> set[str]:
    """Session groups of v2 holding a photo the baseline saw (v1 train or val)."""
    seen = set(v1.loc[v1["split"].isin(["train", "val"]), "path"])
    return set(split.loc[split["path"].isin(seen), "group_id"])


def read_photos(split_dir: Path = SPLIT) -> list[Photo]:
    frame = pd.read_parquet(
        split_dir, columns=["path", "split", "class_id", "group_id", "image_type"]
    )
    frame = frame[frame["image_type"] == CONSUMER].sort_values(["split", "path"])
    return [
        Photo(r.path, r.split, int(r.class_id), str(r.group_id))
        for r in frame.itertuples(index=False)
    ]


def process(photo: Photo, out: Path = OUT, images: Path = IMAGES) -> dict[str, str]:
    """Label one photo; write its resized image and YOLO labels when the boxes are trusted."""
    disc, _, name = photo.path.split("/")
    file = images / disc / name
    row = {
        "file": f"{photo.stem}.jpg",
        "path": photo.path,
        "split": photo.split,
        "class_id": str(photo.class_id),
        "group_id": photo.group_id,
        "boxes": "0",
        "flags": "",
        "ok": "0",
    }
    if not file.exists():
        row["flags"] = "missing"
        return row
    image = read_image(file)
    found = detect_pills(image)
    row.update(boxes=str(len(found.boxes)), flags=",".join(found.flags), ok=str(int(found.ok)))
    if not found.ok:
        return row
    scale = DETECT_SIZE / max(found.width, found.height)
    small = cv2.resize(
        image,
        (round(found.width * scale), round(found.height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    (out / "images" / photo.split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / photo.split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "images" / photo.split / row["file"]), small)
    lines = yolo_lines(found.boxes, found.width, found.height)
    (out / "labels" / photo.split / f"{photo.stem}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return row


def write_outputs(rows: list[dict[str, str]], seen_groups: set[str], out: Path = OUT) -> dict:
    """index.csv, test_holdout.txt, data.yaml and summary.json of the kept photos."""
    kept = [r for r in rows if r["ok"] == "1"]
    for r in kept:
        r["holdout"] = str(int(r["split"] == "test" and r["group_id"] not in seen_groups))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["file", "path", "split", "class_id", "group_id", "boxes", "holdout"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    holdout = sorted(r["file"] for r in kept if r["holdout"] == "1")
    (out / "test_holdout.txt").write_text("\n".join(holdout) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: pill\n",
        encoding="utf-8",
    )
    flags: Counter[str] = Counter()
    for r in rows:
        flags.update(filter(None, r["flags"].split(",")))
    summary = {
        "dataset": "c3pi-v2-detect",
        "photos": {s: sum(r["split"] == s for r in rows) for s in SPLITS},
        "kept": {s: sum(r["split"] == s for r in kept) for s in SPLITS},
        "test_holdout": len(holdout),
        "test_seen_by_baseline": sum(r["split"] == "test" for r in kept) - len(holdout),
        "boxes": sum(int(r["boxes"]) for r in kept),
        "classes": len({r["class_id"] for r in kept}),
        "flags": dict(sorted(flags.items())),
        "detect_size": DETECT_SIZE,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.data.prepare_detect_v2")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="first N photos per split (tests)")
    parser.add_argument("--out", type=Path, default=OUT, help="default: the DVC output")
    args = parser.parse_args(argv)
    if args.out.exists():
        shutil.rmtree(args.out)  # no stale image from an earlier run

    photos = read_photos()
    if args.limit:
        photos = [p for s in SPLITS for p in [q for q in photos if q.split == s][: args.limit]]
    split = pd.read_parquet(SPLIT, columns=["path", "group_id"])
    seen = baseline_groups(split, pd.read_csv(V1_LIST))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(partial(process, out=args.out), photos, chunksize=16))
    summary = write_outputs(rows, seen, args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
