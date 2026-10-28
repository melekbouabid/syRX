"""Build the training datasets from the downloaded C3PI subset.

    python -m tujya_vision.prepare            # labels + both datasets
    python -m tujya_vision.prepare --sheet    # also a contact sheet of boxes to check by eye

Outputs (all under data/vision/, tracked by DVC, never by git):
- c3pi/labels/autolabel.csv     one row per image: boxes, flags, split, class
- datasets/detect/              YOLO detection dataset, one class "pill"; consumer photos
                                without flags only; long side resized to 640 px
- datasets/classify/            YOLO classification dataset: square crops around each box,
                                224 px, folders <split>/<class_id:03d>/

The split of every image comes from config/splits/selected_images.csv, so a photo and all
its crops stay in one split.
"""

from __future__ import annotations

import argparse
import csv
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tujya_vision.autolabel import Box, detect_pills, detect_reference, read_image, square_crop

ROOT = Path(__file__).resolve().parents[2]
VISION = ROOT / "data" / "vision"
IMAGES = VISION / "c3pi" / "images"
SELECTED = ROOT / "config" / "splits" / "selected_images.csv"
LABELS = VISION / "c3pi" / "labels" / "autolabel.csv"
DETECT = VISION / "datasets" / "detect"
CLASSIFY = VISION / "datasets" / "classify"
DETECT_SIZE = 640
CROP_SIZE = 224
MAX_CROPS = 2  # front and back of the same pill at most


@dataclass
class Item:
    path: str
    class_id: int
    image_type: str
    split: str

    @property
    def file(self) -> Path:
        disc, _, name = self.path.split("/")
        return IMAGES / disc / name

    @property
    def stem(self) -> str:
        disc, _, name = self.path.split("/")
        return f"{disc}_{Path(name).stem}".replace("!", "_")


def read_items() -> list[Item]:
    with SELECTED.open(newline="", encoding="utf-8") as handle:
        return [
            Item(r["path"], int(r["class_id"]), r["image_type"], r["split"])
            for r in csv.DictReader(handle)
        ]


def _format_boxes(boxes: list[Box]) -> str:
    return ";".join(" ".join(str(v) for v in box) for box in boxes)


def _parse_boxes(text: str) -> list[Box]:
    out = []
    for chunk in filter(None, text.split(";")):
        x1, y1, x2, y2 = (int(v) for v in chunk.split())
        out.append((x1, y1, x2, y2))
    return out


def process(item: Item) -> dict[str, str]:
    """Label one image and write its detection sample and its crops."""
    row = {
        "path": item.path,
        "class_id": str(item.class_id),
        "image_type": item.image_type,
        "split": item.split,
        "width": "",
        "height": "",
        "boxes": "",
        "flags": "",
        "ok": "0",
    }
    if not item.file.exists():
        row["flags"] = "missing"
        return row
    image = read_image(item.file)
    found = detect_pills(image) if item.image_type == "C3PI_Test" else detect_reference(image)
    row.update(
        width=str(found.width),
        height=str(found.height),
        boxes=_format_boxes(found.boxes),
        flags=",".join(found.flags),
        ok=str(int(found.ok)),
    )

    # Doubtful consumer photos are left out of both datasets; reference images only need a box.
    usable = found.ok if item.image_type == "C3PI_Test" else bool(found.boxes)
    for k, box in enumerate(found.boxes[:MAX_CROPS] if usable else []):
        folder = CLASSIFY / item.split / f"{item.class_id:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{item.stem}_{k}.jpg"), square_crop(image, box, size=CROP_SIZE))

    if item.image_type == "C3PI_Test" and found.ok:
        scale = DETECT_SIZE / max(found.width, found.height)
        small = cv2.resize(
            image,
            (round(found.width * scale), round(found.height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        (DETECT / "images" / item.split).mkdir(parents=True, exist_ok=True)
        (DETECT / "labels" / item.split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(DETECT / "images" / item.split / f"{item.stem}.jpg"), small)
        lines = []
        for x1, y1, x2, y2 in found.boxes:
            cx, cy = (x1 + x2) / 2 / found.width, (y1 + y2) / 2 / found.height
            bw, bh = (x2 - x1) / found.width, (y2 - y1) / found.height
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (DETECT / "labels" / item.split / f"{item.stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    return row


def write_detect_yaml() -> None:
    (DETECT / "data.yaml").write_text(
        f"path: {DETECT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: pill\n",
        encoding="utf-8",
    )


def contact_sheet(rows: list[dict[str, str]], out: Path, n: int = 48, seed: int = 0) -> None:
    """Grid of random consumer photos with their boxes (green = ok, red = flagged)."""
    sample = [r for r in rows if r["image_type"] == "C3PI_Test" and r["width"]]
    random.Random(seed).shuffle(sample)
    tile, cols = 240, 8
    sheet = np.full(((n + cols - 1) // cols * tile, cols * tile, 3), 255, np.uint8)
    for i, row in enumerate(sample[:n]):
        item = Item(row["path"], int(row["class_id"]), row["image_type"], row["split"])
        image = read_image(item.file)
        colour = (0, 200, 0) if row["ok"] == "1" else (0, 0, 255)
        for x1, y1, x2, y2 in _parse_boxes(row["boxes"]):
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, max(2, image.shape[1] // 150))
        scale = tile / max(image.shape[:2])
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        sheet[r * tile : r * tile + small.shape[0], c * tile : c * tile + small.shape[1]] = small
        cv2.putText(
            sheet,
            row["flags"][:28],
            (c * tile + 3, r * tile + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tujya_vision.prepare")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="only the first N images (tests)")
    parser.add_argument("--sheet", type=Path, default=None, help="write a QA contact sheet")
    args = parser.parse_args(argv)

    items = read_items()
    if args.limit:
        items = items[: args.limit]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(process, items, chunksize=16))
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    with LABELS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_detect_yaml()

    consumer = [r for r in rows if r["image_type"] == "C3PI_Test"]
    counts: dict[str, int] = {}
    for r in consumer:
        for flag in filter(None, r["flags"].split(",")):
            counts[flag] = counts.get(flag, 0) + 1
    ok = sum(1 for r in consumer if r["ok"] == "1")
    n_boxes = [len(_parse_boxes(r["boxes"])) for r in consumer if r["boxes"]]
    print(f"{len(rows)} images, {len(consumer)} consumer: {ok} ok, flags {counts}")
    print(
        f"boxes per labelled consumer photo: "
        f"{ {k: n_boxes.count(k) for k in sorted(set(n_boxes))} }"
    )
    if args.sheet:
        contact_sheet(rows, args.sheet)
        print(f"contact sheet: {args.sheet}")


if __name__ == "__main__":
    main()
