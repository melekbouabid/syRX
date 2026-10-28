"""Command line for the C3PI subset: `python -m tujya_vision.data_cli <command>`.

Run from the `vision/` folder. Paths default to the repository layout:
metadata in data/vision/c3pi/metadata, images in data/vision/c3pi/images,
class list in config/pill_classes.csv, split lists in config/splits/.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from tujya_vision import c3pi

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "vision" / "c3pi" / "metadata"
IMAGES = ROOT / "data" / "vision" / "c3pi" / "images"
CLASSES = ROOT / "config" / "pill_classes.csv"
SPLITS = ROOT / "config" / "splits"


def _selected(index: Path) -> list[c3pi.ImageRecord]:
    """Images listed in config/splits/selected_images.csv, in that order (class by class)."""
    by_path: dict[str, c3pi.ImageRecord] = {}
    for record in c3pi.read_index(index):
        by_path.setdefault(record.path, record)
    return [by_path[row["path"]] for row in _read_csv(SPLITS / "selected_images.csv")]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cmd_metadata(args: argparse.Namespace) -> None:
    written = c3pi.download_metadata(META)
    print(f"{len(written)} metadata files written to {META}")


def cmd_index(args: argparse.Namespace) -> None:
    records = c3pi.build_index(args.xml_dir)
    c3pi.write_index(records, args.out)
    types = Counter((r.image_type, r.file_type) for r in records)
    print(f"{len(records)} images -> {args.out}")
    for (image_type, file_type), n in sorted(types.items()):
        print(f"  {image_type:30s} {file_type:4s} {n}")


def cmd_select(args: argparse.Namespace) -> None:
    records = c3pi.read_index(args.index)
    classes, chosen = c3pi.select_classes(
        records,
        max_classes=args.max_classes,
        min_images=args.min_images,
        max_images_per_class=args.max_per_class,
        budget_bytes=int(args.budget_gb * 1e9),
        exclude_cameras=args.exclude_camera,
        seed=args.seed,
    )
    c3pi.write_classes(classes, args.classes)
    SPLITS.mkdir(parents=True, exist_ok=True)
    class_ids = {c.class_key: c.class_id for c in classes}
    split = c3pi.split_images(chosen, val=args.val, test=args.test, seed=args.seed)
    with (SPLITS / "selected_images.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "class_id", "image_type", "file_size", "sha1", "split"])
        for r in sorted(chosen, key=lambda r: (class_ids[r.class_key], r.path)):
            writer.writerow(
                [r.path, class_ids[r.class_key], r.image_type, r.file_size, r.sha1, split[r.path]]
            )
    total = sum(r.file_size for r in chosen)
    print(f"{len(classes)} classes, {len(chosen)} images, {total / 1e9:.2f} GB")
    print(Counter(split.values()))


def cmd_download(args: argparse.Namespace) -> None:
    records = _selected(args.index)
    if args.limit:
        records = records[: args.limit]
    done, skipped, failed = c3pi.download(records, args.images, workers=args.workers)
    print(f"downloaded={done} skipped={skipped} failed={failed}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tujya_vision.data_cli")
    parser.add_argument("--index", type=Path, default=META / "images.csv")
    parser.add_argument("--classes", type=Path, default=CLASSES)
    parser.add_argument("--images", type=Path, default=IMAGES)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("metadata", help="download the READMEs, directory and ALLXML files")
    p.set_defaults(func=cmd_metadata)

    p = sub.add_parser("index", help="ALLXML/*.xml -> images.csv")
    p.add_argument("--xml-dir", type=Path, default=META / "ALLXML")
    p.add_argument("--out", type=Path, default=META / "images.csv")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("select", help="rank classes, write pill_classes.csv and the splits")
    p.add_argument("--max-classes", type=int, default=200)
    p.add_argument("--min-images", type=int, default=20)
    p.add_argument("--max-per-class", type=int, default=40)
    p.add_argument("--budget-gb", type=float, default=4.0)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--exclude-camera",
        action="append",
        default=[],
        help="camera name to leave out (repeatable); see docs/DATASET_C3PI.md",
    )
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("download", help="download the selected images (resumable)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_download)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
