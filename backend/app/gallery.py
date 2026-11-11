"""The demo gallery list, shared by the API (routers/demo.py) and the Grad-CAM precompute
(`src/models/gradcam_gallery.py`, training venv): standard library only.

`config/demo_gallery.csv` is chosen by hand (decisions D52 and D105). A photo is kept only if
no served model saw it in training: it must be a consumer photo of the C3PI v2 **test** split
(`config/splits/c3pi_v2/test.csv`, the detector's split, D89) and must not be in the v1 train
or val split (`config/splits/selected_images.csv`, the classifier's split), and the file must
exist under the C3PI image folder.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

CONSUMER = "C3PI_Test"  # consumer photos; the other image types are studio or challenge shots


@dataclass(frozen=True)
class DemoPhoto:
    path: str  # as in the CSV files: PillProjectDisc20/images/B3U.JPG
    file: Path
    class_id: int
    why: str
    gradcam: Path  # may not exist: computed for the listed photos only (D104)


def gradcam_name(path: str) -> str:
    """`PillProjectDisc20/images/B3U.JPG` -> `PillProjectDisc20_B3U.jpg` (one folder)."""
    disc, _, name = path.split("/")
    return f"{disc}_{Path(name).stem}.jpg"


def never_trained_on(holdout_file: Path, split_file: Path) -> set[str]:
    """Consumer photos of the v2 test split that are not in the v1 train or val split."""
    with holdout_file.open(newline="", encoding="utf-8") as handle:
        test = {row["path"] for row in csv.DictReader(handle) if row["image_type"] == CONSUMER}
    with split_file.open(newline="", encoding="utf-8") as handle:
        trained = {row["path"] for row in csv.DictReader(handle) if row["split"] != "test"}
    return test - trained


def load_gallery(
    gallery_file: Path,
    split_file: Path,
    holdout_file: Path,
    images_dir: Path,
    gradcam_dir: Path,
) -> list[DemoPhoto]:
    """The listed photos that may be shown, in the order of the list (their API ids)."""
    if not all(f.is_file() for f in (gallery_file, split_file, holdout_file)):
        return []
    allowed = never_trained_on(holdout_file, split_file)
    photos = []
    with gallery_file.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["path"] not in allowed:
                continue
            disc, _, name = row["path"].split("/")
            file = images_dir / disc / name
            if file.is_file():
                photos.append(
                    DemoPhoto(
                        row["path"],
                        file,
                        int(row["class_id"]),
                        row.get("why", ""),
                        gradcam_dir / gradcam_name(row["path"]),
                    )
                )
    return photos
