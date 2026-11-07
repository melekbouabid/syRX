"""Classification dataset of C3PI v2 (src/data/prepare_classify_v2.py) on synthetic images
and rows."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data import prepare_classify_v2 as P


def test_segmented_pill_box_above_the_banner() -> None:
    image = np.zeros((100, 120, 4), np.uint16)
    image[:, :, :3] = 200 * 257
    image[5:60, 10:110, 3] = 3 * 257  # faint alpha noise around the pill (as the NLM PNGs)
    image[20:50, 30:70, 3] = 65535  # the pill
    image[80:, :, 3] = 65535  # the opaque banner at the bottom
    composite, boxes = P.segmented_pill(image)
    assert boxes == [(30, 20, 70, 50)]
    assert composite.dtype == np.uint8 and composite.shape == (100, 120, 3)
    assert composite[5, 5].tolist() == [P.SEG_BACKGROUND] * 3  # transparent -> grey
    with pytest.raises(ValueError, match="alpha"):
        P.segmented_pill(np.zeros((10, 10, 3), np.uint8))


def test_crops_are_written_in_the_folder_of_their_class(tmp_path: Path) -> None:
    images = tmp_path / "images" / "Disc1"
    images.mkdir(parents=True)
    photo = np.full((300, 400, 3), 60, np.uint8)
    cv2.rectangle(photo, (150, 100), (230, 160), (240, 240, 240), -1)
    cv2.imwrite(str(images / "A.JPG"), photo)
    source = P.Source("Disc1/images/A.JPG", "train", 7, "g1", "challenge")
    rows = P.process(source, out=tmp_path / "out", images=tmp_path / "images")
    assert [r["ok"] for r in rows] == ["1"]
    crop = cv2.imread(str(tmp_path / "out" / rows[0]["file"]))
    assert rows[0]["file"] == "train/007/Disc1_A_0.jpg"
    assert crop.shape == (P.CROP_SIZE, P.CROP_SIZE, 3)
    missing = P.process(
        P.Source("Disc1/images/B.JPG", "val", 7, "g2", "consumer"),
        out=tmp_path / "out",
        images=tmp_path / "images",
    )
    assert missing[0]["ok"] == "0" and missing[0]["flags"] == "missing"


def test_holdout_excludes_sessions_seen_by_v1(tmp_path: Path) -> None:
    v2 = tmp_path / "classes_v2.csv"
    v2.write_text(
        "class_id,class_key,name,generic_name,in_v1\n0,k0,A,a,True\n1,k1,B,b,False\n", "utf-8"
    )
    v1 = tmp_path / "classes_v1.csv"
    v1.write_text("class_id,class_key\n5,k0\n", encoding="utf-8")

    def row(file: str, split: str, group: str, cls: str = "0", ok: str = "1") -> dict[str, str]:
        return {
            "file": file,
            "split": split,
            "class_id": cls,
            "group_id": group,
            "role": "consumer",
            "source": file,
            "box": "",
            "flags": "",
            "ok": ok,
        }

    rows = [
        row("train/000/a_0.jpg", "train", "g1"),
        row("test/000/b_0.jpg", "test", "seen"),
        row("test/000/c_0.jpg", "test", "new"),
        row("test/001/d_0.jpg", "test", "new2", cls="1"),
        row("", "test", "new3", ok="0"),
    ]
    summary = P.write_outputs(rows, {"seen"}, tmp_path / "out", v2_classes=v2, v1_classes=v1)
    holdout = (tmp_path / "out" / "test_holdout.txt").read_text(encoding="utf-8").split()
    assert holdout == ["test/000/c_0.jpg", "test/001/d_0.jpg"]
    assert summary["test_holdout"] == 2 and summary["test_seen_by_v1"] == 1
    assert summary["test_holdout_v1_classes"] == 1
    assert summary["images_without_crop"] == 1
    classes = (tmp_path / "out" / "classes.csv").read_text(encoding="utf-8").splitlines()
    assert classes[0].startswith("class_id,class_key,name,generic_name,in_v1,v1_class_id")
    assert classes[1].startswith("0,k0,A,a,True,5,1,0,2,1")
