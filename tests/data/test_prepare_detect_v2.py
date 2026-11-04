"""Detection dataset of C3PI v2 (src/data/prepare_detect_v2.py) on synthetic rows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.prepare_detect_v2 import Photo, baseline_groups, write_outputs, yolo_lines


def test_yolo_lines_are_normalised() -> None:
    assert yolo_lines([(10, 20, 30, 60)], 100, 200) == ["0 0.200000 0.200000 0.200000 0.200000"]


def test_baseline_groups_are_sessions_with_a_v1_train_or_val_photo() -> None:
    split = pd.DataFrame(
        {"path": ["a", "b", "c", "d", "e"], "group_id": ["g1", "g1", "g2", "g3", "g4"]}
    )
    v1 = pd.DataFrame({"path": ["a", "c", "d", "z"], "split": ["train", "val", "test", "train"]})
    # d was in the v1 test split: the baseline never trained on it, its session stays usable.
    assert baseline_groups(split, v1) == {"g1", "g2"}


def test_photo_stem_is_safe_for_file_names() -> None:
    photo = Photo("PillProjectDisc7/images/AB!C.JPG", "test", 3, "g")
    assert photo.stem == "PillProjectDisc7_AB_C"


def row(file: str, split: str, group: str, ok: str = "1", flags: str = "") -> dict[str, str]:
    return {
        "file": file,
        "path": f"D/images/{file}",
        "split": split,
        "class_id": "1",
        "group_id": group,
        "boxes": "1" if ok == "1" else "0",
        "flags": flags,
        "ok": ok,
    }


def test_outputs_holdout_excludes_sessions_seen_by_the_baseline(tmp_path: Path) -> None:
    rows = [
        row("t1.jpg", "train", "g0"),
        row("v1.jpg", "val", "g1"),
        row("s1.jpg", "test", "g2"),
        row("s2.jpg", "test", "seen"),
        row("s3.jpg", "test", "g3", ok="0", flags="no_region"),
    ]
    summary = write_outputs(rows, {"seen"}, tmp_path)
    assert (tmp_path / "test_holdout.txt").read_text(encoding="utf-8") == "s1.jpg\n"
    assert summary["kept"] == {"train": 1, "val": 1, "test": 2}
    assert summary["photos"]["test"] == 3
    assert summary["test_holdout"] == 1
    assert summary["test_seen_by_baseline"] == 1
    assert summary["flags"] == {"no_region": 1}
    index = pd.read_csv(tmp_path / "index.csv")
    assert list(index["file"]) == ["t1.jpg", "v1.jpg", "s1.jpg", "s2.jpg"]
    assert list(index["holdout"]) == [0, 0, 1, 0]
    assert "names:\n  0: pill" in (tmp_path / "data.yaml").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == summary
