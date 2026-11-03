"""Job `split`: groups kept together, stratified by class, no leakage between splits."""

from __future__ import annotations

import csv
import json

import pandas as pd
import pytest

pytest.importorskip("pyspark")

from src.data.spark import split_job


def test_assign_groups_fills_test_then_val_by_photo_share():
    groups = pd.DataFrame(
        {
            "class_key": "X",
            "group_id": [f"g{i}" for i in range(11)],
            "n_consumer": [2] * 10 + [0],  # the last group has no consumer photo
        }
    )
    out = split_job.assign_groups(groups, val=0.2, test=0.2, seed=0)
    photos = out.groupby("split")["n_consumer"].sum().to_dict()
    assert photos == {"test": 4, "val": 4, "train": 12}
    assert out.loc[out["n_consumer"] == 0, "split"].tolist() == ["train"]
    again = split_job.assign_groups(groups.sample(frac=1, random_state=3), 0.2, 0.2, 0)
    assert dict(zip(out["group_id"], out["split"], strict=True)) == dict(
        zip(again["group_id"], again["split"], strict=True)
    )  # independent of the row order (Spark partitioning)


def test_assign_groups_skips_a_session_too_large_for_the_share():
    sizes = [12, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]  # 36 photos: test and val want 5 each
    groups = pd.DataFrame(
        {"class_key": "X", "group_id": [f"g{i}" for i in range(13)], "n_consumer": sizes}
    )
    out = split_job.assign_groups(groups, val=0.15, test=0.15, seed=0)
    photos = out.groupby("split")["n_consumer"].sum().to_dict()
    assert photos["test"] == photos["val"] == 4  # never above the share of 5
    assert out.loc[out["n_consumer"] == 12, "split"].item() == "train"


def test_assign_groups_keeps_one_group_for_train():
    two = pd.DataFrame({"class_key": "X", "group_id": ["a", "b"], "n_consumer": [5, 5]})
    assert sorted(split_job.assign_groups(two, 0.15, 0.15, 0)["split"]) == ["test", "train"]


def test_merge_groups_follows_near_duplicates_and_identical_files():
    images = pd.DataFrame(
        {
            "path": ["p1", "p2", "p3", "p4", "p5"],
            "group_id": ["s1", "s1", "s2", "c1", "s3"],
            "sha1": ["a", "b", "c", "d", "c"],
        }
    )
    pairs = pd.DataFrame({"path_a": ["p2", "p9"], "path_b": ["p4", "p1"]})
    merged = split_job.merge_groups(images, pairs).tolist()
    assert merged[0] == merged[1] == merged[3]  # p4 near-identical to p2
    assert merged[2] == merged[4]  # p3 and p5 are the same file
    assert merged[0] != merged[2]


def _selection_rows():
    rows = []

    def add(key, name, role="consumer", session="s", in_class=True, image_type="C3PI_Test"):
        camera, _, date = session.partition("@")
        rows.append(
            {
                "path": f"D1/images/{key}_{name}",
                "class_key": key,
                "role": role,
                "in_class": in_class,
                "image_type": image_type,
                "ndc11": f"{key}0000000001",
                "camera": camera,
                "background": "card",
                "illumination": "lamp",
                "acquisition_date": date or "Monday",
                "sha1": f"{key}{name}",
                "file_size": 100,
            }
        )

    for i in range(10):
        add("X", f"{i}.JPG", session=f"cam{i // 2}")  # 5 sessions of 2 photos
    add("X", "seg.PNG", role="seg", image_type="MC_C3PI_REFERENCE_SEG_V1.6")
    add("X", "ref.JPG", role="reference", image_type="MC_API_RXNAV_V1.3")
    add("X", "ch.JPG", role="challenge", image_type="MC_CHALLENGE_V1.0")
    for i in range(4):
        add("Y", f"{i}.JPG", session=f"cam{i}")
    for i in range(6):
        add("Z", f"{i}.JPG", session=f"cam{i // 2}")
    add("W", "ch.JPG", role="challenge", in_class=False, image_type="MC_CHALLENGE_V1.0")
    return rows


@pytest.fixture
def result(spark, params, tmp_path):
    index, quality = tmp_path / "index", tmp_path / "quality"
    rows = _selection_rows()
    spark.createDataFrame(pd.DataFrame(rows)).write.parquet(str(index / "selection"))
    info = pd.DataFrame(
        [
            {
                "class_key": k,
                "n_consumer": 9,
                "rank": i,
                "ndc11_list": f"{k}0000000001",
                "ndc9": f"{k}00000000",
                "part": 1,
                "name": f"{k} 10MG",
                "generic_name": k,
                "labeled_by": "LAB",
                "imprint": "A",
                "color": "WHITE",
                "shape": "ROUND",
            }
            for i, k in enumerate("XYZ")
        ]
    )
    spark.createDataFrame(info).write.parquet(str(index / "classes"))
    judged = pd.DataFrame(
        {
            "path": [r["path"] for r in rows],
            "reason": [
                "too_dark" if r["path"].endswith(("Y_0.JPG", "Y_1.JPG")) else None for r in rows
            ],
            "dhash": "0" * 64,
        }
    )
    spark.createDataFrame(judged, "path string, reason string, dhash string").write.parquet(
        str(quality / "judged")
    )
    # The challenge image of X is near-identical to photo X_3 (session cam1).
    pairs = pd.DataFrame(
        {
            "path_a": ["D1/images/X_3.JPG"],
            "path_b": ["D1/images/X_ch.JPG"],
            "class_a": ["X"],
            "class_b": ["X"],
            "bits": [1],
        }
    )
    spark.createDataFrame(pairs).write.parquet(str(quality / "near_duplicates"))
    v1 = tmp_path / "pill_classes.csv"
    v1.write_text("class_id,class_key\n0,X\n", encoding="utf-8")
    params["index"]["min_images"] = 3
    summary = split_job.run(
        spark,
        params,
        index=index,
        quality=quality,
        out=tmp_path / "split",
        splits_dir=tmp_path / "splits",
        classes_v1=v1,
    )
    images = spark.read.parquet(str(tmp_path / "split" / "images")).toPandas()
    return summary, images, tmp_path / "splits"


def test_classes_need_n_usable_photos(result):
    summary, images, splits = result
    with (splits / "classes.csv").open(encoding="utf-8") as handle:
        classes = list(csv.DictReader(handle))
    assert [(c["class_id"], c["class_key"], c["in_v1"]) for c in classes] == [
        ("0", "X", "True"),
        ("1", "Z", "False"),
    ]  # Y has 2 usable photos after the quality checks: dropped
    assert summary["classes"] == 2 and summary["candidates_dropped_after_quality"] == 1
    assert "W" not in set(images["class_key"])  # challenge image of no class


def test_no_leakage_and_every_split_present(result):
    summary, images, splits = result
    assert summary["leakage"] == {
        "groups_in_two_splits": 0,
        "files_in_train_and_heldout": 0,
        "near_duplicates_across_splits": 0,
    }
    assert (images.groupby("group_id")["split"].nunique() == 1).all()
    consumer = images[images["role"] == "consumer"]
    for key in ("X", "Z"):
        assert set(consumer.loc[consumer["class_key"] == key, "split"]) == {"train", "val", "test"}
    # Sessions stay whole: the two photos of camera cam0 share their split.
    x = images.set_index("path")["split"]
    assert x["D1/images/X_0.JPG"] == x["D1/images/X_1.JPG"]
    # Studio images stay in train; the challenge copy follows its photo.
    assert x["D1/images/X_seg.PNG"] == x["D1/images/X_ref.JPG"] == "train"
    assert x["D1/images/X_ch.JPG"] == x["D1/images/X_3.JPG"]
    held = set()
    for name in ("val", "test"):
        with (splits / f"{name}.csv").open(encoding="utf-8") as handle:
            listed = {r["path"] for r in csv.DictReader(handle)}
        assert listed == set(images.loc[images["split"] == name, "path"])
        held |= listed
    assert not held & set(images.loc[images["split"] == "train", "path"])
    assert json.loads((splits / "summary.json").read_text())["classes"] == 2


def test_check_leakage_reports_a_shared_file(spark):
    images = spark.createDataFrame(
        [("a", "g1", "s", "train"), ("b", "g2", "s", "test")],
        "path string, group_id string, sha1 string, split string",
    )
    pairs = spark.createDataFrame([], "path_a string, path_b string")
    assert split_job.check_leakage(images, pairs)["files_in_train_and_heldout"] == 1
