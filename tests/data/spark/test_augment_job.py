"""Job `augment`: rebalancing of the rare classes, from the train split only."""

from __future__ import annotations

import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("pyspark")

from src.data.spark import augment_job
from tests.data.spark.fixtures import photo, segmented


def test_plan_fills_up_to_the_target_using_each_source_in_turn():
    sources = pd.DataFrame(
        {
            "path": ["b", "a"],
            "class_id": [3, 3],
            "class_key": ["K", "K"],
            "role": ["consumer", "seg"],
        }
    )
    out = augment_job.plan(sources, target=6, seed=0)
    assert out["source"].tolist() == ["a", "b", "a", "b"]
    assert out["index"].tolist() == [0, 1, 2, 3]
    assert out["seed"].nunique() == 4
    assert out.equals(augment_job.plan(sources, target=6, seed=0))  # reproducible
    assert augment_job.plan(sources, target=2, seed=0).empty  # already at the target
    assert augment_job.plan(sources.iloc[0:0], target=6, seed=0).empty


def test_cut_pill_drops_the_banner_and_crops_to_the_mask(tmp_path):
    with Image.open(segmented(tmp_path / "s.PNG")) as image:
        pill = augment_job.cut_pill(image)
    assert 36 <= pill.width <= 42 and 36 <= pill.height <= 42  # the 20 px radius disc


def test_augment_returns_rgb_of_the_requested_size(tmp_path):
    for source in (photo(tmp_path / "p.JPG", 1), segmented(tmp_path / "s.PNG")):
        image, operations = augment_job.augment(source, seed=7, size=64)
        assert image.mode == "RGB" and max(image.size) == 64
        assert operations.startswith("rotate(") and "brightness(" in operations
        assert augment_job.augment(source, seed=7, size=64)[1] == operations


@pytest.fixture
def split_table(tmp_path):
    images = tmp_path / "images"
    rows = []

    def add(class_id, key, name, split, role="consumer", exists=True):
        path = f"D1/images/{key}_{name}"
        if exists:
            file = images / "D1" / f"{key}_{name}"
            segmented(file) if name.endswith(".PNG") else photo(file, len(rows))
        rows.append(
            {"path": path, "class_id": class_id, "class_key": key, "role": role, "split": split}
        )

    add(0, "X", "0.JPG", "train")
    add(0, "X", "1.JPG", "train")
    add(0, "X", "s.PNG", "train", role="seg")
    add(0, "X", "r.JPG", "train", role="reference")  # not a source of augmentation
    # Held-out files do not even exist: reading one would fail the generation.
    add(0, "X", "v.JPG", "val", exists=False)
    add(0, "X", "t.JPG", "test", exists=False)
    for i in range(7):
        add(1, "Z", f"{i}.JPG", "train")
    return images, pd.DataFrame(rows)


def test_job_augments_rare_classes_from_train_only(spark, params, tmp_path, split_table):
    images, table = split_table
    split_dir = tmp_path / "split"
    spark.createDataFrame(table).write.parquet(str(split_dir))
    out = tmp_path / "augmented"
    summary = augment_job.run(
        spark, params, split_images=split_dir, images=images, out=out, partitions=2
    )
    manifest = spark.read.parquet(str(out / "manifest")).toPandas()
    assert manifest["ok"].all(), manifest["error"].tolist()
    assert sorted(manifest["class_key"]) == ["X"] * 3  # target 6, X has 3 sources, Z has 7
    train = set(table.loc[(table["split"] == "train") & (table["role"] != "reference"), "path"])
    assert set(manifest["source"]) <= train
    assert summary["sources_not_train"] == 0
    assert summary["generated"] == 3 and summary["min_train_after"] == 6
    files = sorted(p.name for p in (out / "images" / "000").iterdir())
    assert files == ["X_aug000.jpg", "X_aug001.jpg", "X_aug002.jpg"]
    with Image.open(out / "images" / "000" / "X_aug000.jpg") as image:
        assert max(image.size) == 64
