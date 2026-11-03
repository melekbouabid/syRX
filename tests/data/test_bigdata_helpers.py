"""Host-side helpers of the Big Data step (no Spark needed): the cluster launcher, the
before / after table, the v2 ingredient mapping and the download list."""

from __future__ import annotations

import csv
from pathlib import Path

from src.data import c3pi, data_cli
from src.data.corpus.ingredients import IngredientRow
from src.data.spark import log_stats, rag_sources, submit


def test_submit_runs_the_job_in_the_master_container():
    command = submit.spark_submit_command("split", driver_memory="2g", executor_memory="4g")
    assert command[:7] == ["docker", "compose", "--profile", "spark", "exec", "-T", "spark-master"]
    assert command[command.index("--master") + 1] == "spark://spark-master:7077"
    assert "spark.driver.host=spark-master" in command
    assert command[-4:] == [
        "/opt/tujia/src/data/spark/run.py",
        "split",
        "--master",
        submit.MASTER_URL,
    ]
    assert command[command.index("--driver-memory") + 1] == "2g"


def test_before_after_table():
    v1 = {
        "classes": 150,
        "images": 4206,
        "consumer_photos": 4000,
        "bytes": 4.77e9,
        "consumer_per_class_mean": 26.7,
    }
    v2 = {
        "classes": 380,
        "images": 20000,
        "consumer_photos": 12000,
        "bytes": 39e9,
        "augmented_bytes": 0.2e9,
        "augmented_images": 3000,
        "consumer_per_class_mean": 32.1,
    }
    table = {r["measure"]: (r["before"], r["after"]) for r in log_stats.before_after(v1, v2)}
    assert table["classes"] == (150, 380)
    assert table["augmented train images"] == (0, 3000)
    assert table["GB"] == (4.77, 39.2)


def test_flatten_params():
    out: dict = {}
    log_stats._flatten("bigdata", {"seed": 0, "split": {"val": 0.15}}, out)
    assert out == {"bigdata.seed": 0, "bigdata.split.val": 0.15}


def _row(class_id: int, key: str, ingredient: str) -> IngredientRow:
    return IngredientRow(class_id, key, "1", "ACTIVE", "9", "p", "7", ingredient, "ndc")


def test_v2_classes_reuse_the_v1_mapping(tmp_path, monkeypatch):
    classes = [
        {"class_id": "0", "class_key": "B_1", "ndc11_list": "2", "generic_name": "B"},
        {"class_id": "1", "class_key": "A_1", "ndc11_list": "1", "generic_name": "A"},
    ]
    v1 = [_row(5, "A_1", "warfarin"), _row(5, "A_1", "aspirin")]
    known, missing = rag_sources.reuse_v1(classes, v1)
    assert [(r.class_id, r.ingredient) for r in known] == [(1, "warfarin"), (1, "aspirin")]
    assert [p["class_key"] for p in missing] == ["B_1"]
    monkeypatch.setattr(rag_sources, "CLASS_INGREDIENTS", tmp_path / "class_ingredients.csv")
    assert rag_sources.write_class_ingredients(known, classes) == 1
    with (tmp_path / "class_ingredients.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["class_key"], r["active_ingredients"]) for r in rows] == [
        ("B_1", ""),
        ("A_1", "aspirin; warfarin"),
    ]


def test_download_list_selects_index_records(tmp_path: Path):
    records = [
        c3pi.ImageRecord(
            disc="PillProjectDisc1",
            file_name=name,
            file_type="JPG",
            file_size=1,
            sha1="x",
            image_type="C3PI_Test",
            ndc11="1",
            ndc9="1",
            part=1,
            parts=1,
            generic_name="",
            proprietary_name="",
            labeled_by="",
            imprint="",
            color="",
            shape="",
            size_mm="",
            camera="",
            illumination="",
            background="",
        )
        for name in ("A.JPG", "B.JPG", "C.JPG")
    ]
    index = tmp_path / "images.csv"
    c3pi.write_index(records, index)
    listing = tmp_path / "list.csv"
    listing.write_text(
        "path,sha1\nPillProjectDisc1/images/C.JPG,x\nPillProjectDisc1/images/A.JPG,x\n",
        encoding="utf-8",
    )
    assert [r.file_name for r in data_cli._listed(index, listing)] == ["C.JPG", "A.JPG"]
