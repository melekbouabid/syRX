"""Job `index` on a tiny fake archive: one row per image, statistics, class selection."""

from __future__ import annotations

import csv
import json

import pytest

pytest.importorskip("pyspark")

from src.data.spark import index_job
from tests.data.spark.fixtures import write_metadata

A, B, C = "11111111101", "22222222201", "33333333301"


def archive() -> dict[str, list[dict]]:
    disc1 = [
        *(
            {"ndc11": A, "file": f"A{i}.JPG", "image_type": "C3PI_Test", "camera": f"cam{i % 2}"}
            for i in range(4)
        ),
        # Listed twice by NLM: counted once.
        {"ndc11": A, "file": "A0.JPG", "image_type": "C3PI_Test", "camera": "cam0"},
        {"ndc11": A, "file": "A.WMV", "image_type": "C3PI_Test"},
        {"ndc11": A, "file": "A.CR2", "image_type": "C3PI_Reference", "size": 23_000_000},
        {"ndc11": A, "file": "ARX.JPG", "image_type": "MC_API_RXNAV_V1.3"},
        {"ndc11": A, "file": "ANLM.JPG", "image_type": "MC_API_NLMIMAGE_V1.3"},
        *(
            {
                "ndc11": A,
                "file": f"AS{i}.PNG",
                "image_type": "MC_C3PI_REFERENCE_SEG_V1.6",
                "sha1": f"{i:040d}",
            }
            for i in range(3)
        ),
        {"ndc11": A, "file": "ACH.JPG", "image_type": "MC_CHALLENGE_V1.0"},
        {"ndc11": A, "file": "ACAL.PNG", "image_type": "MC_COOKED_CALIBRATED_V1.2"},
    ]
    disc2 = [
        *({"ndc11": B, "file": f"B{i}.JPG", "image_type": "C3PI_Test"} for i in range(2)),
        {"ndc11": B, "file": "BCH.JPG", "image_type": "MC_CHALLENGE_V1.0"},
        *(
            {
                "ndc11": C,
                "part": 2,
                "file": f"C{i}.JPG",
                "image_type": "C3PI_Test",
                "name": "ÉPI® 5MG",
            }
            for i in range(3)
        ),
        {
            "ndc11": "33333333302",
            "part": 2,
            "file": "CSPL.JPG",
            "image_type": "MC_SPL_SPLIMAGE_V3.0",
        },
    ]
    return {"PillProjectDisc1": disc1, "PillProjectDisc2": disc2}


@pytest.fixture
def result(spark, params, tmp_path):
    metadata = write_metadata(tmp_path / "metadata", archive())
    out = tmp_path / "index"
    summary = index_job.run(spark, params, metadata=metadata, out=out)
    return spark, out, summary


def test_one_row_per_image_with_its_type(result):
    spark, out, summary = result
    images = spark.read.parquet(str(out / "images"))
    assert summary["xml_rows"] == 21
    assert images.count() == summary["images"] == 20  # A0.JPG once
    assert images.where("in_directory").count() == 20
    assert {p.name for p in (out / "images").iterdir() if p.is_dir()} >= {
        "image_type=C3PI_Test",
        "image_type=MC_CHALLENGE_V1.0",
    }
    row = images.where("path = 'PillProjectDisc2/images/C0.JPG'").first()
    assert (row.ndc9, row.part, row.class_key) == ("333333333", 2, "333333333_2")
    assert row.name == "ÉPI® 5MG"  # ISO-8859-1 XML decoded


def test_statistics_per_type_and_class(result):
    spark, out, summary = result
    assert summary["by_type"]["C3PI_Test JPG"] == {"files": 9, "bytes": 9000, "products": 3}
    assert summary["by_type"]["C3PI_Test WMV"]["files"] == 1
    stats = {r.class_key: r for r in spark.read.parquet(str(out / "class_stats")).collect()}
    assert stats["111111111_1"].consumer_jpg == 4
    assert stats["111111111_1"]["MC_C3PI_REFERENCE_SEG_V1.6"] == 3
    assert stats["222222222_1"].consumer_jpg == 2


def test_classes_need_n_consumer_photos(result):
    spark, out, summary = result
    classes = spark.read.parquet(str(out / "classes")).orderBy("rank").collect()
    assert [(c.class_key, c.n_consumer, c.rank) for c in classes] == [
        ("111111111_1", 4, 0),
        ("333333333_2", 3, 1),
    ]
    assert classes[1].ndc11_list == "33333333301;33333333302"
    assert summary["candidate_classes"] == 2


def test_selection_roles(result):
    spark, out, summary = result
    selection = {
        r.path.split("/")[-1]: r for r in spark.read.parquet(str(out / "selection")).collect()
    }
    roles = {name: r.role for name, r in selection.items()}
    assert {n for n, r in roles.items() if r == "consumer"} == {
        "A0.JPG",
        "A1.JPG",
        "A2.JPG",
        "A3.JPG",
        "C0.JPG",
        "C1.JPG",
        "C2.JPG",
    }
    # One RxImage JPG per package, RxNav first; SPL when it is the only one.
    assert {n for n, r in roles.items() if r == "reference"} == {"ARX.JPG", "CSPL.JPG"}
    assert {n for n, r in roles.items() if r == "seg"} == {"AS0.PNG", "AS1.PNG"}  # 2 per class
    assert roles["ACH.JPG"] == roles["BCH.JPG"] == "challenge"  # challenge: all
    assert selection["ACH.JPG"].in_class and not selection["BCH.JPG"].in_class
    assert not {"B0.JPG", "A.WMV", "A.CR2", "ACAL.PNG"} & set(roles)
    with (out / "download_list.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == summary["selection_files"] == 13
    assert json.loads((out / "summary.json").read_text())["selection"]["seg"]["files"] == 2


def test_camera_filter_and_challenge_of_classes_only(spark, params, tmp_path):
    metadata = write_metadata(tmp_path / "metadata", archive())
    params["index"].update(exclude_cameras=["cam1"], challenge="classes")
    summary = index_job.run(spark, params, metadata=metadata, out=tmp_path / "index")
    # A keeps 2 photos (cam0) < 3: only C is left, and no challenge image belongs to C.
    assert summary["candidate_classes"] == 1
    assert "challenge" not in summary["selection"]
