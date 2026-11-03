"""Job `epillid`: overlap of the ePillID benchmark with C3PI and with the classes."""

from __future__ import annotations

import pytest

pytest.importorskip("pyspark")

from src.data.spark import epillid_job


def test_overlap_by_ndc9_and_kind(spark, params, tmp_path):
    labels = tmp_path / "all_labels.csv"
    labels.write_text(
        "images,pilltype_id,label_code_id,prod_code_id,is_ref,is_front,is_new,image_path,label\n"
        "0.jpg,00002-3228-30_X,2,3228,False,False,False,fcn_mix_weight/dc_224/0.jpg,a\n"
        "1.jpg,00002-3228-30_X,2,3228,True,True,False,fcn_mix_weight/dr_224/1.jpg,a\n"
        "2.jpg,0002-3270_0,2,3270,True,True,True,segmented_nih_pills_224/2.jpg,b\n"
        "3.jpg,99999-0001-01_Y,99999,1,False,False,False,fcn_mix_weight/dc_224/3.jpg,c\n",
        encoding="utf-8",
    )
    index = tmp_path / "images"
    spark.createDataFrame([("000023228",), ("000023270",)], "ndc9 string").write.parquet(str(index))
    v2 = tmp_path / "classes_v2.csv"
    v2.write_text("class_id,class_key,name\n0,000023228_1,A 10MG\n1,000025555_1,B\n", "utf-8")
    v1 = tmp_path / "classes_v1.csv"
    v1.write_text("class_id,class_key\n0,000025555_1\n", encoding="utf-8")
    out = tmp_path / "epillid"
    summary = epillid_job.run(
        spark,
        params,
        epillid_csv=labels,
        index=index,
        classes_v2=v2,
        classes_v1=v1,
        out=out,
    )
    assert summary["images"] == 4 and summary["pill_types"] == 3
    assert summary["pill_types_in_c3pi"] == 2  # 99999-0001 is not in the archive
    assert summary["by_kind"]["consumer"] == {
        "images": 2,
        "pill_types": 2,
        "images_in_c3pi": 1,
        "images_in_v2_classes": 1,
        "images_in_v1_classes": 0,
    }
    assert summary["by_kind"]["pillbox_reference"]["images_in_c3pi"] == 1
    classes = spark.read.parquet(str(out / "classes")).collect()
    assert [(c.class_key, c.consumer, c.challenge_reference) for c in classes] == [
        ("000023228_1", 1, 1)
    ]
    assert summary["v2_classes_with_epillid_consumer"] == 1
