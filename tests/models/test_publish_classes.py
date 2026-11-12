"""A classifier is published only with a class list the API resolves to the same products
(src/models/publish.py `check_class_table`, D101)."""

from __future__ import annotations

import json
from pathlib import Path

import onnx
import pytest

from src.models import publish
from tests.models.classification.helpers import tiny_classifier

TABLE = "class_id,class_key,name\n0,111111111_1,A\n1,222222222_1,B\n2,333333333_1,C\n"


@pytest.fixture
def table(tmp_path: Path) -> Path:
    path = tmp_path / "pill_classes.csv"
    path.write_text(TABLE, encoding="utf-8")
    return path


def _model(tmp_path: Path, names: dict[int, str], keys: dict[int, str] | None) -> Path:
    metadata = {"names": repr(names)}
    if keys is not None:
        metadata["class_keys"] = json.dumps({str(k): v for k, v in keys.items()})
        metadata["label_space"] = "c3pi-v2"
    path = tmp_path / "classifier.onnx"
    onnx.save(tiny_classifier(classes=len(names), metadata=metadata), str(path))
    return path


def test_an_ultralytics_export_without_class_keys_is_published_as_before(
    tmp_path: Path, table: Path
) -> None:
    publish.check_class_table(_model(tmp_path, {0: "000", 1: "001"}, None), table)


def test_outputs_naming_the_same_products_are_accepted(tmp_path: Path, table: Path) -> None:
    # Output 0 is class 2 of the table, output 1 class 0: same products, another order.
    model = _model(tmp_path, {0: "002", 1: "000"}, {0: "333333333_1", 1: "111111111_1"})
    publish.check_class_table(model, table)


def test_a_model_of_another_label_space_is_refused(tmp_path: Path, table: Path) -> None:
    # v2 ids: output 1 is folder "001", but in the API's table class 1 is another product,
    # and output 2 (class 7) is not in the table at all.
    names = {0: "000", 1: "001", 2: "007"}
    keys = {0: "111111111_1", 1: "999999999_1", 2: "444444444_1"}
    with pytest.raises(SystemExit, match="2 outputs would name another product"):
        publish.check_class_table(_model(tmp_path, names, keys), table)
