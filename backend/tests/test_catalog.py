"""GET /api/catalog/summary and /showcase: public counts, one real scan of a demo photo (D128)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.fake_models import write_models

NAMES = ["005", "007", "009", "011", "013", "015"]
GREEN_BOX = [(46.4, 32.0, 16.0, 19.2, 0.95)]  # one pill, the green rectangle below
SPLIT = "path,class_id,image_type,file_size,sha1,split\nDisc1/images/G.JPG,7,C3PI_Test,1,x,test\n"
HOLDOUT = "path,class_id,image_type,sha1,group_id\nDisc1/images/G.JPG,7,C3PI_Test,x,g1\n"
GALLERY = "path,class_id,why\nDisc1/images/G.JPG,7,major\n"


@pytest.fixture
def gallery(app: FastAPI) -> FastAPI:
    """One demo photo: a green pill, which the fake classifier calls class 7 (warfarin)."""
    settings = app.state.settings
    settings.demo_split_file.write_text(SPLIT, encoding="utf-8")
    settings.demo_holdout_file.write_text(HOLDOUT, encoding="utf-8")
    settings.demo_gallery_file.write_text(GALLERY, encoding="utf-8")
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    image[20:80, 120:170] = (0, 255, 0)  # BGR green
    file: Path = settings.demo_images_dir / "Disc1" / "G.JPG"
    file.parent.mkdir(parents=True)
    cv2.imwrite(str(file), image)
    write_models(settings.models_dir, GREEN_BOX, NAMES)
    return app


def test_summary_is_public_and_counts_the_tables(
    anonymous: TestClient, gallery: FastAPI, pharma: str
) -> None:
    response = anonymous.get("/api/catalog/summary")
    assert response.status_code == 200, response.text
    body = response.json()
    # tujya_testing/pharma_data.py: 4 pill classes, 6 ingredients, 4 interaction pairs.
    assert body["counts"] == {"pills": 4, "ingredients": 6, "interaction_pairs": 4}
    assert body["confirm_threshold"] == 0.85
    assert "consultez un pharmacien ou un médecin" in body["disclaimer"]


def test_showcase_is_a_real_scan_of_a_gallery_photo(
    anonymous: TestClient, gallery: FastAPI, pharma: str
) -> None:
    showcase = anonymous.get("/api/catalog/showcase").json()["showcase"]
    assert showcase["photo_id"] == 0
    assert showcase["pill_name"] == "WARFARIN 5MG"
    assert showcase["active_ingredients"] == ["warfarin"]
    assert 0.85 <= showcase["probability"] <= 1
    # The most severe interaction of warfarin in the tables: Major with fluconazole.
    assert showcase["interaction"] == {
        "ingredient_a": "warfarin",
        "ingredient_b": "fluconazole",
        "severity": "Major",
        "source": "DDInter",
    }


def test_no_showcase_below_the_confirmation_threshold(
    app: FastAPI, anonymous: TestClient, gallery: FastAPI, pharma: str
) -> None:
    app.state.settings = app.state.settings.model_copy(update={"scan_confirm_threshold": 0.9999})
    assert anonymous.get("/api/catalog/showcase").json()["showcase"] is None


def test_showcase_without_medicine_tables_is_none(anonymous: TestClient, gallery: FastAPI) -> None:
    assert anonymous.get("/api/catalog/showcase").json()["showcase"] is None


def test_without_models_or_tables_nothing_is_invented(anonymous: TestClient) -> None:
    assert anonymous.get("/api/catalog/summary").json()["counts"] is None
    assert anonymous.get("/api/catalog/showcase").json()["showcase"] is None
