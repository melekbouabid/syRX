"""Demo gallery: only listed test-split photos that exist are served, read only."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

SPLIT = """path,class_id,image_type,file_size,sha1,split
Disc1/images/A.JPG,5,C3PI_Test,1,x,test
Disc1/images/B.JPG,7,C3PI_Test,1,x,train
Disc2/images/C.JPG,7,C3PI_Test,1,x,test
Disc2/images/D.JPG,9,MC_API_RXNAV_V1.3,1,x,test
Disc2/images/E.JPG,9,C3PI_Test,1,x,test
"""
GALLERY = """path,class_id,why
Disc1/images/A.JPG,5,major
Disc1/images/B.JPG,7,major
Disc2/images/C.JPG,7,confirm
Disc2/images/D.JPG,9,major
Disc2/images/E.JPG,9,major
"""


@pytest.fixture
def photos(app: FastAPI) -> FastAPI:
    settings = app.state.settings
    settings.demo_split_file.write_text(SPLIT, encoding="utf-8")
    settings.demo_gallery_file.write_text(GALLERY, encoding="utf-8")
    image = np.full((480, 640, 3), 200, dtype=np.uint8)
    # E.JPG is listed and in the test split but missing on disk.
    for name in ("Disc1/A.JPG", "Disc1/B.JPG", "Disc2/C.JPG", "Disc2/D.JPG"):
        file: Path = settings.demo_images_dir / name
        file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(file), image)
    return app


def test_gallery_requires_sign_in(anonymous: TestClient, photos: FastAPI) -> None:
    assert anonymous.get("/api/demo/images").status_code == 401
    assert anonymous.get("/api/demo/images/0").status_code == 401


def test_only_test_split_consumer_photos_are_listed(
    user: TestClient, photos: FastAPI, pharma: str
) -> None:
    body = user.get("/api/demo/images").json()
    assert [(i["id"], i["class_id"], i["why"]) for i in body["items"]] == [
        (0, 5, "major"),  # A; B is train, D a reference image, E missing
        (1, 7, "confirm"),  # C
    ]
    assert body["items"][0]["expected_name"] == "HYZAAR 50MG"
    assert body["items"][1]["url"] == "/api/demo/images/1"
    assert "test" in body["note"]


def test_gallery_without_medicine_tables_still_lists(user: TestClient, photos: FastAPI) -> None:
    items = user.get("/api/demo/images").json()["items"]
    assert len(items) == 2 and items[0]["expected_name"] == ""


def test_photos_and_thumbnails_are_served(user: TestClient, photos: FastAPI) -> None:
    full = user.get("/api/demo/images/0")
    assert full.status_code == 200 and full.headers["content-type"] == "image/jpeg"
    thumb = user.get("/api/demo/images/0", params={"thumbnail": "true"})
    assert thumb.status_code == 200
    decoded = cv2.imdecode(np.frombuffer(thumb.content, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == 360 and decoded.shape[0] == 270
    assert user.get("/api/demo/images/2").status_code == 404
    assert user.get("/api/demo/images/-1").status_code == 404


def test_no_gallery_file_means_an_empty_gallery(user: TestClient) -> None:
    assert user.get("/api/demo/images").json()["items"] == []
