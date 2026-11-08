"""Demo gallery: only listed test-split photos that exist are served, read only.

A photo must be in the v2 test split (the detector's) and not in the v1 train or val split
(the classifier's): never seen in training by a served model (D105). Grad-CAM pictures are
served next to the photos when they were precomputed (D104).
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import PROJECT_DIR
from app.routers.demo import gradcam_name

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
# v2 test split (config/splits/c3pi_v2/test.csv): every photo above except B.
HOLDOUT = """path,class_id,image_type,sha1,group_id
Disc1/images/A.JPG,5,C3PI_Test,x,g1
Disc2/images/C.JPG,7,C3PI_Test,x,g2
Disc2/images/D.JPG,9,MC_API_RXNAV_V1.3,x,g3
Disc2/images/E.JPG,9,C3PI_Test,x,g4
"""


@pytest.fixture
def photos(app: FastAPI) -> FastAPI:
    settings = app.state.settings
    settings.demo_split_file.write_text(SPLIT, encoding="utf-8")
    settings.demo_holdout_file.write_text(HOLDOUT, encoding="utf-8")
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


# --- never seen in training by the served models (D105) --------------------------------------

# F: in the v2 test split, not in v1 at all (the v1 classifier never had it): served.
# G: in the v1 test split but not in the v2 test split (the v2 detector trained on it): refused.
# H: in the v2 test split but in the v1 train split (the classifier trained on it): refused.
SPLIT_V1 = """path,class_id,image_type,file_size,sha1,split
Disc3/images/G.JPG,1,C3PI_Test,1,x,test
Disc3/images/H.JPG,2,C3PI_Test,1,x,train
"""
HOLDOUT_V2 = """path,class_id,image_type,sha1,group_id
Disc3/images/F.JPG,30,C3PI_Test,x,g1
Disc3/images/H.JPG,31,C3PI_Test,x,g2
"""
GALLERY_V2 = """path,class_id,why
Disc3/images/G.JPG,1,major
Disc3/images/F.JPG,3,major
Disc3/images/H.JPG,2,major
"""


@pytest.fixture
def unseen(app: FastAPI) -> FastAPI:
    settings = app.state.settings
    settings.demo_split_file.write_text(SPLIT_V1, encoding="utf-8")
    settings.demo_holdout_file.write_text(HOLDOUT_V2, encoding="utf-8")
    settings.demo_gallery_file.write_text(GALLERY_V2, encoding="utf-8")
    image = np.full((300, 400, 3), 120, dtype=np.uint8)
    for name in ("G", "F", "H"):
        file: Path = settings.demo_images_dir / "Disc3" / f"{name}.JPG"
        file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(file), image)
    return app


def test_only_photos_no_served_model_trained_on_are_listed(
    user: TestClient, unseen: FastAPI
) -> None:
    body = user.get("/api/demo/images").json()
    assert [(i["id"], i["class_id"]) for i in body["items"]] == [(0, 3)]  # F only
    assert "ni le détecteur ni le classifieur" in body["note"]
    assert user.get("/api/demo/images/1").status_code == 404


def test_no_v2_test_split_file_means_an_empty_gallery(user: TestClient, photos: FastAPI) -> None:
    photos.state.settings.demo_holdout_file.unlink()
    assert user.get("/api/demo/images").json()["items"] == []


# --- Grad-CAM (D104) --------------------------------------------------------------------------


def test_gradcam_name_flattens_the_disc_folder() -> None:
    assert gradcam_name("PillProjectDisc20/images/B3U!X.JPG") == "PillProjectDisc20_B3U!X.jpg"


def test_gradcam_is_served_only_when_it_was_computed(user: TestClient, photos: FastAPI) -> None:
    settings = photos.state.settings
    items = user.get("/api/demo/images").json()["items"]
    assert [i["gradcam_url"] for i in items] == [None, None]
    missing = user.get("/api/demo/images/0/gradcam")
    assert missing.status_code == 404
    assert missing.json()["detail"] == "No Grad-CAM for this photo"

    picture = np.zeros((240, 320, 3), np.uint8)
    picture[:, :, 2] = 255
    settings.demo_gradcam_dir.mkdir(parents=True)
    cv2.imwrite(str(settings.demo_gradcam_dir / gradcam_name("Disc1/images/A.JPG")), picture)
    items = user.get("/api/demo/images").json()["items"]
    assert items[0]["gradcam_url"] == "/api/demo/images/0/gradcam"
    assert items[1]["gradcam_url"] is None  # C has none
    served = user.get("/api/demo/images/0/gradcam")
    assert served.status_code == 200 and served.headers["content-type"] == "image/jpeg"
    decoded = cv2.imdecode(np.frombuffer(served.content, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (240, 320, 3)
    assert user.get("/api/demo/images/1/gradcam").status_code == 404
    assert user.get("/api/demo/images/7/gradcam").status_code == 404  # no such photo


def test_gradcam_requires_sign_in(anonymous: TestClient, photos: FastAPI) -> None:
    assert anonymous.get("/api/demo/images/0/gradcam").status_code == 401


# --- the shipped gallery ------------------------------------------------------------------------


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_the_shipped_gallery_was_never_seen_in_training() -> None:
    """config/demo_gallery.csv: v2 test consumer photos, absent from the v1 train/val split."""
    config = PROJECT_DIR / "config"
    gallery = _rows(config / "demo_gallery.csv")
    test_v2 = {r["path"]: r for r in _rows(config / "splits" / "c3pi_v2" / "test.csv")}
    v1 = {r["path"]: r for r in _rows(config / "splits" / "selected_images.csv")}
    classes = {r["class_id"] for r in _rows(config / "pill_classes.csv")}
    assert len(gallery) == 8
    for row in gallery:
        assert row["path"] in test_v2 and test_v2[row["path"]]["image_type"] == "C3PI_Test"
        assert v1.get(row["path"], {}).get("split", "test") == "test", row["path"]
        assert row["class_id"] in classes  # a class of the served classifier (v1 ids)


def test_the_shipped_gallery_is_in_the_detection_holdout() -> None:
    """Stricter, with the DVC data: test_holdout.txt of D93 (no photo of the same session in
    the v1 train or val split) and the same product in v1 and v2."""
    holdout = PROJECT_DIR / "data" / "processed" / "c3pi" / "datasets_v2" / "detect"
    if not (holdout / "test_holdout.txt").is_file():
        pytest.skip("DVC data not pulled (data/processed/c3pi/datasets_v2/detect)")
    held = set((holdout / "test_holdout.txt").read_text(encoding="utf-8").split())
    index = {r["path"]: r for r in _rows(holdout / "index.csv")}
    config = PROJECT_DIR / "config"
    v1_keys = {r["class_id"]: r["class_key"] for r in _rows(config / "pill_classes.csv")}
    v2_keys = {
        r["class_id"]: r["class_key"] for r in _rows(config / "splits" / "c3pi_v2" / "classes.csv")
    }
    for row in _rows(config / "demo_gallery.csv"):
        entry = index[row["path"]]
        assert entry["file"] in held and entry["holdout"] == "1", row["path"]
        assert v2_keys[entry["class_id"]] == v1_keys[row["class_id"]], row["path"]
