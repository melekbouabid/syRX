"""POST /api/scan with tiny fake ONNX models (tests/fake_models.py), no real weights needed."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from app.limits import BodySizeLimit
from app.scanner import letterbox, nms, square_crop
from tests.fake_models import write_models

NAMES = ["005", "007", "009", "011", "013", "015"]
# A 200 x 100 photo letterboxed to 64 x 64: scale 0.32, 16 px of padding above.
RED = (20, 30, 60, 70)  # x1, y1, x2, y2 in the photo
GREEN = (120, 20, 170, 80)
BOXES = [
    (12.8, 32.0, 12.8, 12.8, 0.90),  # red square
    (13.0, 32.2, 12.8, 12.8, 0.80),  # the same pill found twice: removed by NMS
    (46.4, 32.0, 16.0, 19.2, 0.95),  # green rectangle
    (30.0, 30.0, 5.0, 5.0, 0.10),  # below the detection threshold
]


def photo(*, fmt: str = ".png") -> bytes:
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    x1, y1, x2, y2 = RED
    image[y1:y2, x1:x2] = (0, 0, 255)  # BGR
    x1, y1, x2, y2 = GREEN
    image[y1:y2, x1:x2] = (0, 255, 0)
    ok, encoded = cv2.imencode(fmt, image)
    assert ok
    return encoded.tobytes()


def upload(client: TestClient, data: bytes, content_type: str = "image/png") -> object:
    return client.post("/api/scan", files={"image": ("pills.png", data, content_type)})


@pytest.fixture
def models(app: FastAPI) -> Path:
    directory: Path = app.state.settings.models_dir
    write_models(directory, BOXES, NAMES)
    return directory


def test_scan_requires_sign_in(anonymous: TestClient) -> None:
    response = anonymous.post("/api/scan", files={"image": ("a.png", photo(), "image/png")})
    assert response.status_code == 401


def test_scan_finds_and_identifies_every_pill(user: TestClient, models: Path, pharma: str) -> None:
    response = user.post("/api/scan", files={"image": ("a.png", photo(), "image/png")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["image_width"], body["image_height"]) == (200, 100)
    assert body["inference_ms"] >= 0 and body["timings"]["total_ms"] == body["inference_ms"]
    assert body["message"] is None
    assert "consultez un pharmacien ou un médecin" in body["disclaimer"]
    assert {s["name"] for s in body["sources"]} == {"c3pi", "rxnav"}
    assert body["models"]["classify"]["version"] == 0

    green, red = body["pills"]  # most confident first; the duplicate and the faint box are gone
    assert green["detection_confidence"] == pytest.approx(0.95)
    for pill, expected in ((red, RED), (green, GREEN)):
        box = pill["box"]
        assert [box[k] for k in ("x1", "y1", "x2", "y2")] == pytest.approx(expected, abs=0.6)
        norm = pill["box_normalized"]
        assert norm["x1"] == pytest.approx(expected[0] / 200, abs=0.01)
        assert norm["y2"] == pytest.approx(expected[3] / 100, abs=0.01)

    top_red, top_green = red["candidates"][0], green["candidates"][0]
    assert top_red["class_id"] == 5 and top_red["name"] == "HYZAAR 50MG"
    assert top_red["generic_name"] == "LOSARTAN AND HYDROCHLOROTHIAZIDE"
    assert top_red["active_ingredients"] == ["hydrochlorothiazide", "losartan"]
    assert top_green["class_id"] == 7 and top_green["active_ingredients"] == ["warfarin"]
    for pill in body["pills"]:
        probabilities = [c["probability"] for c in pill["candidates"]]
        assert len(probabilities) == 5
        assert probabilities == sorted(probabilities, reverse=True)
        assert sum(probabilities) <= 1.0001
        assert pill["needs_confirmation"] is False  # top-1 above the 0.85 default
    # A class the medicine tables do not know is still returned, without inventing details.
    unknown = next(c for c in red["candidates"] if c["class_id"] == 13)
    assert unknown["name"] == "class 13" and unknown["active_ingredients"] == []


def test_low_confidence_asks_for_confirmation(
    app: FastAPI, user: TestClient, models: Path, pharma: str
) -> None:
    app.state.settings = app.state.settings.model_copy(update={"scan_confirm_threshold": 0.9999})
    body = upload(user, photo()).json()
    assert body["confirm_threshold"] == 0.9999
    assert all(pill["needs_confirmation"] for pill in body["pills"])


def test_photo_without_pills(app: FastAPI, user: TestClient, pharma: str) -> None:
    write_models(app.state.settings.models_dir, [(10, 10, 4, 4, 0.05)], NAMES)
    body = upload(user, photo(fmt=".jpg"), "image/jpeg").json()
    assert body["pills"] == []
    assert body["message"].startswith("Aucun comprimé détecté")


def test_missing_models_answer_503(user: TestClient, pharma: str) -> None:
    response = upload(user, photo())
    assert response.status_code == 503
    assert "tujya_vision.publish" in response.json()["detail"]


def test_missing_medicine_tables_answer_503(user: TestClient, models: Path) -> None:
    response = upload(user, photo())
    assert response.status_code == 503
    assert "pharma_loader" in response.json()["detail"]


def test_uploads_must_be_images(user: TestClient, models: Path, pharma: str) -> None:
    assert upload(user, b"hello", "text/plain").status_code == 415
    broken = upload(user, b"\x89PNG not really", "image/png")
    assert broken.status_code == 415
    assert "not an image" in broken.json()["detail"]
    assert upload(user, b"", "image/png").status_code == 415


def test_uploads_have_a_size_limit(
    app: FastAPI, user: TestClient, models: Path, pharma: str
) -> None:
    app.state.settings = app.state.settings.model_copy(update={"scan_max_bytes": 1000})
    response = upload(user, photo(fmt=".bmp"))  # about 60 kB
    assert response.status_code == 413


def test_body_limit_refuses_on_content_length() -> None:
    async def echo(request: object) -> PlainTextResponse:
        return PlainTextResponse("ok")

    inner = Starlette(routes=[Route("/", echo, methods=["POST"])])
    client = TestClient(BodySizeLimit(inner, max_bytes=10))
    assert client.post("/", content=b"small").status_code == 200
    assert client.post("/", content=b"x" * 11).status_code == 413


def test_letterbox_keeps_the_aspect_ratio() -> None:
    padded, ratio, (left, top) = letterbox(np.zeros((100, 200, 3), np.uint8), 64)
    assert padded.shape == (64, 64, 3)
    assert ratio == pytest.approx(0.32) and (left, top) == (0, 16)
    assert padded[0, 0].tolist() == [114, 114, 114]  # Ultralytics' grey padding


def test_nms_keeps_the_best_of_overlapping_boxes() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    assert nms(boxes, np.array([0.8, 0.9, 0.5]), iou=0.5) == [1, 2]


def test_square_crop_pads_at_the_border() -> None:
    image = np.zeros((50, 50, 3), np.uint8)
    crop = square_crop(image, (0, 0, 20, 10), 32)
    assert crop.shape == (32, 32, 3)
