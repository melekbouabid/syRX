"""Endpoints of the course guide: GET /health, GET /model/metadata, POST /predict(/visualize).

Tiny fake ONNX models (tests/fake_models.py) stand in for the published ones; every success
case and every error code (400 invalid image, 413 too large, 503 models not loaded).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.vision import class_colour, text_colour
from tests.fake_models import write_models
from tests.test_scan import BOXES, GREEN, NAMES, RED, photo

MANIFEST = {
    "published_at": "2026-11-06T13:27:28+00:00",
    "detect": {
        "file": "detector.onnx",
        "registered_model": "tujya-pill-detector",
        "version": 6,
        "aliases": ["served"],
        "precision": "fp16",
        "variant_of": 2,
        "arch": "yolov8n",
        "run_id": "abc",
        "sha256": "0" * 64,
        "source": "mlflow",
        "metrics": {"test_v2_mAP50_95": 0.8757, "val_mAP50_95": 0.8563},
    },
    "classify": {
        "file": "classifier.onnx",
        "registered_model": "tujya-pill-classifier",
        "version": 1,
        "aliases": ["champion", "served"],
        "precision": "fp32",
        "arch": "yolo11s-cls",
        "metrics": {"test_top1": 0.9371},
    },
}


@pytest.fixture
def models(app: FastAPI) -> Path:
    directory: Path = app.state.settings.models_dir
    write_models(directory, BOXES, NAMES)
    (directory / "models.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    return directory


def send(client: TestClient, path: str, data: bytes, content_type: str = "image/png") -> object:
    return client.post(path, files={"image": ("pills.png", data, content_type)})


# --- GET /health ----------------------------------------------------------------------------


def test_health_reports_models_gpu_and_versions(anonymous: TestClient, models: Path) -> None:
    response = anonymous.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["api_version"] == "0.3.0"
    assert "CPUExecutionProvider" in body["gpu"]["available_providers"]
    assert body["gpu"]["active_providers"] == ["CPUExecutionProvider"]  # settings: cpu
    assert isinstance(body["gpu"]["available"], bool)
    detect, classify = body["models"]
    assert (detect["task"], detect["version"], detect["precision"]) == ("detect", 6, "fp16")
    assert detect["aliases"] == ["served"] and detect["variant_of"] == 2
    assert (classify["task"], classify["version"]) == ("classify", 1)
    assert body["published_at"] == MANIFEST["published_at"]


def test_health_without_models_is_degraded_not_an_error(anonymous: TestClient) -> None:
    response = anonymous.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded" and body["model_loaded"] is False
    assert "python -m src.models.publish" in body["detail"]
    assert body["models"] == [] and body["gpu"]["active_providers"] == []


def test_old_health_route_is_unchanged(anonymous: TestClient) -> None:
    assert anonymous.get("/api/health").json() == {"status": "ok"}


# --- GET /model/metadata --------------------------------------------------------------------


def test_model_metadata(anonymous: TestClient, models: Path) -> None:
    response = anonymous.get("/model/metadata")
    assert response.status_code == 200
    body = response.json()
    detector, classifier = body["detector"], body["classifier"]
    assert detector["name"] == "tujya-pill-detector"
    assert detector["architecture"] == "yolov8n"
    assert (detector["version"], detector["precision"], detector["variant_of"]) == (6, "fp16", 2)
    assert detector["metrics"] == {"test_v2_mAP50_95": 0.8757, "val_mAP50_95": 0.8563}
    assert detector["classes"] == ["pill"] and detector["input_size"] == 64
    assert classifier["architecture"] == "yolo11s-cls"
    assert classifier["aliases"] == ["champion", "served"]
    assert classifier["classes"] == NAMES and classifier["num_classes"] == len(NAMES)
    assert classifier["input_size"] == 16
    assert "consultez un pharmacien" in body["disclaimer"]


def test_model_metadata_without_a_manifest(anonymous: TestClient, app: FastAPI) -> None:
    directory: Path = app.state.settings.models_dir
    write_models(directory, BOXES, NAMES)
    (directory / "models.json").unlink()
    body = anonymous.get("/model/metadata").json()
    assert body["detector"]["name"] == "detector.onnx"
    assert body["detector"]["architecture"] == "unknown"  # no Registry tag, no description
    assert body["detector"]["metrics"] == {} and body["classifier"]["version"] is None


def test_model_metadata_without_models_is_503(anonymous: TestClient) -> None:
    response = anonymous.get("/model/metadata")
    assert response.status_code == 503
    assert "python -m src.models.publish" in response.json()["detail"]


# --- POST /predict --------------------------------------------------------------------------


def test_predict_returns_boxes_top5_and_latency(anonymous: TestClient, models: Path) -> None:
    response = send(anonymous, "/predict", photo())
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["image_width"], body["image_height"]) == (200, 100)
    assert body["count"] == 2 == len(body["pills"])
    green, red = body["pills"]  # most confident first, duplicate and faint box removed
    assert green["confidence"] == pytest.approx(0.95)
    for pill, expected in ((red, RED), (green, GREEN)):
        box = pill["box"]
        assert [box[k] for k in ("x1", "y1", "x2", "y2")] == pytest.approx(expected, abs=0.6)
        assert pill["box_normalized"]["x2"] == pytest.approx(expected[2] / 200, abs=0.01)
        probabilities = [c["probability"] for c in pill["top5"]]
        assert len(probabilities) == 5 and probabilities == sorted(probabilities, reverse=True)
    assert red["top5"][0]["class_id"] == 5 and green["top5"][0]["class_id"] == 7
    timings = body["timings"]
    assert timings["inference_ms"] == pytest.approx(
        timings["detection_ms"] + timings["classification_ms"], abs=0.11
    )
    assert timings["server_ms"] >= timings["inference_ms"] >= 0
    assert body["model_versions"] == {"detect": 6, "classify": 1}
    assert "consultez un pharmacien" in body["disclaimer"]


def test_predict_needs_no_account(anonymous: TestClient, models: Path) -> None:
    assert send(anonymous, "/predict", photo()).status_code == 200


def test_predict_without_pills(anonymous: TestClient, app: FastAPI) -> None:
    write_models(app.state.settings.models_dir, [(10, 10, 4, 4, 0.05)], NAMES)
    body = send(anonymous, "/predict", photo(fmt=".jpg"), "image/jpeg").json()
    assert body["count"] == 0 and body["pills"] == []


@pytest.mark.parametrize("path", ["/predict", "/predict/visualize"])
def test_not_an_image_is_400(anonymous: TestClient, models: Path, path: str) -> None:
    response = send(anonymous, path, b"%PDF-1.7 not an image", "application/pdf")
    assert response.status_code == 400
    assert "JPEG, PNG or WebP" in response.json()["detail"]


@pytest.mark.parametrize("path", ["/predict", "/predict/visualize"])
def test_unreadable_image_is_400(anonymous: TestClient, models: Path, path: str) -> None:
    response = send(anonymous, path, b"\x89PNG broken bytes", "image/png")
    assert response.status_code == 400
    assert "not an image that can be read" in response.json()["detail"]


@pytest.mark.parametrize("path", ["/predict", "/predict/visualize"])
def test_empty_file_is_400(anonymous: TestClient, models: Path, path: str) -> None:
    assert send(anonymous, path, b"", "image/png").status_code == 400


@pytest.mark.parametrize("path", ["/predict", "/predict/visualize"])
def test_too_large_image_is_413(
    anonymous: TestClient, app: FastAPI, models: Path, path: str
) -> None:
    app.state.settings = app.state.settings.model_copy(update={"scan_max_bytes": 100})
    assert send(anonymous, path, photo()).status_code == 413


@pytest.mark.parametrize("path", ["/predict", "/predict/visualize"])
def test_models_not_loaded_is_503(anonymous: TestClient, path: str) -> None:
    response = send(anonymous, path, photo())
    assert response.status_code == 503
    assert "Model files missing" in response.json()["detail"]


def test_missing_file_field_is_422(anonymous: TestClient, models: Path) -> None:
    assert anonymous.post("/predict").status_code == 422


# --- POST /predict/visualize ----------------------------------------------------------------


def test_visualize_draws_coloured_boxes(anonymous: TestClient, models: Path) -> None:
    response = send(anonymous, "/predict/visualize", photo())
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-pill-count"] == "2"
    assert float(response.headers["x-server-ms"]) >= float(response.headers["x-inference-ms"])
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (100, 200, 3)  # same size as the upload (below max_side)
    # The red pill's box edge is drawn in the colour of its top-1 class (005).
    x1, y1, _, y2 = RED
    edge = image[(y1 + y2) // 2, x1]
    assert tuple(int(v) for v in edge) == class_colour(5)
    # The white background far from the pills is untouched.
    assert tuple(int(v) for v in image[95, 100]) == (255, 255, 255)


def test_visualize_downscales_to_max_side(anonymous: TestClient, models: Path) -> None:
    response = anonymous.post(
        "/predict/visualize?max_side=128", files={"image": ("a.png", photo(), "image/png")}
    )
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[:2] == (64, 128)


def test_class_colours_are_stable_and_distinct() -> None:
    assert class_colour(5) == class_colour(5)
    assert len({class_colour(c) for c in range(20)}) == 20


def test_class_colours_follow_the_palette_and_labels_stay_readable() -> None:
    # Same values as frontend/src/lib/vision.test.ts: boxes of the app and of the PNG agree.
    assert class_colour(0) == (0x38, 0x38, 0xFF)  # BGR of #FF3838
    assert class_colour(9) == (0xBB, 0xD4, 0x00)  # #00D4BB
    assert class_colour(120) == class_colour(0)  # 120 mod 20
    assert class_colour(-1) == class_colour(19)
    assert text_colour(class_colour(4)) == (0, 0, 0)  # light yellow-green: black text
    assert text_colour(class_colour(16)) == (255, 255, 255)  # dark violet: white text


def test_swagger_documents_the_guide_endpoints(anonymous: TestClient) -> None:
    paths = anonymous.get("/openapi.json").json()["paths"]
    assert {"/health", "/model/metadata", "/predict", "/predict/visualize"} <= set(paths)
    responses = paths["/predict"]["post"]["responses"]
    assert {"200", "400", "413", "503"} <= set(responses)
    assert "image/png" in paths["/predict/visualize"]["post"]["responses"]["200"]["content"]
