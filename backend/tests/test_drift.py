"""Data drift monitor (app/drift.py): statistics, KS + PSI, sliding window, route."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import drift as D
from tests.fake_models import write_models
from tests.test_scan import BOXES, NAMES
from tests.test_scan import photo as scan_photo


def photo(level: int = 128, noise: float = 30.0, size: tuple[int, int] = (300, 400)) -> np.ndarray:
    rng = np.random.default_rng(level)
    image = rng.normal(level, noise, (*size, 3))
    return np.clip(image, 0, 255).astype(np.uint8)


def test_image_stats_follow_brightness_blur_and_size() -> None:
    base = D.image_stats(photo(), [0.9, 0.8], [0.95, 0.7])
    assert set(base) == set(D.FEATURES)
    assert base["boxes"] == 2 and base["detector_confidence"] == pytest.approx(0.85)
    assert base["classifier_confidence"] == pytest.approx(0.825)
    assert base["megapixels"] == pytest.approx(0.12) and base["aspect_ratio"] == pytest.approx(
        4 / 3
    )
    dark = D.image_stats(photo(40))
    assert dark["brightness"] < base["brightness"] - 50
    import cv2

    blurred = D.image_stats(cv2.GaussianBlur(photo(), (0, 0), 4))
    assert blurred["sharpness"] < base["sharpness"] - 1
    assert "detector_confidence" not in dark  # no pill found: not counted
    assert dark["boxes"] == 0


def test_psi_is_zero_for_the_same_sample_and_large_for_a_shift() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 2000)
    assert D.psi(reference, reference) == pytest.approx(0.0, abs=1e-9)
    assert D.psi(reference, rng.normal(0, 1, 500)) < 0.1
    assert D.psi(reference, rng.normal(1.5, 1, 500)) > 0.25
    # constant reference (e.g. always one pill): its value, and the rest
    assert D.psi(np.ones(100), np.ones(50)) == pytest.approx(0.0, abs=1e-6)
    assert D.psi(np.ones(100), np.zeros(50)) > 0.25


def synthetic_reference(n: int = 600) -> D.Reference:
    rng = np.random.default_rng(1)
    values = {name: rng.normal(10, 2, n).tolist() for name in D.FEATURES}
    return D.Reference(values, dict(D.DEFAULT_THRESHOLDS), {"photos": n})


def records(n: int, shift: float = 0.0, seed: int = 2) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        row = {name: float(rng.normal(10, 2)) for name in D.FEATURES}
        row["brightness"] += shift
        rows.append(row)
    return rows


def test_report_ok_then_drift_then_insufficient() -> None:
    reference = synthetic_reference()
    same = D.drift_report(reference, records(200))
    assert same["status"] == "ok" and same["drifted"] == []
    shifted = D.drift_report(reference, records(200, shift=-6))
    assert shifted["status"] == "drift" and shifted["drifted"] == ["brightness"]
    feature = next(f for f in shifted["features"] if f["feature"] == "brightness")
    assert feature["ks_pvalue"] < 0.01 and feature["psi"] >= 0.25
    few = D.drift_report(reference, records(5, shift=-6))
    assert few["status"] == "insufficient"


def test_monitor_keeps_a_window_and_no_image() -> None:
    monitor = D.DriftMonitor(synthetic_reference(), window=50)
    for row in records(120, shift=-6):
        monitor.record(row)
    report = monitor.report()
    assert report["window_size"] == 50 and report["photos_seen"] == 120
    assert report["status"] == "drift"
    assert all(set(row) <= set(D.FEATURES) for row in monitor.window)  # numbers only
    monitor.reset()
    assert monitor.report()["status"] == "insufficient"
    assert D.DriftMonitor(None).report()["status"] == "unavailable"


def test_reference_file_round_trip(tmp_path) -> None:
    reference = synthetic_reference(10)
    path = tmp_path / "reference.json"
    import json

    path.write_text(json.dumps(reference.to_json()), encoding="utf-8")
    loaded = D.Reference.load(path)
    assert loaded.values == reference.values and loaded.thresholds["psi_drift"] == 0.25
    assert D.DriftMonitor.from_file(tmp_path / "missing.json").reference is None


def test_drift_route_records_numbers_of_every_scanned_photo(
    app: FastAPI, anonymous: TestClient, user: TestClient, admin: TestClient, pharma: str
) -> None:
    write_models(app.state.settings.models_dir, BOXES, NAMES)
    app.state.drift = D.DriftMonitor(synthetic_reference(), window=100)
    before = anonymous.get("/monitoring/drift")
    assert before.status_code == 200  # no sign-in: aggregate numbers only, like /health
    assert before.json()["status"] == "insufficient" and before.json()["window_size"] == 0
    picture = scan_photo()
    assert anonymous.post("/predict", files={"image": ("a.png", picture, "image/png")}).is_success
    sent = user.post("/api/scan", files={"image": ("b.png", picture, "image/png")})
    assert sent.status_code == 200, sent.text
    report = anonymous.get("/monitoring/drift").json()
    assert report["window_size"] == 2 and report["photos_seen"] == 2
    assert {f["feature"] for f in report["features"]} == set(D.FEATURES)
    # Only the eight numbers are kept: no image, no file name, nothing about the user.
    assert all(set(row) <= set(D.FEATURES) for row in app.state.drift.window)
    assert app.state.drift.window[0]["boxes"] == 2
    assert user.post("/monitoring/drift/reset").status_code == 403
    assert admin.post("/monitoring/drift/reset").status_code == 204
    assert anonymous.get("/monitoring/drift").json()["window_size"] == 0


def test_without_reference_the_route_says_unavailable_and_scans_still_work(
    app: FastAPI, anonymous: TestClient
) -> None:
    write_models(app.state.settings.models_dir, BOXES, NAMES)
    assert app.state.drift.reference is None  # the fixture's reference file does not exist
    picture = scan_photo()
    assert anonymous.post("/predict", files={"image": ("a.png", picture, "image/png")}).is_success
    report = anonymous.get("/monitoring/drift").json()
    assert report["status"] == "unavailable" and report["window_size"] == 0
