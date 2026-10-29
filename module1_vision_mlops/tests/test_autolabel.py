"""Auto-labelling on synthetic photos: pills drawn on a vignetted, noisy background."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tujya_vision.autolabel import detect_pills, read_image, square_crop


def _photo(pills: list[tuple[int, int, int, int, tuple[int, int, int]]]) -> np.ndarray:
    """1200x900 grey card with vignetting and noise; each pill is an ellipse (cx, cy, a, b)."""
    h, w = 900, 1200
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    base = (150 - 45 * r**2).clip(0, 255)
    image = np.dstack([base, base, base + 5]).astype(np.uint8)
    rng = np.random.default_rng(0)
    image = cv2.add(image, rng.integers(0, 8, image.shape, dtype=np.uint8))
    for cx, cy, a, b, colour in pills:
        cv2.ellipse(image, (cx + 8, cy + 8), (a, b), 0, 0, 360, (90, 90, 95), -1)  # shadow
        cv2.ellipse(image, (cx, cy), (a, b), 0, 0, 360, colour, -1)
    return image


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area


def test_one_coloured_pill() -> None:
    found = detect_pills(_photo([(600, 450, 60, 35, (40, 60, 220))]))
    assert found.ok and len(found.boxes) == 1
    assert _iou(found.boxes[0], (540, 415, 660, 485)) > 0.7


def test_front_and_back_side_by_side() -> None:
    pills = [(450, 450, 50, 50, (60, 160, 240)), (750, 450, 50, 50, (60, 160, 240))]
    found = detect_pills(_photo(pills))
    assert found.ok and len(found.boxes) == 2
    truth = [(400, 400, 500, 500), (700, 400, 800, 500)]
    for box in found.boxes:
        assert max(_iou(box, t) for t in truth) > 0.6


def test_white_pill_on_grey() -> None:
    found = detect_pills(_photo([(600, 450, 70, 30, (235, 235, 235))]))
    assert found.ok and len(found.boxes) == 1
    assert _iou(found.boxes[0], (530, 420, 670, 480)) > 0.6


def test_empty_background_is_flagged() -> None:
    found = detect_pills(_photo([]))
    assert not found.ok and found.flags


def test_square_crop_pads_at_the_border() -> None:
    image = _photo([(30, 30, 25, 25, (40, 60, 220))])
    crop = square_crop(image, (5, 5, 55, 55), size=64)
    assert crop.shape == (64, 64, 3)


def test_read_image_handles_unusual_names(tmp_path: Path) -> None:
    path = tmp_path / "!!0MOPX!Z.JPG"
    ok, data = cv2.imencode(".jpg", _photo([]))
    assert ok
    path.write_bytes(data.tobytes())
    assert read_image(path).shape == (900, 1200, 3)
    with pytest.raises(ValueError, match="cannot decode"):
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not a jpeg")
        read_image(bad)
