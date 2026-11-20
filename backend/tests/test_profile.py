"""The user's own account: phone, profile photo and signature (app/images.py)."""

from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient


def png(image: np.ndarray) -> bytes:
    return cv2.imencode(".png", image)[1].tobytes()


def photo_bytes() -> bytes:
    image = np.zeros((300, 500, 3), np.uint8)
    image[:, :250] = (0, 0, 255)
    return cv2.imencode(".jpg", image)[1].tobytes()


def signature_bytes() -> bytes:
    """A dark stroke on white paper, like a scanned signature."""
    image = np.full((400, 1200, 3), 255, np.uint8)
    cv2.line(image, (100, 200), (1100, 220), (20, 20, 20), 12)
    return png(image)


def test_phone(user: TestClient) -> None:
    me = user.patch("/api/auth/me", json={"full_name": "Demo", "phone": "+216 20 123 456"}).json()
    assert me["phone"] == "+216 20 123 456"
    me = user.patch("/api/auth/me", json={"full_name": "Demo"}).json()
    assert me["phone"] == "+216 20 123 456"  # null keeps it
    assert (
        user.patch("/api/auth/me", json={"full_name": "Demo", "phone": ""}).json()["phone"] is None
    )
    bad = user.patch("/api/auth/me", json={"full_name": "Demo", "phone": "<script>"})
    assert bad.status_code == 422


def test_photo_is_cropped_and_reencoded(user: TestClient, admin: TestClient) -> None:
    me = user.put("/api/auth/me/photo", files={"image": ("p.jpg", photo_bytes(), "image/jpeg")})
    assert me.status_code == 200, me.text
    url = me.json()["photo_url"]
    assert url.startswith(f"/api/users/{me.json()['id']}/photo?v=")

    stored = user.get(url)
    assert stored.headers["content-type"] == "image/jpeg"
    image = cv2.imdecode(np.frombuffer(stored.content, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (256, 256, 3)
    assert admin.get(url).status_code == 200  # administrators see photos

    assert user.delete("/api/auth/me/photo").json()["photo_url"] is None
    assert user.get(url).status_code == 404


def test_signature_is_private(user: TestClient, admin: TestClient) -> None:
    me = user.put(
        "/api/auth/me/signature", files={"image": ("s.png", signature_bytes(), "image/png")}
    ).json()
    url = me["signature_url"]
    stored = user.get(url)
    image = cv2.imdecode(np.frombuffer(stored.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert image.shape[2] == 4  # transparent background
    assert image.shape[1] <= 600 and image.shape[0] <= 200
    assert image[:, :, 3].max() > 200 and image[0, 0, 3] == 0  # ink, and empty paper

    assert admin.get(url).status_code == 404  # not even administrators
    listed = admin.get("/api/admin/users", params={"q": "user@"}).json()["items"][0]
    assert listed["signature_url"] is None


def test_rejected_images(user: TestClient) -> None:
    not_image = user.put("/api/auth/me/photo", files={"image": ("x.jpg", b"hello", "image/jpeg")})
    assert not_image.status_code == 422
    blank = png(np.full((100, 300, 3), 255, np.uint8))
    empty = user.put("/api/auth/me/signature", files={"image": ("s.png", blank, "image/png")})
    assert empty.status_code == 422
    assert "empty" in empty.json()["detail"]


def test_image_routes_need_sign_in(anonymous: TestClient) -> None:
    assert anonymous.get("/api/users/1/photo").status_code == 401
    files = {"image": ("p.jpg", photo_bytes(), "image/jpeg")}
    assert anonymous.put("/api/auth/me/photo", files=files).status_code == 401
