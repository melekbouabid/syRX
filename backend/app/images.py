"""Profile photo and signature: decoded and re-encoded before storage, which drops all metadata."""

from __future__ import annotations

import cv2
import numpy as np

PHOTO_SIZE = 256  # square, in pixels
SIGNATURE_MAX = (600, 200)  # width, height
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 25_000_000


class ImageRejected(ValueError):
    """Not an image, too large, or (for a signature) nothing drawn."""


def _decode(data: bytes, flags: int) -> np.ndarray:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageRejected(f"The image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
    if image is None or image.size == 0:
        raise ImageRejected("Send a JPEG, PNG or WebP image.")
    if image.shape[0] * image.shape[1] > MAX_PIXELS:
        raise ImageRejected("The image has too many pixels.")
    return image


def profile_photo(data: bytes) -> bytes:
    """A centred square crop, PHOTO_SIZE pixels wide, as JPEG."""
    image = _decode(data, cv2.IMREAD_COLOR)
    height, width = image.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    square = image[top : top + side, left : left + side]
    small = cv2.resize(square, (PHOTO_SIZE, PHOTO_SIZE), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:  # pragma: no cover - OpenCV always encodes a valid BGR array
        raise ImageRejected("The photo could not be encoded.")
    return encoded.tobytes()


def signature(data: bytes) -> bytes:
    """Dark ink on a transparent background, cropped to the strokes, as PNG.

    Works for a drawing from the signature pad (transparent PNG) and for a photo or scan of a
    signature on white paper.
    """
    image = _decode(data, cv2.IMREAD_UNCHANGED)
    if image.ndim == 2:
        gray, alpha = image, None
    elif image.shape[2] == 4:
        gray, alpha = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY), image[:, :, 3]
    else:
        gray, alpha = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY), None
    if gray.dtype != np.uint8:
        gray = cv2.convertScaleAbs(gray, alpha=255.0 / max(1.0, float(gray.max())))
    # Ink: dark pixels, and only the visible ones when the image is transparent.
    ink = (255 - gray).astype(np.float32)
    ink[ink < 60] = 0  # paper grain and light shadows are not ink
    if alpha is not None:
        ink *= alpha.astype(np.float32) / 255.0
    rows = np.flatnonzero(ink.max(axis=1) > 0)
    cols = np.flatnonzero(ink.max(axis=0) > 0)
    if rows.size == 0 or cols.size == 0:
        raise ImageRejected("The signature is empty: draw it first.")
    pad = 8
    top, bottom = max(0, rows[0] - pad), min(ink.shape[0], rows[-1] + pad + 1)
    left, right = max(0, cols[0] - pad), min(ink.shape[1], cols[-1] + pad + 1)
    ink = ink[top:bottom, left:right]
    scale = min(1.0, SIGNATURE_MAX[0] / ink.shape[1], SIGNATURE_MAX[1] / ink.shape[0])
    if scale < 1.0:
        size = (max(1, round(ink.shape[1] * scale)), max(1, round(ink.shape[0] * scale)))
        ink = cv2.resize(ink, size, interpolation=cv2.INTER_AREA)
    out = np.zeros((ink.shape[0], ink.shape[1], 4), np.uint8)
    out[:, :, :3] = (17, 24, 39)  # ink colour (BGR), near black
    out[:, :, 3] = np.clip(ink * 1.6, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", out)
    if not ok:  # pragma: no cover
        raise ImageRejected("The signature could not be encoded.")
    return encoded.tobytes()
