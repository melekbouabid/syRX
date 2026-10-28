"""Automatic pill bounding boxes for C3PI consumer photos (classical computer vision).

C3PI gives no boxes. Consumer photos show one or two pills (often the front and the back of
the same product) on a fairly plain background, usually with vignetting and a shadow. The
pills are found as the regions whose colour differs from the background:

1. downscale to `work_size` pixels on the long side;
2. background = a smooth quadratic surface per Lab channel, fitted robustly (pixels that
   do not fit are dropped and the fit repeated), which absorbs vignetting and gradients;
3. distance of every pixel to that surface, darker-and-grey differences (shadows) weighted
   down, thresholded at several times the background noise;
4. connected regions that touch the border (vignetting, table edges) are dropped, as are
   regions much smaller than the largest one;
5. coins (C3PI's size reference: grey with relief texture) are ignored; a region that is
   not compact (a shadow crescent next to a pale pill) is refined with GrabCut, and flagged
   when that fails;
6. each kept region becomes a box, mapped back to full resolution.

Each result carries flags so doubtful images can be left out of detection training and
checked by eye (`python -m tujya_vision.prepare --sheet`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels of the original image
INFORMATIONAL = {"coin_ignored", "refined"}  # flags that do not make an image doubtful


@dataclass
class Detection:
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """At least one box and no flag that makes the boxes doubtful."""
        doubtful = [f for f in self.flags if f not in INFORMATIONAL]
        return bool(self.boxes) and not doubtful


def read_image(path: Path) -> np.ndarray:
    """BGR image; works with any Windows path (cv2.imread does not)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode {path}")
    return image


def _design(h: int, w: int, step: int = 1) -> np.ndarray:
    """Quadratic surface basis 1, x, y, x^2, xy, y^2 on a grid, x and y in [-1, 1]."""
    ys, xs = np.mgrid[0:h:step, 0:w:step].astype(np.float32)
    x, y = xs / max(1, w - 1) * 2 - 1, ys / max(1, h - 1) * 2 - 1
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1).reshape(-1, 6)


def _distance(residual: np.ndarray) -> np.ndarray:
    """Colour distance to the background; darker-and-grey (shadows) counts less."""
    lightness, a, b = residual[..., 0], residual[..., 1], residual[..., 2]
    lightness = np.where(lightness < 0, 0.4 * lightness, lightness)
    return np.sqrt(lightness**2 + a**2 + b**2)


def _foreground(
    small: np.ndarray, *, k: float = 6.0, floor: float = 10.0, cap: float = 25.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Pixels that differ from a smooth background surface (handles vignetting).

    Returns the filled mask, the distance map and the threshold used.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    step = max(1, min(h, w) // 120)
    basis = _design(h, w, step)
    samples = lab[::step, ::step].reshape(-1, 3)
    keep = np.ones(len(samples), dtype=bool)
    for _ in range(5):  # robust fit: refit without the pixels that do not look like background
        coef, *_ = np.linalg.lstsq(basis[keep], samples[keep], rcond=None)
        dist = _distance((samples - basis @ coef)[:, None, :])[:, 0]
        noise = float(np.median(dist[keep])) * 1.5 + 1e-3
        # Bounded: a background the surface cannot follow (a sharp dark band) must not
        # raise the threshold above what any pill exceeds.
        threshold = min(max(k * noise, floor), cap)
        keep = dist < threshold
    background = (_design(h, w) @ coef).reshape(h, w, 3)
    dist = _distance(lab - background)
    mask = (dist > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    # Fill holes (imprints, reflections) so each pill is one region.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled, dist, threshold


def _is_coin(small: np.ndarray, labels: np.ndarray, index: int) -> bool:
    """A coin (C3PI's size reference): grey, with fine relief texture inside, unlike a pill.

    Measured on C3PI photos: coins have chroma < 9 and a mean |Laplacian| of 9 to 22 inside
    the region; pills (white ones included) stay between 2 and 8.
    """
    inner = cv2.erode((labels == index).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if inner.sum() < 30:
        return False
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    pixels = lab[inner].astype(np.float32)
    chroma = float(np.hypot(pixels[:, 1] - 128, pixels[:, 2] - 128).mean())
    lightness = cv2.GaussianBlur(lab[..., 0], (3, 3), 0)
    texture = float(np.abs(cv2.Laplacian(lightness, cv2.CV_32F))[inner].mean())
    return chroma < 9 and texture > 8


def _refine(small: np.ndarray, x: int, y: int, bw: int, bh: int) -> tuple[int, ...] | None:
    """GrabCut around an irregular region (often a shadow next to a pale pill).

    Returns (area, x, y, w, h, solidity) of the refined region, or None when GrabCut does not
    find a compact object that stays inside the search square.
    """
    h, w = small.shape[:2]
    side = 2.4 * max(bw, bh)
    cx, cy = x + bw / 2, y + bh / 2
    rx, ry = max(1, int(cx - side / 2)), max(1, int(cy - side / 2))
    rw, rh = min(w - 2, int(cx + side / 2)) - rx, min(h - 2, int(cy + side / 2)) - ry
    if rw < 8 or rh < 8:
        return None
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, (rx, ry, rw, rh), bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if count < 2:
        return None
    best = 1 + int(np.argmax(stats[1:, 4]))
    nx, ny, nw, nh, area = (int(v) for v in stats[best])
    if nx <= rx or ny <= ry or nx + nw >= rx + rw or ny + nh >= ry + rh:
        return None
    return (area, nx, ny, nw, nh, _solidity(labels == best, area))


def _solidity(region: np.ndarray, area: int) -> float:
    """Area over convex hull area: about 1 for a pill, low for a crescent-shaped shadow."""
    contours, _ = cv2.findContours(
        region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    hull = cv2.contourArea(cv2.convexHull(max(contours, key=cv2.contourArea)))
    return area / max(hull, 1.0)


def detect_pills(
    image: np.ndarray,
    *,
    work_size: int = 640,
    min_area_frac: float = 0.0004,
    max_area_frac: float = 0.6,
    keep_ratio: float = 0.25,
    min_solidity: float = 0.8,
    max_pills: int = 4,
    min_contrast: float = 1.2,
    refine: bool = True,
) -> Detection:
    """Boxes of the pills in one consumer photo, largest first."""
    height, width = image.shape[:2]
    scale = work_size / max(height, width)
    small = cv2.resize(image, (round(width * scale), round(height * scale)), cv2.INTER_AREA)
    h, w = small.shape[:2]
    mask, distance, threshold = _foreground(small)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    result = Detection(width, height)
    for i in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        touches = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        if touches or area < min_area_frac * h * w:
            continue
        if float(distance[labels == i].mean()) < min_contrast * threshold:
            continue  # a faint blob of uneven light, not an object
        solidity = _solidity(labels == i, area)
        if _is_coin(small, labels, i):
            result.flags.append("coin_ignored")  # informational, not a failure
            continue
        regions.append((area, x, y, bw, bh, solidity))
    if not regions:
        result.flags = ["no_region"]
        return result
    regions.sort(reverse=True)
    largest = regions[0][0]
    candidates = [r for r in regions if r[0] >= keep_ratio * largest]
    if len(candidates) > max_pills:
        result.flags.append("too_many_regions")
    for area, x, y, bw, bh, solidity in candidates[:max_pills]:
        if solidity < min_solidity and refine:
            refined = _refine(small, x, y, bw, bh)
            if refined is not None and refined[5] >= min_solidity and refined[0] > area:
                area, x, y, bw, bh, solidity = refined
                result.flags.append("refined")  # informational: GrabCut fixed the region
        if area > max_area_frac * h * w:
            result.flags.append("region_too_large")
        if solidity < min_solidity:
            result.flags.append("irregular_region")  # a shadow crescent or half a pill
        if max(bw, bh) / max(1, min(bw, bh)) > 4.5:
            result.flags.append("elongated_region")
        result.boxes.append(
            (
                max(0, int(x / scale)),
                max(0, int(y / scale)),
                min(width, int((x + bw) / scale)),
                min(height, int((y + bh) / scale)),
            )
        )
    return result


def detect_reference(image: np.ndarray) -> Detection:
    """Pills of an RxNav reference image: drop the ruler (left, bottom) and the banner first."""
    height, width = image.shape[:2]
    top, left = 0, round(0.07 * width)
    inner = image[top : round(0.78 * height), left:]
    found = detect_pills(inner, min_solidity=0.0)
    found.width, found.height = width, height
    found.boxes = [(x1 + left, y1 + top, x2 + left, y2 + top) for x1, y1, x2, y2 in found.boxes]
    return found


def square_crop(image: np.ndarray, box: Box, *, margin: float = 0.2, size: int = 224) -> np.ndarray:
    """Square crop centred on the box, `margin` added on each side, resized to size x size."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = round(cx - side / 2), round(cy - side / 2)
    right, bottom = round(cx + side / 2), round(cy + side / 2)
    # Pad with the edge colour when the square leaves the image.
    pad = max(0, -left, -top, right - width, bottom - height)
    if pad:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        left, top, right, bottom = left + pad, top + pad, right + pad, bottom + pad
    crop = image[top:bottom, left:right]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
