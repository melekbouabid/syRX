"""Data drift of the scanned photos (step 2.8, docs/module1/derive.md).

The models were trained on the photos of C3PI v2 (consumer-grade photos of one pill on a
plain background). A scan that looks different (dark, blurred, a phone photo of another size,
something that is not a pill) is scored with less certainty, and the confidence the app shows
means less. This module compares the photos the API receives with reference photos of the same kind
(the C3PI v2 validation split, never trained on: D122):

- `image_stats` / `scan_stats`: eight numbers per photo, the same in the reference stage
  (src/data/drift_reference.py) and in the API: brightness, contrast, sharpness, size, aspect
  ratio, number of pills found, mean confidence of the detector and of the classifier;
- `DriftMonitor`: a sliding window of the last `window` photos, in memory only. No image and
  nothing about the user is kept, only these eight numbers; a restart empties the window;
- `drift_report`: per statistic, two-sample Kolmogorov-Smirnov test (scipy) and Population
  Stability Index (PSI, bins = deciles of the reference) against the reference sample;
  thresholds from params.yaml `drift`, written into the reference file by the DVC stage.

A statistic drifts when the KS p-value is below `ks_alpha` AND its PSI reaches `psi_drift`
(with a few hundred photos a KS test flags differences too small to matter; the PSI says
how large the shift is); it is in `warning` when the p-value is below `ks_alpha` and the PSI
reaches only `psi_warning` (a PSI alone moves by about 0.05 between two samples of a few
hundred photos of the same distribution).
"""

from __future__ import annotations

import json
import math
import threading
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np

FEATURES: tuple[str, ...] = (
    "brightness",
    "contrast",
    "sharpness",
    "megapixels",
    "aspect_ratio",
    "boxes",
    "detector_confidence",
    "classifier_confidence",
)
DESCRIPTIONS = {
    "brightness": "mean grey level (0-255)",
    "contrast": "standard deviation of the grey levels",
    "sharpness": "log10 of the variance of the Laplacian, image resized to 512 px",
    "megapixels": "width x height / 1e6",
    "aspect_ratio": "long side / short side",
    "boxes": "pills found by the detector",
    "detector_confidence": "mean confidence of the pills found (none: not counted)",
    "classifier_confidence": "mean top-1 probability of the pills found (none: not counted)",
}
SHARPNESS_SIDE = 512  # sharpness does not depend on the photo size
Status = Literal["insufficient", "ok", "warning", "drift"]

DEFAULT_THRESHOLDS: dict[str, float] = {
    "window": 200,
    "min_samples": 30,
    "ks_alpha": 0.01,
    "psi_warning": 0.1,
    "psi_drift": 0.25,
    "bins": 10,
}


def _grey(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def image_stats(
    image: np.ndarray,
    detector_confidences: Sequence[float] = (),
    classifier_confidences: Sequence[float] = (),
) -> dict[str, float]:
    """The eight statistics of one BGR photo and of what the models found in it."""
    grey = _grey(image)
    height, width = grey.shape[:2]
    scale = SHARPNESS_SIDE / max(height, width)
    small = cv2.resize(grey, (max(1, round(width * scale)), max(1, round(height * scale))))
    laplacian = float(cv2.Laplacian(small, cv2.CV_64F).var())
    stats = {
        "brightness": float(grey.mean()),
        "contrast": float(grey.std()),
        "sharpness": math.log10(1.0 + laplacian),
        "megapixels": width * height / 1e6,
        "aspect_ratio": max(width, height) / max(1, min(width, height)),
        "boxes": float(len(detector_confidences)),
    }
    if len(detector_confidences):
        stats["detector_confidence"] = float(np.mean(detector_confidences))
    if len(classifier_confidences):
        stats["classifier_confidence"] = float(np.mean(classifier_confidences))
    return stats


class _Found(Protocol):
    confidence: float
    probabilities: list[float]


class _Scan(Protocol):
    @property
    def detections(self) -> Sequence[_Found]: ...


def scan_stats(image: np.ndarray, output: _Scan) -> dict[str, float]:
    """`image_stats` of a photo and of the served scan of it (app/inference.py ScanOutput)."""
    return image_stats(
        image,
        [d.confidence for d in output.detections],
        [d.probabilities[0] for d in output.detections if d.probabilities],
    )


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index, bins = quantiles of the reference (epsilon for empty bins).

    0 = same distribution; by convention < 0.1 stable, 0.1-0.25 moderate, > 0.25 large shift.
    """
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # a constant reference: one bin for its value, one for the rest
        same = np.isclose(current, edges[0])
        expected = np.array([1.0, 0.0])
        actual = np.array([same.mean(), 1 - same.mean()])
    else:
        inner = edges[1:-1]
        expected = np.bincount(
            np.searchsorted(inner, reference, side="right"), minlength=len(edges) - 1
        )
        actual = np.bincount(
            np.searchsorted(inner, current, side="right"), minlength=len(edges) - 1
        )
        expected = expected / expected.sum()
        actual = actual / actual.sum()
    eps = 1e-4
    expected, actual = np.clip(expected, eps, None), np.clip(actual, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


@dataclass(frozen=True)
class Reference:
    """Statistics of the reference photos (written by the DVC stage `drift_reference`)."""

    values: dict[str, list[float]]  # feature -> sample of the reference photos
    thresholds: dict[str, float]
    meta: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> Reference:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            {k: [float(x) for x in v] for k, v in data["values"].items()},
            {**DEFAULT_THRESHOLDS, **data.get("thresholds", {})},
            dict(data.get("meta", {})),
        )

    def to_json(self) -> dict[str, Any]:
        return {"meta": self.meta, "thresholds": self.thresholds, "values": self.values}


def feature_report(
    name: str, reference: Sequence[float], current: Sequence[float], thresholds: dict[str, float]
) -> dict[str, Any]:
    """KS test and PSI of one statistic; `warning` and `drift` need a small p-value and a
    PSI of at least `psi_warning` / `psi_drift`."""
    from scipy.stats import ks_2samp

    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    out: dict[str, Any] = {
        "feature": name,
        "description": DESCRIPTIONS.get(name, name),
        "reference_n": int(ref.size),
        "current_n": int(cur.size),
        "reference_mean": float(ref.mean()) if ref.size else None,
        "current_mean": float(cur.mean()) if cur.size else None,
        "ks_statistic": None,
        "ks_pvalue": None,
        "psi": None,
        "status": "insufficient",
    }
    if ref.size == 0 or cur.size < int(thresholds["min_samples"]):
        return out
    test = ks_2samp(ref, cur)
    value = psi(ref, cur, int(thresholds["bins"]))
    status: Status = "ok"
    if float(test.pvalue) < thresholds["ks_alpha"]:  # a difference, then its size
        if value >= thresholds["psi_drift"]:
            status = "drift"
        elif value >= thresholds["psi_warning"]:
            status = "warning"
    out.update(
        ks_statistic=float(test.statistic), ks_pvalue=float(test.pvalue), psi=value, status=status
    )
    return out


RANK: dict[str, int] = {"insufficient": 0, "ok": 1, "warning": 2, "drift": 3}


def drift_report(reference: Reference, records: Iterable[dict[str, float]]) -> dict[str, Any]:
    """Report of a window of photo statistics against the reference."""
    rows = list(records)
    thresholds = reference.thresholds
    features = [
        feature_report(
            name,
            reference.values.get(name, []),
            [r[name] for r in rows if name in r and math.isfinite(r[name])],
            thresholds,
        )
        for name in FEATURES
    ]
    checked = [f for f in features if f["status"] != "insufficient"]
    if len(rows) < int(thresholds["min_samples"]) or not checked:
        status: str = "insufficient"
    else:
        status = max((f["status"] for f in checked), key=RANK.__getitem__)
    return {
        "status": status,
        "window_size": len(rows),
        "window_max": int(thresholds["window"]),
        "drifted": [f["feature"] for f in features if f["status"] == "drift"],
        "features": features,
        "thresholds": thresholds,
        "reference": reference.meta,
    }


class DriftMonitor:
    """Sliding window of the statistics of the last scanned photos (memory only)."""

    def __init__(self, reference: Reference | None, window: int | None = None) -> None:
        self.reference = reference
        size = window or int((reference.thresholds if reference else DEFAULT_THRESHOLDS)["window"])
        self.window: deque[dict[str, float]] = deque(maxlen=size)
        self.total = 0
        self._lock = threading.Lock()

    @classmethod
    def from_file(cls, path: Path) -> DriftMonitor:
        """A monitor without reference when the file is missing (the report says so)."""
        return cls(Reference.load(path) if Path(path).is_file() else None)

    def record(self, stats: dict[str, float]) -> None:
        with self._lock:
            self.window.append(dict(stats))
            self.total += 1

    def record_scan(self, image: np.ndarray, output: _Scan) -> None:
        """Add a scanned photo to the window; only when a reference is loaded."""
        if self.reference is not None:
            self.record(scan_stats(image, output))

    def reset(self) -> None:
        with self._lock:
            self.window.clear()

    def report(self) -> dict[str, Any]:
        with self._lock:
            rows = list(self.window)
            total = self.total
        if self.reference is None:
            return {
                "status": "unavailable",
                "window_size": len(rows),
                "window_max": self.window.maxlen,
                "drifted": [],
                "features": [],
                "thresholds": DEFAULT_THRESHOLDS,
                "reference": {},
                "photos_seen": total,
            }
        return {**drift_report(self.reference, rows), "photos_seen": total}
