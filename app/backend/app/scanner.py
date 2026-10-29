"""Pill scan with ONNX Runtime: detect every pill, crop it, identify it.

Same two stages and the same pre- and post-processing as the training code in `vision/`
(Ultralytics YOLO11 letterbox and NMS for the detector, the 20 %-margin square crop of
`tujya_vision.autolabel.square_crop` for the classifier), re-implemented with numpy and
OpenCV so the API needs neither PyTorch nor Ultralytics.

The model directory (`Settings.models_dir`) holds what `python -m tujya_vision.publish`
writes:

    detector.onnx     YOLO11 detection, one class "pill", input 1x3xSxS
    classifier.onnx   YOLO11 classification, outputs probabilities, input 1x3xCxC
    models.json       optional: where the two files come from (MLflow model and version)

Class names come from the ONNX metadata that Ultralytics writes (`names`: index -> folder
name such as "007"); the folder name is the class_id of config/pill_classes.csv.
"""

from __future__ import annotations

import ast
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Decoding is bounded before it starts: a small JPEG can claim a huge size.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(50_000_000))
import cv2
import onnxruntime as ort

DETECTOR_FILE = "detector.onnx"
CLASSIFIER_FILE = "classifier.onnx"
MANIFEST_FILE = "models.json"
LETTERBOX_FILL = 114  # Ultralytics' padding grey
CROP_MARGIN = 0.2  # same as the training crops (docs/VISION.md)


class ModelsMissing(RuntimeError):
    """The model directory does not contain the two ONNX files."""


class InvalidImage(ValueError):
    """The upload is not an image OpenCV can decode, or it is too large."""


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels of the original image
    confidence: float
    classes: list[int]  # top-k class ids, most probable first
    probabilities: list[float]


@dataclass(frozen=True)
class ScanOutput:
    width: int
    height: int
    detections: list[Detection]
    detection_ms: float
    classification_ms: float
    models: dict[str, Any] = field(default_factory=dict)


def decode_image(data: bytes, *, max_pixels: int = 50_000_000) -> np.ndarray:
    """BGR image from encoded bytes (JPEG, PNG, WebP...), EXIF orientation applied."""
    if not data:
        raise InvalidImage("The file is empty.")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImage("The file is not an image that can be read (JPEG, PNG or WebP).")
    if image.shape[0] * image.shape[1] > max_pixels:
        raise InvalidImage(f"The image has more than {max_pixels:,} pixels.")
    return image


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping the aspect ratio and pad to size x size (Ultralytics LetterBox)."""
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_w, new_h = round(width * ratio), round(height * ratio)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    if (new_w, new_h) != (width, height):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(LETTERBOX_FILL,) * 3
    )
    return padded, ratio, (left, top)


def to_tensor(image: np.ndarray) -> np.ndarray:
    """BGR uint8 HxWx3 -> RGB float32 1x3xHxW in [0, 1], as Ultralytics feeds its models."""
    rgb = image[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(rgb, dtype=np.float32)[None] / 255.0


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Greedy non-maximum suppression; boxes are x1, y1, x2, y2."""
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size:
        best, rest = order[0], order[1:]
        keep.append(int(best))
        xx1 = np.maximum(boxes[best, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[best, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[best, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[best, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        overlap = inter / np.maximum(areas[best] + areas[rest] - inter, 1e-9)
        order = rest[overlap <= iou]
    return keep


def square_crop(image: np.ndarray, box: tuple[float, ...], size: int) -> np.ndarray:
    """Square crop centred on the box with a 20 % margin each side, size x size (as trained)."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1 + 2 * CROP_MARGIN)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = round(cx - side / 2), round(cy - side / 2)
    right, bottom = round(cx + side / 2), round(cy + side / 2)
    pad = max(0, -left, -top, right - width, bottom - height)
    if pad:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        left, top, right, bottom = left + pad, top + pad, right + pad, bottom + pad
    crop = image[top:bottom, left:right]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def _metadata(session: ort.InferenceSession) -> dict[str, Any]:
    raw = session.get_modelmeta().custom_metadata_map
    meta: dict[str, Any] = dict(raw)
    for key in ("names", "imgsz"):
        if key in raw:
            meta[key] = ast.literal_eval(raw[key])
    return meta


def session_options(threads: int = 0) -> ort.SessionOptions:
    """No busy-waiting between runs: spinning ONNX Runtime threads starve OpenCV's resize
    threads (and other requests) on a shared CPU. `threads` 0 = ONNX Runtime's default."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options


def available_providers(choice: str) -> list[str]:
    """`cpu` (default, always works) or `auto` (CUDA first when this onnxruntime has it)."""
    if choice == "auto":
        wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return [p for p in wanted if p in ort.get_available_providers()]
    return ["CPUExecutionProvider"]


class PillScanner:
    def __init__(
        self,
        detector: ort.InferenceSession,
        classifier: ort.InferenceSession,
        *,
        manifest: dict[str, Any] | None = None,
        conf: float = 0.25,
        iou: float = 0.7,
        max_pills: int = 20,
        top_k: int = 5,
    ) -> None:
        self.detector, self.classifier = detector, classifier
        det_meta, cls_meta = _metadata(detector), _metadata(classifier)
        self.det_size = _square_input(detector, det_meta)
        self.cls_size = _square_input(classifier, cls_meta)
        names = cls_meta.get("names") or {}
        # Output index -> class_id. Ultralytics sorts the class folders by name, so index i
        # is folder "007" etc.; without names the index is the class id.
        n_out = int(classifier.get_outputs()[0].shape[-1])
        self.class_ids = [int(names.get(i, i)) for i in range(n_out)]
        self.manifest = manifest or {}
        self.conf, self.iou, self.max_pills, self.top_k = conf, iou, max_pills, top_k

    @classmethod
    def from_dir(
        cls, directory: Path, *, providers: str = "cpu", threads: int = 0, **options: Any
    ) -> PillScanner:
        detector, classifier = directory / DETECTOR_FILE, directory / CLASSIFIER_FILE
        missing = [p.name for p in (detector, classifier) if not p.is_file()]
        if missing:
            raise ModelsMissing(
                f"Model files missing in {directory}: {', '.join(missing)}. "
                "Export them with `python -m tujya_vision.publish` (see docs/VISION.md)."
            )
        manifest_path = directory / MANIFEST_FILE
        manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.is_file() else {}
        chosen = available_providers(providers)
        return cls(
            ort.InferenceSession(str(detector), session_options(threads), providers=chosen),
            ort.InferenceSession(str(classifier), session_options(threads), providers=chosen),
            manifest=manifest,
            **options,
        )

    def detect(self, image: np.ndarray) -> list[tuple[tuple[float, float, float, float], float]]:
        """Boxes in original pixels with their confidence, most confident first."""
        height, width = image.shape[:2]
        padded, ratio, (left, top) = letterbox(image, self.det_size)
        name = self.detector.get_inputs()[0].name
        output = self.detector.run(None, {name: to_tensor(padded)})[0][0]  # (4 + nc, N)
        predictions = output.T
        scores = predictions[:, 4:].max(axis=1)
        mask = scores >= self.conf
        if not mask.any():
            return []
        cx, cy, w, h = predictions[mask, :4].T
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        kept = nms(boxes, scores[mask], self.iou)[: self.max_pills]
        boxes = (boxes[kept] - [left, top, left, top]) / ratio
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        confidences = scores[mask][kept]
        return [
            (tuple(float(v) for v in box), float(c))  # type: ignore[misc]
            for box, c in zip(boxes, confidences, strict=True)
        ]

    def classify(self, crop: np.ndarray) -> tuple[list[int], list[float]]:
        name = self.classifier.get_inputs()[0].name
        probs = self.classifier.run(None, {name: to_tensor(crop)})[0][0]
        order = probs.argsort()[::-1][: self.top_k]
        return [self.class_ids[i] for i in order], [float(probs[i]) for i in order]

    def scan(self, image: np.ndarray) -> ScanOutput:
        height, width = image.shape[:2]
        start = time.perf_counter()
        found = self.detect(image)
        detected = time.perf_counter()
        detections = []
        for box, confidence in found:
            classes, probabilities = self.classify(square_crop(image, box, self.cls_size))
            detections.append(Detection(box, confidence, classes, probabilities))
        done = time.perf_counter()
        return ScanOutput(
            width=width,
            height=height,
            detections=detections,
            detection_ms=(detected - start) * 1000,
            classification_ms=(done - detected) * 1000,
            models=self.manifest,
        )


def _square_input(session: ort.InferenceSession, meta: dict[str, Any]) -> int:
    shape = session.get_inputs()[0].shape
    if isinstance(shape[-1], int):
        return int(shape[-1])
    imgsz = meta.get("imgsz", [640])
    return int(imgsz[0] if isinstance(imgsz, list | tuple) else imgsz)


class ScannerHolder:
    """Loads the models on first use, so the API starts (and answers 503) without them."""

    def __init__(self, directory: Path, *, providers: str = "cpu", **options: Any) -> None:
        self.directory, self.providers, self.options = directory, providers, options
        self._scanner: PillScanner | None = None
        self._lock = threading.Lock()

    def get(self) -> PillScanner:
        with self._lock:
            if self._scanner is None:
                self._scanner = PillScanner.from_dir(
                    self.directory, providers=self.providers, **self.options
                )
            return self._scanner
