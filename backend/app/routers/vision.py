"""Endpoints of the course guide: GET /health, GET /model/metadata, POST /predict(/visualize).

The model alone, as the guide asks: no account and no medicine database (decision D97).
The app's own route, POST /api/scan (routers/scan.py), adds the medicine names, the sources
and the confirmation rule on top of the same ONNX Runtime scan (app/inference.py).

Errors: 400 when the upload is not a readable image, 413 when it is too large, 503 when the
models are not loaded (no `models/detector.onnx` / `classifier.onnx`: `python -m
src.models.publish`).
"""

from __future__ import annotations

import time
from typing import Annotated, Any

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.deps import SettingsDep
from app.inference import (
    CLASSIFIER_FILE,
    DETECTOR_FILE,
    InvalidImage,
    ModelsMissing,
    PillScanner,
    ScannerHolder,
    ScanOutput,
    _metadata,
    decode_image,
)
from app.medicines import DISCLAIMER
from app.schemas import (
    ClassScore,
    GpuStatus,
    HealthOut,
    LoadedModel,
    MetadataOut,
    ModelMetadata,
    NormalizedBox,
    PixelBox,
    PredictedPill,
    PredictOut,
    PredictTimings,
)

router = APIRouter(tags=["model (course guide)"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
FILES = {"detect": DETECTOR_FILE, "classify": CLASSIFIER_FILE}
ERRORS: dict[int | str, dict[str, Any]] = {
    400: {"description": "Not a readable JPEG, PNG or WebP image"},
    413: {"description": "Image larger than the upload limit"},
    503: {"description": "Models not loaded (publish them with python -m src.models.publish)"},
}


def _holder(request: Request) -> ScannerHolder:
    holder: ScannerHolder = request.app.state.scanner
    return holder


def loaded_scanner(request: Request) -> PillScanner:
    try:
        return _holder(request).get()
    except ModelsMissing as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


def loaded_models(manifest: dict[str, Any]) -> list[LoadedModel]:
    return [
        LoadedModel(
            task=task,  # type: ignore[arg-type]
            file=str((manifest.get(task) or {}).get("file", filename)),
            **{
                key: value
                for key, value in (manifest.get(task) or {}).items()
                if key in LoadedModel.model_fields and key not in ("task", "file")
            },
        )
        for task, filename in FILES.items()
    ]


# --- GET /health ----------------------------------------------------------------------------


@router.get("/health", response_model=HealthOut)
async def health(request: Request) -> HealthOut:
    """API state, GPU availability and the model versions loaded (MLflow version and aliases).

    Always 200: `status` is `degraded` and `model_loaded` false when the models cannot be
    loaded, so a monitoring probe sees the reason instead of an error.
    """
    available = list(ort.get_available_providers())
    try:
        scanner = await run_in_threadpool(_holder(request).get)
    except Exception as exc:  # missing or unreadable model files: report, do not fail
        return HealthOut(
            status="degraded",
            model_loaded=False,
            detail=str(exc),
            api_version=request.app.version,
            gpu=GpuStatus(
                available="CUDAExecutionProvider" in available,
                available_providers=available,
                active_providers=[],
            ),
            models=[],
        )
    return HealthOut(
        status="ok",
        model_loaded=True,
        api_version=request.app.version,
        gpu=GpuStatus(
            available="CUDAExecutionProvider" in available,
            available_providers=available,
            active_providers=list(scanner.detector.get_providers()),
        ),
        models=loaded_models(scanner.manifest),
        published_at=scanner.manifest.get("published_at"),
    )


# --- GET /model/metadata --------------------------------------------------------------------


def _architecture(entry: dict[str, Any], meta: dict[str, Any]) -> str:
    """Registry tag `arch` (models.json), else the Ultralytics description of the ONNX file."""
    if entry.get("arch"):
        return str(entry["arch"])
    description = str(meta.get("description", ""))
    words = [w for w in description.split() if w.lower().startswith("yolo")]
    return words[0].lower() if words else "unknown"


def model_metadata(
    task: str, session: ort.InferenceSession, size: int, classes: list[str], manifest: dict
) -> ModelMetadata:
    entry: dict[str, Any] = manifest.get(task) or {}
    meta = _metadata(session)
    return ModelMetadata(
        task=task,  # type: ignore[arg-type]
        name=str(entry.get("registered_model") or FILES[task]),
        architecture=_architecture(entry, meta),
        version=entry.get("version"),
        aliases=list(entry.get("aliases", [])),
        precision=str(entry.get("precision", "fp32")),
        variant_of=entry.get("variant_of"),
        run_id=entry.get("run_id"),
        input_size=size,
        metrics={k: float(v) for k, v in (entry.get("metrics") or {}).items()},
        classes=classes,
        num_classes=len(classes),
    )


@router.get(
    "/model/metadata",
    response_model=MetadataOut,
    responses={503: ERRORS[503]},
)
async def metadata(request: Request) -> MetadataOut:
    """Name, architecture, Registry version and aliases, validation and test metrics, classes.

    The metrics are those of the published version in the MLflow Model Registry (models.json,
    written by `python -m src.models.publish`); an FP16 / INT8 variant shows its own test
    scores and latency next to the validation scores of the champion it comes from.
    """
    scanner = await run_in_threadpool(loaded_scanner, request)
    names = _metadata(scanner.detector).get("names") or {0: "pill"}
    detector_classes = [str(names[i]) for i in sorted(names)]
    return MetadataOut(
        detector=model_metadata(
            "detect", scanner.detector, scanner.det_size, detector_classes, scanner.manifest
        ),
        classifier=model_metadata(
            "classify",
            scanner.classifier,
            scanner.cls_size,
            [f"{c:03d}" for c in scanner.class_ids],
            scanner.manifest,
        ),
        published_at=scanner.manifest.get("published_at"),
        disclaimer=DISCLAIMER,
    )


# --- POST /predict and /predict/visualize ---------------------------------------------------


async def read_image(image: UploadFile, settings: Settings) -> bytes:
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Send a JPEG, PNG or WebP image.")
    data = await image.read(settings.scan_max_bytes + 1)
    if len(data) > settings.scan_max_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"The image is larger than {settings.scan_max_bytes // (1024 * 1024)} MB.",
        )
    return data


def _scan(scanner: PillScanner, data: bytes, settings: Settings) -> tuple[np.ndarray, ScanOutput]:
    """Blocking part (decoding, ONNX Runtime), run in the thread pool."""
    try:
        picture = decode_image(data, max_pixels=settings.scan_max_pixels)
    except InvalidImage as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return picture, scanner.scan(picture)


async def predict_upload(
    image: UploadFile, request: Request, settings: Settings
) -> tuple[np.ndarray, ScanOutput, PillScanner]:
    data = await read_image(image, settings)
    scanner = await run_in_threadpool(loaded_scanner, request)
    picture, output = await run_in_threadpool(_scan, scanner, data, settings)
    return picture, output, scanner


def to_predict_out(output: ScanOutput, started: float, manifest: dict[str, Any]) -> PredictOut:
    pills = []
    for detection in output.detections:
        x1, y1, x2, y2 = detection.box
        pills.append(
            PredictedPill(
                box=PixelBox(x1=x1, y1=y1, x2=x2, y2=y2),
                box_normalized=NormalizedBox(
                    x1=x1 / output.width,
                    y1=y1 / output.height,
                    x2=x2 / output.width,
                    y2=y2 / output.height,
                ),
                confidence=min(1.0, max(0.0, detection.confidence)),
                top5=[
                    ClassScore(class_id=c, probability=min(1.0, max(0.0, p)))
                    for c, p in zip(detection.classes, detection.probabilities, strict=True)
                ],
            )
        )
    inference = output.detection_ms + output.classification_ms
    return PredictOut(
        image_width=output.width,
        image_height=output.height,
        count=len(pills),
        pills=pills,
        timings=PredictTimings(
            detection_ms=round(output.detection_ms, 1),
            classification_ms=round(output.classification_ms, 1),
            inference_ms=round(inference, 1),
            server_ms=round((time.perf_counter() - started) * 1000, 1),
        ),
        model_versions={task: (manifest.get(task) or {}).get("version") for task in FILES},
        disclaimer=DISCLAIMER,
    )


UploadImage = Annotated[UploadFile, File(description="JPEG, PNG or WebP photo of the pills")]


@router.post("/predict", response_model=PredictOut, responses=ERRORS)
async def predict(image: UploadImage, request: Request, settings: SettingsDep) -> PredictOut:
    """Detect every pill, then give the top-5 classes of each one with their probabilities.

    Boxes in pixels of the uploaded image and normalised (0-1); server latency in
    `timings` (`inference_ms`: ONNX Runtime only, `server_ms`: the whole request). An
    identification is a suggestion, never a certainty: read the probabilities.
    """
    started = time.perf_counter()
    _, output, scanner = await predict_upload(image, request, settings)
    return to_predict_out(output, started, scanner.manifest)


def class_colour(class_id: int) -> tuple[int, int, int]:
    """A stable, saturated BGR colour per class (the same class keeps its colour)."""
    hue = (class_id * 47) % 180
    pixel = np.array([[[hue, 220, 230]]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def annotate(picture: np.ndarray, output: ScanOutput) -> np.ndarray:
    """Boxes coloured by top-1 class, label "class probability (detection confidence)"."""
    canvas = picture.copy()
    thickness = max(2, round(max(canvas.shape[:2]) / 300))
    scale = max(0.5, max(canvas.shape[:2]) / 1200)
    for detection in output.detections:
        x1, y1, x2, y2 = (round(v) for v in detection.box)
        top = detection.classes[0] if detection.classes else -1
        colour = class_colour(top)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)
        probability = detection.probabilities[0] if detection.probabilities else 0.0
        label = f"{top:03d} {probability:.0%} (det {detection.confidence:.2f})"
        (width, height), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        y_text = y1 - base - 2 if y1 - height - base - 4 >= 0 else y2 + height + 2
        cv2.rectangle(
            canvas, (x1, y_text - height - 2), (x1 + width + 4, y_text + base), colour, -1
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 2, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


@router.post(
    "/predict/visualize",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "Annotated image"}, **ERRORS},
)
async def predict_visualize(
    image: UploadImage,
    request: Request,
    settings: SettingsDep,
    max_side: Annotated[
        int, Query(ge=128, le=4096, description="Longest side of the returned image")
    ] = 1280,
) -> Response:
    """The uploaded photo with each pill's box, top-1 class and probabilities drawn on it (PNG).

    Boxes are coloured by class. Headers: `X-Pill-Count`, `X-Inference-Ms` (ONNX Runtime) and
    `X-Server-Ms` (whole request); the JSON of the same prediction is POST /predict.
    """
    started = time.perf_counter()
    picture, output, _ = await predict_upload(image, request, settings)

    def render() -> bytes:
        canvas = annotate(picture, output)
        longest = max(canvas.shape[:2])
        if longest > max_side:
            ratio = max_side / longest
            size = (round(canvas.shape[1] * ratio), round(canvas.shape[0] * ratio))
            canvas = cv2.resize(canvas, size, interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".png", canvas)
        if not ok:  # pragma: no cover - OpenCV always encodes a BGR uint8 image
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "PNG encoding failed.")
        return encoded.tobytes()

    body = await run_in_threadpool(render)
    inference = output.detection_ms + output.classification_ms
    return Response(
        body,
        media_type="image/png",
        headers={
            "X-Pill-Count": str(len(output.detections)),
            "X-Inference-Ms": f"{inference:.1f}",
            "X-Server-Ms": f"{(time.perf_counter() - started) * 1000:.1f}",
        },
    )
