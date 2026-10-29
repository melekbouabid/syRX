"""POST /api/scan: one photo in, every detected pill out with its top-5 identifications.

Sign-in required, like the catalog: the demo uses the demo account (decision D46).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.deps import PharmaSession, SettingsDep, get_current_user
from app.medicines import DISCLAIMER, PharmaUnavailable, pill_infos, sources
from app.scanner import InvalidImage, ModelsMissing, PillScanner, ScannerHolder, decode_image
from app.schemas import (
    NormalizedBox,
    PillCandidate,
    PixelBox,
    ScannedPill,
    ScanResult,
    ScanTimings,
    SourceOut,
)

router = APIRouter(prefix="/api", tags=["scan"], dependencies=[Depends(get_current_user)])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
NO_PILL = "Aucun comprimé détecté. Essayez une photo nette, le comprimé entier, sur un fond uni."


def pharma_unavailable() -> HTTPException:
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "The medicine tables are not loaded. Run `python -m app.pharma_loader` (docs/INFRA.md).",
    )


def get_scanner(request: Request) -> PillScanner:
    holder: ScannerHolder = request.app.state.scanner
    try:
        return holder.get()
    except ModelsMissing as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


def _run(scanner: PillScanner, data: bytes, settings: Settings, db: Session) -> ScanResult:
    """Blocking part (decode, ONNX Runtime, database), run in the thread pool."""
    try:
        image = decode_image(data, max_pixels=settings.scan_max_pixels)
    except InvalidImage as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    output = scanner.scan(image)
    try:
        infos = pill_infos(db, (c for d in output.detections for c in d.classes))
        cited = sources(db, ["c3pi", "rxnav"])
    except PharmaUnavailable as exc:
        raise pharma_unavailable() from exc

    pills = []
    for detection in output.detections:
        candidates = []
        for class_id, probability in zip(detection.classes, detection.probabilities, strict=True):
            info = infos.get(class_id)
            candidates.append(
                PillCandidate(
                    class_id=class_id,
                    name=info.name if info else f"class {class_id}",
                    generic_name=info.generic_name if info else "",
                    active_ingredients=info.ingredients if info else [],
                    probability=min(1.0, max(0.0, probability)),
                )
            )
        x1, y1, x2, y2 = detection.box
        pills.append(
            ScannedPill(
                box=PixelBox(x1=x1, y1=y1, x2=x2, y2=y2),
                box_normalized=NormalizedBox(
                    x1=x1 / output.width,
                    y1=y1 / output.height,
                    x2=x2 / output.width,
                    y2=y2 / output.height,
                ),
                detection_confidence=min(1.0, detection.confidence),
                candidates=candidates,
                needs_confirmation=candidates[0].probability < settings.scan_confirm_threshold,
            )
        )
    total = output.detection_ms + output.classification_ms
    return ScanResult(
        image_width=output.width,
        image_height=output.height,
        inference_ms=round(total, 1),
        timings=ScanTimings(
            detection_ms=round(output.detection_ms, 1),
            classification_ms=round(output.classification_ms, 1),
            total_ms=round(total, 1),
        ),
        confirm_threshold=settings.scan_confirm_threshold,
        pills=pills,
        message=None if pills else NO_PILL,
        models=output.models,
        sources=[SourceOut.model_validate(s) for s in cited],
        disclaimer=DISCLAIMER,
    )


@router.post(
    "/scan",
    response_model=ScanResult,
    responses={
        413: {"description": "Image larger than the upload limit"},
        415: {"description": "Not a JPEG, PNG or WebP image"},
        503: {"description": "Models or medicine tables not available"},
    },
)
async def scan(
    image: Annotated[UploadFile, File(description="JPEG, PNG or WebP photo of the pills")],
    request: Request,
    settings: SettingsDep,
    db: PharmaSession,
) -> ScanResult:
    """Detect every pill in the photo, then identify each one among the 150 C3PI classes.

    Identification is a suggestion: `needs_confirmation` is true when the most probable
    class is below `confirm_threshold`, and the app then asks the user to pick from the
    top-5 or search the list (GET /api/pills).
    """
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Send a JPEG, PNG or WebP image."
        )
    data = await image.read(settings.scan_max_bytes + 1)
    if len(data) > settings.scan_max_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"The image is larger than {settings.scan_max_bytes // (1024 * 1024)} MB.",
        )
    scanner = await run_in_threadpool(get_scanner, request)
    return await run_in_threadpool(_run, scanner, data, settings, db)
