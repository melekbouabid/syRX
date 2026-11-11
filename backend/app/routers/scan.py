"""POST /api/scan: one photo in, every detected pill out with its top-5 identifications,
or the printed text of a medicine box linked to its active ingredients (step 2.7).

Sign-in required, like the medicine routes: the demo uses the demo account (decision D46).
The photo's kind (pill, box, unknown) comes from app.boxes.decide_kind (D111): a box is
never given a pill class, and a photo that is neither is refused instead of guessed.
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.boxes import BoxMatch, BoxReader, KindThresholds, OcrResult, decide_kind
from app.config import Settings
from app.deps import PharmaSession, SettingsDep, get_current_user
from app.inference import (
    InvalidImage,
    ModelsMissing,
    PillScanner,
    ScannerHolder,
    decode_image,
)
from app.medicines import DISCLAIMER, PharmaUnavailable, pill_infos, sources
from app.pharma import Ingredient
from app.schemas import (
    BoxIngredient,
    BoxReading,
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
NOT_A_PILL = (
    "Cette photo ne ressemble ni à un comprimé ni à une boîte de médicament : aucune "
    "identification n'est proposée. Essayez une photo nette d'un comprimé entier sur un fond "
    "uni, ou de la face d'une boîte où le nom est imprimé."
)
BOX_FOUND = (
    "Boîte de médicament : principe actif lu sur l'emballage. Vérifiez-le sur la boîte et sa "
    "notice."
)
BOX_NOT_RECOGNISED = (
    "Boîte de médicament détectée, mais aucune marque ni aucun principe actif connu n'a été lu : "
    "boîte non reconnue. Essayez une photo nette de la face où le nom est imprimé."
)
MAX_TEXT_LINES = 40


def pharma_unavailable() -> HTTPException:
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "The medicine tables are not loaded. Run `python -m app.pharma_loader` (docs/infra.md).",
    )


def get_scanner(request: Request) -> PillScanner:
    holder: ScannerHolder = request.app.state.scanner
    try:
        return holder.get()
    except ModelsMissing as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


def get_boxes(request: Request) -> BoxReader:
    reader: BoxReader = request.app.state.boxes
    return reader


def _box_reading(reader: BoxReader, ocr: OcrResult, match: BoxMatch) -> BoxReading:
    return BoxReading(
        text=[line.text for line in ocr.lines][:MAX_TEXT_LINES],
        brand=match.brand.brand if match.brand else None,
        brand_country=match.brand.country if match.brand else None,
        brand_text=match.brand_text or None,
        ingredients=[
            BoxIngredient(
                name=i.name,
                method=i.method,
                matched_text=i.matched_text,
                confidence=min(1.0, max(0.0, i.score)),
                source=i.source,
                reference=i.reference,
                url=i.url,
                in_interaction_table=i.in_interaction_table,
            )
            for i in match.ingredients
        ],
        recognised=match.recognised,
        confidence=min(1.0, max(0.0, match.confidence)),
        rxnav=reader.rxnav_status,
    )


def _run(
    scanner: PillScanner,
    data: bytes,
    settings: Settings,
    db: Session,
    boxes: BoxReader | None = None,
) -> ScanResult:
    """Blocking part (decode, ONNX Runtime, OCR, database), run in the thread pool."""
    try:
        image = decode_image(data, max_pixels=settings.scan_max_pixels)
    except InvalidImage as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    output = scanner.scan(image)
    ocr, match = None, None
    engine = boxes.ocr.get() if boxes is not None else None
    try:
        if boxes is not None and engine is not None:
            ocr = engine.read(image)
            matcher = boxes.matcher(lambda: db.scalars(select(Ingredient.name)).all())
            match = matcher.match(ocr.lines)
        infos = pill_infos(db, (c for d in output.detections for c in d.classes))
        cited = sources(db, ["c3pi", "rxnav"])
    except PharmaUnavailable as exc:
        raise pharma_unavailable() from exc
    kind = decide_kind(
        [d.confidence for d in output.detections],
        ocr,
        match,
        KindThresholds(settings.scan_pill_min_confidence, settings.box_text_chars),
    )
    pills = []
    for detection in output.detections if kind == "pill" else []:
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
    if kind == "pill":
        message = None if pills else NO_PILL
    elif kind == "box":
        message = BOX_FOUND if match is not None and match.recognised else BOX_NOT_RECOGNISED
    else:
        message = NOT_A_PILL if output.detections else NO_PILL
    total = output.detection_ms + output.classification_ms
    return ScanResult(
        kind=kind,
        box=(
            _box_reading(boxes, ocr, match)
            if kind == "box" and boxes is not None and ocr is not None and match is not None
            else None
        ),
        image_width=output.width,
        image_height=output.height,
        inference_ms=round(total, 1),
        timings=ScanTimings(
            detection_ms=round(output.detection_ms, 1),
            classification_ms=round(output.classification_ms, 1),
            total_ms=round(total, 1),
            ocr_ms=round(ocr.ocr_ms, 1) if ocr is not None else None,
        ),
        confirm_threshold=settings.scan_confirm_threshold,
        pills=pills,
        message=message,
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
    image: Annotated[
        UploadFile, File(description="JPEG, PNG or WebP photo of pills or of a medicine box")
    ],
    request: Request,
    settings: SettingsDep,
    db: PharmaSession,
) -> ScanResult:
    """Detect every pill in the photo, then identify each one among the 150 C3PI classes;
    or read a medicine box and link its text to active ingredients.

    `kind` says what the photo shows: `pill` (pills below), `box` (`box`: lines read, brand,
    ingredients with their source; no pill class is given to a box) or `unknown` (neither:
    refused, nothing is guessed). Identification is a suggestion: `needs_confirmation` is
    true when the most probable class is below `confirm_threshold`, and the app then asks
    the user to pick from the top-5 or search the list (GET /api/pills). `timings.server_ms`
    is the whole request on the server; the app shows it next to the latency it measures in
    the browser.
    """
    started = time.perf_counter()
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
    result = await run_in_threadpool(_run, scanner, data, settings, db, get_boxes(request))
    result.timings.server_ms = round((time.perf_counter() - started) * 1000, 1)
    return result
