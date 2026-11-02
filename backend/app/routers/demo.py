"""Demo gallery: a few C3PI test-split photos, so the scan can be shown without real pills.

The list is `config/demo_gallery.csv` (chosen by hand, decision D52). A photo is served only
if `config/splits/selected_images.csv` puts it in the **test** split (never seen in
training) and the file exists under the C3PI image folder; nothing else on disk can be read
through these routes. Read only; sign-in required like the scan.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import FileResponse

from app.deps import PharmaSession, SettingsDep, get_current_user
from app.medicines import PharmaUnavailable, pill_infos
from app.schemas import DemoGallery, DemoImage

router = APIRouter(prefix="/api/demo", tags=["demo"], dependencies=[Depends(get_current_user)])

THUMB_WIDTH = 360
CACHE = "private, max-age=3600"


@dataclass(frozen=True)
class DemoPhoto:
    file: Path
    class_id: int
    why: str


def _gallery(gallery_file: Path, split_file: Path, images_dir: Path) -> list[DemoPhoto]:
    if not gallery_file.is_file() or not split_file.is_file():
        return []
    with split_file.open(newline="", encoding="utf-8") as handle:
        test = {
            row["path"]
            for row in csv.DictReader(handle)
            if row["split"] == "test" and row["image_type"] == "C3PI_Test"
        }
    photos = []
    with gallery_file.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["path"] not in test:
                continue
            disc, _, name = row["path"].split("/")
            file = images_dir / disc / name
            if file.is_file():
                photos.append(DemoPhoto(file, int(row["class_id"]), row.get("why", "")))
    return photos


def gallery(settings: SettingsDep) -> list[DemoPhoto]:
    return _gallery(settings.demo_gallery_file, settings.demo_split_file, settings.demo_images_dir)


@lru_cache(maxsize=32)
def _thumbnail(file: Path, mtime: float) -> bytes:
    image = cv2.imdecode(np.fromfile(str(file), dtype=np.uint8), cv2.IMREAD_COLOR)
    height, width = image.shape[:2]
    small = cv2.resize(
        image, (THUMB_WIDTH, round(height * THUMB_WIDTH / width)), interpolation=cv2.INTER_AREA
    )
    ok, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError(f"cannot encode a thumbnail of {file}")
    return encoded.tobytes()


@router.get("/images", response_model=DemoGallery)
def list_images(settings: SettingsDep, db: PharmaSession) -> DemoGallery:
    photos = gallery(settings)
    try:
        names = {k: v.name for k, v in pill_infos(db, (p.class_id for p in photos)).items()}
    except PharmaUnavailable:
        names = {}  # the gallery still works; the scan will then explain what is missing
    return DemoGallery(
        items=[
            DemoImage(
                id=index,
                class_id=photo.class_id,
                expected_name=names.get(photo.class_id, ""),
                why=photo.why,
                url=f"/api/demo/images/{index}",
                thumbnail_url=f"/api/demo/images/{index}?thumbnail=true",
            )
            for index, photo in enumerate(photos)
        ],
        note=(
            "Photos du split de test C3PI (NLM, ODbL, usage recherche) : le modèle ne les a "
            "jamais vues pendant l'entraînement."
        ),
    )


@router.get(
    "/images/{image_id}",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}, 404: {"description": "No such photo"}},
)
def read_image(image_id: int, settings: SettingsDep, thumbnail: bool = False) -> Response:
    photos = gallery(settings)
    if not 0 <= image_id < len(photos):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such demo photo")
    file = photos[image_id].file
    if thumbnail:
        data = _thumbnail(file, file.stat().st_mtime)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": CACHE})
    return FileResponse(file, media_type="image/jpeg", headers={"Cache-Control": CACHE})
