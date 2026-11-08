"""Demo gallery: a few C3PI test photos, so the scan can be shown without real pills.

The list is `config/demo_gallery.csv` (chosen by hand, decisions D52 and D105). A photo is
served only if no served model saw it in training: it must be a consumer photo of the C3PI
v2 **test** split (`config/splits/c3pi_v2/test.csv`, the detector's split, D89) and must not
be in the v1 train or val split (`config/splits/selected_images.csv`, the classifier's
split). The file must also exist under the C3PI image folder; nothing else on disk can be
read through these routes. Read only; sign-in required like the scan.

Grad-CAM: `GET /api/demo/images/{id}/gradcam` serves the picture precomputed for that photo
with the PyTorch weights of the served classifier (`python -m src.models.gradcam_gallery`,
decision D104); the API itself runs ONNX models only, without PyTorch.
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
CONSUMER = "C3PI_Test"  # consumer photos; the other image types are studio or challenge shots


@dataclass(frozen=True)
class DemoPhoto:
    file: Path
    class_id: int
    why: str
    gradcam: Path  # may not exist: only computed for the listed photos (D104)


def gradcam_name(path: str) -> str:
    """`PillProjectDisc20/images/B3U.JPG` -> `PillProjectDisc20_B3U.jpg` (one folder)."""
    disc, _, name = path.split("/")
    return f"{disc}_{Path(name).stem}.jpg"


def _never_trained_on(holdout_file: Path, split_file: Path) -> set[str]:
    """Consumer photos of the v2 test split that are not in the v1 train or val split."""
    with holdout_file.open(newline="", encoding="utf-8") as handle:
        test = {row["path"] for row in csv.DictReader(handle) if row["image_type"] == CONSUMER}
    with split_file.open(newline="", encoding="utf-8") as handle:
        trained = {row["path"] for row in csv.DictReader(handle) if row["split"] != "test"}
    return test - trained


def _gallery(
    gallery_file: Path,
    split_file: Path,
    holdout_file: Path,
    images_dir: Path,
    gradcam_dir: Path,
) -> list[DemoPhoto]:
    if not all(f.is_file() for f in (gallery_file, split_file, holdout_file)):
        return []
    allowed = _never_trained_on(holdout_file, split_file)
    photos = []
    with gallery_file.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["path"] not in allowed:
                continue
            disc, _, name = row["path"].split("/")
            file = images_dir / disc / name
            if file.is_file():
                photos.append(
                    DemoPhoto(
                        file,
                        int(row["class_id"]),
                        row.get("why", ""),
                        gradcam_dir / gradcam_name(row["path"]),
                    )
                )
    return photos


def gallery(settings: SettingsDep) -> list[DemoPhoto]:
    return _gallery(
        settings.demo_gallery_file,
        settings.demo_split_file,
        settings.demo_holdout_file,
        settings.demo_images_dir,
        settings.demo_gradcam_dir,
    )


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
                gradcam_url=(
                    f"/api/demo/images/{index}/gradcam" if photo.gradcam.is_file() else None
                ),
            )
            for index, photo in enumerate(photos)
        ],
        note=(
            "Photos du split de test C3PI (NLM, ODbL, usage recherche) : ni le détecteur ni "
            "le classifieur servis ne les ont vues pendant l'entraînement."
        ),
    )


def _photo(image_id: int, settings: SettingsDep) -> DemoPhoto:
    photos = gallery(settings)
    if not 0 <= image_id < len(photos):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such demo photo")
    return photos[image_id]


@router.get(
    "/images/{image_id}",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}, 404: {"description": "No such photo"}},
)
def read_image(image_id: int, settings: SettingsDep, thumbnail: bool = False) -> Response:
    file = _photo(image_id, settings).file
    if thumbnail:
        data = _thumbnail(file, file.stat().st_mtime)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": CACHE})
    return FileResponse(file, media_type="image/jpeg", headers={"Cache-Control": CACHE})


@router.get(
    "/images/{image_id}/gradcam",
    response_class=Response,
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "Grad-CAM of each detected pill"},
        404: {"description": "No such photo, or no Grad-CAM computed for it"},
    },
)
def read_gradcam(image_id: int, settings: SettingsDep) -> Response:
    """The photo with the Grad-CAM heat map of the classifier over each detected pill.

    Precomputed offline with the PyTorch weights of the served classifier, for the top-1
    class the API gives (decision D104). Same proportions as the photo, so the boxes of the scan
    fit on it.
    """
    file = _photo(image_id, settings).gradcam
    if not file.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No Grad-CAM for this photo")
    return FileResponse(file, media_type="image/jpeg", headers={"Cache-Control": CACHE})
