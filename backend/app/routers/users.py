"""The stored images of an account: its profile photo and its signature.

A photo is shown to its owner and to administrators; a signature only to its owner (it is
printed on the owner's reports, inside the report itself, never through this route).
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import select

from app.deps import CurrentUser, DbSession
from app.models import Profile, Role

router = APIRouter(prefix="/api/users", tags=["users"])

MEDIA = {"photo": "image/jpeg", "signature": "image/png"}


@router.get(
    "/{user_id}/{kind}",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}, "image/png": {}}}, 404: {}},
    summary="Profile photo (owner or administrator) or signature (owner only)",
)
def read_image(
    user_id: int, kind: Literal["photo", "signature"], viewer: CurrentUser, db: DbSession
) -> Response:
    allowed = viewer.id == user_id or (kind == "photo" and viewer.role == Role.ADMIN)
    missing = HTTPException(status.HTTP_404_NOT_FOUND, "No image")
    if not allowed:
        raise missing  # the same answer as "no image": nothing is revealed about the account
    column = Profile.photo if kind == "photo" else Profile.signature
    data = db.scalar(select(column).where(Profile.user_id == user_id))
    if not data:
        raise missing
    # The address carries ?v=<upload time>, so it can be cached; it stays private to the browser.
    return Response(
        data, media_type=MEDIA[kind], headers={"Cache-Control": "private, max-age=86400"}
    )
