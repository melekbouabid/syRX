"""GET /monitoring/drift: data drift of the scanned photos (aggregate numbers only, no sign-in).

POST /monitoring/drift/reset (administrator) empties the window.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, status

from app.activity import add_event
from app.deps import AdminUser, DbSession
from app.drift import DriftMonitor
from app.models import Action
from app.schemas import DriftReport

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


def monitor(request: Request) -> DriftMonitor:
    found: DriftMonitor = request.app.state.drift
    return found


@router.get(
    "/drift",
    response_model=DriftReport,
    summary="Data drift of the scanned photos against the reference photos",
)
def drift(request: Request) -> Any:
    return monitor(request).report()


@router.post(
    "/drift/reset",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Empty the window of scanned photos (administrator)",
)
def reset(request: Request, admin: AdminUser, db: DbSession) -> None:
    monitor(request).reset()
    add_event(db, admin.id, Action.DRIFT_RESET, request)
    db.commit()
