"""Data drift of the scanned photos: GET /monitoring/drift (step 2.8, docs/module1/derive.md).

Every photo scanned by POST /predict, /predict/visualize or /api/scan adds eight numbers to a
sliding window in memory (app/drift.py): no image, no file name, nothing about the user. This
route compares the window with the statistics of the training photos (Kolmogorov-Smirnov test
and PSI per statistic) and says whether the photos drift. No sign-in needed, like /health:
the report holds aggregate numbers only. POST /monitoring/drift/reset (administrator) empties
the window, e.g. before the demonstration of docs/module1/derive.md.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, status

from app.deps import AdminUser
from app.drift import DriftMonitor
from app.schemas import DriftReport

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


def monitor(request: Request) -> DriftMonitor:
    found: DriftMonitor = request.app.state.drift
    return found


@router.get(
    "/drift",
    response_model=DriftReport,
    summary="Data drift of the scanned photos against the training photos",
)
def drift(request: Request) -> Any:
    return monitor(request).report()


@router.post(
    "/drift/reset",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Empty the window of scanned photos (administrator)",
)
def reset(request: Request, _: AdminUser) -> None:
    monitor(request).reset()
