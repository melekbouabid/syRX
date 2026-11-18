"""The activity log that administrators read (GET /api/admin/activity and /api/admin/usage).

An event says who did what and when, from which address and browser, with a few counts. It
never holds a photo, a pill or medicine name, or the text of a question: those are health
data and stay with the user (docs/project_rules.md).
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from fastapi import Request
from sqlalchemy import delete
from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool

from app.db import utc_now
from app.models import Action, ActivityEvent, User

log = logging.getLogger("tujya.api")

# The filters of the admin Activity page; every Action belongs to exactly one group.
GROUPS: dict[str, tuple[Action, ...]] = {
    "access": (
        Action.SIGN_UP,
        Action.SIGN_IN,
        Action.SIGN_OUT,
    ),
    "security": (
        Action.SIGN_IN_FAILED,
        Action.SIGN_IN_BLOCKED,
        Action.PASSWORD_CHANGE,
    ),
    "usage": (
        Action.SCAN,
        Action.INTERACTIONS,
        Action.CHAT,
        Action.TOXICITY,
        Action.MEDICINE_SAVE,
    ),
    "reports": (
        Action.REPORT_CREATE,
        Action.REPORT_SHARE,
        Action.REPORT_UNSHARE,
        Action.REPORT_DELETE,
    ),
    "account": (
        Action.PROFILE_UPDATE,
        Action.PHOTO_UPDATE,
        Action.SIGNATURE_UPDATE,
    ),
    "admin": (
        Action.USER_CREATE,
        Action.USER_UPDATE,
        Action.PASSWORD_RESET,
        Action.USER_DELETE,
        Action.DRIFT_RESET,
        Action.DATA_EXPORT,
    ),
}
GROUP_OF: dict[str, str] = {
    action.value: group for group, acts in GROUPS.items() for action in acts
}


def client_ip(request: Request) -> str | None:
    """The browser's address.

    Behind the nginx of docker compose it is the last X-Forwarded-For entry, the one nginx
    added; anything before it was written by the browser and is not trusted. Without that
    header (Vite dev proxy, tests) it is the direct client's address.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()[:45] or None
    return request.client.host[:45] if request.client else None


def add_event(
    db: Session,
    user_id: int,
    action: Action,
    request: Request | None = None,
    *,
    target: User | None = None,
    **details: Any,
) -> None:
    """Add an event to `db`; the caller's commit saves it with the change it records."""
    agent = request.headers.get("user-agent") if request is not None else None
    db.add(
        ActivityEvent(
            user_id=user_id,
            action=action.value,
            target_id=target.id if target is not None else None,
            target_email=target.email if target is not None else None,
            details=details,
            ip=client_ip(request) if request is not None else None,
            user_agent=agent[:300] if agent else None,
        )
    )


def _record(request: Request, user_id: int, action: Action, details: dict[str, Any]) -> None:
    factory: sessionmaker[Session] = request.app.state.session_factory
    try:
        with factory() as db:
            add_event(db, user_id, action, request, **details)
            db.commit()
    except Exception:
        # The log must never break a scan or an answer.
        log.exception("could not record the activity event %s", action.value)


async def record(request: Request, user_id: int, action: Action, **details: Any) -> None:
    """Record an event in its own transaction, from an async route that writes nothing else."""
    await run_in_threadpool(_record, request, user_id, action, details)


def prune(db: Session, days: int) -> int:
    """Delete the events older than `days` days (0 keeps everything). Returns how many."""
    if days <= 0:
        return 0
    result = db.execute(
        delete(ActivityEvent).where(ActivityEvent.created_at < utc_now() - timedelta(days=days))
    )
    db.commit()
    return int(getattr(result, "rowcount", 0) or 0)
