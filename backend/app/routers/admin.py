"""Administration: every route here requires a signed-in administrator.

Accounts (list, create, edit, reset a password, delete), the numbers of the dashboard, the
activity log, the daily usage and the CSV exports. What an administrator does to an account
is itself written to the activity log.
"""

from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from datetime import timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.accounts import commit_or_conflict, create_user, ensure_email_free, revoke_sessions
from app.activity import GROUP_OF, GROUPS, add_event
from app.db import utc_now
from app.deps import AdminUser, CurrentSession, DbSession, require_admin
from app.models import Action, ActivityEvent, AuthSession, Report, Role, User
from app.schemas import (
    ActivityEventOut,
    ActivityPage,
    AdminPasswordReset,
    AdminStats,
    AdminUserCreate,
    AdminUserUpdate,
    UsageDay,
    UsageOut,
    UserOut,
    UserPage,
)
from app.security import hash_password

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])

Group = Literal["access", "security", "usage", "reports", "account", "admin"]
# Daily usage: which actions fill which column of UsageDay.
USAGE_COLUMNS: dict[str, tuple[Action, ...]] = {
    "sign_ins": (Action.SIGN_IN,),
    "sign_ups": (Action.SIGN_UP,),
    "scans": (Action.SCAN,),
    "interactions": (Action.INTERACTIONS,),
    "questions": (Action.CHAT,),
    "toxicity": (Action.TOXICITY,),
    "reports": (Action.REPORT_CREATE,),
}
COLUMN_OF = {a.value: column for column, actions in USAGE_COLUMNS.items() for a in actions}


def _get_user(db: Session, user_id: int) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    return user


def _view(user: User) -> UserOut:
    """An account as administrators see it: never its signature."""
    return UserOut.model_validate(user).model_copy(update={"signature_url": None})


def _count_events(db: Session, since: Any, *actions: Action) -> int:
    statement = (
        select(func.count())
        .select_from(ActivityEvent)
        .where(
            ActivityEvent.created_at >= since, ActivityEvent.action.in_([a.value for a in actions])
        )
    )
    return db.scalar(statement) or 0


@router.get("/stats", response_model=AdminStats)
def read_stats(db: DbSession) -> AdminStats:
    now = utc_now()
    week = now - timedelta(days=7)

    def count(*conditions: object) -> int:
        return db.scalar(select(func.count()).select_from(User).where(*conditions)) or 0

    total = count()
    active = count(User.is_active)
    sessions = select(func.count()).select_from(AuthSession).where(AuthSession.expires_at > now)
    return AdminStats(
        users_total=total,
        users_active=active,
        users_inactive=total - active,
        admins=count(User.role == Role.ADMIN, User.is_active),
        signups_last_7_days=count(User.created_at >= week),
        active_sessions=db.scalar(sessions) or 0,
        scans_last_7_days=_count_events(db, week, Action.SCAN),
        questions_last_7_days=_count_events(db, week, Action.CHAT),
        reports_total=db.scalar(select(func.count()).select_from(Report)) or 0,
        failed_sign_ins_last_7_days=_count_events(
            db, week, Action.SIGN_IN_FAILED, Action.SIGN_IN_BLOCKED
        ),
    )


# ------------------------------------------------------------------ accounts


@router.get("/users", response_model=UserPage)
def list_users(
    db: DbSession,
    q: Annotated[str, Query(max_length=100, description="Part of a name or an email")] = "",
    role: Role | None = None,
    state: Annotated[Literal["active", "inactive"] | None, Query(alias="status")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> UserPage:
    conditions = []
    if q.strip():
        needle = q.strip()
        conditions.append(
            or_(
                User.email.icontains(needle, autoescape=True),
                User.full_name.icontains(needle, autoescape=True),
            )
        )
    if role is not None:
        conditions.append(User.role == role)
    if state is not None:
        conditions.append(User.is_active.is_(state == "active"))

    total = db.scalar(select(func.count()).select_from(User).where(*conditions)) or 0
    rows = db.scalars(
        select(User)
        .where(*conditions)
        .order_by(User.created_at.desc(), User.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return UserPage(items=[_view(row) for row in rows], total=total, page=page, page_size=page_size)


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_account(
    payload: AdminUserCreate, request: Request, admin: AdminUser, db: DbSession
) -> UserOut:
    ensure_email_free(db, payload.email)
    user = create_user(
        db,
        email=payload.email,
        full_name=payload.full_name,
        password=payload.password,
        role=payload.role,
        is_active=payload.is_active,
    )
    add_event(db, admin.id, Action.USER_CREATE, request, target=user, role=user.role)
    commit_or_conflict(db)
    return _view(user)


@router.get("/users/{user_id}", response_model=UserOut)
def read_account(user_id: int, db: DbSession) -> UserOut:
    return _view(_get_user(db, user_id))


@router.patch("/users/{user_id}", response_model=UserOut)
def update_account(
    user_id: int, payload: AdminUserUpdate, request: Request, admin: AdminUser, db: DbSession
) -> UserOut:
    user = _get_user(db, user_id)
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)

    # An administrator cannot lock themselves out. Since only another administrator can
    # demote or deactivate one, at least one active administrator always remains.
    if user.id == admin.id:
        if changes.get("role", Role.ADMIN) != Role.ADMIN:
            raise HTTPException(status.HTTP_409_CONFLICT, "You cannot change your own role")
        if changes.get("is_active") is False:
            raise HTTPException(status.HTTP_409_CONFLICT, "You cannot deactivate your own account")

    changed: list[str] = []
    if "email" in changes and changes["email"] != user.email:
        ensure_email_free(db, changes["email"], except_user_id=user.id)
        user.email = changes["email"]
        changed.append("email")
    if "full_name" in changes and changes["full_name"] != user.full_name:
        user.full_name = changes["full_name"]
        changed.append("full_name")
    if "role" in changes and changes["role"].value != user.role:
        user.role = changes["role"].value
        changed.append("role")
    if "is_active" in changes and changes["is_active"] != user.is_active:
        user.is_active = changes["is_active"]
        changed.append("is_active")
        if not user.is_active:
            revoke_sessions(db, user.id)
    if changed:
        add_event(
            db,
            admin.id,
            Action.USER_UPDATE,
            request,
            target=user,
            fields=changed,
            role=user.role,
            is_active=user.is_active,
        )
    commit_or_conflict(db)
    return _view(user)


@router.post("/users/{user_id}/password", status_code=status.HTTP_204_NO_CONTENT)
def reset_password(
    user_id: int,
    payload: AdminPasswordReset,
    request: Request,
    session: CurrentSession,
    db: DbSession,
) -> None:
    """Set a new password and sign the user out everywhere.

    Passwords are stored as Argon2 hashes: they can be replaced, never read back.
    """
    user = _get_user(db, user_id)
    user.password_hash = hash_password(payload.new_password)
    # An administrator resetting their own password stays signed in on this browser.
    revoke_sessions(db, user.id, keep=session if session.user_id == user.id else None)
    add_event(db, session.user_id, Action.PASSWORD_RESET, request, target=user)
    db.commit()


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_account(user_id: int, request: Request, admin: AdminUser, db: DbSession) -> None:
    user = _get_user(db, user_id)
    if user.id == admin.id:
        raise HTTPException(status.HTTP_409_CONFLICT, "You cannot delete your own account")
    # Recorded under the administrator, so it stays after the account's own events are gone.
    add_event(db, admin.id, Action.USER_DELETE, request, target=user, name=user.full_name)
    db.delete(user)
    db.commit()


# ------------------------------------------------------------------ activity log


def _event_conditions(
    user_id: int | None, group: str | None, action: str | None, q: str, days: int
) -> list[Any]:
    conditions: list[Any] = []
    if days:
        conditions.append(ActivityEvent.created_at >= utc_now() - timedelta(days=days))
    if user_id is not None:
        # An account's events, and what administrators did to it.
        conditions.append(or_(ActivityEvent.user_id == user_id, ActivityEvent.target_id == user_id))
    if group is not None:
        conditions.append(ActivityEvent.action.in_([a.value for a in GROUPS[group]]))
    if action:
        conditions.append(ActivityEvent.action == action)
    if q.strip():
        needle = q.strip()
        conditions.append(
            or_(
                User.email.icontains(needle, autoescape=True),
                User.full_name.icontains(needle, autoescape=True),
                ActivityEvent.target_email.icontains(needle, autoescape=True),
                ActivityEvent.ip.icontains(needle, autoescape=True),
            )
        )
    return conditions


def _event_out(event: ActivityEvent, user: User) -> ActivityEventOut:
    return ActivityEventOut(
        id=event.id,
        created_at=event.created_at,
        user_id=event.user_id,
        user_name=user.full_name,
        user_email=user.email,
        action=event.action,
        group=GROUP_OF.get(event.action, "usage"),
        target_id=event.target_id,
        target_email=event.target_email,
        details=event.details or {},
        ip=event.ip,
        user_agent=event.user_agent,
    )


@router.get("/activity", response_model=ActivityPage)
def list_activity(
    db: DbSession,
    user_id: int | None = None,
    group: Group | None = None,
    action: Annotated[str, Query(max_length=32)] = "",
    q: Annotated[str, Query(max_length=100, description="Name, email or IP address")] = "",
    days: Annotated[int, Query(ge=0, le=3650, description="0 = all time")] = 30,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 50,
) -> ActivityPage:
    """Newest first. Counts per group follow every filter except the group itself."""
    conditions = _event_conditions(user_id, group, action, q, days)
    base = select(ActivityEvent, User).join(User, User.id == ActivityEvent.user_id)
    total = db.scalar(
        select(func.count())
        .select_from(ActivityEvent)
        .join(User, User.id == ActivityEvent.user_id)
        .where(*conditions)
    )
    rows = db.execute(
        base.where(*conditions)
        .order_by(ActivityEvent.created_at.desc(), ActivityEvent.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    per_action = db.execute(
        select(ActivityEvent.action, func.count())
        .join(User, User.id == ActivityEvent.user_id)
        .where(*_event_conditions(user_id, None, action, q, days))
        .group_by(ActivityEvent.action)
    ).all()
    counts = dict.fromkeys(GROUPS, 0)
    for name, n in per_action:
        counts[GROUP_OF.get(name, "usage")] += n
    return ActivityPage(
        items=[_event_out(event, user) for event, user in rows],
        total=total or 0,
        page=page,
        page_size=page_size,
        counts=counts,
    )


@router.get("/usage", response_model=UsageOut)
def read_usage(db: DbSession, days: Annotated[int, Query(ge=7, le=365)] = 30) -> UsageOut:
    """What the app was used for, day by day (UTC), and the most active accounts."""
    today = utc_now().date()
    first = today - timedelta(days=days - 1)
    rows = db.execute(
        select(ActivityEvent.created_at, ActivityEvent.action, ActivityEvent.user_id).where(
            ActivityEvent.created_at
            >= utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
            - timedelta(days=days - 1)
        )
    ).all()
    per_day: dict[str, Counter[str]] = defaultdict(Counter)
    users_per_day: dict[str, set[int]] = defaultdict(set)
    per_user: Counter[int] = Counter()
    for created_at, action, user_id in rows:
        day = created_at.date().isoformat()
        column = COLUMN_OF.get(action)
        if column is not None:
            per_day[day][column] += 1
            per_user[user_id] += 1
        if action not in (Action.SIGN_IN_FAILED.value, Action.SIGN_IN_BLOCKED.value):
            users_per_day[day].add(user_id)
    out_days = []
    for offset in range(days):
        day = (first + timedelta(days=offset)).isoformat()
        out_days.append(
            UsageDay(
                day=day,
                **{column: per_day[day][column] for column in USAGE_COLUMNS},
                active_users=len(users_per_day[day]),
            )
        )
    totals = {column: sum(getattr(d, column) for d in out_days) for column in USAGE_COLUMNS}
    totals["active_users"] = len(set().union(*users_per_day.values())) if users_per_day else 0
    top_ids = [user_id for user_id, _ in per_user.most_common(5)]
    names = (
        {u.id: u for u in db.scalars(select(User).where(User.id.in_(top_ids))).all()}
        if top_ids
        else {}
    )
    top_users = [
        {
            "user_id": user_id,
            "name": names[user_id].full_name,
            "email": names[user_id].email,
            "actions": per_user[user_id],
        }
        for user_id in top_ids
        if user_id in names
    ]
    return UsageOut(days=out_days, totals=totals, top_users=top_users)


# ------------------------------------------------------------------ CSV exports


def _cell(value: Any) -> str:
    """A CSV cell that a spreadsheet will not run as a formula."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def _csv(name: str, header: list[str], rows: list[list[Any]]) -> Response:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows([[_cell(v) for v in row] for row in rows])
    stamp = utc_now().strftime("%Y%m%d-%H%M")
    return Response(
        "﻿" + buffer.getvalue(),  # BOM: Excel then reads the accents correctly
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="tujya-{name}-{stamp}.csv"'},
    )


@router.get(
    "/export/users.csv",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}}},
    summary="Every account as CSV (no password hash, no image)",
)
def export_users(request: Request, admin: AdminUser, db: DbSession) -> Response:
    users = db.scalars(select(User).order_by(User.id)).all()
    rows = [
        [u.id, u.email, u.full_name, u.role, u.is_active, u.phone, u.created_at, u.last_login_at]
        for u in users
    ]
    add_event(db, admin.id, Action.DATA_EXPORT, request, table="users", rows=len(rows))
    db.commit()
    header = ["id", "email", "full_name", "role", "is_active", "phone", "created_at", "last_login"]
    return _csv("users", header, rows)


@router.get(
    "/export/activity.csv",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}}},
    summary="The activity log as CSV, with the same filters as /api/admin/activity",
)
def export_activity(
    request: Request,
    admin: AdminUser,
    db: DbSession,
    user_id: int | None = None,
    group: Group | None = None,
    action: Annotated[str, Query(max_length=32)] = "",
    q: Annotated[str, Query(max_length=100)] = "",
    days: Annotated[int, Query(ge=0, le=3650)] = 30,
) -> Response:
    conditions = _event_conditions(user_id, group, action, q, days)
    rows = db.execute(
        select(ActivityEvent, User)
        .join(User, User.id == ActivityEvent.user_id)
        .where(*conditions)
        .order_by(ActivityEvent.created_at.desc(), ActivityEvent.id.desc())
        .limit(50_000)
    ).all()
    data = [
        [
            e.created_at,
            u.email,
            e.action,
            GROUP_OF.get(e.action, ""),
            e.target_email,
            e.details,
            e.ip,
            e.user_agent,
        ]
        for e, u in rows
    ]
    add_event(db, admin.id, Action.DATA_EXPORT, request, table="activity", rows=len(data))
    db.commit()
    header = ["created_at", "user", "action", "group", "target", "details", "ip", "user_agent"]
    return _csv("activity", header, data)
