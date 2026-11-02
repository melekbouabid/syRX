"""Administration: every route here requires a signed-in administrator."""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.accounts import commit_or_conflict, create_user, ensure_email_free, revoke_sessions
from app.db import utc_now
from app.deps import AdminUser, CurrentSession, DbSession, require_admin
from app.models import AuthSession, Role, User
from app.schemas import (
    AdminPasswordReset,
    AdminStats,
    AdminUserCreate,
    AdminUserUpdate,
    UserOut,
    UserPage,
)
from app.security import hash_password

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


def _get_user(db: Session, user_id: int) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    return user


@router.get("/stats", response_model=AdminStats)
def read_stats(db: DbSession) -> AdminStats:
    now = utc_now()

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
        signups_last_7_days=count(User.created_at >= now - timedelta(days=7)),
        active_sessions=db.scalar(sessions) or 0,
    )


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
    return UserPage(
        items=[UserOut.model_validate(row) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_account(payload: AdminUserCreate, db: DbSession) -> User:
    ensure_email_free(db, payload.email)
    user = create_user(
        db,
        email=payload.email,
        full_name=payload.full_name,
        password=payload.password,
        role=payload.role,
        is_active=payload.is_active,
    )
    commit_or_conflict(db)
    return user


@router.get("/users/{user_id}", response_model=UserOut)
def read_account(user_id: int, db: DbSession) -> User:
    return _get_user(db, user_id)


@router.patch("/users/{user_id}", response_model=UserOut)
def update_account(user_id: int, payload: AdminUserUpdate, admin: AdminUser, db: DbSession) -> User:
    user = _get_user(db, user_id)
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)

    # An administrator cannot lock themselves out. Since only another administrator can
    # demote or deactivate one, at least one active administrator always remains.
    if user.id == admin.id:
        if changes.get("role", Role.ADMIN) != Role.ADMIN:
            raise HTTPException(status.HTTP_409_CONFLICT, "You cannot change your own role")
        if changes.get("is_active") is False:
            raise HTTPException(status.HTTP_409_CONFLICT, "You cannot deactivate your own account")

    if "email" in changes:
        ensure_email_free(db, changes["email"], except_user_id=user.id)
        user.email = changes["email"]
    if "full_name" in changes:
        user.full_name = changes["full_name"]
    if "role" in changes:
        user.role = changes["role"].value
    if "is_active" in changes:
        user.is_active = changes["is_active"]
        if not user.is_active:
            revoke_sessions(db, user.id)
    commit_or_conflict(db)
    return user


@router.post("/users/{user_id}/password", status_code=status.HTTP_204_NO_CONTENT)
def reset_password(
    user_id: int, payload: AdminPasswordReset, session: CurrentSession, db: DbSession
) -> None:
    """Set a new password and sign the user out everywhere.

    Passwords are stored as Argon2 hashes: they can be replaced, never read back.
    """
    user = _get_user(db, user_id)
    user.password_hash = hash_password(payload.new_password)
    # An administrator resetting their own password stays signed in on this browser.
    revoke_sessions(db, user.id, keep=session if session.user_id == user.id else None)
    db.commit()


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_account(user_id: int, admin: AdminUser, db: DbSession) -> None:
    user = _get_user(db, user_id)
    if user.id == admin.id:
        raise HTTPException(status.HTTP_409_CONFLICT, "You cannot delete your own account")
    db.delete(user)
    db.commit()
