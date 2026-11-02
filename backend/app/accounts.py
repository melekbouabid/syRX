"""Account operations shared by the sign-in and administration routes."""

from __future__ import annotations

import logging
from datetime import timedelta

from fastapi import HTTPException, Response, status
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import DEFAULT_ADMIN_PASSWORD, Settings
from app.db import utc_now
from app.models import AuthSession, Role, User
from app.security import hash_password, hash_token, new_session_token, verify_password

log = logging.getLogger("synergyrx.api")


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == email.strip().lower()))


def ensure_email_free(db: Session, email: str, *, except_user_id: int | None = None) -> None:
    existing = get_user_by_email(db, email)
    if existing is not None and existing.id != except_user_id:
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")


def create_user(
    db: Session,
    *,
    email: str,
    full_name: str,
    password: str,
    role: Role = Role.USER,
    is_active: bool = True,
) -> User:
    user = User(
        email=email.strip().lower(),
        full_name=full_name,
        password_hash=hash_password(password),
        role=role.value,
        is_active=is_active,
    )
    db.add(user)
    db.flush()
    return user


def start_session(db: Session, user: User, response: Response, settings: Settings) -> None:
    """Record a sign-in and hand the browser its session cookie."""
    now = utc_now()
    db.execute(delete(AuthSession).where(AuthSession.expires_at <= now))
    token = new_session_token()
    lifetime = timedelta(hours=settings.session_hours)
    db.add(AuthSession(token_hash=hash_token(token), user_id=user.id, expires_at=now + lifetime))
    user.last_login_at = now
    response.set_cookie(
        settings.session_cookie,
        token,
        max_age=int(lifetime.total_seconds()),
        httponly=True,  # not readable from JavaScript
        samesite="lax",  # not sent with cross-site POST requests
        secure=settings.cookie_secure,
        path="/",
    )


def end_session_cookie(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        settings.session_cookie,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )


def revoke_sessions(db: Session, user_id: int, *, keep: AuthSession | None = None) -> None:
    """Sign a user out everywhere, optionally keeping the session making the request."""
    statement = delete(AuthSession).where(AuthSession.user_id == user_id)
    if keep is not None:
        statement = statement.where(AuthSession.id != keep.id)
    db.execute(statement)


def seed_accounts(db: Session, settings: Settings) -> None:
    """Create the first administrator (and the demo user) on an empty database."""
    if db.scalar(select(func.count()).select_from(User)):
        return
    create_user(
        db,
        email=settings.admin_email,
        full_name="Administrator",
        password=settings.admin_password,
        role=Role.ADMIN,
    )
    if settings.seed_demo_user:
        create_user(
            db,
            email=settings.demo_user_email,
            full_name="Demo User",
            password=settings.demo_user_password,
        )
    db.commit()
    log.warning("Created the first administrator account: %s", settings.admin_email)


def warn_about_default_password(db: Session, settings: Settings) -> None:
    """Repeat at every start until the first administrator has chosen a password."""
    admin = get_user_by_email(db, settings.admin_email)
    if admin is not None and verify_password(DEFAULT_ADMIN_PASSWORD, admin.password_hash):
        log.warning(
            "%s still has the default password. Change it under Admin > My account before "
            "this server is reachable by anyone else.",
            admin.email,
        )


def commit_or_conflict(db: Session) -> None:
    """Commit; a duplicate email that slipped past the check becomes a 409, not a 500."""
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, "An account with this email already exists"
        ) from None
