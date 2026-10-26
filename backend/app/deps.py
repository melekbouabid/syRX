"""Request dependencies: database session, settings, signed-in user, admin check."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import utc_now
from app.models import AuthSession, Role, User
from app.security import hash_token


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_db(request: Request) -> Iterator[Session]:
    with request.app.state.session_factory() as session:
        yield session


SettingsDep = Annotated[Settings, Depends(get_settings)]
DbSession = Annotated[Session, Depends(get_db)]


def get_auth_session(request: Request, db: DbSession, settings: SettingsDep) -> AuthSession:
    """The session behind the request's cookie. 401 when missing, expired or deactivated."""
    unauthorized = HTTPException(status.HTTP_401_UNAUTHORIZED, "Not signed in")
    token = request.cookies.get(settings.session_cookie)
    if not token:
        raise unauthorized
    session = db.scalar(select(AuthSession).where(AuthSession.token_hash == hash_token(token)))
    if session is None or session.expires_at <= utc_now() or not session.user.is_active:
        raise unauthorized
    return session


CurrentSession = Annotated[AuthSession, Depends(get_auth_session)]


def get_current_user(session: CurrentSession) -> User:
    return session.user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_admin(user: CurrentUser) -> User:
    # The role is read from the database on every request, so a demotion applies at once.
    if user.role != Role.ADMIN:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Administrator access required")
    return user


AdminUser = Annotated[User, Depends(require_admin)]
