"""Sign up, sign in, sign out and the signed-in user's own account."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy import delete

from app.accounts import (
    commit_or_conflict,
    create_user,
    end_session_cookie,
    ensure_email_free,
    get_user_by_email,
    revoke_sessions,
    start_session,
)
from app.deps import CurrentSession, CurrentUser, DbSession, SettingsDep
from app.models import AuthSession, User
from app.schemas import LoginIn, PasswordChange, ProfileUpdate, SignupIn, UserOut
from app.security import hash_password, hash_token, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/signup", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def sign_up(payload: SignupIn, response: Response, db: DbSession, settings: SettingsDep) -> User:
    """Create a regular account and sign it in. Administrators are only made by administrators."""
    ensure_email_free(db, payload.email)
    user = create_user(
        db, email=payload.email, full_name=payload.full_name, password=payload.password
    )
    start_session(db, user, response, settings)
    commit_or_conflict(db)
    return user


@router.post("/login", response_model=UserOut)
def sign_in(payload: LoginIn, response: Response, db: DbSession, settings: SettingsDep) -> User:
    user = get_user_by_email(db, payload.email)
    # Always verify, even for an unknown email, and answer both failures the same way.
    if not verify_password(payload.password, user.password_hash if user else None) or not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    if not user.is_active:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "This account has been deactivated. Contact an administrator.",
        )
    start_session(db, user, response, settings)
    db.commit()
    return user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def sign_out(request: Request, response: Response, db: DbSession, settings: SettingsDep) -> None:
    """Always succeeds, so that an expired session can still be cleared from the browser."""
    token = request.cookies.get(settings.session_cookie)
    if token:
        db.execute(delete(AuthSession).where(AuthSession.token_hash == hash_token(token)))
        db.commit()
    end_session_cookie(response, settings)


@router.get("/me", response_model=UserOut)
def read_me(user: CurrentUser) -> User:
    return user


@router.patch("/me", response_model=UserOut)
def update_me(payload: ProfileUpdate, user: CurrentUser, db: DbSession) -> User:
    user.full_name = payload.full_name
    db.commit()
    return user


@router.post("/me/password", status_code=status.HTTP_204_NO_CONTENT)
def change_my_password(payload: PasswordChange, session: CurrentSession, db: DbSession) -> None:
    """Change the password and sign out every other browser of this account."""
    user = session.user
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The current password is incorrect")
    user.password_hash = hash_password(payload.new_password)
    revoke_sessions(db, user.id, keep=session)
    db.commit()
