"""Sign up, sign in, sign out and the signed-in user's own account (details, photo,
signature, password). Every change is written to the activity log (app/activity.py)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile, status
from sqlalchemy import delete, select

from app import images
from app.accounts import (
    commit_or_conflict,
    create_user,
    end_session_cookie,
    ensure_email_free,
    get_user_by_email,
    revoke_sessions,
    start_session,
)
from app.activity import add_event
from app.db import utc_now
from app.deps import CurrentSession, CurrentUser, DbSession, SettingsDep
from app.models import Action, AuthSession, Profile, User
from app.schemas import LoginIn, PasswordChange, ProfileUpdate, SignupIn, UserOut
from app.security import hash_password, hash_token, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])

ImageFile = Annotated[UploadFile, File(description="JPEG, PNG or WebP image, 5 MB at most")]


@router.post("/signup", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def sign_up(
    payload: SignupIn, request: Request, response: Response, db: DbSession, settings: SettingsDep
) -> User:
    """Create a regular account and sign it in. Administrators are only made by administrators."""
    ensure_email_free(db, payload.email)
    user = create_user(
        db, email=payload.email, full_name=payload.full_name, password=payload.password
    )
    start_session(db, user, response, settings)
    add_event(db, user.id, Action.SIGN_UP, request)
    commit_or_conflict(db)
    return user


@router.post("/login", response_model=UserOut)
def sign_in(
    payload: LoginIn, request: Request, response: Response, db: DbSession, settings: SettingsDep
) -> User:
    user = get_user_by_email(db, payload.email)
    # Always verify, even for an unknown email, and answer both failures the same way.
    if not verify_password(payload.password, user.password_hash if user else None) or not user:
        if user is not None:  # an attempt on a real account is worth an administrator's eye
            add_event(db, user.id, Action.SIGN_IN_FAILED, request)
            db.commit()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    if not user.is_active:
        add_event(db, user.id, Action.SIGN_IN_BLOCKED, request)
        db.commit()
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "This account has been deactivated. Contact an administrator.",
        )
    start_session(db, user, response, settings)
    add_event(db, user.id, Action.SIGN_IN, request)
    db.commit()
    return user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def sign_out(request: Request, response: Response, db: DbSession, settings: SettingsDep) -> None:
    """Always succeeds, so that an expired session can still be cleared from the browser."""
    token = request.cookies.get(settings.session_cookie)
    if token:
        user_id = db.scalar(
            select(AuthSession.user_id).where(AuthSession.token_hash == hash_token(token))
        )
        if user_id is not None:
            add_event(db, user_id, Action.SIGN_OUT, request)
        db.execute(delete(AuthSession).where(AuthSession.token_hash == hash_token(token)))
        db.commit()
    end_session_cookie(response, settings)


@router.get("/me", response_model=UserOut)
def read_me(user: CurrentUser) -> User:
    return user


def _profile(user: User) -> Profile:
    if user.profile is None:
        user.profile = Profile(user_id=user.id)
    return user.profile


@router.patch("/me", response_model=UserOut)
def update_me(payload: ProfileUpdate, request: Request, user: CurrentUser, db: DbSession) -> User:
    changed = ["full_name"] if user.full_name != payload.full_name else []
    user.full_name = payload.full_name
    if payload.phone is not None and payload.phone != (user.phone or ""):
        _profile(user).phone = payload.phone or None
        changed.append("phone")
    if changed:
        add_event(db, user.id, Action.PROFILE_UPDATE, request, fields=changed)
    db.commit()
    return user


def _read(upload: UploadFile) -> bytes:
    return upload.file.read(images.MAX_UPLOAD_BYTES + 1)


def _store(user: User, db: DbSession, request: Request, kind: str, data: bytes | None) -> User:
    profile = _profile(user)
    now = utc_now() if data is not None else None
    if kind == "photo":
        profile.photo, profile.photo_at = data, now
        action = Action.PHOTO_UPDATE
    else:
        profile.signature, profile.signature_at = data, now
        action = Action.SIGNATURE_UPDATE
    add_event(db, user.id, action, request, removed=data is None)
    db.commit()
    return user


def _clean(kind: str, data: bytes) -> bytes:
    try:
        return images.profile_photo(data) if kind == "photo" else images.signature(data)
    except images.ImageRejected as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc


@router.put("/me/photo", response_model=UserOut)
def set_my_photo(image: ImageFile, request: Request, user: CurrentUser, db: DbSession) -> User:
    """Square crop, 256 pixels, re-encoded as JPEG (metadata dropped)."""
    return _store(user, db, request, "photo", _clean("photo", _read(image)))


@router.delete("/me/photo", response_model=UserOut)
def delete_my_photo(request: Request, user: CurrentUser, db: DbSession) -> User:
    return _store(user, db, request, "photo", None)


@router.put("/me/signature", response_model=UserOut)
def set_my_signature(image: ImageFile, request: Request, user: CurrentUser, db: DbSession) -> User:
    """The signature printed on the user's reports: ink cropped onto a transparent PNG."""
    return _store(user, db, request, "signature", _clean("signature", _read(image)))


@router.delete("/me/signature", response_model=UserOut)
def delete_my_signature(request: Request, user: CurrentUser, db: DbSession) -> User:
    return _store(user, db, request, "signature", None)


@router.post("/me/password", status_code=status.HTTP_204_NO_CONTENT)
def change_my_password(
    payload: PasswordChange, request: Request, session: CurrentSession, db: DbSession
) -> None:
    """Change the password and sign out every other browser of this account."""
    user = session.user
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The current password is incorrect")
    user.password_hash = hash_password(payload.new_password)
    revoke_sessions(db, user.id, keep=session)
    add_event(db, user.id, Action.PASSWORD_CHANGE, request)
    db.commit()
