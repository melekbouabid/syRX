"""Database tables: accounts, their sign-in sessions, their profile, the activity log that
administrators read, and what each user keeps (saved medicines and reports).

The new tables never change `users`: the project creates its tables with `create_all` and has
no migrations, so an existing `app.db` gains the new tables at the next start and keeps working.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, ForeignKey, LargeBinary, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base, UTCDateTime, utc_now


class Role(StrEnum):
    ADMIN = "admin"
    USER = "user"


class User(Base):
    __tablename__ = "users"
    # Never hand the id of a deleted account to a new one: an administrator's open
    # "edit user 7" dialog must not end up editing somebody else.
    __table_args__ = ({"sqlite_autoincrement": True},)

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)  # lower case
    full_name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default=Role.USER.value)  # a Role value
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)
    last_login_at: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)

    sessions: Mapped[list[AuthSession]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    # Loaded with the account (one extra query per list, not per row); the images inside it
    # are deferred and read only by the image routes.
    profile: Mapped[Profile | None] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="selectin", passive_deletes=True
    )
    # Deleting an account deletes its events, its saved medicines and its reports. What an
    # administrator did to it stays on record: those events belong to the administrator.
    events: Mapped[list[ActivityEvent]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
    saved_medicines: Mapped[list[SavedMedicine]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
    reports: Mapped[list[Report]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )

    @property
    def phone(self) -> str | None:
        return self.profile.phone if self.profile else None

    @property
    def photo_url(self) -> str | None:
        return _image_url(self.id, "photo", self.profile.photo_at if self.profile else None)

    @property
    def signature_url(self) -> str | None:
        stamp = self.profile.signature_at if self.profile else None
        return _image_url(self.id, "signature", stamp)


def _image_url(user_id: int, kind: str, stamp: datetime | None) -> str | None:
    """Address of a stored image. It changes with each upload, so browsers may cache it."""
    if stamp is None:
        return None
    return f"/api/users/{user_id}/{kind}?v={int(stamp.timestamp() * 1000)}"


class AuthSession(Base):
    """One signed-in browser. Only the SHA-256 of the cookie value is stored."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime)

    user: Mapped[User] = relationship(back_populates="sessions")


class Profile(Base):
    """What a user adds to their account: phone, photo (JPEG) and signature (PNG).

    Both images are decoded and re-encoded by the server (app/images.py) before they are
    stored, so a file is never kept as it was uploaded.
    """

    __tablename__ = "profiles"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    phone: Mapped[str | None] = mapped_column(String(32), default=None)
    photo: Mapped[bytes | None] = mapped_column(LargeBinary, deferred=True, default=None)
    photo_at: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)
    signature: Mapped[bytes | None] = mapped_column(LargeBinary, deferred=True, default=None)
    signature_at: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)

    user: Mapped[User] = relationship(back_populates="profile")


class Action(StrEnum):
    """What an activity event records. The admin pages group them (app/activity.py GROUPS)."""

    SIGN_UP = "sign_up"
    SIGN_IN = "sign_in"
    SIGN_IN_FAILED = "sign_in_failed"  # wrong password for an existing account
    SIGN_IN_BLOCKED = "sign_in_blocked"  # right password, deactivated account
    SIGN_OUT = "sign_out"
    PROFILE_UPDATE = "profile_update"
    PHOTO_UPDATE = "photo_update"
    SIGNATURE_UPDATE = "signature_update"
    PASSWORD_CHANGE = "password_change"  # noqa: S105  an action name, not a password
    SCAN = "scan"
    INTERACTIONS = "interactions"
    CHAT = "chat"
    TOXICITY = "toxicity"
    MEDICINE_SAVE = "medicine_save"
    REPORT_CREATE = "report_create"
    REPORT_SHARE = "report_share"
    REPORT_UNSHARE = "report_unshare"
    REPORT_DELETE = "report_delete"
    # Done by an administrator to another account: the target's id and email are in the event.
    USER_CREATE = "user_create"
    USER_UPDATE = "user_update"
    PASSWORD_RESET = "password_reset"  # noqa: S105  an action name, not a password
    USER_DELETE = "user_delete"
    DRIFT_RESET = "drift_reset"
    DATA_EXPORT = "data_export"


class ActivityEvent(Base):
    """One thing an account did, for the administrators' activity log.

    What happened and when, with counts only: never a photo, a pill or medicine name, or the
    text of a question. Those are health data and stay with the user (docs/project_rules.md).
    """

    __tablename__ = "activity_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    action: Mapped[str] = mapped_column(String(32), index=True)  # an Action value
    # The account an administrator acted on; no foreign key, so the event outlives a deletion.
    target_id: Mapped[int | None] = mapped_column(default=None, index=True)
    target_email: Mapped[str | None] = mapped_column(String(254), default=None)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    ip: Mapped[str | None] = mapped_column(String(45), default=None)
    user_agent: Mapped[str | None] = mapped_column(String(300), default=None)

    user: Mapped[User] = relationship(back_populates="events")


class SavedMedicine(Base):
    """A pill class in the user's "My medicines" list. Only its owner ever reads it.

    Only the class_id is kept: name and ingredients are read from the medicine tables, so they
    always match the current data.
    """

    __tablename__ = "saved_medicines"
    __table_args__ = (UniqueConstraint("user_id", "class_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    class_id: Mapped[int]
    # The model's probability when confirmed from a scan; null when picked from the list.
    probability: Mapped[float | None] = mapped_column(default=None)
    added_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)

    user: Mapped[User] = relationship(back_populates="saved_medicines")


class Report(Base):
    """A saved interaction report, kept as it was on the day it was made.

    `content` is a snapshot built by the server (pills, interaction check, sealed chat
    answers), so a report never changes when the medicine tables do. A share link is stored
    as the SHA-256 of its token, like a session cookie: the token is shown once, to the owner.
    """

    __tablename__ = "reports"
    # Never reuse the id of a deleted report: an old browser tab must not open another one.
    __table_args__ = ({"sqlite_autoincrement": True},)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, index=True)
    title: Mapped[str] = mapped_column(String(120))
    content: Mapped[dict[str, Any]] = mapped_column(JSON)
    share_hash: Mapped[str | None] = mapped_column(String(64), unique=True, default=None)
    share_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)
    share_views: Mapped[int] = mapped_column(default=0)

    user: Mapped[User] = relationship(back_populates="reports")
