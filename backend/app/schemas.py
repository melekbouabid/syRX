"""Request and response bodies."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, EmailStr, Field, StringConstraints

from app.models import Role

# Emails are compared case-insensitively, so they are stored in lower case.
Email = Annotated[EmailStr, AfterValidator(str.lower)]
Password = Annotated[str, StringConstraints(min_length=8, max_length=128)]
FullName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    full_name: str
    role: Role
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None


# --- sign in, sign up, own account ----------------------------------------------------------


class SignupIn(BaseModel):
    email: Email
    full_name: FullName
    password: Password


class LoginIn(BaseModel):
    # No format or length rule here: a wrong value is simply a failed sign-in.
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)


class ProfileUpdate(BaseModel):
    full_name: FullName


class PasswordChange(BaseModel):
    current_password: str = Field(max_length=128)
    new_password: Password


# --- administration -------------------------------------------------------------------------


class AdminUserCreate(BaseModel):
    email: Email
    full_name: FullName
    password: Password
    role: Role = Role.USER
    is_active: bool = True


class AdminUserUpdate(BaseModel):
    email: Email | None = None
    full_name: FullName | None = None
    role: Role | None = None
    is_active: bool | None = None


class AdminPasswordReset(BaseModel):
    new_password: Password


class UserPage(BaseModel):
    items: list[UserOut]
    total: int
    page: int
    page_size: int


class AdminStats(BaseModel):
    users_total: int
    users_active: int
    users_inactive: int
    admins: int
    signups_last_7_days: int
    active_sessions: int


# --- catalog --------------------------------------------------------------------------------


class CatalogSummary(BaseModel):
    """Row counts of the processed tables; null when a table has not been built yet."""

    drugs: int | None
    cell_lines: int | None
    synergy_measurements: int | None
    abstracts: int | None


class CatalogPage(BaseModel):
    items: list[dict[str, Any]]
    total: int
    page: int
    page_size: int
