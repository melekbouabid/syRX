"""Request and response bodies."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    StringConstraints,
    model_validator,
)

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


# --- Tujya: pills, scan, interactions -------------------------------------------------------
# Every answer about medicines carries its sources and the disclaimer (CLAUDE.md safety rules).

IngredientName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=2, max_length=100)
]
ClassId = Annotated[int, Field(ge=0, le=9999)]


class SourceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    title: str
    url: str
    license: str
    citation: str


class PillOut(BaseModel):
    """One identification class: a C3PI product (NDC9 + part) and its active ingredients."""

    class_id: int
    name: str
    generic_name: str
    labeled_by: str
    imprint: str
    color: str
    shape: str
    active_ingredients: list[str]


class PillList(BaseModel):
    items: list[PillOut]
    total: int
    sources: list[SourceOut]


class PillCandidate(BaseModel):
    class_id: int
    name: str
    generic_name: str
    active_ingredients: list[str]
    probability: float = Field(ge=0, le=1)


class PixelBox(BaseModel):
    """x1, y1 top-left and x2, y2 bottom-right, in pixels of the uploaded image."""

    x1: float
    y1: float
    x2: float
    y2: float


class NormalizedBox(BaseModel):
    """Same box divided by the image width (x) and height (y): values between 0 and 1."""

    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)


class ScannedPill(BaseModel):
    box: PixelBox
    box_normalized: NormalizedBox
    detection_confidence: float = Field(ge=0, le=1)
    candidates: list[PillCandidate] = Field(description="Top-5 classes, most probable first")
    needs_confirmation: bool = Field(
        description="True when the top-1 probability is below confirm_threshold"
    )


class ScanTimings(BaseModel):
    detection_ms: float
    classification_ms: float
    total_ms: float


class ScanResult(BaseModel):
    image_width: int
    image_height: int
    inference_ms: float = Field(description="Detection plus identification, ONNX Runtime")
    timings: ScanTimings
    confirm_threshold: float
    pills: list[ScannedPill]
    message: str | None = None
    models: dict[str, Any]
    sources: list[SourceOut]
    disclaimer: str


class InteractionsIn(BaseModel):
    """At least two things to compare, pill classes (from a scan) and/or ingredient names."""

    pills: list[ClassId] = Field(default_factory=list, max_length=10)
    ingredients: list[IngredientName] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def _two_to_ten(self) -> InteractionsIn:
        count = len(self.pills) + len(self.ingredients)
        if not 2 <= count <= 10:
            raise ValueError("give between 2 and 10 pills and ingredients in total")
        return self


class CheckedItem(BaseModel):
    kind: Literal["pill", "ingredient"]
    label: str
    class_id: int | None
    found: bool = Field(description="False when this database does not know the item")
    active_ingredients: list[str]


class InteractionOut(BaseModel):
    ingredient_a: str
    ingredient_b: str
    severity: str = Field(description="DDInter level: Major, Moderate, Minor or Unknown")
    description: str
    source: SourceOut


class PairOut(BaseModel):
    item_a: CheckedItem
    item_b: CheckedItem
    highest_severity: str | None
    interactions: list[InteractionOut]
    shared_ingredients: list[str] = Field(
        description="Active ingredients present in both items (same substance twice)"
    )
    message: str | None = Field(
        description="Set when nothing is listed: no known interaction, or nothing to check"
    )


class InteractionsResult(BaseModel):
    items: list[CheckedItem]
    pairs: list[PairOut]
    highest_severity: str | None
    message: str
    sources: list[SourceOut]
    disclaimer: str


class DemoImage(BaseModel):
    id: int
    class_id: int = Field(description="Class the photo shows (C3PI label), to compare with")
    expected_name: str
    why: str = Field(description="Why it is in the gallery: major, confirm, no-ingredient...")
    url: str
    thumbnail_url: str


class DemoGallery(BaseModel):
    items: list[DemoImage]
    note: str
