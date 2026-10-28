"""Tujya medicine tables: pills, active ingredients, their interactions, and the sources.

These tables have their own declarative base and their own database (PostgreSQL from
docker-compose.yml, or a SQLite file for development and tests), separate from the
accounts database `app.db`.

    pills ──< pill_ingredients >── ingredients ──< interactions >── ingredients
       └────────────── sources ──────────────┘          (each row cites a source)
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import CheckConstraint, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.db import UTCDateTime, utc_now


class PharmaBase(DeclarativeBase):
    pass


class Source(PharmaBase):
    """Where a row comes from: shown to the user as the citation."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)  # c3pi, ddinter, rxnav...
    title: Mapped[str] = mapped_column(String(255))
    url: Mapped[str] = mapped_column(String(500))
    license: Mapped[str] = mapped_column(String(120))
    citation: Mapped[str] = mapped_column(Text, default="")
    loaded_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now)


class Pill(PharmaBase):
    """One class of the identification model (a C3PI product: NDC9 + part)."""

    __tablename__ = "pills"

    id: Mapped[int] = mapped_column(primary_key=True)
    class_id: Mapped[int] = mapped_column(unique=True, index=True)  # model output index
    class_key: Mapped[str] = mapped_column(String(32), unique=True)
    ndc9: Mapped[str] = mapped_column(String(9), index=True)
    part: Mapped[int] = mapped_column(default=1)
    ndc11_list: Mapped[str] = mapped_column(Text, default="")  # ";"-joined package NDCs
    name: Mapped[str] = mapped_column(String(255))
    labeled_by: Mapped[str] = mapped_column(String(255), default="")
    imprint: Mapped[str] = mapped_column(String(120), default="")
    color: Mapped[str] = mapped_column(String(120), default="")
    shape: Mapped[str] = mapped_column(String(60), default="")
    n_images: Mapped[int] = mapped_column(default=0)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))

    source: Mapped[Source] = relationship()
    ingredients: Mapped[list[PillIngredient]] = relationship(
        back_populates="pill", cascade="all, delete-orphan"
    )


class Ingredient(PharmaBase):
    """An active ingredient (moiety), named in lower case, e.g. "hydrochlorothiazide"."""

    __tablename__ = "ingredients"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    rxcui: Mapped[str | None] = mapped_column(String(16), default=None)
    ddinter_id: Mapped[str | None] = mapped_column(String(32), default=None)


class PillIngredient(PharmaBase):
    __tablename__ = "pill_ingredients"

    pill_id: Mapped[int] = mapped_column(
        ForeignKey("pills.id", ondelete="CASCADE"), primary_key=True
    )
    ingredient_id: Mapped[int] = mapped_column(
        ForeignKey("ingredients.id", ondelete="CASCADE"), primary_key=True
    )
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))

    pill: Mapped[Pill] = relationship(back_populates="ingredients")
    ingredient: Mapped[Ingredient] = relationship()
    source: Mapped[Source] = relationship()


class Interaction(PharmaBase):
    """One interaction between two ingredients, as one source states it.

    The pair is stored once, with the smaller ingredient id first, so a lookup of (a, b)
    and (b, a) finds the same row.
    """

    __tablename__ = "interactions"
    __table_args__ = (
        CheckConstraint("ingredient_a_id < ingredient_b_id", name="ordered_pair"),
        UniqueConstraint("ingredient_a_id", "ingredient_b_id", "source_id", name="one_per_source"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    ingredient_a_id: Mapped[int] = mapped_column(
        ForeignKey("ingredients.id", ondelete="CASCADE"), index=True
    )
    ingredient_b_id: Mapped[int] = mapped_column(
        ForeignKey("ingredients.id", ondelete="CASCADE"), index=True
    )
    severity: Mapped[str] = mapped_column(String(16))  # Major, Moderate, Minor, Unknown
    description: Mapped[str] = mapped_column(Text, default="")
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))

    ingredient_a: Mapped[Ingredient] = relationship(foreign_keys=[ingredient_a_id])
    ingredient_b: Mapped[Ingredient] = relationship(foreign_keys=[ingredient_b_id])
    source: Mapped[Source] = relationship()
