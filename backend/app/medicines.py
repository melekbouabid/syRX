"""Read access to the Tujya medicine tables (app/pharma.py): pill classes, their active
ingredients, and the interactions between ingredients, each with its source.

Safety wording (CLAUDE.md): a pair absent from the table is "aucune interaction connue dans
cette base de données", never "safe"; DDInter's "Unknown" severity is an interaction whose
level is not graded, and is always reported as an interaction.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import combinations

from sqlalchemy import Select, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from app.pharma import Ingredient, Interaction, Pill, PillIngredient, Source

DISCLAIMER = "Information uniquement, consultez un pharmacien ou un médecin."
NO_KNOWN_INTERACTION = "Aucune interaction connue dans cette base de données."
SEVERITY_ORDER = {"Major": 0, "Moderate": 1, "Minor": 2, "Unknown": 3}


class PharmaUnavailable(RuntimeError):
    """The medicine tables are missing or empty: load them with app.pharma_loader."""


@dataclass(frozen=True)
class PillInfo:
    class_id: int
    name: str
    generic_name: str
    labeled_by: str
    imprint: str
    color: str
    shape: str
    ingredients: list[str]


@dataclass(frozen=True)
class Item:
    """One thing the user asked about: a pill class or an ingredient typed by name."""

    kind: str  # "pill" or "ingredient"
    label: str
    ingredients: list[str]
    found: bool
    class_id: int | None = None


@dataclass(frozen=True)
class Found:
    ingredient_a: str
    ingredient_b: str
    severity: str
    description: str
    source: Source


@dataclass
class PairResult:
    a: Item
    b: Item
    interactions: list[Found] = field(default_factory=list)
    shared_ingredients: list[str] = field(default_factory=list)

    @property
    def message(self) -> str | None:
        if not self.a.found or not self.b.found:
            return (
                "Au moins un des deux éléments est absent de cette base de données : "
                "aucune vérification n'est possible."
            )
        if not self.a.ingredients or not self.b.ingredients:
            return (
                "Aucun principe actif connu pour au moins un des deux comprimés dans cette "
                "base de données : aucune vérification n'est possible."
            )
        if not self.interactions:
            return NO_KNOWN_INTERACTION
        return None


def _info(pill: Pill) -> PillInfo:
    return PillInfo(
        class_id=pill.class_id,
        name=pill.name,
        generic_name=pill.generic_name,
        labeled_by=pill.labeled_by,
        imprint=pill.imprint,
        color=pill.color,
        shape=pill.shape,
        ingredients=sorted(link.ingredient.name for link in pill.ingredients),
    )


def _pill_query() -> Select[tuple[Pill]]:
    return select(Pill).options(
        selectinload(Pill.ingredients).selectinload(PillIngredient.ingredient)
    )


def pill_infos(db: Session, class_ids: Iterable[int]) -> dict[int, PillInfo]:
    """Pill classes by class_id. Raises PharmaUnavailable when the tables are not loaded."""
    wanted = sorted(set(class_ids))
    try:
        pills = db.scalars(_pill_query().where(Pill.class_id.in_(wanted))).all()
        if not pills and db.scalar(select(Pill.id).limit(1)) is None:
            raise PharmaUnavailable("the pills table is empty")
    except SQLAlchemyError as exc:
        raise PharmaUnavailable(str(exc.__class__.__name__)) from exc
    return {pill.class_id: _info(pill) for pill in pills}


def search_pills(db: Session, q: str, limit: int) -> list[PillInfo]:
    """Pill classes whose name, generic name, imprint, maker or ingredient contains every word
    of `q` (case-insensitive); a number alone also matches the class_id. Name order."""
    try:
        query = _pill_query().order_by(Pill.name, Pill.class_id)
        words = q.split()
        for word in words:
            ingredient_match = (
                select(PillIngredient.pill_id)
                .join(Ingredient)
                .where(Ingredient.name.icontains(word, autoescape=True))
            )
            conditions = [
                Pill.name.icontains(word, autoescape=True),
                Pill.generic_name.icontains(word, autoescape=True),
                Pill.imprint.icontains(word, autoescape=True),
                Pill.labeled_by.icontains(word, autoescape=True),
                Pill.id.in_(ingredient_match),
            ]
            if word.isdigit():
                conditions.append(Pill.class_id == int(word))
            query = query.where(or_(*conditions))
        pills = db.scalars(query.limit(limit)).all()
        if not pills and db.scalar(select(Pill.id).limit(1)) is None:
            raise PharmaUnavailable("the pills table is empty")
    except SQLAlchemyError as exc:
        raise PharmaUnavailable(str(exc.__class__.__name__)) from exc
    return [_info(pill) for pill in pills]


def resolve(db: Session, class_ids: list[int], names: list[str]) -> list[Item]:
    """The items to check, in the order given: pills first, then typed ingredient names."""
    infos = pill_infos(db, class_ids)
    items = []
    for class_id in class_ids:
        info = infos.get(class_id)
        if info is None:
            items.append(Item("pill", f"classe {class_id}", [], False, class_id))
        else:
            items.append(Item("pill", info.name, info.ingredients, True, class_id))
    wanted = {name.strip().lower() for name in names}
    try:
        known = set(db.scalars(select(Ingredient.name).where(Ingredient.name.in_(wanted))))
    except SQLAlchemyError as exc:
        raise PharmaUnavailable(str(exc.__class__.__name__)) from exc
    for name in names:
        key = name.strip().lower()
        items.append(Item("ingredient", key, [key] if key in known else [], key in known))
    return items


def check(db: Session, items: list[Item]) -> list[PairResult]:
    """Every pair of items, with every interaction between their ingredients."""
    ingredient_names = {name for item in items for name in item.ingredients}
    try:
        by_name = {
            row.name: row
            for row in db.scalars(select(Ingredient).where(Ingredient.name.in_(ingredient_names)))
        }
        ids = [row.id for row in by_name.values()]
        rows = db.scalars(
            select(Interaction)
            .options(
                selectinload(Interaction.ingredient_a),
                selectinload(Interaction.ingredient_b),
                selectinload(Interaction.source),
            )
            .where(Interaction.ingredient_a_id.in_(ids), Interaction.ingredient_b_id.in_(ids))
        ).all()
    except SQLAlchemyError as exc:
        raise PharmaUnavailable(str(exc.__class__.__name__)) from exc
    table: dict[frozenset[str], list[Interaction]] = {}
    for row in rows:
        table.setdefault(frozenset((row.ingredient_a.name, row.ingredient_b.name)), []).append(row)

    results = []
    for a, b in combinations(items, 2):
        pair = PairResult(a, b, shared_ingredients=sorted(set(a.ingredients) & set(b.ingredients)))
        seen: set[int] = set()  # two items with ingredients in common can reach a row twice
        for name_a in a.ingredients:
            for name_b in b.ingredients:
                for row in table.get(frozenset((name_a, name_b)), []):
                    if row.id in seen:
                        continue
                    seen.add(row.id)
                    pair.interactions.append(
                        Found(name_a, name_b, row.severity, row.description, row.source)
                    )
        pair.interactions.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.ingredient_a))
        results.append(pair)
    results.sort(
        key=lambda p: min((SEVERITY_ORDER.get(f.severity, 9) for f in p.interactions), default=10)
    )
    return results


def sources(db: Session, names: Iterable[str]) -> list[Source]:
    try:
        return list(
            db.scalars(select(Source).where(Source.name.in_(list(names))).order_by(Source.name))
        )
    except SQLAlchemyError as exc:
        raise PharmaUnavailable(str(exc.__class__.__name__)) from exc
