"""GET /api/catalog/summary and /showcase: what the public start and sign-in pages show.

Public, read only, and never anything about an account. Two routes:

- `summary`: the size of the catalogue, counted in the medicine tables loaded by `pharma_loader`
  (pill classes the classifier knows, active ingredients, interaction pairs), so the start
  page never shows a figure typed by hand;
- `showcase`: one real scan of a demo gallery photo (C3PI test split, never seen in
  training: D105) with the served models, and the most severe interaction its active
  ingredient has in DDInter.
  The floating cards of the start page show it; without it they are marked as an example.

Both are kept in memory (counts for ten minutes, the scan per published model version), so
the route cannot be used to load the server: at most one inference per model version.
Decision D128.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from fastapi import APIRouter, Request
from sqlalchemy import func, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.deps import PharmaSession, SettingsDep
from app.gallery import DemoPhoto, load_gallery
from app.inference import ScannerHolder, decode_image
from app.medicines import DISCLAIMER, SEVERITY_ORDER, PharmaUnavailable, pill_infos
from app.pharma import Ingredient, Interaction, Pill, PillIngredient
from app.schemas import (
    CatalogCounts,
    CatalogShowcase,
    CatalogSummary,
    ShowcaseInteraction,
    ShowcaseOut,
)

router = APIRouter(prefix="/api/catalog", tags=["catalog"])

COUNTS_TTL = 600.0  # seconds
SHOWCASE_PHOTOS = 8  # gallery photos tried, in list order, to find one identified correctly


@dataclass
class _Cache:
    lock: threading.Lock
    showcase_lock: threading.Lock
    counts: CatalogCounts | None = None
    counted_at: float = 0.0
    showcase_key: str | None = None
    showcase: CatalogShowcase | None = None


def _cache(request: Request) -> _Cache:
    state = request.app.state
    if not hasattr(state, "catalog_cache"):
        state.catalog_cache = _Cache(threading.Lock(), threading.Lock())
    cache: _Cache = state.catalog_cache
    return cache


def count_catalog(db: Session) -> CatalogCounts | None:
    """Rows of the medicine tables; None when they are not loaded."""
    try:
        pills = db.scalar(select(func.count()).select_from(Pill)) or 0
        ingredients = db.scalar(select(func.count()).select_from(Ingredient)) or 0
        # One row per pair and source; DDInter is the only interaction source, so this is
        # the number of pairs. Counted as distinct pairs anyway, in case a source is added.
        pairs = select(Interaction.ingredient_a_id, Interaction.ingredient_b_id).distinct()
        interactions = db.scalar(select(func.count()).select_from(pairs.subquery())) or 0
    except SQLAlchemyError:
        return None
    if pills == 0:
        return None
    return CatalogCounts(pills=pills, ingredients=ingredients, interaction_pairs=interactions)


def worst_interaction(db: Session, ingredient: str) -> ShowcaseInteraction | None:
    """The most severe interaction of one ingredient in the tables (then by partner name)."""
    own = db.scalar(select(Ingredient).where(Ingredient.name == ingredient))
    if own is None:
        return None
    rows = db.scalars(
        select(Interaction).where(
            or_(Interaction.ingredient_a_id == own.id, Interaction.ingredient_b_id == own.id)
        )
    ).all()
    if not rows:
        return None
    partners = {
        i.id: i.name
        for i in db.scalars(
            select(Ingredient).where(
                Ingredient.id.in_(
                    {
                        r.ingredient_b_id if r.ingredient_a_id == own.id else r.ingredient_a_id
                        for r in rows
                    }
                )
            )
        )
    }

    def other(row: Interaction) -> str:
        return partners[
            row.ingredient_b_id if row.ingredient_a_id == own.id else row.ingredient_a_id
        ]

    best = min(rows, key=lambda r: (SEVERITY_ORDER.get(r.severity, 9), other(r)))
    return ShowcaseInteraction(
        ingredient_a=ingredient, ingredient_b=other(best), severity=best.severity, source="DDInter"
    )


def _has_ingredients(db: Session) -> bool:
    return db.scalar(select(PillIngredient.pill_id).limit(1)) is not None


def _has_ingredients_safe(db: Session) -> bool:
    try:
        return _has_ingredients(db)
    except SQLAlchemyError:
        return False


def find_showcase(
    holder: ScannerHolder, photos: list[DemoPhoto], db: Session, settings: Settings
) -> CatalogShowcase | None:
    """The first gallery photo with one pill, identified as its true class above the
    confirmation threshold; None when the models, the photos or the tables are missing."""
    try:
        scanner = holder.get()
    except Exception:  # missing or unreadable model files: the cards say "example"
        return None
    for index, photo in enumerate(photos[:SHOWCASE_PHOTOS]):
        image = decode_image(photo.file.read_bytes(), max_pixels=settings.scan_max_pixels)
        output = scanner.scan(image)
        if len(output.detections) != 1:
            continue
        detection = output.detections[0]
        top, probability = detection.classes[0], detection.probabilities[0]
        if top != photo.class_id or probability < settings.scan_confirm_threshold:
            continue
        try:
            info = pill_infos(db, [top]).get(top)
        except PharmaUnavailable:
            return None
        if info is None or not info.ingredients:
            continue
        interaction = worst_interaction(db, info.ingredients[0])
        classify = output.models.get("classify") or {}
        return CatalogShowcase(
            photo_id=index,
            pill_name=info.name,
            active_ingredients=info.ingredients,
            probability=round(float(probability), 4),
            model=str(classify.get("registered_model") or "classifier"),
            model_version=classify.get("version"),
            interaction=interaction,
        )
    return None


def _model_key(holder: ScannerHolder) -> str | None:
    try:
        manifest = holder.get().manifest
    except Exception:
        return None
    classify = manifest.get("classify") or {}
    detect = manifest.get("detect") or {}
    return f"{manifest.get('published_at')}|{classify.get('version')}|{detect.get('version')}"


@router.get("/summary", response_model=CatalogSummary)
async def summary(request: Request, settings: SettingsDep, db: PharmaSession) -> CatalogSummary:
    """Catalogue counts for the public pages, counted in the medicine tables.

    `counts` is null when the tables are not loaded: the app then shows no figure rather
    than an invented one. Nothing here depends on who asks.
    """
    cache = _cache(request)

    def compute() -> CatalogSummary:
        with cache.lock:
            now = time.monotonic()
            if cache.counts is None or now - cache.counted_at > COUNTS_TTL:
                cache.counts = count_catalog(db)  # tables not loaded yet: asked again next time
                cache.counted_at = now
            return CatalogSummary(
                counts=cache.counts,
                confirm_threshold=settings.scan_confirm_threshold,
                disclaimer=DISCLAIMER,
            )

    return await run_in_threadpool(compute)


@router.get("/showcase", response_model=ShowcaseOut)
async def showcase(request: Request, settings: SettingsDep, db: PharmaSession) -> ShowcaseOut:
    """One real scan of a C3PI test photo with the served models, for the start page cards.

    `showcase` is null when the models, the demo photos or the medicine tables are missing;
    the app then marks its cards as an example. Computed once per published model version
    (the first call may take a few seconds while the models load).
    """
    cache = _cache(request)
    holder: ScannerHolder = request.app.state.scanner

    def compute() -> ShowcaseOut:
        with cache.showcase_lock:
            model = _model_key(holder)
            tables = _has_ingredients_safe(db)
            # A scan found while the tables were missing is computed again once they are in.
            key = None if model is None else f"{model}|{tables}"
            if key is None:
                cache.showcase_key, cache.showcase = None, None
            elif key != cache.showcase_key:
                photos = load_gallery(
                    settings.demo_gallery_file,
                    settings.demo_split_file,
                    settings.demo_holdout_file,
                    settings.demo_images_dir,
                    settings.demo_gradcam_dir,
                )
                cache.showcase = find_showcase(holder, photos, db, settings) if tables else None
                cache.showcase_key = key
            return ShowcaseOut(
                showcase=cache.showcase, confirm_threshold=settings.scan_confirm_threshold
            )

    return await run_in_threadpool(compute)
