"""Pill search (for manual confirmation in the app) and interaction checks.

Both read the medicine tables loaded by app.pharma_loader; sign-in required.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.deps import PharmaSession, get_current_user
from app.medicines import (
    DISCLAIMER,
    NO_KNOWN_INTERACTION,
    SEVERITY_ORDER,
    Item,
    PharmaUnavailable,
    check,
    resolve,
    search_pills,
    sources,
)
from app.routers.scan import pharma_unavailable
from app.schemas import (
    CheckedItem,
    InteractionOut,
    InteractionsIn,
    InteractionsResult,
    PairOut,
    PillList,
    PillOut,
    SourceOut,
)

router = APIRouter(prefix="/api", tags=["medicines"], dependencies=[Depends(get_current_user)])


def _item(item: Item) -> CheckedItem:
    return CheckedItem(
        kind="pill" if item.kind == "pill" else "ingredient",
        label=item.label,
        class_id=item.class_id,
        found=item.found,
        active_ingredients=item.ingredients,
    )


def _highest(severities: list[str]) -> str | None:
    return min(severities, key=lambda s: SEVERITY_ORDER.get(s, 9), default=None)


def interactions_report(
    session: Session, pill_ids: list[int], ingredients: list[str]
) -> InteractionsResult:
    """The /api/interactions answer; also used by the agent's check_interaction tool."""
    items = resolve(session, pill_ids, ingredients)
    pairs = []
    for pair in check(session, items):
        found = [
            InteractionOut(
                ingredient_a=f.ingredient_a,
                ingredient_b=f.ingredient_b,
                severity=f.severity,
                description=f.description,
                source=SourceOut.model_validate(f.source),
            )
            for f in pair.interactions
        ]
        pairs.append(
            PairOut(
                item_a=_item(pair.a),
                item_b=_item(pair.b),
                highest_severity=_highest([f.severity for f in found]),
                interactions=found,
                shared_ingredients=pair.shared_ingredients,
                message=pair.message,
            )
        )
    highest = _highest([p.highest_severity for p in pairs if p.highest_severity])
    with_interaction = sum(1 for p in pairs if p.interactions)
    if with_interaction:
        message = (
            f"{with_interaction} paire(s) sur {len(pairs)} avec une interaction connue "
            "dans cette base de données."
        )
    else:
        message = NO_KNOWN_INTERACTION
    unchecked = sum(1 for p in pairs if p.message and p.message != NO_KNOWN_INTERACTION)
    if unchecked:
        message += f" {unchecked} paire(s) n'ont pas pu être vérifiées (voir le détail)."
    cited = ["ddinter"] + (["c3pi", "rxnav"] if pill_ids else [])
    return InteractionsResult(
        items=[_item(i) for i in items],
        pairs=pairs,
        highest_severity=highest,
        message=message,
        sources=[SourceOut.model_validate(s) for s in sources(session, cited)],
        disclaimer=DISCLAIMER,
    )


@router.get("/pills", response_model=PillList)
async def list_pills(
    db: PharmaSession,
    q: Annotated[str, Query(max_length=100, description="Name, ingredient, imprint, maker")] = "",
    limit: Annotated[int, Query(ge=1, le=150)] = 20,
) -> PillList:
    """Search the identification classes, e.g. to correct a scan by hand."""

    def run(session: Session) -> PillList:
        found = search_pills(session, q, limit)
        return PillList(
            items=[
                PillOut(
                    class_id=p.class_id,
                    name=p.name,
                    generic_name=p.generic_name,
                    labeled_by=p.labeled_by,
                    imprint=p.imprint,
                    color=p.color,
                    shape=p.shape,
                    active_ingredients=p.ingredients,
                )
                for p in found
            ],
            total=len(found),
            sources=[SourceOut.model_validate(s) for s in sources(session, ["c3pi", "rxnav"])],
        )

    try:
        return await run_in_threadpool(run, db)
    except PharmaUnavailable as exc:
        raise pharma_unavailable() from exc


@router.post("/interactions", response_model=InteractionsResult)
async def check_interactions(payload: InteractionsIn, db: PharmaSession) -> InteractionsResult:
    """Every pair among the pills and ingredients given, with DDInter's severity and source.

    A pair with nothing listed is reported as "aucune interaction connue dans cette base de
    données": it is never called safe. Severity "Unknown" is an interaction whose level
    DDInter does not grade. No dose or timing advice is given.
    """

    try:
        return await run_in_threadpool(interactions_report, db, payload.pills, payload.ingredients)
    except PharmaUnavailable as exc:
        raise pharma_unavailable() from exc
