"""POST /api/toxicity (bonus): Tox21 in vitro assay probabilities for one or two molecules.

"Prédiction d'un modèle sur des tests in vitro, pas une évaluation clinique" (docs/TOXICITY.md).
Each molecule is predicted on its own; nothing is ever said about their combination.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.deps import get_current_user
from app.pharma import Ingredient
from app.rag.names import NameIndex
from app.schemas import ToxicityIn, ToxicityResult
from app.toxicity import ToxHolder, ToxicityUnavailable, predict_molecules

router = APIRouter(prefix="/api", tags=["toxicity"], dependencies=[Depends(get_current_user)])


def _name_resolver(request: Request) -> NameIndex | None:
    """French or English ingredient names (warfarine -> warfarin), if the tables are loaded."""
    cached = getattr(request.app.state, "tox_names", None)
    if cached is not None:
        return cached
    try:
        with Session(request.app.state.pharma_engine) as session:
            index = NameIndex(session.scalars(select(Ingredient.name)).all())
    except Exception:  # tables not loaded: names are then used as given
        return None
    request.app.state.tox_names = index
    return index


@router.post("/toxicity", response_model=ToxicityResult)
async def toxicity(body: ToxicityIn, request: Request) -> ToxicityResult:
    holder: ToxHolder = request.app.state.toxicity
    names = await run_in_threadpool(_name_resolver, request)
    resolve = names.lookup if names else None
    try:
        result = await run_in_threadpool(predict_molecules, holder, body.molecules, resolve)
    except ToxicityUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return ToxicityResult.model_validate(result)
