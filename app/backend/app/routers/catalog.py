"""Drugs and cell lines for any signed-in user; table sizes for everyone."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.catalog import Catalog, TableUnavailable, public, search
from app.deps import get_current_user
from app.schemas import CatalogPage, CatalogSummary

router = APIRouter(
    prefix="/api/catalog", tags=["catalog"], dependencies=[Depends(get_current_user)]
)
# No sign-in needed. Keep this to what the introduction page may show to anyone.
public_router = APIRouter(prefix="/api/public", tags=["public"])

SearchQuery = Annotated[str, Query(max_length=100, description="Words to look for")]
PageNumber = Annotated[int, Query(ge=1)]
PageSize = Annotated[int, Query(ge=1, le=100)]


def get_catalog(request: Request) -> Catalog:
    catalog: Catalog = request.app.state.catalog
    return catalog


CatalogDep = Annotated[Catalog, Depends(get_catalog)]


def _page(rows: list[dict[str, Any]], q: str, page: int, page_size: int) -> CatalogPage:
    matches = search(rows, q)
    start = (page - 1) * page_size
    return CatalogPage(
        items=[public(row) for row in matches[start : start + page_size]],
        total=len(matches),
        page=page,
        page_size=page_size,
    )


def _unavailable(exc: TableUnavailable) -> HTTPException:
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        f"The {exc} table is not available yet. Build it with `synergyrx clean`.",
    )


@public_router.get("/summary", response_model=CatalogSummary)
def read_summary(catalog: CatalogDep) -> CatalogSummary:
    """How many rows each table has. Counts only: reading a row still needs an account."""
    return CatalogSummary(
        drugs=catalog.row_count("drugs"),
        cell_lines=catalog.row_count("cell_lines"),
        synergy_measurements=catalog.row_count("synergy_pairs"),
        abstracts=catalog.row_count("abstracts"),
    )


@router.get("/drugs", response_model=CatalogPage)
def list_drugs(
    catalog: CatalogDep, q: SearchQuery = "", page: PageNumber = 1, page_size: PageSize = 25
) -> CatalogPage:
    try:
        return _page(catalog.drugs(), q, page, page_size)
    except TableUnavailable as exc:
        raise _unavailable(exc) from exc


@router.get("/cell-lines", response_model=CatalogPage)
def list_cell_lines(
    catalog: CatalogDep, q: SearchQuery = "", page: PageNumber = 1, page_size: PageSize = 25
) -> CatalogPage:
    try:
        return _page(catalog.cell_lines(), q, page, page_size)
    except TableUnavailable as exc:
        raise _unavailable(exc) from exc
