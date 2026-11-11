"""GET /api/documents: the documents indexed for the RAG, their metadata and chunks (D114).

The same chunks the agent searches (Qdrant collection tujya_chunks by default), grouped
into documents (src/rag/documents.py). Filters: type, ingredient, section, language,
source, text query. Sign-in required, as for the chat that cites them.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.deps import get_current_user
from app.schemas import (
    DocumentChunkOut,
    DocumentDetail,
    DocumentOut,
    DocumentsPage,
    IndexInfoOut,
)
from src.agent.runtime import AgentHolder
from src.agent.safety import redact_doses
from src.rag.documents import Document
from src.rag.retrieval import Retriever
from src.rag.stores import STORE_ERRORS

router = APIRouter(
    prefix="/api/documents", tags=["documents"], dependencies=[Depends(get_current_user)]
)


def get_retriever(request: Request) -> Retriever:
    holder: AgentHolder = request.app.state.agents
    try:
        retriever = holder.toolbox().retrievers.get()
    except STORE_ERRORS as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"The vector store is not reachable ({type(exc).__name__}). Is the qdrant service "
            "running? See docs/infra.md.",
        ) from exc
    if retriever is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The document index is empty. Run `python -m src.rag.index` (docs/module2/rag.md).",
        )
    return retriever


def index_info(request: Request, retriever: Retriever) -> IndexInfoOut:
    store = retriever.store
    meta, info = store.get_meta(), store.info()
    catalog = retriever.catalog
    settings = request.app.state.settings
    return IndexInfoOut(
        store=store.kind,
        collection=info["collection"],
        points=info["points"],
        embedding_model=meta.get("model", ""),
        dim=int(meta.get("dim", store.dim)),
        indexed_at=meta.get("indexed_at", ""),
        documents=len(catalog.documents),
        chunks=catalog.chunks,
        payload_indexes=info["payload_indexes"],
        dashboard_url=settings.qdrant_dashboard_url if store.kind == "qdrant" else None,
    )


def summary(document: Document) -> DocumentOut:
    return DocumentOut(
        doc_id=document.doc_id,
        doc_type=document.doc_type,
        title=document.title,
        source_name=document.source_name,
        url=document.url,
        date=document.date,
        version=document.version,
        license=document.license,
        language=document.language,
        article_type=document.article_type,
        is_retracted=document.is_retracted,
        ingredients=list(document.ingredients),
        mentioned_ingredients=list(document.mentioned_ingredients),
        sections=list(document.sections),
        chunk_count=len(document.chunks),
    )


@router.get(
    "",
    response_model=DocumentsPage,
    responses={503: {"description": "Vector store not reachable or index empty"}},
)
def list_documents(
    request: Request,
    doc_type: Annotated[str | None, Query(description="label, abstract or interaction")] = None,
    ingredient: Annotated[str | None, Query(description="about or mentioning it")] = None,
    section: str | None = None,
    language: str | None = None,
    source: Annotated[str | None, Query(description="openFDA drug label, PubMed, ...")] = None,
    q: Annotated[str | None, Query(max_length=200, description="words of the title")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentsPage:
    """The indexed documents with their metadata, filtered and paginated."""
    retriever = get_retriever(request)
    found = retriever.catalog.search(
        doc_type=doc_type,
        ingredient=ingredient.strip().lower() if ingredient else None,
        section=section,
        language=language,
        source=source,
        query=q,
    )
    return DocumentsPage(
        total=len(found),
        offset=offset,
        limit=limit,
        items=[summary(d) for d in found[offset : offset + limit]],
        facets=retriever.catalog.facets(found),
        index=index_info(request, retriever),
    )


@router.get(
    "/{doc_id:path}",
    response_model=DocumentDetail,
    responses={404: {"description": "No such document"}, 503: {"description": "No index"}},
)
def get_document(doc_id: str, request: Request) -> DocumentDetail:
    """One document: its metadata and every chunk as indexed (doses shown as [...])."""
    document = get_retriever(request).catalog.get(doc_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No indexed document {doc_id!r}")
    return DocumentDetail(
        **summary(document).model_dump(),
        chunks=[
            DocumentChunkOut(
                chunk_id=c.chunk_id,
                chunk_index=c.chunk_index,
                section=c.section,
                title=c.title,
                text=redact_doses(c.text),
                content_hash=content_hash,
            )
            for c, content_hash in zip(document.chunks, document.hashes, strict=True)
        ],
    )
