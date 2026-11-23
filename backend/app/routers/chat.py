"""POST /api/chat: a question about medicines, answered by the agent with citations.

`stream: true` returns Server-Sent Events: `tool`, `token`, then `final` or `error`.
Sign-in required.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app import reports
from app.activity import record
from app.deps import CurrentUser, get_current_user
from app.medicines import DISCLAIMER, PharmaUnavailable
from app.models import Action
from app.schemas import ChatIn, ChatResult, CitationOut, CitationRef, ToolCallOut
from src.agent.graph import AgentState, TujyaAgent
from src.agent.runtime import AgentHolder
from src.agent.safety import redact_doses
from src.rag.init_qdrant import EmbeddingError
from src.rag.stores import STORE_ERRORS

router = APIRouter(prefix="/api", tags=["chat"], dependencies=[Depends(get_current_user)])


def get_agent(request: Request) -> tuple[AgentHolder, TujyaAgent]:
    holder: AgentHolder = request.app.state.agents
    agent = holder.agent()
    if holder.toolbox().retrievers.get() is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The document index is empty. Run `python -m src.rag.init_qdrant` "
            "(docs/module2/README.md).",
        )
    return holder, agent


def to_result(state: AgentState, model: str, started: float) -> ChatResult:
    cited = set(state.get("safety", {}).get("cited", []))
    timings = {
        **state.get("timings", {}),
        "total_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    return ChatResult(
        question=state["question"],
        answer=state.get("answer", ""),
        language=state.get("language", "fr"),
        ingredients=state.get("ingredients", []),
        citations=[
            CitationOut(
                n=s["n"],
                title=s["title"],
                url=s["url"],
                section=s.get("section", ""),
                snippet=redact_doses(s.get("snippet", ""))[:500],
                source_name=s.get("source_name", ""),
                doc_type=s.get("doc_type", ""),
                date=s.get("date", ""),
                cited=s["n"] in cited,
                doc_id=s.get("doc_id", ""),
                license=s.get("license", ""),
                rerank_score=s.get("rerank_score"),
            )
            for s in state.get("sources", [])
        ],
        tool_calls=[ToolCallOut(**c) for c in state.get("tool_calls", [])],
        safety=state.get("safety", {}),
        model=model,
        timings=timings,
        disclaimer=DISCLAIMER,
    )


def sealed(result: ChatResult, key: bytes) -> ChatResult:
    """The result with its seal, so the user can add this exact answer to a report."""
    refs = [CitationRef(n=c.n, title=c.title, url=c.url) for c in result.citations]
    result.seal = reports.seal(key, result.question, result.answer, refs)
    return result


def unavailable(exc: Exception) -> HTTPException:
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        f"The agent is not available ({type(exc).__name__}). Is the ollama service running? "
        "See README.md.",
    )


def _event(name: str, data: Any) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.post(
    "/chat",
    response_model=ChatResult,
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "JSON, or SSE if stream"},
        503: {"description": "Document index, LLM or medicine tables not available"},
    },
)
async def chat(payload: ChatIn, request: Request, user: CurrentUser) -> Any:
    """Answer a question about medicines from openFDA labels, PubMed and DDInter, with
    numbered citations. Never gives a dose, never calls a combination safe."""
    started = time.perf_counter()
    try:
        holder, agent = get_agent(request)
    except (PharmaUnavailable, EmbeddingError, *STORE_ERRORS) as exc:
        raise unavailable(exc) from exc
    key: bytes = request.app.state.report_key
    # Counted, never the question itself (app/activity.py).
    await record(request, user.id, Action.CHAT, stream=payload.stream, pills=len(payload.pills))

    if not payload.stream:
        try:
            state = await agent.run(payload.question, payload.pills)
        except (EmbeddingError, PharmaUnavailable, ConnectionError, *STORE_ERRORS) as exc:
            raise unavailable(exc) from exc
        return sealed(to_result(state, holder.model_name, started), key)

    async def events() -> AsyncIterator[str]:
        try:
            async for event in agent.stream(payload.question, payload.pills):
                if event["type"] == "final":
                    result = sealed(to_result(event["state"], holder.model_name, started), key)
                    yield _event("final", result.model_dump(mode="json"))
                else:
                    yield _event(event["type"], event)
        except Exception as exc:
            yield _event("error", {"detail": f"{type(exc).__name__}: {exc}"[:300]})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
