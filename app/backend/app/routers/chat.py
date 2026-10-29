"""POST /api/chat: a question about medicines, answered by the Tujya agent with citations.

`stream: false` returns one JSON `ChatResult`. `stream: true` returns Server-Sent Events:
`tool` (each tool call, as it happens), `token` (answer text, sentence by sentence, after
the sentence-level safety filter), then `final` (the checked `ChatResult`, which is the
authoritative answer) or `error`. Sign-in required.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.deps import get_current_user
from app.medicines import DISCLAIMER, PharmaUnavailable
from app.rag.agent import AgentState, TujyaAgent
from app.rag.embeddings import EmbeddingError
from app.rag.runtime import AgentHolder
from app.rag.safety import redact_doses
from app.schemas import ChatIn, ChatResult, CitationOut, ToolCallOut

router = APIRouter(prefix="/api", tags=["chat"], dependencies=[Depends(get_current_user)])


def get_agent(request: Request) -> tuple[AgentHolder, TujyaAgent]:
    holder: AgentHolder = request.app.state.agents
    agent = holder.agent()
    if holder.toolbox().retrievers.get() is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The document index is empty. Run `python -m app.rag.index` (docs/RAG.md).",
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
            )
            for s in state.get("sources", [])
        ],
        tool_calls=[ToolCallOut(**c) for c in state.get("tool_calls", [])],
        safety=state.get("safety", {}),
        model=model,
        timings=timings,
        disclaimer=DISCLAIMER,
    )


def unavailable(exc: Exception) -> HTTPException:
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        f"The agent is not available ({type(exc).__name__}). Is the ollama service running? "
        "See docs/INFRA.md.",
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
async def chat(payload: ChatIn, request: Request) -> Any:
    """Answer a question about medicines from openFDA labels, PubMed and DDInter, with
    numbered citations. Never gives a dose, never calls a combination safe."""
    started = time.perf_counter()
    try:
        holder, agent = get_agent(request)
    except (PharmaUnavailable, EmbeddingError, OSError) as exc:
        raise unavailable(exc) from exc

    if not payload.stream:
        try:
            state = await agent.run(payload.question, payload.pills)
        except (EmbeddingError, PharmaUnavailable, ConnectionError, OSError) as exc:
            raise unavailable(exc) from exc
        return to_result(state, holder.model_name, started)

    async def events() -> AsyncIterator[str]:
        try:
            async for event in agent.stream(payload.question, payload.pills):
                if event["type"] == "final":
                    result = to_result(event["state"], holder.model_name, started)
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
