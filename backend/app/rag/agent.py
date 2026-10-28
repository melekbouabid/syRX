"""The Tujya agent, a LangGraph graph:

    understand -> tools -> retrieve -> answer -> safety
        \\______________ dose question: refuse ______/

- understand: drug names in the question (French or English) and in the confirmed pills,
  language, and whether the question asks for a dose (refused without any LLM call).
- tools: the LLM chooses tool calls (Ollama tool calling); the calls a question of that
  kind always needs are added when the model forgets them (check_interaction for two or
  more drugs, search_documents always), then every call is run.
- retrieve: the documents found by search_documents (and DDInter rows, label excerpts)
  become the numbered sources [1], [2], ...
- answer: the LLM answers from those sources only, citing them; tokens are streamed
  sentence by sentence through the safety filter.
- safety: the rules of app/rag/safety.py on the whole answer; rewrite or refuse.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.medicines import DISCLAIMER, pill_infos
from app.rag.embeddings import fold
from app.rag.safety import (
    CROSS_REFERENCE,
    REFUSAL_DOSE,
    SentenceFilter,
    check_answer,
    is_dose_question,
)
from app.rag.tools import CHAT_TOOLS, Toolbox, ToolError, langchain_tools

MAX_SOURCES = 8
MAX_TOOL_CALLS = 6
SOURCE_CHARS = 900
FRENCH_HINTS = frozenset(
    [
        "le",
        "la",
        "les",
        "des",
        "est",
        "une",
        "avec",
        "quels",
        "quelles",
        "quel",
        "quelle",
        "peut",
        "entre",
        "interaction",
        "effets",
        "indesirables",
        "prendre",
        "puis",
        "je",
        "mon",
        "ma",
        "mes",
        "et",
        "ou",
        "pour",
        "sur",
        "contre",
    ]
)


class AgentState(TypedDict, total=False):
    question: str
    pills: list[int]
    language: str
    ingredients: list[str]
    pill_labels: list[str]
    dose_question: bool
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    draft: str
    answer: str
    safety: dict[str, Any]
    timings: dict[str, float]


def detect_language(question: str) -> str:
    words = re.findall(r"[a-z]+", fold(question))
    french = sum(1 for w in words if w in FRENCH_HINTS)
    accents = bool(re.search(r"[éèêàùçôîâ]", question.lower()))
    return "fr" if french >= 2 or accents else "en"


TOOLS_PROMPT = """You are the tool planner of Tujya, a medicine-information assistant.
Call the tools needed to answer the user's question about medicines:
- check_interaction when the question is about two or more drugs together;
- get_drug_info for facts about one drug (contraindications, warnings, mechanism);
- search_documents for anything else, with an English query and the drug names as filters.
Use the active ingredient names given. Do not answer the question yourself."""

ANSWER_PROMPT = {
    "fr": """Tu es Tujya, un assistant d'information sur les médicaments. Réponds en français.
Règles absolues :
- Réponds UNIQUEMENT à partir des sources numérotées ci-dessous. Si elles ne répondent pas, dis-le.
- Après chaque phrase, cite la ou les sources qui la soutiennent, par leur numéro : [1], [2].
- Ne donne JAMAIS de dose, de posologie, de nombre de comprimés ou de fréquence de prise.
- N'écris JAMAIS qu'une association est « sûre » ou « sans danger ». S'il n'y a pas d'interaction
  dans les sources, écris : « aucune interaction connue dans cette base de données ».
- Ne donne aucun conseil d'horaire ou d'espacement des prises sauf si une source le dit ; cite-la.
- 3 à 6 phrases courtes, sans titre ni liste : d'abord la réponse directe (par exemple
  la sévérité DDInter), puis ce que disent les notices ou les articles (mécanisme, effets,
  surveillance).""",
    "en": """You are Tujya, a medicine-information assistant. Answer in English.
Absolute rules:
- Answer ONLY from the numbered sources below. If they do not answer, say so.
- After each sentence, cite the source(s) that support it by number: [1], [2].
- NEVER give a dose, a dosage, a number of tablets or how often to take a medicine.
- NEVER write that a combination is "safe". If the sources show no interaction, write:
  "no known interaction in this database".
- Give no timing or spacing advice unless a source says it; cite it.
- 3 to 6 short sentences, no heading, no list: first the direct answer (e.g. the DDInter
  severity), then what the labels or articles say (mechanism, effects, monitoring).""",
}


def _summary(name: str, result: dict[str, Any] | None, error: str | None) -> str:
    if error:
        return f"error: {error}"
    assert result is not None
    if name == "check_interaction":
        return str(result.get("message", ""))
    if name == "search_documents":
        return f"{len(result.get('results', []))} documents"
    if name == "get_drug_info":
        return (
            f"{result.get('ingredient')}: {len(result.get('pill_classes', []))} pill classes, "
            f"{len(result.get('label_sections', []))} label sections"
        )
    return "ok"


def interaction_sources(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Each DDInter interaction found by check_interaction, as a citable source."""
    found = []
    for pair in result.get("pairs", []):
        for row in pair.get("interactions", []):
            source = row.get("source", {})
            found.append(
                {
                    "chunk_id": f"ddinter:{row['ingredient_a']}:{row['ingredient_b']}",
                    "doc_type": "interaction",
                    "title": f"DDInter 2.0: {row['ingredient_a']} + {row['ingredient_b']} "
                    f"({row['severity']})",
                    "section": "interaction",
                    "url": source.get("url", "https://ddinter2.scbdd.com/"),
                    "source_name": source.get("title", "DDInter 2.0"),
                    "date": "",
                    "license": source.get("license", ""),
                    "text": row.get("description", ""),
                    "snippet": row.get("description", ""),
                }
            )
        if not pair.get("interactions") and pair.get("message"):
            a, b = pair["item_a"]["label"], pair["item_b"]["label"]
            found.append(
                {
                    "chunk_id": f"ddinter-none:{a}:{b}",
                    "doc_type": "interaction",
                    "title": f"DDInter 2.0: {a} + {b}",
                    "section": "interaction",
                    "url": "https://ddinter2.scbdd.com/",
                    "source_name": "DDInter 2.0 (Tujya database)",
                    "date": "",
                    "license": "CC BY-NC-SA 4.0",
                    "text": f"{a} + {b}: {pair['message']}",
                    "snippet": f"{a} + {b}: {pair['message']}",
                }
            )
    return found


class TujyaAgent:
    def __init__(self, llm: BaseChatModel, toolbox: Toolbox, *, top_k: int = 6) -> None:
        self.llm, self.toolbox, self.top_k = llm, toolbox, top_k
        self.tools = {t.name: t for t in langchain_tools(toolbox, CHAT_TOOLS)}
        graph = StateGraph(AgentState)
        graph.add_node("understand", self.understand)
        graph.add_node("tools", self.call_tools)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("answer", self.answer)
        graph.add_node("safety", self.safety)
        graph.add_edge(START, "understand")
        graph.add_conditional_edges(
            "understand",
            lambda state: "safety" if state.get("dose_question") else "tools",
            {"safety": "safety", "tools": "tools"},
        )
        graph.add_edge("tools", "retrieve")
        graph.add_edge("retrieve", "answer")
        graph.add_edge("answer", "safety")
        graph.add_edge("safety", END)
        self.graph = graph.compile()

    # -------------------------------------------------------------- nodes

    async def understand(self, state: AgentState) -> AgentState:
        started = time.perf_counter()
        question = state["question"]
        found = self.toolbox.names().find(question)
        labels: list[str] = []
        pills = state.get("pills", [])
        if pills:
            with Session(self.toolbox.pharma_engine) as session:
                infos = pill_infos(session, pills)
            for class_id in pills:
                info = infos.get(class_id)
                if info:
                    labels.append(f"{info.name} ({', '.join(info.ingredients) or '?'})")
                    found += [i for i in info.ingredients if i not in found]
        return {
            "language": detect_language(question),
            "ingredients": found,
            "pill_labels": labels,
            "dose_question": is_dose_question(question),
            "timings": {"understand_ms": round((time.perf_counter() - started) * 1000, 1)},
        }

    async def _planned_calls(self, state: AgentState) -> list[dict[str, Any]]:
        ingredients = state.get("ingredients", [])
        context = (
            f"Question: {state['question']}\n"
            f"Active ingredients mentioned or confirmed: {', '.join(ingredients) or 'none'}"
        )
        try:
            planner = self.llm.bind_tools(list(self.tools.values()))
            message = await planner.ainvoke([SystemMessage(TOOLS_PROMPT), HumanMessage(context)])
            calls = getattr(message, "tool_calls", []) or []
        except Exception:
            calls = []
        planned = [
            {"name": c["name"], "args": dict(c.get("args") or {}), "by": "llm"}
            for c in calls
            if c.get("name") in self.tools
        ][:MAX_TOOL_CALLS]
        names = {c["name"] for c in planned}
        if len(ingredients) >= 2 and "check_interaction" not in names:
            planned.append(
                {"name": "check_interaction", "args": {"drugs": ingredients}, "by": "rule"}
            )
        if "search_documents" not in names:
            planned.append(
                {
                    "name": "search_documents",
                    "args": {"query": state["question"], "ingredients": ingredients},
                    "by": "rule",
                }
            )
        return planned

    async def call_tools(self, state: AgentState) -> AgentState:
        started = time.perf_counter()
        writer = get_stream_writer()
        calls, results = [], []
        for call in await self._planned_calls(state):
            tool = self.tools[call["name"]]
            result, error = None, None
            try:
                result = await tool.ainvoke(call["args"])
            except ToolError as exc:
                error = str(exc)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:300]
            record = {
                "name": call["name"],
                "args": call["args"],
                "chosen_by": call["by"],
                "ok": error is None,
                "summary": _summary(call["name"], result, error),
            }
            calls.append(record)
            results.append({"name": call["name"], "result": result})
            writer({"type": "tool", "tool": record})
        timings = {**state.get("timings", {}), "tools_ms": _ms(started)}
        return {"tool_calls": calls, "tool_results": results, "timings": timings}

    async def retrieve(self, state: AgentState) -> AgentState:
        started = time.perf_counter()
        sources: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(items: list[dict[str, Any]], limit: int) -> None:
            for item in items[:limit]:
                if item["chunk_id"] not in seen and len(sources) < MAX_SOURCES:
                    seen.add(item["chunk_id"])
                    sources.append(item)

        results = state.get("tool_results", [])
        for entry in results:
            if entry["name"] == "check_interaction" and entry["result"]:
                add(interaction_sources(entry["result"]), 3)
        for entry in results:
            if entry["name"] == "get_drug_info" and entry["result"]:
                add(entry["result"].get("label_sections", []), 2)
        for entry in results:
            if entry["name"] == "search_documents" and entry["result"]:
                add(entry["result"].get("results", []), self.top_k)
        for number, source in enumerate(sources, start=1):
            source["n"] = number
        timings = {**state.get("timings", {}), "retrieve_ms": _ms(started)}
        return {"sources": sources, "timings": timings}

    def _answer_messages(self, state: AgentState) -> list[Any]:
        language = state.get("language", "fr")
        blocks = []
        for source in state.get("sources", []):
            text = CROSS_REFERENCE.sub("", source.get("text") or source.get("snippet") or "")
            blocks.append(
                f"[{source['n']}] {source['title']} ({source['source_name']}"
                f"{', ' + source['date'] if source.get('date') else ''})\n{text[:SOURCE_CHARS]}"
            )
        pills = state.get("pill_labels", [])
        context = "\n\n".join(blocks)
        user = (
            f"Comprimés confirmés : {'; '.join(pills)}\n" if pills else ""
        ) + f"Sources :\n{context}\n\nQuestion : {state['question']}"
        return [SystemMessage(ANSWER_PROMPT[language]), HumanMessage(user)]

    async def answer(self, state: AgentState) -> AgentState:
        started = time.perf_counter()
        if not state.get("sources"):
            return {"draft": "", "timings": {**state.get("timings", {}), "answer_ms": 0.0}}
        writer = get_stream_writer()
        sentences = SentenceFilter()
        parts: list[str] = []
        async for chunk in self.llm.astream(self._answer_messages(state)):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if not text:
                continue
            parts.append(text)
            for sentence in sentences.feed(text):
                writer({"type": "token", "text": sentence})
        for sentence in sentences.flush():
            writer({"type": "token", "text": sentence})
        timings = {**state.get("timings", {}), "answer_ms": _ms(started)}
        return {"draft": "".join(parts).strip(), "timings": timings}

    async def safety(self, state: AgentState) -> AgentState:
        if state.get("dose_question"):
            answer = f"{REFUSAL_DOSE}\n\n{DISCLAIMER}"
            report = {"action": "refused", "violations": ["dose_question"], "cited": []}
            return {"answer": answer, "safety": report, "sources": [], "tool_calls": []}
        found = any(
            pair.get("interactions")
            for entry in state.get("tool_results", [])
            if entry["name"] == "check_interaction" and entry["result"]
            for pair in entry["result"].get("pairs", [])
        )
        sources = state.get("sources", [])
        checked = check_answer(
            state.get("draft", ""),
            len(sources),
            interaction_found=found,
            source_texts=[f"{s['title']} {s.get('text') or s.get('snippet', '')}" for s in sources],
        )
        return {
            "answer": checked.answer,
            "safety": {
                "action": checked.action,
                "violations": checked.violations,
                "cited": checked.cited,
            },
        }

    # ------------------------------------------------------------ running

    async def run(self, question: str, pills: list[int] | None = None) -> AgentState:
        state: AgentState = {"question": question, "pills": pills or []}
        result: AgentState = await self.graph.ainvoke(state)
        return result

    async def stream(
        self, question: str, pills: list[int] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Events: {"type": "tool"}, {"type": "token"}, then one {"type": "final"}."""
        state: AgentState = {"question": question, "pills": pills or []}
        final: dict[str, Any] = dict(state)
        async for mode, payload in self.graph.astream(state, stream_mode=["custom", "updates"]):
            if mode == "custom":
                yield payload
            else:
                for update in payload.values():
                    if update:
                        final.update(update)
        yield {"type": "final", "state": final}


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def tool_message_json(result: Any) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)[:4000]


__all__ = ["AIMessage", "AgentState", "TujyaAgent", "detect_language"]
