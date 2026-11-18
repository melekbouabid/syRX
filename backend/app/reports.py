"""Saved reports: what they hold, their share links, and the seal on chat answers.

A report is built by the server: the pills and ingredients come in as ids and names, and the
interaction check is run again here, so a report (which can be shared) never holds a result
the server did not compute. Chat answers come back from the browser, so each must carry the
seal the server put on it when it answered (`ChatResult.seal`): an edited answer, for
example one that calls a combination safe, is refused.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from datetime import datetime
from typing import Any

from app.medicines import DISCLAIMER
from app.models import Report
from app.schemas import (
    ChatExcerpt,
    CitationRef,
    InteractionsResult,
    ReportPill,
    ShareState,
)
from app.security import hash_token

# ------------------------------------------------------------------ seal on chat answers


def signing_key(configured: str) -> bytes:
    """The configured key, or a random one for this run."""
    return configured.encode("utf-8") if configured else secrets.token_bytes(32)


def _canonical(question: str, answer: str, citations: list[CitationRef]) -> bytes:
    body = {"q": question, "a": answer, "c": [[c.n, c.title, c.url] for c in citations]}
    return json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def seal(key: bytes, question: str, answer: str, citations: list[CitationRef]) -> str:
    return hmac.new(key, _canonical(question, answer, citations), hashlib.sha256).hexdigest()


def seal_matches(key: bytes, excerpt: ChatExcerpt, given: str) -> bool:
    return hmac.compare_digest(
        seal(key, excerpt.question, excerpt.answer, excerpt.citations), given
    )


# ------------------------------------------------------------------ share links


def new_share_token() -> tuple[str, str]:
    """A link token (192 random bits, not guessable) and the hash that is stored."""
    token = secrets.token_urlsafe(24)
    return token, hash_token(token)


def share_state(report: Report, now: datetime) -> ShareState | None:
    if report.share_expires_at is None:
        return None
    active = report.share_hash is not None and report.share_expires_at > now
    return ShareState(active=active, expires_at=report.share_expires_at, views=report.share_views)


# ------------------------------------------------------------------ content


def content(
    pills: list[ReportPill],
    ingredients: list[str],
    interactions: InteractionsResult | None,
    chat: list[ChatExcerpt],
) -> dict[str, Any]:
    return {
        "pills": [p.model_dump() for p in pills],
        "ingredients": ingredients,
        "interactions": interactions.model_dump(mode="json") if interactions else None,
        "chat": [c.model_dump() for c in chat],
    }


def _interactions(report: Report) -> InteractionsResult | None:
    data = report.content.get("interactions")
    return InteractionsResult.model_validate(data) if data else None


def signature_data_url(png: bytes | None) -> str | None:
    if not png:
        return None
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def body(report: Report) -> dict[str, Any]:
    """The fields of ReportBody: the owner's view and the shared one."""
    interactions = _interactions(report)
    profile = report.user.profile
    return {
        "id": report.id,
        "created_at": report.created_at,
        "title": report.title,
        "pills": [ReportPill.model_validate(p) for p in report.content.get("pills", [])],
        "ingredients": list(report.content.get("ingredients", [])),
        "interactions": interactions,
        "chat": [ChatExcerpt.model_validate(c) for c in report.content.get("chat", [])],
        "highest_severity": interactions.highest_severity if interactions else None,
        "author": report.user.full_name,
        "signature": signature_data_url(profile.signature if profile else None),
        "disclaimer": DISCLAIMER,
    }


def summary(report: Report, now: datetime) -> dict[str, Any]:
    interactions = _interactions(report)
    pills = report.content.get("pills", [])
    typed = report.content.get("ingredients", [])
    return {
        "id": report.id,
        "created_at": report.created_at,
        "title": report.title,
        "pill_names": [p["name"] for p in pills],
        "ingredients": sorted({i for p in pills for i in p["active_ingredients"]} | set(typed)),
        "highest_severity": interactions.highest_severity if interactions else None,
        "pairs_with_interaction": (
            sum(1 for pair in interactions.pairs if pair.interactions) if interactions else 0
        ),
        "chat_answers": len(report.content.get("chat", [])),
        "share": share_state(report, now),
    }


def default_title(pills: list[ReportPill], ingredients: list[str]) -> str:
    names = [p.name for p in pills] + ingredients
    title = " + ".join(names[:3]) + (f" + {len(names) - 3}" if len(names) > 3 else "")
    return title[:120] or "Report"
