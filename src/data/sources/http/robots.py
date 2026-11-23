"""robots.txt handling following RFC 9309 (wildcards, `$`, longest match wins).

The standard-library parser does plain prefix matching in file order, which gets real files
wrong: Zenodo disallows `/api` but allows `/api/records/*/files`.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

RobotsFetcher = Callable[[str], Awaitable[tuple[int | None, str]]]


@dataclass
class _Group:
    agents: list[str] = field(default_factory=list)
    rules: list[tuple[bool, str]] = field(default_factory=list)
    crawl_delay: float | None = None


def _compile(pattern: str) -> re.Pattern[str]:
    anchored = pattern.endswith("$")
    body = pattern[:-1] if anchored else pattern
    regex = "".join(".*" if ch == "*" else re.escape(ch) for ch in body)
    return re.compile(regex + ("$" if anchored else ""))


class RobotsPolicy:
    """Parsed robots.txt for one origin."""

    def __init__(
        self,
        text: str = "",
        *,
        allow_all: bool = False,
        disallow_all: bool = False,
        unreachable: bool = False,
    ):
        self.allow_all = allow_all
        self.disallow_all = disallow_all
        # True when robots.txt itself could not be fetched (as opposed to it saying no).
        self.unreachable = unreachable
        self._groups: list[_Group] = [] if (allow_all or disallow_all) else self._parse(text)

    @classmethod
    def from_response(cls, status: int | None, text: str) -> RobotsPolicy:
        """RFC 9309 section 2.3.1: 4xx means no restrictions, 5xx / unreachable means stop."""
        if status is None or status >= 500:
            return cls(disallow_all=True, unreachable=True)
        if status >= 400:
            return cls(allow_all=True)
        return cls(text)

    @staticmethod
    def _parse(text: str) -> list[_Group]:
        groups: list[_Group] = []
        current: _Group | None = None
        last_was_agent = False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip()
            if key == "user-agent":
                if current is None or not last_was_agent:
                    current = _Group()
                    groups.append(current)
                current.agents.append(value.lower())
                last_was_agent = True
                continue
            last_was_agent = False
            if current is None:
                continue
            if key in ("allow", "disallow"):
                if value:
                    current.rules.append((key == "allow", value))
            elif key == "crawl-delay":
                try:
                    current.crawl_delay = float(value)
                except ValueError:
                    continue
        return groups

    def _groups_for(self, agent: str) -> list[_Group]:
        agent = agent.lower()
        specific = [
            (len(name), group)
            for group in self._groups
            for name in group.agents
            if name != "*" and name in agent
        ]
        if specific:
            best = max(length for length, _ in specific)
            return [group for length, group in specific if length == best]
        return [group for group in self._groups if "*" in group.agents]

    def allowed(self, path: str, agent: str) -> bool:
        if self.allow_all or path == "/robots.txt":
            return True
        if self.disallow_all:
            return False
        best_len, verdict = -1, True
        for group in self._groups_for(agent):
            for allow, pattern in group.rules:
                if _compile(pattern).match(path) is None:
                    continue
                # Longest pattern wins; on a tie the less restrictive rule (allow) wins.
                if len(pattern) > best_len or (len(pattern) == best_len and allow):
                    best_len, verdict = len(pattern), allow
        return verdict

    def crawl_delay(self, agent: str) -> float | None:
        delays = [g.crawl_delay for g in self._groups_for(agent) if g.crawl_delay is not None]
        return max(delays) if delays else None


class RobotsChecker:
    """Fetches each origin's robots.txt once per run and answers `allowed(url)`."""

    def __init__(self, fetcher: RobotsFetcher, agent: str) -> None:
        self._fetch = fetcher
        self.agent = agent
        self._policies: dict[str, RobotsPolicy] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def policy(self, url: str) -> RobotsPolicy:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:
            if origin not in self._policies:
                status, text = await self._fetch(f"{origin}/robots.txt")
                self._policies[origin] = RobotsPolicy.from_response(status, text)
        return self._policies[origin]

    async def check(self, url: str) -> tuple[bool, float | None]:
        """Return (allowed, crawl_delay_seconds) for `url`."""
        policy = await self.policy(url)
        parts = urlsplit(url)
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"
        return policy.allowed(path, self.agent), policy.crawl_delay(self.agent)
