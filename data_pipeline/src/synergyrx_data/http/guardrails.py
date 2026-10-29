"""Compliance guardrails enforced before and after every request.

The client refuses to: talk to a host no connector declared, send credentials or cookies,
pretend to be a browser, forward-spoof an IP address, or carry on past a login wall,
CAPTCHA or bot check. There is deliberately no proxy pool and no way to add one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from synergyrx_data.http.errors import GuardrailViolation
from synergyrx_data.logging_utils import SECRET_PARAMS

BROWSER_MARKERS = (
    "mozilla",
    "chrome/",
    "safari/",
    "applewebkit",
    "gecko",
    "firefox",
    "edg/",
    "opera",
    "msie",
    "trident",
    "googlebot",
    "bingbot",
)

# Headers a connector may never set: sessions, credentials, and address spoofing.
FORBIDDEN_HEADERS = frozenset(
    {
        "cookie",
        "authorization",
        "proxy-authorization",
        "user-agent",
        "x-forwarded-for",
        "forwarded",
        "x-real-ip",
        "client-ip",
        "true-client-ip",
        "x-originating-ip",
        "via",
    }
)

# Text that marks an access wall when HTML arrives where data was expected.
WALL_MARKERS = (
    "captcha",
    "verifying you",
    "not a bot",
    "are you a robot",
    "are you human",
    "cf-chl",
    "challenge-platform",
    "unusual traffic",
    "access denied",
    "please log in",
    "sign in to continue",
    "login required",
    "cookies must be enabled",
)


def build_user_agent(
    product: str, version: str, contact_email: str | None, project_url: str | None = None
) -> str:
    """`product/version (purpose; contact; url)`: says who we are and how to reach us."""
    details = ["research data pipeline"]
    details.append(f"mailto:{contact_email}" if contact_email else "contact not configured")
    if project_url:
        details.append(f"+{project_url}")
    user_agent = f"{product}/{version} ({'; '.join(details)})"
    assert_honest_user_agent(user_agent)
    return user_agent


def assert_honest_user_agent(user_agent: str) -> None:
    lowered = user_agent.lower()
    for marker in BROWSER_MARKERS:
        if marker in lowered:
            raise GuardrailViolation(
                f"User-Agent must not imitate a browser or another crawler (found {marker!r})"
            )


def strip_secret_params(url: str) -> str:
    """Drop secret query parameters; the result is safe to store, hash and cite."""
    parts = urlsplit(url)
    if not parts.query:
        return url
    kept = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in SECRET_PARAMS
    ]
    return urlunsplit(parts._replace(query=urlencode(kept)))


def detect_wall(headers: Mapping[str, str], head: bytes) -> str | None:
    """Return a short reason if the response looks like a CAPTCHA, login or bot check."""
    if headers.get("cf-mitigated", "").lower() == "challenge":
        return "cloudflare challenge"
    text = head[:4096].decode("utf-8", "replace").lower()
    for marker in WALL_MARKERS:
        if marker in text:
            return f"page contains {marker!r}"
    return None


class Guardrails:
    def __init__(self, allowed_hosts: Iterable[str]) -> None:
        self.allowed_hosts = frozenset(h.lower() for h in allowed_hosts)

    def check_url(self, url: str) -> str:
        """Validate a URL before it is requested and return its host."""
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if parts.scheme != "https":
            raise GuardrailViolation(f"only https is allowed, got {parts.scheme!r}", url=url)
        if parts.username or parts.password:
            raise GuardrailViolation("credentials in URLs are not allowed", url=url)
        if host not in self.allowed_hosts:
            raise GuardrailViolation(
                f"host {host!r} is not declared by any connector; this pipeline only talks to "
                "its registered sources",
                url=url,
            )
        return host

    @staticmethod
    def check_redirect(url: str) -> None:
        parts = urlsplit(url)
        if parts.scheme != "https":
            raise GuardrailViolation(f"redirect to non-https URL refused: {parts.scheme}", url=url)
        if parts.username or parts.password:
            raise GuardrailViolation("redirect to a URL with credentials refused", url=url)

    @staticmethod
    def check_headers(headers: Mapping[str, str] | None) -> None:
        for name in headers or {}:
            if name.lower() in FORBIDDEN_HEADERS:
                raise GuardrailViolation(f"header {name!r} may not be set by connectors")
