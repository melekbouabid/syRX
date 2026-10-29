"""The polite HTTP client every connector goes through.

One request passes, in order: guardrails (declared host, https, no credentials), robots.txt
(HTML and bulk files), the source's circuit breaker, the disk cache, the host's concurrency
semaphore and token bucket, then the network with retries. Responses are size-limited and
their content type is checked; an access wall is reported, never worked around.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar
from urllib.parse import urlencode, urlsplit

import httpx
from tenacity import RetryCallState

from synergyrx_data import __version__
from synergyrx_data.config import HostLimit, Settings
from synergyrx_data.http.cache import HttpCache
from synergyrx_data.http.circuit import CircuitBreaker
from synergyrx_data.http.errors import (
    AccessControlError,
    ContentTypeError,
    GuardrailViolation,
    HttpError,
    IntegrityError,
    NotFoundError,
    ResponseTooLargeError,
    RetryableHttpError,
    RobotsDisallowedError,
)
from synergyrx_data.http.guardrails import (
    Guardrails,
    build_user_agent,
    detect_wall,
    strip_secret_params,
)
from synergyrx_data.http.ratelimit import HostRateLimiter
from synergyrx_data.http.retry import RETRYABLE_STATUS, build_retrying, parse_retry_after
from synergyrx_data.http.robots import RobotsChecker
from synergyrx_data.logging_utils import SECRET_PARAMS, get_logger, log_event
from synergyrx_data.manifest import hash_file

Kind = Literal["api", "bulk", "html"]
T = TypeVar("T")
log = get_logger("http")

# Acceptable Content-Type fragments per expected kind. "binary" accepts anything but HTML.
_KIND_TYPES: dict[str, tuple[str, ...]] = {
    "json": ("json",),
    "xml": ("xml",),
    "csv": ("csv", "text/plain", "octet-stream", "tab-separated"),
    "text": ("text/plain", "text/csv", "tab-separated", "octet-stream"),
    "html": ("html",),
}


@dataclass
class FetchResult:
    url: str
    status: int
    headers: dict[str, str]
    content: bytes
    from_cache: bool = False

    def json(self) -> Any:
        return json.loads(self.content)

    def text(self) -> str:
        return self.content.decode("utf-8", "replace")


@dataclass
class DownloadResult:
    path: Path
    url: str
    size: int
    sha256: str
    md5: str
    etag: str | None = None
    last_modified: str | None = None
    resumed: bool = False


@dataclass
class ProbeResult:
    url: str
    status: int | None
    final_url: str | None = None
    content_type: str | None = None
    note: str = ""


@dataclass
class ClientStats:
    requests: Counter[str] = field(default_factory=Counter)
    cache_hits: int = 0
    revalidated: int = 0
    retries: int = 0
    bytes: int = 0


def check_content(
    expect: tuple[str, ...], headers: Mapping[str, str], head: bytes, url: str
) -> None:
    """Refuse a response that is not the kind of content the caller asked for."""
    content_type = headers.get("content-type", "").lower()
    sniffed_html = head[:256].lstrip().lower().startswith((b"<!doctype html", b"<html"))
    if ("html" in content_type or sniffed_html) and "html" not in expect:
        if reason := detect_wall(headers, head):
            raise AccessControlError(f"access wall instead of data ({reason})", url=url)
        raise ContentTypeError(f"expected {'/'.join(expect)}, got an HTML page", url=url)
    if "binary" in expect or not content_type:
        return
    accepted = tuple(fragment for kind in expect for fragment in _KIND_TYPES.get(kind, ()))
    if not any(fragment in content_type for fragment in accepted):
        raise ContentTypeError(f"expected {'/'.join(expect)}, got {content_type!r}", url=url)


def _total_from_content_range(value: str | None) -> int | None:
    if value and "/" in value:
        total = value.rsplit("/", 1)[1]
        return int(total) if total.isdigit() else None
    return None


class PoliteClient:
    def __init__(
        self,
        settings: Settings,
        *,
        allowed_hosts: Iterable[str],
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: Callable[[], float] = random.random,
    ) -> None:
        http = settings.http
        self.settings = settings
        self._sleep = sleep
        self._clock = clock
        self._rng = rng
        self.user_agent = build_user_agent(
            http.product, __version__, settings.secrets.contact_email, http.project_url
        )
        self.guardrails = Guardrails(allowed_hosts)
        self.limiter = HostRateLimiter(
            http.default_limit,
            self._host_limits(settings),
            safety_factor=http.safety_factor,
            clock=clock,
            sleep=sleep,
        )
        self.cache = HttpCache(settings.cache_dir)
        self.robots = RobotsChecker(self._fetch_robots, http.product)
        self.stats = ClientStats()
        self._breakers: dict[str, CircuitBreaker] = {}
        self._http = httpx.AsyncClient(
            headers={"User-Agent": self.user_agent, "Accept": "*/*"},
            timeout=httpx.Timeout(http.read_timeout, connect=http.connect_timeout),
            follow_redirects=True,
            max_redirects=5,
            transport=transport,
            event_hooks={"request": [self._on_request]},
        )

    @staticmethod
    def _host_limits(settings: Settings) -> dict[str, HostLimit]:
        """Configured limits, with NCBI hosts set by whether an API key is present."""
        http = settings.http
        hosts = dict(http.hosts)
        ncbi_rate = (
            http.ncbi.rate_with_key if settings.secrets.api_key else http.ncbi.rate_without_key
        )
        for host in http.ncbi.hosts:
            limit = hosts.get(host) or HostLimit(rate=ncbi_rate, burst=1, max_concurrency=3)
            hosts[host] = limit.model_copy(update={"rate": min(limit.rate, ncbi_rate)})
        return hosts

    async def __aenter__(self) -> PoliteClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------ checks

    async def _on_request(self, request: httpx.Request) -> None:
        # Runs for every hop, so redirects are held to the same rules.
        Guardrails.check_redirect(str(request.url))

    def _ensure_identity(self) -> None:
        if self.settings.http.require_contact_email and not self.settings.secrets.contact_email:
            raise GuardrailViolation(
                "CONTACT_EMAIL is not set. Add it to .env so providers can reach you; "
                "live requests are refused without it."
            )

    def breaker(self, source: str) -> CircuitBreaker:
        if source not in self._breakers:
            circuit = self.settings.http.circuit
            self._breakers[source] = CircuitBreaker(
                source, circuit.failure_threshold, circuit.reset_seconds, clock=self._clock
            )
        return self._breakers[source]

    async def _check_robots(self, url: str) -> None:
        allowed, delay = await self.robots.check(url)
        host = urlsplit(url).hostname or ""
        if delay and self.settings.http.honor_crawl_delay:
            self.limiter.set_min_interval(host, delay)
        if not allowed:
            unreachable = (await self.robots.policy(url)).unreachable
            raise RobotsDisallowedError(
                f"robots.txt of {host} could not be fetched (host down or server error), so "
                "nothing is requested from it"
                if unreachable
                else f"robots.txt of {host} disallows this path for automated clients",
                url=strip_secret_params(url),
            )

    async def _fetch_robots(self, robots_url: str) -> tuple[int | None, str]:
        host = urlsplit(robots_url).hostname or ""
        try:
            async with self.limiter.semaphore(host):
                await self.limiter.acquire(host)
                self.stats.requests[host] += 1
                response = await self._http.get(robots_url)
            return response.status_code, response.text[:500_000]
        except httpx.HTTPError:
            return None, ""

    async def _prepare(self, url: str, kind: Kind, headers: Mapping[str, str] | None) -> str:
        self._ensure_identity()
        host = self.guardrails.check_url(url)
        Guardrails.check_headers(headers)
        if kind == "html" or (kind == "bulk" and self.settings.http.robots_for_bulk):
            await self._check_robots(url)
        return host

    # ------------------------------------------------------------- primitives

    @staticmethod
    def _raise_for_status(response: httpx.Response, url: str) -> None:
        status = response.status_code
        if status < 400:
            return
        safe = strip_secret_params(url)
        if status in RETRYABLE_STATUS:
            raise RetryableHttpError(
                f"HTTP {status}",
                status=status,
                url=safe,
                retry_after=parse_retry_after(response.headers.get("retry-after")),
            )
        if status in (401, 403, 407, 451):
            raise AccessControlError(f"HTTP {status}: access refused", status=status, url=safe)
        if status in (404, 410):
            raise NotFoundError(f"HTTP {status}", status=status, url=safe)
        raise HttpError(f"HTTP {status}", status=status, url=safe)

    def _on_retry(self, state: RetryCallState) -> None:
        self.stats.retries += 1
        exc = state.outcome.exception() if state.outcome else None
        log_event(
            log,
            logging.WARNING,
            "http.retry",
            attempt=state.attempt_number,
            wait_seconds=round(state.next_action.sleep, 2) if state.next_action else None,
            error=str(exc),
            url=getattr(exc, "url", None),
        )

    async def _run(self, source: str, fn: Callable[[], Awaitable[T]]) -> T:
        """Run `fn` with retries, feeding the outcome to the source's circuit breaker."""
        breaker = self.breaker(source)
        breaker.before_call()
        retrying = build_retrying(
            self.settings.http.retry, sleep=self._sleep, rng=self._rng, on_retry=self._on_retry
        )

        async def attempt() -> T:
            # tenacity only awaits real coroutine functions, and `fn` may be a lambda.
            return await fn()

        try:
            result: T = await retrying(attempt)
        except NotFoundError:
            breaker.record_success()  # the server answered properly
            raise
        except AccessControlError as exc:
            breaker.record_failure(str(exc), trip=True)
            raise
        except HttpError as exc:
            breaker.record_failure(str(exc))
            raise
        except httpx.HTTPError as exc:
            breaker.record_failure(type(exc).__name__)
            raise HttpError(f"network error: {type(exc).__name__}: {exc}") from exc
        breaker.record_success()
        return result

    async def _send(
        self,
        method: str,
        url: str,
        *,
        host: str,
        params: Mapping[str, Any] | None,
        data: Mapping[str, Any] | None,
        headers: Mapping[str, str],
        max_bytes: int,
        expect: tuple[str, ...],
    ) -> tuple[httpx.Response, bytes]:
        async with self.limiter.semaphore(host):
            await self.limiter.acquire(host)
            self.stats.requests[host] += 1
            request = self._http.build_request(
                method, url, params=params, data=data, headers=dict(headers)
            )
            response = await self._http.send(request, stream=True)
            try:
                self._raise_for_status(response, url)
                if response.status_code == 304:
                    return response, b""
                declared = response.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > max_bytes:
                    raise ResponseTooLargeError(
                        f"response of {declared} bytes exceeds the {max_bytes} byte limit",
                        url=strip_secret_params(url),
                    )
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise ResponseTooLargeError(
                            f"response exceeds the {max_bytes} byte limit",
                            url=strip_secret_params(url),
                        )
                    chunks.append(chunk)
                body = b"".join(chunks)
            finally:
                await response.aclose()
                self._http.cookies.clear()  # no session is ever carried between requests
        check_content(expect, response.headers, body[:4096], strip_secret_params(url))
        self.stats.bytes += len(body)
        return response, body

    # ------------------------------------------------------------- public API

    async def fetch(
        self,
        url: str,
        *,
        source: str,
        method: str = "GET",
        params: Mapping[str, Any] | None = None,
        data: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        kind: Kind = "api",
        expect: tuple[str, ...] = ("json",),
        use_cache: bool = True,
        ttl: float | None = None,
        max_bytes: int | None = None,
    ) -> FetchResult:
        """GET or POST a document, using the disk cache and conditional requests."""
        host = await self._prepare(url, kind, headers)
        full_url = str(httpx.URL(url).copy_merge_params(dict(params))) if params else url
        safe_url = strip_secret_params(full_url)
        safe_body = (
            urlencode(sorted((k, str(v)) for k, v in data.items() if k not in SECRET_PARAMS))
            if data
            else ""
        )
        key = HttpCache.make_key(method, safe_url, safe_body.encode() or None)
        entry = self.cache.get(key) if use_cache else None
        lifetime = self.settings.http.cache_ttl_seconds if ttl is None else ttl
        if entry and not self.settings.refresh and entry.is_fresh(lifetime, self.cache.now()):
            self.stats.cache_hits += 1
            cached_headers = {"content-type": entry.content_type or ""}
            return FetchResult(safe_url, entry.status, cached_headers, entry.read(), True)

        request_headers = dict(headers or {})
        if entry:
            request_headers.update(entry.conditional_headers())
        limit = max_bytes or self.settings.http.max_response_bytes
        response, body = await self._run(
            source,
            lambda: self._send(
                method,
                url,
                host=host,
                params=params,
                data=data,
                headers=request_headers,
                max_bytes=limit,
                expect=expect,
            ),
        )
        log_event(
            log,
            logging.DEBUG,
            "http.response",
            source=source,
            url=safe_url,
            status=response.status_code,
            bytes=len(body),
        )
        if response.status_code == 304 and entry:
            self.cache.touch(entry)
            self.stats.revalidated += 1
            cached_headers = {"content-type": entry.content_type or ""}
            return FetchResult(safe_url, entry.status, cached_headers, entry.read(), True)
        if use_cache:
            self.cache.put(
                key, url=safe_url, status=response.status_code, headers=response.headers, body=body
            )
        return FetchResult(safe_url, response.status_code, dict(response.headers), body)

    async def download(
        self,
        url: str,
        dest: Path,
        *,
        source: str,
        kind: Kind = "bulk",
        expected_size: int | None = None,
        expected_md5: str | None = None,
        max_bytes: int | None = None,
        on_progress: Callable[[int, int | None], None] | None = None,
    ) -> DownloadResult:
        """Download a large file to `dest`, resuming a previous attempt when possible.

        Bytes go to `<dest>.part`; the file only appears under its final name after its size
        and checksum were verified (atomic rename).
        """
        host = await self._prepare(url, kind, None)
        safe_url = strip_secret_params(url)
        limit = max_bytes or self.settings.http.max_download_bytes
        if expected_size is not None and expected_size > limit:
            raise ResponseTooLargeError(
                f"file of {expected_size} bytes exceeds the {limit} byte download limit",
                url=safe_url,
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        meta_path = dest.with_name(dest.name + ".part.json")
        info: dict[str, Any] = {"resumed": False, "etag": None, "last_modified": None}

        def reset_part() -> None:
            for stale in (part, meta_path):
                with contextlib.suppress(FileNotFoundError):
                    stale.unlink()

        async def attempt() -> None:
            offset = part.stat().st_size if part.exists() else 0
            meta: dict[str, Any] = {}
            if offset:
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    meta = {}
                if meta.get("url") != safe_url:
                    reset_part()
                    offset, meta = 0, {}
            headers = {"Accept-Encoding": "identity"}  # byte offsets must match the file
            if offset:
                headers["Range"] = f"bytes={offset}-"
                if validator := (meta.get("etag") or meta.get("last_modified")):
                    headers["If-Range"] = validator
            async with self.limiter.semaphore(host):
                await self.limiter.acquire(host)
                self.stats.requests[host] += 1
                request = self._http.build_request("GET", url, headers=headers)
                response = await self._http.send(request, stream=True)
                try:
                    status = response.status_code
                    if status == 416:
                        total = _total_from_content_range(response.headers.get("content-range"))
                        if total is not None and total == offset:
                            return  # the part file is already complete
                        reset_part()
                        raise RetryableHttpError(
                            "partial download is stale, restarting", status=416, url=safe_url
                        )
                    self._raise_for_status(response, url)
                    resumed = status == 206 and offset > 0
                    if resumed:
                        start = (response.headers.get("content-range") or "").removeprefix("bytes ")
                        if not start.startswith(f"{offset}-"):
                            reset_part()
                            raise RetryableHttpError(
                                "server resumed at the wrong offset", status=206, url=safe_url
                            )
                        info["resumed"] = True
                    else:
                        offset = 0
                        info["resumed"] = False
                        meta = {
                            "url": safe_url,
                            "etag": response.headers.get("etag"),
                            "last_modified": response.headers.get("last-modified"),
                        }
                        meta_path.write_text(json.dumps(meta), encoding="utf-8")
                    info["etag"], info["last_modified"] = (
                        meta.get("etag"),
                        meta.get("last_modified"),
                    )
                    declared = response.headers.get("content-length", "")
                    total_size = offset + int(declared) if declared.isdigit() else None
                    if total_size is not None and total_size > limit:
                        raise ResponseTooLargeError(
                            f"file of {total_size} bytes exceeds the download limit", url=safe_url
                        )
                    written = offset
                    first = True
                    with part.open("ab" if resumed else "wb") as handle:
                        async for chunk in response.aiter_raw(1 << 16):
                            if first:
                                check_content(("binary",), response.headers, chunk, safe_url)
                                first = False
                            written += len(chunk)
                            if written > limit:
                                raise ResponseTooLargeError(
                                    "file exceeds the download limit", url=safe_url
                                )
                            handle.write(chunk)
                            if on_progress:
                                on_progress(written, total_size)
                    self.stats.bytes += written - offset
                finally:
                    await response.aclose()
                    self._http.cookies.clear()

        await self._run(source, attempt)

        size = part.stat().st_size
        sha256, md5 = await asyncio.to_thread(hash_file, part)
        if expected_size is not None and size != expected_size:
            reset_part()
            raise IntegrityError(
                f"downloaded {size} bytes but expected {expected_size}", url=safe_url
            )
        if expected_md5 and md5 != expected_md5.lower():
            reset_part()
            raise IntegrityError("MD5 checksum does not match the published value", url=safe_url)
        part.replace(dest)
        with contextlib.suppress(FileNotFoundError):
            meta_path.unlink()
        log_event(
            log,
            logging.INFO,
            "download.complete",
            source=source,
            url=safe_url,
            bytes=size,
            resumed=info["resumed"],
        )
        return DownloadResult(
            dest, safe_url, size, sha256, md5, info["etag"], info["last_modified"], info["resumed"]
        )

    async def head_lines(
        self,
        url: str,
        n_lines: int,
        *,
        source: str,
        kind: Kind = "bulk",
        max_bytes: int | None = None,
    ) -> bytes:
        """Fetch only the first `n_lines` complete lines of a text file (sample mode).

        Asks for a byte range and stops reading as soon as enough lines arrived, so a sample
        of a multi-hundred-megabyte file costs a few kilobytes.
        """
        host = await self._prepare(url, kind, None)
        safe_url = strip_secret_params(url)
        limit = max_bytes or self.settings.http.sample_max_bytes

        async def attempt() -> tuple[bytes, bool]:
            headers = {"Range": f"bytes=0-{limit - 1}", "Accept-Encoding": "identity"}
            async with self.limiter.semaphore(host):
                await self.limiter.acquire(host)
                self.stats.requests[host] += 1
                request = self._http.build_request("GET", url, headers=headers)
                response = await self._http.send(request, stream=True)
                try:
                    self._raise_for_status(response, url)
                    buffer = bytearray()
                    lines = 0
                    complete = True
                    async for chunk in response.aiter_raw(1 << 16):
                        if not buffer:
                            check_content(("binary",), response.headers, chunk, safe_url)
                        buffer.extend(chunk)
                        lines += chunk.count(b"\n")
                        if lines >= n_lines or len(buffer) >= limit:
                            complete = False
                            break
                    total = _total_from_content_range(response.headers.get("content-range"))
                    if total is not None and total > len(buffer):
                        complete = False
                finally:
                    await response.aclose()
                    self._http.cookies.clear()
            return bytes(buffer), complete

        data, complete = await self._run(source, attempt)
        self.stats.bytes += len(data)
        end = -1
        for _ in range(n_lines):
            nxt = data.find(b"\n", end + 1)
            if nxt == -1:
                return data if complete else data[: end + 1]
            end = nxt
        return data[: end + 1]

    async def probe(self, url: str, *, kind: Kind = "html") -> ProbeResult:
        """Check that a link answers, reading at most a few kilobytes. Never raises."""
        try:
            self._ensure_identity()
            host = self.guardrails.check_url(url)
            if kind != "api":
                await self._check_robots(url)
        except RobotsDisallowedError as exc:
            return ProbeResult(url, None, note=f"not requested: {exc}")
        except GuardrailViolation as exc:
            return ProbeResult(url, None, note=f"not requested: {exc}")
        try:
            async with self.limiter.semaphore(host):
                await self.limiter.acquire(host)
                self.stats.requests[host] += 1
                request = self._http.build_request("GET", url)
                response = await self._http.send(request, stream=True)
                try:
                    head = b""
                    content_type = response.headers.get("content-type", "")
                    if "html" in content_type.lower():
                        async for chunk in response.aiter_bytes():
                            head = chunk
                            break
                    wall = detect_wall(response.headers, head) if head else None
                    note = f"access wall: {wall}" if wall else ""
                    if response.status_code == 202 and not head:
                        # Seen on Figshare+ and Dataverse pages: a challenge, not content.
                        note = "202 with an empty body: probably a bot challenge"
                    return ProbeResult(
                        url,
                        response.status_code,
                        str(response.url),
                        content_type.split(";")[0] or None,
                        note,
                    )
                finally:
                    await response.aclose()
                    self._http.cookies.clear()
        except (httpx.HTTPError, HttpError) as exc:
            return ProbeResult(url, None, note=f"network error: {type(exc).__name__}")
