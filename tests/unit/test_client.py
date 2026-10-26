"""PoliteClient behaviour against a mocked network."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest
import respx

from synergyrx_data.config import Secrets, Settings
from synergyrx_data.http.errors import (
    AccessControlError,
    CircuitOpenError,
    ContentTypeError,
    GuardrailViolation,
    HttpError,
    IntegrityError,
    NotFoundError,
    ResponseTooLargeError,
    RobotsDisallowedError,
)
from synergyrx_data.http.guardrails import build_user_agent, strip_secret_params
from tests.conftest import TEST_EMAIL, ClientFactory, FakeClock

API = "https://example.org/api/item"
JSON = {"content-type": "application/json"}


# ------------------------------------------------------------ identity


async def test_user_agent_names_the_tool_and_the_contact(
    make_client: ClientFactory, mock_http: respx.MockRouter
) -> None:
    route = mock_http.get(API).respond(json={"ok": True})
    await make_client().fetch(API, source="t")
    agent = route.calls.last.request.headers["user-agent"]
    assert agent.startswith("synergyrx-data/")
    assert f"mailto:{TEST_EMAIL}" in agent
    assert "mozilla" not in agent.lower()


def test_user_agent_may_not_imitate_a_browser() -> None:
    with pytest.raises(GuardrailViolation, match="imitate"):
        build_user_agent("Mozilla/5.0 Chrome/120", "1.0", TEST_EMAIL)


async def test_live_requests_are_refused_without_contact_email(
    settings: Settings, make_client: ClientFactory, mock_http: respx.MockRouter
) -> None:
    settings.secrets = Secrets()
    route = mock_http.get(API).respond(json={})
    with pytest.raises(GuardrailViolation, match="CONTACT_EMAIL"):
        await make_client(settings).fetch(API, source="t")
    assert not route.called


# ---------------------------------------------------------- guardrails


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("http://example.org/api", "https"),
        ("https://user:secret@example.org/api", "credentials"),
        ("https://not-a-registered-source.test/data", "not declared"),
    ],
)
async def test_urls_outside_the_rules_never_reach_the_network(
    make_client: ClientFactory, mock_http: respx.MockRouter, url: str, reason: str
) -> None:
    with pytest.raises(GuardrailViolation, match=reason):
        await make_client().fetch(url, source="t")
    assert len(mock_http.calls) == 0


@pytest.mark.parametrize("header", ["Cookie", "Authorization", "X-Forwarded-For", "User-Agent"])
async def test_session_credential_and_spoofing_headers_are_refused(
    make_client: ClientFactory, mock_http: respx.MockRouter, header: str
) -> None:
    with pytest.raises(GuardrailViolation, match="may not be set"):
        await make_client().fetch(API, source="t", headers={header: "x"})
    assert len(mock_http.calls) == 0


async def test_cookies_are_not_carried_between_requests(
    make_client: ClientFactory, mock_http: respx.MockRouter
) -> None:
    route = mock_http.get(API).respond(json={}, headers={"set-cookie": "session=abc; Path=/"})
    client = make_client()
    await client.fetch(API, source="t", use_cache=False)
    await client.fetch(API, source="t", use_cache=False)
    assert "cookie" not in route.calls.last.request.headers


@pytest.mark.parametrize("status", [401, 403])
async def test_access_refusal_is_not_retried_and_opens_the_circuit(
    make_client: ClientFactory, mock_http: respx.MockRouter, status: int
) -> None:
    route = mock_http.get(API).respond(status)
    client = make_client()
    with pytest.raises(AccessControlError):
        await client.fetch(API, source="walled")
    assert route.call_count == 1
    with pytest.raises(CircuitOpenError):
        await client.fetch(API, source="walled")
    assert route.call_count == 1  # the second call never left


async def test_captcha_page_instead_of_data_is_reported_not_parsed(
    make_client: ClientFactory, mock_http: respx.MockRouter
) -> None:
    page = "<!doctype html><html><body>Verifying you are a person, not a bot. CAPTCHA</body></html>"
    route = mock_http.get(API).respond(200, html=page)
    with pytest.raises(AccessControlError, match="access wall"):
        await make_client().fetch(API, source="t")
    assert route.call_count == 1


async def test_wrong_content_type_and_oversized_responses_are_rejected(
    make_client: ClientFactory, mock_http: respx.MockRouter
) -> None:
    mock_http.get(API).respond(200, text="a,b\n1,2\n", headers={"content-type": "text/csv"})
    with pytest.raises(ContentTypeError):
        await make_client().fetch(API, source="t", expect=("json",))
    big = "https://example.org/api/big"
    mock_http.get(big).respond(200, content=b"x" * 5000, headers=JSON)
    with pytest.raises(ResponseTooLargeError):
        await make_client().fetch(big, source="t", max_bytes=1000)


# ------------------------------------------------------- retry behaviour


async def test_429_waits_for_retry_after_then_succeeds(
    make_client: ClientFactory, mock_http: respx.MockRouter, clock: FakeClock
) -> None:
    route = mock_http.get(API).mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "7"}),
            httpx.Response(503),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    client = make_client()
    result = await client.fetch(API, source="t")
    assert result.json() == {"ok": True}
    assert route.call_count == 3
    waits = [s for s in clock.sleeps if s >= 0.3]
    assert waits[0] == 7.0  # the server's Retry-After, not our shorter backoff
    assert waits[1] == pytest.approx(0.75)  # attempt 2: 0.5 * 2, equal jitter at rng=0.5
    assert client.stats.retries == 2


async def test_gives_up_after_max_attempts_and_trips_breaker_eventually(
    make_client: ClientFactory, mock_http: respx.MockRouter
) -> None:
    route = mock_http.get(API).respond(503)
    client = make_client()
    with pytest.raises(HttpError, match="503"):
        await client.fetch(API, source="flaky")
    assert route.call_count == 4  # http.retry.max_attempts in the test settings
    for _ in range(4):
        with pytest.raises(HttpError):
            await client.fetch(API, source="flaky")
    with pytest.raises(CircuitOpenError):
        await client.fetch(API, source="flaky")


async def test_a_long_retry_after_means_stop(
    make_client: ClientFactory, mock_http: respx.MockRouter
) -> None:
    route = mock_http.get(API).respond(429, headers={"retry-after": "86400"})
    with pytest.raises(HttpError):
        await make_client().fetch(API, source="t")
    assert route.call_count == 1


async def test_not_found_is_an_answer_not_a_failure(
    make_client: ClientFactory, mock_http: respx.MockRouter
) -> None:
    route = mock_http.get(API).respond(404)
    client = make_client()
    for _ in range(8):
        with pytest.raises(NotFoundError):
            await client.fetch(API, source="lookups")
    assert route.call_count == 8
    assert client.breaker("lookups").state == "closed"


# ---------------------------------------------------------------- cache


async def test_fresh_cache_entry_avoids_the_network(
    make_client: ClientFactory, mock_http: respx.MockRouter
) -> None:
    route = mock_http.get(API).respond(json={"n": 1}, headers={"etag": '"v1"'})
    first = await make_client().fetch(API, source="t")
    second = await make_client().fetch(API, source="t")  # a new client: the cache is on disk
    assert (first.from_cache, second.from_cache) == (False, True)
    assert second.json() == {"n": 1}
    assert route.call_count == 1


async def test_stale_entry_is_revalidated_with_etag(
    make_client: ClientFactory, mock_http: respx.MockRouter
) -> None:
    route = mock_http.get(API).mock(
        side_effect=[
            httpx.Response(
                200, json={"n": 1}, headers={"etag": '"v1"', "last-modified": "Mon, 01"}
            ),
            httpx.Response(304),
        ]
    )
    client = make_client()
    await client.fetch(API, source="t", ttl=0)
    again = await client.fetch(API, source="t", ttl=0)
    request = route.calls.last.request
    assert request.headers["if-none-match"] == '"v1"'
    assert request.headers["if-modified-since"] == "Mon, 01"
    assert again.from_cache and again.json() == {"n": 1}
    assert client.stats.revalidated == 1


# -------------------------------------------------------------- secrets


async def test_api_key_never_reaches_cache_files_or_returned_urls(
    settings: Settings, make_client: ClientFactory, mock_http: respx.MockRouter
) -> None:
    secret = "SECRETKEY1234567890"
    route = mock_http.get(API).respond(json={})
    result = await make_client().fetch(
        API, source="t", params={"db": "pubmed", "api_key": secret, "email": TEST_EMAIL}
    )
    assert "api_key=" + secret in str(route.calls.last.request.url)  # it is sent...
    assert secret not in result.url and TEST_EMAIL not in result.url  # ...but never kept
    stored = b"".join(p.read_bytes() for p in settings.cache_dir.rglob("*") if p.is_file())
    assert secret.encode() not in stored
    assert strip_secret_params(f"{API}?db=pubmed&api_key={secret}") == f"{API}?db=pubmed"


# ----------------------------------------------------------- robots.txt


async def test_html_is_only_fetched_when_robots_allows_it(
    make_client: ClientFactory, mock_http: respx.MockRouter
) -> None:
    mock_http.get("https://example.org/robots.txt").respond(
        200, text="User-agent: *\nDisallow: /private\n"
    )
    page = mock_http.get("https://example.org/private/page").respond(200, html="<html></html>")
    open_page = mock_http.get("https://example.org/public").respond(200, html="<html>ok</html>")
    client = make_client()
    with pytest.raises(RobotsDisallowedError):
        await client.fetch(
            "https://example.org/private/page", source="t", kind="html", expect=("html",)
        )
    assert not page.called
    await client.fetch("https://example.org/public", source="t", kind="html", expect=("html",))
    assert open_page.called


async def test_crawl_delay_from_robots_slows_the_host_down(
    make_client: ClientFactory, mock_http: respx.MockRouter, clock: FakeClock
) -> None:
    mock_http.get("https://example.org/robots.txt").respond(
        200, text="User-agent: *\nCrawl-delay: 10\n"
    )
    mock_http.get(url__regex=r"https://example\.org/file\d").respond(200, content=b"data")
    client = make_client()
    for n in range(3):
        await client.fetch(
            f"https://example.org/file{n}", source="t", kind="bulk", expect=("binary",)
        )
    assert clock() >= 20.0  # three requests, ten seconds apart


# ------------------------------------------------------------ downloads

PAYLOAD = bytes(range(256)) * 400  # 102,400 bytes
URL = "https://example.org/files/big.bin"


def _range_handler(payload: bytes, etag: str = '"abc"') -> respx.MockRouter:
    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"etag": etag, "content-type": "application/octet-stream"}
        if "range" in request.headers:
            start = int(request.headers["range"].removeprefix("bytes=").split("-")[0])
            if request.headers.get("if-range", etag) != etag:
                return httpx.Response(200, content=payload, headers=headers)
            if start >= len(payload):
                headers["content-range"] = f"bytes */{len(payload)}"
                return httpx.Response(416, headers=headers)
            headers["content-range"] = f"bytes {start}-{len(payload) - 1}/{len(payload)}"
            return httpx.Response(206, content=payload[start:], headers=headers)
        return httpx.Response(200, content=payload, headers=headers)

    return handler  # type: ignore[return-value]


async def test_download_verifies_and_renames_atomically(
    make_client: ClientFactory, mock_http: respx.MockRouter, allow_robots: None, tmp_path: Path
) -> None:
    mock_http.get(URL).mock(side_effect=_range_handler(PAYLOAD))
    dest = tmp_path / "out" / "big.bin"
    result = await make_client().download(
        URL,
        dest,
        source="t",
        expected_size=len(PAYLOAD),
        expected_md5=hashlib.md5(PAYLOAD).hexdigest(),
    )
    assert dest.read_bytes() == PAYLOAD
    assert result.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    assert not result.resumed
    assert sorted(p.name for p in dest.parent.iterdir()) == ["big.bin"]  # no .part left behind


async def test_interrupted_download_resumes_with_a_range_request(
    make_client: ClientFactory, mock_http: respx.MockRouter, allow_robots: None, tmp_path: Path
) -> None:
    route = mock_http.get(URL).mock(side_effect=_range_handler(PAYLOAD))
    dest = tmp_path / "big.bin"
    # A previous run died after 30,000 bytes.
    dest.with_name("big.bin.part").write_bytes(PAYLOAD[:30_000])
    dest.with_name("big.bin.part.json").write_text(json.dumps({"url": URL, "etag": '"abc"'}))
    result = await make_client().download(URL, dest, source="t", expected_size=len(PAYLOAD))
    request = route.calls.last.request
    assert request.headers["range"] == "bytes=30000-"
    assert request.headers["if-range"] == '"abc"'
    assert result.resumed
    assert dest.read_bytes() == PAYLOAD


async def test_resume_restarts_when_the_remote_file_changed(
    make_client: ClientFactory, mock_http: respx.MockRouter, allow_robots: None, tmp_path: Path
) -> None:
    mock_http.get(URL).mock(side_effect=_range_handler(PAYLOAD, etag='"new"'))
    dest = tmp_path / "big.bin"
    dest.with_name("big.bin.part").write_bytes(b"old-version-bytes" * 100)
    dest.with_name("big.bin.part.json").write_text(json.dumps({"url": URL, "etag": '"old"'}))
    result = await make_client().download(URL, dest, source="t", expected_size=len(PAYLOAD))
    assert not result.resumed
    assert dest.read_bytes() == PAYLOAD


async def test_connection_drop_mid_download_is_resumed_by_the_retry(
    make_client: ClientFactory, mock_http: respx.MockRouter, allow_robots: None, tmp_path: Path
) -> None:
    class Dropping(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield PAYLOAD[:70_000]  # one full 64 KiB chunk reaches the disk before the drop
            raise httpx.ReadError("connection reset")

    calls = {"n": 0}
    resume = _range_handler(PAYLOAD)

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, stream=Dropping(), headers={"etag": '"abc"'})
        return resume(request)  # type: ignore[operator]

    mock_http.get(URL).mock(side_effect=handler)
    dest = tmp_path / "big.bin"
    result = await make_client().download(URL, dest, source="t", expected_size=len(PAYLOAD))
    assert calls["n"] == 2 and result.resumed
    assert mock_http.calls.last.request.headers["range"] == "bytes=65536-"
    assert dest.read_bytes() == PAYLOAD


async def test_checksum_mismatch_leaves_no_file(
    make_client: ClientFactory, mock_http: respx.MockRouter, allow_robots: None, tmp_path: Path
) -> None:
    mock_http.get(URL).mock(side_effect=_range_handler(PAYLOAD))
    dest = tmp_path / "big.bin"
    with pytest.raises(IntegrityError, match="MD5"):
        await make_client().download(URL, dest, source="t", expected_md5="0" * 32)
    assert list(tmp_path.iterdir()) == []


async def test_download_refuses_files_over_the_size_limit(
    make_client: ClientFactory, mock_http: respx.MockRouter, allow_robots: None, tmp_path: Path
) -> None:
    route = mock_http.get(URL).mock(side_effect=_range_handler(PAYLOAD))
    with pytest.raises(ResponseTooLargeError):
        await make_client().download(URL, tmp_path / "x", source="t", max_bytes=1000)
    with pytest.raises(ResponseTooLargeError):
        await make_client().download(
            URL, tmp_path / "y", source="t", expected_size=10**12, max_bytes=10**9
        )
    assert route.call_count == 1  # the second refusal happened before any request


async def test_bulk_download_blocked_by_robots_is_not_attempted(
    make_client: ClientFactory, mock_http: respx.MockRouter, tmp_path: Path
) -> None:
    mock_http.get("https://example.org/robots.txt").respond(
        200, text="User-agent: *\nDisallow: /files/\n"
    )
    route = mock_http.get(URL).respond(200, content=PAYLOAD)
    with pytest.raises(RobotsDisallowedError):
        await make_client().download(URL, tmp_path / "big.bin", source="t")
    assert not route.called


async def test_head_lines_reads_only_the_start_of_a_large_file(
    make_client: ClientFactory, mock_http: respx.MockRouter, allow_robots: None
) -> None:
    lines = b"".join(f"row{n},value\n".encode() for n in range(100_000))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["range"].startswith("bytes=0-")
        return httpx.Response(
            206, content=lines, headers={"content-range": f"bytes 0-{len(lines) - 1}/{len(lines)}"}
        )

    mock_http.get(URL).mock(side_effect=handler)
    head = await make_client().head_lines(URL, 11, source="t")
    assert head.count(b"\n") == 11
    assert head.endswith(b"row10,value\n")


async def test_head_lines_keeps_a_short_file_whole(
    make_client: ClientFactory, mock_http: respx.MockRouter, allow_robots: None
) -> None:
    mock_http.get(URL).respond(200, content=b"h\n1\n2")  # no trailing newline
    assert await make_client().head_lines(URL, 50, source="t") == b"h\n1\n2"


async def test_probe_reports_status_walls_and_robots_without_raising(
    make_client: ClientFactory, mock_http: respx.MockRouter
) -> None:
    mock_http.get("https://example.org/robots.txt").respond(
        200, text="User-agent: *\nDisallow: /blocked\n"
    )
    mock_http.get("https://example.org/gone").respond(410, html="<html>moved</html>")
    mock_http.get("https://example.org/wall").respond(200, html="<html>reCAPTCHA check</html>")
    mock_http.get("https://example.org/down").mock(side_effect=httpx.ConnectError("refused"))
    client = make_client()
    assert (await client.probe("https://example.org/gone")).status == 410
    assert "access wall" in (await client.probe("https://example.org/wall")).note
    assert (await client.probe("https://example.org/down")).status is None
    blocked = await client.probe("https://example.org/blocked/x")
    assert blocked.status is None and "robots.txt" in blocked.note


async def test_unreachable_robots_means_nothing_is_requested_and_says_why(
    make_client: ClientFactory, mock_http: respx.MockRouter, tmp_path: Path
) -> None:
    mock_http.get("https://example.org/robots.txt").mock(side_effect=httpx.ConnectTimeout("down"))
    page = mock_http.get("https://example.org/page").respond(200, html="<html></html>")
    client = make_client()
    result = await client.probe("https://example.org/page")
    assert result.status is None and "could not be fetched" in result.note
    with pytest.raises(RobotsDisallowedError, match="could not be fetched"):
        await client.download("https://example.org/page", tmp_path / "unused", source="t")
    assert not page.called
