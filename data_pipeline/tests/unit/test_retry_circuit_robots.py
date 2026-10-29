from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from synergyrx_data.config import RetrySettings
from synergyrx_data.http.circuit import CircuitBreaker
from synergyrx_data.http.errors import (
    AccessControlError,
    CircuitOpenError,
    NotFoundError,
    RetryableHttpError,
)
from synergyrx_data.http.retry import backoff_delay, is_retryable, parse_retry_after, wait_seconds
from synergyrx_data.http.robots import RobotsChecker, RobotsPolicy
from tests.conftest import FakeClock

# ------------------------------------------------------------------- retry


def test_parse_retry_after_seconds_and_http_date() -> None:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert parse_retry_after("120") == 120.0
    assert parse_retry_after("Thu, 01 Jan 2026 12:00:30 GMT", now=now) == 30.0
    assert parse_retry_after("Thu, 01 Jan 2026 11:00:00 GMT", now=now) == 0.0  # in the past
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None


def test_backoff_is_exponential_jittered_and_capped() -> None:
    low = [backoff_delay(n, 1.0, 60.0, rng=lambda: 0.0) for n in range(1, 9)]
    high = [backoff_delay(n, 1.0, 60.0, rng=lambda: 1.0) for n in range(1, 9)]
    assert low[:4] == [0.5, 1.0, 2.0, 4.0]
    assert high[:4] == [1.0, 2.0, 4.0, 8.0]
    assert max(high) == 60.0  # capped
    assert all(a < b for a, b in zip(low, high, strict=True))  # jitter spreads the delay


def test_retry_after_is_a_floor_for_the_wait() -> None:
    settings = RetrySettings(backoff_base=1.0, backoff_max=60.0)
    polite = RetryableHttpError("429", status=429, retry_after=45.0)
    assert wait_seconds(polite, 1, settings, rng=lambda: 0.0) == 45.0
    assert wait_seconds(RetryableHttpError("503", status=503), 3, settings, lambda: 1.0) == 4.0


def test_what_is_retried_and_what_is_not() -> None:
    assert is_retryable(RetryableHttpError("503", status=503), 300)
    assert is_retryable(httpx.ReadTimeout("slow"), 300)
    assert not is_retryable(RetryableHttpError("429", status=429, retry_after=3600), 300)
    assert not is_retryable(AccessControlError("403"), 300)
    assert not is_retryable(NotFoundError("404"), 300)


# ----------------------------------------------------------------- circuit


def test_circuit_opens_after_threshold_and_recovers(clock: FakeClock) -> None:
    breaker = CircuitBreaker("src", failure_threshold=3, reset_seconds=60, clock=clock)
    for _ in range(2):
        breaker.record_failure("boom")
    breaker.before_call()  # still closed
    breaker.record_failure("boom")
    assert breaker.state == "open"
    with pytest.raises(CircuitOpenError, match="src"):
        breaker.before_call()
    clock.now += 61
    assert breaker.state == "half_open"
    breaker.before_call()  # one trial call is allowed
    breaker.record_failure("still down")
    assert breaker.state == "open"  # a failed trial re-opens at once
    clock.now += 61
    breaker.record_success()
    assert breaker.state == "closed"


def test_access_refusal_trips_the_circuit_immediately(clock: FakeClock) -> None:
    breaker = CircuitBreaker("src", failure_threshold=5, clock=clock)
    breaker.record_failure("HTTP 403", trip=True)
    assert breaker.state == "open"


# ------------------------------------------------------------------ robots

ZENODO = """
User-agent: *
Disallow: /search
Disallow: /api
Allow: /api/records/*/files
Disallow: /api/records/*/files-archive
Crawl-delay: 10
"""
NCI_WIKI = """
User-agent: *
Crawl-delay: 15
Disallow: /
Allow: /$
Allow: /spaces/*/pages/
"""
AGENT = "synergyrx-data"


def test_wildcards_and_longest_match_like_real_zenodo_rules() -> None:
    policy = RobotsPolicy(ZENODO)
    allowed = "/api/records/11102665/files/summary_table_v1.4.csv/content"
    assert policy.allowed(allowed, AGENT)
    assert not policy.allowed("/api/records/11102665", AGENT)
    assert not policy.allowed("/api/records/11102665/files-archive", AGENT)
    assert policy.allowed("/records/11102665", AGENT)
    assert policy.crawl_delay(AGENT) == 10.0


def test_nci_wiki_blocks_downloads_but_not_pages() -> None:
    policy = RobotsPolicy(NCI_WIKI)
    assert not policy.allowed("/download/attachments/338237347/ComboDrugGrowth_Nov2017.zip", AGENT)
    assert policy.allowed("/spaces/NCIDTPdata/pages/338237347/NCI-ALMANAC", AGENT)
    assert policy.allowed("/", AGENT)  # `Allow: /$` matches only the root


def test_specific_user_agent_group_overrides_the_wildcard() -> None:
    text = "User-agent: *\nDisallow: /\n\nUser-agent: synergyrx-data\nDisallow: /private\n"
    policy = RobotsPolicy(text)
    assert policy.allowed("/data.csv", AGENT)
    assert not policy.allowed("/private/x", AGENT)
    assert not policy.allowed("/data.csv", "otherbot")


@pytest.mark.parametrize(
    ("status", "expected"),
    [(404, True), (410, True), (200, True), (500, False), (503, False), (None, False)],
)
def test_unavailable_means_allowed_unreachable_means_stop(
    status: int | None, expected: bool
) -> None:
    assert RobotsPolicy.from_response(status, "").allowed("/anything", AGENT) is expected


async def test_checker_fetches_each_origin_once() -> None:
    calls: list[str] = []

    async def fetcher(url: str) -> tuple[int | None, str]:
        calls.append(url)
        return 200, ZENODO

    checker = RobotsChecker(fetcher, AGENT)
    assert await checker.check("https://zenodo.org/records/1") == (True, 10.0)
    assert await checker.check("https://zenodo.org/api/records/1?x=1") == (False, 10.0)
    assert calls == ["https://zenodo.org/robots.txt"]
