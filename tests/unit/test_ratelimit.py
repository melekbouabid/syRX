"""The rate limiter must provably keep requests under the configured limit."""

from __future__ import annotations

import asyncio
import itertools
import time

import pytest

from synergyrx_data.config import HostLimit
from synergyrx_data.http.ratelimit import HostRateLimiter, TokenBucket
from tests.conftest import FakeClock


def max_in_any_window(times: list[float], window: float) -> int:
    """Largest number of grants inside any half-open window of `window` seconds."""
    return max(sum(1 for t in times if start <= t < start + window) for start in times)


@pytest.mark.parametrize(("rate", "burst"), [(3.0, 1), (10.0, 1), (5.0, 3), (0.1, 1)])
async def test_never_exceeds_configured_rate(clock: FakeClock, rate: float, burst: int) -> None:
    bucket = TokenBucket(rate, burst, clock=clock, sleep=clock.sleep)
    grants: list[float] = []
    for _ in range(60):
        await bucket.acquire()
        grants.append(clock())
    window = 1.0 if rate >= 1 else 1.0 / rate
    allowed = max(rate, 1.0) * (window if rate >= 1 else rate) + burst
    # In any window the limiter may hand out the steady rate plus at most one burst.
    assert max_in_any_window(grants, window) <= allowed
    # Over the whole run the average rate is bounded by the configured one.
    elapsed = grants[-1] - grants[0]
    assert (len(grants) - burst) / elapsed <= rate * 1.0001


async def test_requests_are_evenly_spaced_without_burst(clock: FakeClock) -> None:
    bucket = TokenBucket(4.0, 1, clock=clock, sleep=clock.sleep)
    grants = []
    for _ in range(5):
        await bucket.acquire()
        grants.append(clock())
    gaps = [b - a for a, b in itertools.pairwise(grants)]
    assert gaps == pytest.approx([0.25] * 4)


async def test_concurrent_callers_share_one_budget(clock: FakeClock) -> None:
    bucket = TokenBucket(2.0, 1, clock=clock, sleep=clock.sleep)
    grants: list[float] = []

    async def worker() -> None:
        await bucket.acquire()
        grants.append(clock())

    await asyncio.gather(*(worker() for _ in range(9)))
    assert max_in_any_window(sorted(grants), 1.0) <= 2
    assert max(grants) == pytest.approx(4.0)  # 9 requests at 2 per second


async def test_real_clock_respects_limit() -> None:
    """Same property against the real clock: 8 requests at 20 per second take >= 0.35 s."""
    bucket = TokenBucket(20.0, 1)
    started = time.monotonic()
    for _ in range(8):
        await bucket.acquire()
    assert time.monotonic() - started >= 7 / 20 * 0.98


async def test_hosts_are_limited_independently(clock: FakeClock) -> None:
    limiter = HostRateLimiter(
        HostLimit(rate=1.0),
        {"fast.example": HostLimit(rate=10.0), "ncbi.nlm.nih.gov": HostLimit(rate=3.0)},
        clock=clock,
        sleep=clock.sleep,
    )
    for _ in range(3):
        await limiter.acquire("fast.example")
    assert clock() == pytest.approx(0.2)
    assert limiter.limit_for("unknown.example").rate == 1.0
    # A subdomain inherits the limit of its configured parent domain.
    assert limiter.limit_for("eutils.ncbi.nlm.nih.gov").rate == 3.0


async def test_safety_factor_and_crawl_delay_only_slow_down(clock: FakeClock) -> None:
    limiter = HostRateLimiter(
        HostLimit(rate=10.0), safety_factor=0.5, clock=clock, sleep=clock.sleep
    )
    assert limiter.bucket("a.example").rate == 5.0
    limiter.set_min_interval("a.example", 10.0)  # robots.txt Crawl-delay: 10
    assert limiter.bucket("a.example").rate == pytest.approx(0.1)
    limiter.set_min_interval("a.example", 0.01)  # would be faster: ignored
    assert limiter.bucket("a.example").rate == pytest.approx(0.1)
    await limiter.acquire("a.example")
    await limiter.acquire("a.example")
    assert clock() >= 10.0 - 1e-6


def test_invalid_limits_are_rejected() -> None:
    with pytest.raises(ValueError, match="rate"):
        TokenBucket(0)
    with pytest.raises(ValueError, match="capacity"):
        TokenBucket(1, 0)
