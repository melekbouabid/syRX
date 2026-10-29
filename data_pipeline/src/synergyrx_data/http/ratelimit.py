"""Per-host rate limiting with token buckets."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from synergyrx_data.config import HostLimit

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


class TokenBucket:
    """Token bucket: `rate` tokens per second, at most `capacity` stored.

    Waiters are served in arrival order because the lock is held while sleeping.
    With `capacity=1` requests are evenly spaced at `1 / rate` seconds.
    """

    def __init__(
        self,
        rate: float,
        capacity: int = 1,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.rate = float(rate)
        self.capacity = capacity
        self._clock = clock
        self._sleep = sleep
        self._tokens = float(capacity)
        self._updated = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
        self._updated = now

    def slow_down(self, rate: float) -> None:
        """Lower the rate (never raise it), e.g. to honour a robots.txt Crawl-delay."""
        if 0 < rate < self.rate:
            self._refill()
            self.rate = rate

    async def acquire(self) -> float:
        """Take one token, waiting if needed. Returns the seconds spent waiting."""
        waited = 0.0
        async with self._lock:
            self._refill()
            while self._tokens < 1.0 - 1e-9:
                delay = (1.0 - self._tokens) / self.rate
                await self._sleep(delay)
                waited += delay
                self._refill()
            self._tokens = max(0.0, self._tokens - 1.0)
        return waited


class HostRateLimiter:
    """One token bucket and one concurrency semaphore per host."""

    def __init__(
        self,
        default: HostLimit,
        hosts: dict[str, HostLimit] | None = None,
        *,
        safety_factor: float = 1.0,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._default = default
        self._hosts = {h.lower(): limit for h, limit in (hosts or {}).items()}
        self._safety = safety_factor
        self._clock = clock
        self._sleep = sleep
        self._buckets: dict[str, TokenBucket] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    def limit_for(self, host: str) -> HostLimit:
        """Exact host first, then the closest configured parent domain, then the default."""
        host = host.lower()
        if host in self._hosts:
            return self._hosts[host]
        labels = host.split(".")
        for i in range(1, len(labels) - 1):
            parent = ".".join(labels[i:])
            if parent in self._hosts:
                return self._hosts[parent]
        return self._default

    def bucket(self, host: str) -> TokenBucket:
        host = host.lower()
        if host not in self._buckets:
            limit = self.limit_for(host)
            self._buckets[host] = TokenBucket(
                limit.rate * self._safety, limit.burst, clock=self._clock, sleep=self._sleep
            )
        return self._buckets[host]

    def semaphore(self, host: str) -> asyncio.Semaphore:
        host = host.lower()
        if host not in self._semaphores:
            self._semaphores[host] = asyncio.Semaphore(self.limit_for(host).max_concurrency)
        return self._semaphores[host]

    def set_min_interval(self, host: str, seconds: float) -> None:
        if seconds > 0:
            self.bucket(host).slow_down(1.0 / seconds)

    async def acquire(self, host: str) -> float:
        return await self.bucket(host).acquire()
