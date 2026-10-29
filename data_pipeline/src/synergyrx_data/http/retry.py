"""Retry policy: exponential backoff with jitter, honouring `Retry-After`."""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt

from synergyrx_data.config import RetrySettings
from synergyrx_data.http.errors import RetryableHttpError

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse a `Retry-After` header (delta seconds or HTTP date) into seconds from now."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


def backoff_delay(
    attempt: int, base: float, cap: float, rng: Callable[[], float] = random.random
) -> float:
    """Equal-jitter backoff: half of the exponential ceiling plus a random half."""
    ceiling = min(cap, base * 2 ** max(0, attempt - 1))
    return ceiling / 2 + rng() * ceiling / 2


def is_retryable(exc: BaseException, max_retry_after: float) -> bool:
    if isinstance(exc, RetryableHttpError):
        # A server asking for a very long pause is telling us to stop for now.
        return exc.retry_after is None or exc.retry_after <= max_retry_after
    return isinstance(exc, httpx.TransportError)


def wait_seconds(
    exc: BaseException | None,
    attempt: int,
    settings: RetrySettings,
    rng: Callable[[], float] = random.random,
) -> float:
    """Never wait less than the server asked for; otherwise use jittered backoff."""
    delay = backoff_delay(attempt, settings.backoff_base, settings.backoff_max, rng)
    if isinstance(exc, RetryableHttpError) and exc.retry_after is not None:
        delay = max(delay, exc.retry_after)
    return delay


def build_retrying(
    settings: RetrySettings,
    *,
    sleep: Callable[[float], Awaitable[None]],
    rng: Callable[[], float] = random.random,
    on_retry: Callable[[RetryCallState], None] | None = None,
) -> AsyncRetrying:
    def _wait(state: RetryCallState) -> float:
        exc = state.outcome.exception() if state.outcome else None
        return wait_seconds(exc, state.attempt_number, settings, rng)

    return AsyncRetrying(
        stop=stop_after_attempt(settings.max_attempts),
        wait=_wait,
        retry=retry_if_exception(lambda exc: is_retryable(exc, settings.max_retry_after)),
        sleep=sleep,
        before_sleep=on_retry,
        reraise=True,
    )
