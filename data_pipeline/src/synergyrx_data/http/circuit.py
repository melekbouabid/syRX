"""Circuit breaker: stop calling a source that keeps failing."""

from __future__ import annotations

import time
from collections.abc import Callable

from synergyrx_data.http.errors import CircuitOpenError


class CircuitBreaker:
    """Closed until `failure_threshold` consecutive failures, then open for `reset_seconds`.

    After the pause a single trial call is allowed (half-open): success closes the circuit,
    failure opens it again.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_seconds: float = 300.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self.last_error: str | None = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self.reset_seconds:
            return "half_open"
        return "open"

    def before_call(self) -> None:
        if self.state == "open":
            assert self._opened_at is not None
            remaining = self.reset_seconds - (self._clock() - self._opened_at)
            raise CircuitOpenError(
                f"circuit for {self.name!r} is open for another {remaining:.0f}s "
                f"(last error: {self.last_error})"
            )

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self, error: str, *, trip: bool = False) -> None:
        """Count a failure. `trip=True` opens the circuit at once (access refused)."""
        self.last_error = error
        self._failures += 1
        if trip or self._failures >= self.failure_threshold or self._opened_at is not None:
            self._opened_at = self._clock()
