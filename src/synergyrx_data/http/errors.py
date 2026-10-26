"""Exceptions raised by the HTTP layer."""

from __future__ import annotations


class HttpError(Exception):
    """Base class for every failure of the polite HTTP client."""

    def __init__(self, message: str, *, status: int | None = None, url: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.url = url


class RetryableHttpError(HttpError):
    """Transient failure (429, 5xx): retried with backoff, honouring `Retry-After`."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        url: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, status=status, url=url)
        self.retry_after = retry_after


class NotFoundError(HttpError):
    """404 / 410. A normal answer for lookups, a dead link for downloads. Never retried."""


class AccessControlError(HttpError):
    """401 / 403, a login wall, a CAPTCHA or a bot check. Never retried, never bypassed."""


class RobotsDisallowedError(HttpError):
    """robots.txt forbids this path for automated clients."""


class GuardrailViolation(HttpError):
    """A request was refused by the compliance guardrails before reaching the network."""


class ResponseTooLargeError(HttpError):
    """The response exceeded the configured maximum size."""


class ContentTypeError(HttpError):
    """The response is not the kind of content that was requested."""


class IntegrityError(HttpError):
    """A downloaded file does not match its expected size or checksum."""


class CircuitOpenError(HttpError):
    """The source failed repeatedly and is paused until its reset time."""
