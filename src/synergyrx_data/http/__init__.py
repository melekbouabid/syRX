"""Polite HTTP stack: client, rate limiting, retries, cache, robots.txt, guardrails."""

from synergyrx_data.http.client import DownloadResult, FetchResult, PoliteClient, ProbeResult
from synergyrx_data.http.errors import (
    AccessControlError,
    CircuitOpenError,
    ContentTypeError,
    GuardrailViolation,
    HttpError,
    IntegrityError,
    NotFoundError,
    ResponseTooLargeError,
    RetryableHttpError,
    RobotsDisallowedError,
)

__all__ = [
    "AccessControlError",
    "CircuitOpenError",
    "ContentTypeError",
    "DownloadResult",
    "FetchResult",
    "GuardrailViolation",
    "HttpError",
    "IntegrityError",
    "NotFoundError",
    "PoliteClient",
    "ProbeResult",
    "ResponseTooLargeError",
    "RetryableHttpError",
    "RobotsDisallowedError",
]
