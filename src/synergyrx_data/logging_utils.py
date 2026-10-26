"""Structured JSON logging with secret redaction."""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_LOGGER = "synergyrx"
REDACTED = "[REDACTED]"

# Query-string parameters whose values must never reach a log line or a file on disk.
SECRET_PARAMS = ("api_key", "apikey", "access_token", "token", "email")
_PARAM_RE = re.compile(rf"(?i)\b({'|'.join(SECRET_PARAMS)})=([^&\s\"']+)")
_secret_values: set[str] = set()

_STANDARD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


def register_secrets(values: list[str]) -> None:
    """Remember literal secret values so they are masked wherever they appear."""
    _secret_values.update(v for v in values if v and len(v) >= 4)


def redact(text: str) -> str:
    """Mask secret query parameters and any registered secret value."""
    text = _PARAM_RE.sub(lambda m: f"{m.group(1)}={REDACTED}", text)
    for value in _secret_values:
        if value in text:
            text = text.replace(value, REDACTED)
    return text


def _redact_any(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: _redact_any(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_redact_any(v) for v in value]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": redact(record.getMessage()),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = _redact_any(value)
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extras = {
            k: _redact_any(v)
            for k, v in record.__dict__.items()
            if k not in _STANDARD_ATTRS and not k.startswith("_")
        }
        suffix = " " + " ".join(f"{k}={v}" for k, v in extras.items()) if extras else ""
        return f"{record.levelname:<7} {record.name}: {redact(record.getMessage())}{suffix}"


def setup_logging(
    level: str = "INFO",
    *,
    json_logs: bool = True,
    file: Path | None = None,
    secrets: list[str] | None = None,
) -> logging.Logger:
    """Configure the package logger. Logs go to stderr so stdout stays clean for reports."""
    register_secrets(secrets or [])
    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(level.upper())
    logger.handlers.clear()
    logger.propagate = False
    formatter: logging.Formatter = JsonFormatter() if json_logs else TextFormatter()
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if file is not None:
        file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(file, encoding="utf-8")
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    # httpx logs full request URLs at INFO, which would include api_key.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit one structured event: `event` is a stable name, `fields` are its attributes."""
    logger.log(level, event, extra=fields)
