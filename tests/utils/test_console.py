"""Console output helper (src/utils/console.py)."""

from __future__ import annotations

import io
import sys

import pytest

from src.utils.console import console_utf8


def test_console_utf8_escapes_what_the_console_cannot_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    console_utf8()
    print("run \U0001f3c3 done")
    stream.flush()
    assert buffer.getvalue().decode("cp1252").strip() == r"run \U0001f3c3 done"
