"""Console output of the command-line tools on Windows (UTF-8)."""

from __future__ import annotations

import sys


def console_utf8() -> None:
    """MLflow prints run links with emoji: on a Windows console or pipe in cp1252 that raises
    UnicodeEncodeError in the middle of a run. Undecodable characters are escaped instead."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")
