"""Pre-commit hook: reject staged files that contain a hard-coded secret.

Finds secret-like assignments, AWS key ids and `sk-...` keys; never prints the value.
A line with `pragma: allowlist secret` is skipped.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ALLOWLIST_MARKER = "pragma: allowlist secret"

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "hard-coded credential",
        re.compile(r"""(?ix)
            (?<![a-z0-9])[a-z0-9_-]*?(?:api[_-]?key|secret|token|password)(?:[_-]key)?(?![a-z0-9])
            ["']?\s*                    # closing quote of a dict key
            (?::\s*[\w\[\]|., ]+?\s*)?  # type annotation: password: str = "..."
            [:=]\s*
            [rbuf]{0,2}(["'])[^"'\n]{8,}\1
            """),
    ),
    ("AWS access key id", re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])")),
    ("API key sk-...", re.compile(r"(?<![\w-])sk-[A-Za-z0-9_-]{10,}")),
]


def find_secrets(lines: list[str]) -> list[tuple[int, str]]:
    """(line number, kind) of every line that looks like it holds a secret."""
    findings = []
    for number, line in enumerate(lines, start=1):
        if ALLOWLIST_MARKER in line:
            continue
        for kind, pattern in PATTERNS:
            if pattern.search(line):
                findings.append((number, kind))
                break
    return findings


def check_file(path: Path) -> list[str]:
    """The error lines for one file (empty when it is clean)."""
    # errors="ignore": a file that looks binary must not crash the hook.
    with path.open(encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()
    return [
        f"{path.as_posix()}:{number}: potential secret detected ({kind})"
        for number, kind in find_secrets(lines)
    ]


def main(argv: list[str]) -> int:
    errors = [error for name in argv for error in check_file(Path(name))]
    for error in errors:
        print(error)
    if errors:
        print("Move the value to .env (git-ignored) and read it from the environment.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
