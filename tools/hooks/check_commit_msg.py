"""Commit-msg hook: the first line must be `type(scope): description` (Conventional Commits).

Git's own merge, revert, fixup!, squash! and amend! messages are accepted too.
Run: `python tools/hooks/check_commit_msg.py .git/COMMIT_EDITMSG`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TYPES = ("feat", "fix", "exp", "data", "docs", "refactor", "chore", "test", "ci", "perf", "style")

CONVENTIONAL = re.compile(rf"^(?:{'|'.join(TYPES)})(?:\([A-Za-z0-9_/-]+\))?!?: \S.{{4,}}$")
GENERATED_BY_GIT = re.compile(r"^(?:Merge |Revert \"|(?:fixup|squash|amend)! )")

HELP = f"""Invalid commit message: {{subject!r}}
Expected: type(scope): description   or   type: description
  type        one of {", ".join(TYPES)} (lower case)
  scope       optional; letters, digits, _, - or /
  description at least 5 characters
Examples: "docs: add a note to the README", "feat(models): add config loader"
"""


def subject(message: str) -> str:
    """The first line of the message, skipping the comment lines Git adds (`#`)."""
    for line in message.splitlines():
        if not line.startswith("#"):
            return line.rstrip()
    return ""


def is_valid(message: str) -> bool:
    first = subject(message)
    return bool(CONVENTIONAL.match(first) or GENERATED_BY_GIT.match(first))


def main(argv: list[str]) -> int:
    message = Path(argv[0]).read_text(encoding="utf-8", errors="ignore")
    if is_valid(message):
        return 0
    print(HELP.format(subject=subject(message)), end="")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
