"""Acceptance tests of the two custom Git hooks (tools/hooks), with the cases of the advanced
Git lab (accepted / rejected) and a few cases of this repository.

The hooks run as pre-commit runs them: a separate Python process given file paths, judged
by its exit code and its output.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parents[1] / "hooks"


def run_hook(script: str, *args: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 (the interpreter running the tests, fixed scripts)
        [sys.executable, str(HOOKS / script), *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )


# --- secret detection (pre-commit) ------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        # Lab, step D.
        'API_KEY = "sk-abcdef123456"',
        'aws_id = "AKIAIOSFODNN7EXAMPLE"',
        # Other names, syntaxes and files of this repository.
        'password = "hunter2hunter2"',
        'DB_PASSWORD = "postgres-prod-1"',
        'SECRET_KEY = "change-me-please"',
        'api-key: "0123456789abcdef"',
        'settings = {"token": "ghp_0123456789"}',
        'password: str = "correct-horse"',
        'client = Client(apikey="0123456789")',
        'headers = {"Authorization": "Bearer " + "sk-proj-0123456789abcdef"}',
    ],
)
def test_secret_is_rejected_with_file_and_line(tmp_path: Path, line: str) -> None:
    source = tmp_path / "config.py"
    source.write_text(f"import os\n\n{line}\n", encoding="utf-8")

    result = run_hook("check_secrets.py", source)

    assert result.returncode == 1
    assert f"{source.as_posix()}:3: potential secret detected" in result.stdout
    assert line.split("=")[-1].strip().strip('"') not in result.stdout  # never echoes it


@pytest.mark.parametrize(
    "line",
    [
        # Lab, step D.
        'API_KEY = os.environ.get("API_KEY")',
        'password = "abc"',
        # No value, a comparison, a value too short, a name that only contains the word.
        "password = settings.password",
        'if token == "expired-token":',
        'TOKEN = "short"',
        'password_hash = "pbkdf2_sha256$600000"',
        'max_tokens = "4096"',
        'aws = "AKIA-short"',
        'risk = "task-scheduler-option"',
        # Known false positive, marked as such.
        'FAKE_API_KEY = "0123456789abcdef"  # pragma: allowlist secret',
    ],
)
def test_clean_line_is_accepted(tmp_path: Path, line: str) -> None:
    source = tmp_path / "config.py"
    source.write_text(f"import os\n\n{line}\n", encoding="utf-8")

    result = run_hook("check_secrets.py", source)

    assert result.returncode == 0, result.stdout
    assert result.stdout == ""


def test_every_finding_of_every_file_is_reported(tmp_path: Path) -> None:
    first = tmp_path / "a.py"
    first.write_text('x = 1\ntoken = "0123456789"\npassword = "0123456789"\n', encoding="utf-8")
    second = tmp_path / "b.yaml"
    second.write_text('key_id: "AKIAIOSFODNN7EXAMPLE"\n', encoding="utf-8")
    clean = tmp_path / "c.py"
    clean.write_text("y = 2\n", encoding="utf-8")

    result = run_hook("check_secrets.py", first, second, clean)

    assert result.returncode == 1
    reported = [line.split(": potential")[0] for line in result.stdout.splitlines() if ":" in line]
    assert reported[:3] == [
        f"{first.as_posix()}:2",
        f"{first.as_posix()}:3",
        f"{second.as_posix()}:1",
    ]


def test_binary_file_does_not_crash_the_hook(tmp_path: Path) -> None:
    blob = tmp_path / "weights.bin"
    blob.write_bytes(bytes(range(256)) * 4)

    assert run_hook("check_secrets.py", blob).returncode == 0


def test_no_file_is_a_success() -> None:
    assert run_hook("check_secrets.py").returncode == 0


# --- commit message convention (commit-msg) ----------------------------------------------


def commit_msg(tmp_path: Path, message: str) -> subprocess.CompletedProcess[str]:
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text(message, encoding="utf-8")
    return run_hook("check_commit_msg.py", path)


@pytest.mark.parametrize(
    "message",
    [
        # Lab, step E.
        "docs: add a note to the README",
        "feat(models): add config loader",
        # Types and scopes of this repository.
        "fix(backend): wrap two comment lines over the 100-character limit",
        "exp(detector): yolo11s at 640 px with mosaic augmentation",
        "chore(pre-commit/hooks): add the commit-msg hook",
        "ci: lint the frontend",
        "feat(api)!: rename the scan endpoint",
        # A body and Git's comment lines do not matter; only the first line is checked.
        "# Please enter the commit message\ntest(hooks): cover the lab cases\n\nBody text.\n",
        # Written by Git itself.
        "Merge branch 'chore/conventions-hooks'",
        'Revert "feat(models): add config loader"',
        "fixup! feat(models): add config loader",
        "squash! wip",
    ],
)
def test_conventional_message_is_accepted(tmp_path: Path, message: str) -> None:
    result = commit_msg(tmp_path, message)

    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize(
    "message",
    [
        # Lab, step E.
        "added a note",
        "fix: typo",  # description shorter than 5 characters
        "Docs: update readme",  # type in upper case
        # Other mistakes.
        "feature: add config loader",  # unknown type
        "feat add config loader",  # no colon
        "feat:add config loader",  # no space after the colon
        "feat(): add config loader",  # empty scope
        "feat(my scope): add config loader",  # space in the scope
        "wip",
        "",
        "# only a comment\n",
    ],
)
def test_bad_message_is_rejected_with_help(tmp_path: Path, message: str) -> None:
    result = commit_msg(tmp_path, message)

    assert result.returncode == 1
    assert "Invalid commit message" in result.stdout
    assert "type(scope): description" in result.stdout
    assert "feat, fix, exp, data, docs, refactor, chore" in result.stdout
