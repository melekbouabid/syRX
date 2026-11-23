"""Settings prefix: only TUJYA_* names are read, from the environment or the .env file."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import ENV_PREFIX, Settings

OTHER_PREFIX = "OTHER_"  # any name without the TUJYA_ prefix


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for prefix in (ENV_PREFIX, OTHER_PREFIX):
        for name in ("SESSION_HOURS", "ADMIN_EMAIL", "LLM_MODEL"):
            monkeypatch.delenv(prefix + name, raising=False)


def test_new_prefix_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TUJYA_SESSION_HOURS", "3")
    assert Settings(_env_file=None).session_hours == 3


def test_another_prefix_is_ignored_in_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTHER_SESSION_HOURS", "5")
    assert Settings(_env_file=None).session_hours == 12  # the default


def test_the_prefix_is_read_next_to_other_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTHER_SESSION_HOURS", "5")
    monkeypatch.setenv("TUJYA_SESSION_HOURS", "3")
    assert Settings(_env_file=None).session_hours == 3


def test_only_prefixed_names_of_an_env_file_apply(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OTHER_ADMIN_EMAIL=owner@example.org\nOTHER_LLM_MODEL=llama3.2:3b\n"
        "TUJYA_LLM_MODEL=qwen2.5:3b\n",
        encoding="utf-8",
    )
    settings = Settings(_env_file=env_file)
    assert settings.admin_email != "owner@example.org"  # not prefixed: ignored
    assert settings.llm_model == "qwen2.5:3b"


def test_defaults_apply_without_any_prefix() -> None:
    assert Settings(_env_file=None).session_hours == 12
