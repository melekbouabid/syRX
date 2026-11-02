"""Settings prefixes: TUJYA_* is read first; the legacy SYNERGYRX_* names still work (D86)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import ENV_PREFIX, LEGACY_ENV_PREFIX, Settings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for prefix in (ENV_PREFIX, LEGACY_ENV_PREFIX):
        for name in ("SESSION_HOURS", "ADMIN_EMAIL", "LLM_MODEL"):
            monkeypatch.delenv(prefix + name, raising=False)


def test_new_prefix_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TUJYA_SESSION_HOURS", "3")
    assert Settings(_env_file=None).session_hours == 3


def test_legacy_prefix_is_still_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYNERGYRX_SESSION_HOURS", "5")
    assert Settings(_env_file=None).session_hours == 5


def test_new_prefix_wins_over_the_legacy_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNERGYRX_SESSION_HOURS", "5")
    monkeypatch.setenv("TUJYA_SESSION_HOURS", "3")
    assert Settings(_env_file=None).session_hours == 3


def test_legacy_names_in_an_existing_env_file_still_apply(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SYNERGYRX_ADMIN_EMAIL=owner@example.org\nSYNERGYRX_LLM_MODEL=llama3.2:3b\n"
        "TUJYA_LLM_MODEL=qwen2.5:3b\n",
        encoding="utf-8",
    )
    settings = Settings(_env_file=env_file)
    assert settings.admin_email == "owner@example.org"  # legacy name only: used
    assert settings.llm_model == "qwen2.5:3b"  # both names: the new one wins


def test_defaults_apply_without_any_prefix() -> None:
    assert Settings(_env_file=None).session_hours == 12
