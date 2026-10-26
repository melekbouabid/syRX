from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest
import yaml

from synergyrx_data.config import ConfigError, apply_overrides, load_secrets, load_settings
from synergyrx_data.http.client import PoliteClient
from synergyrx_data.logging_utils import (
    JsonFormatter,
    get_logger,
    log_event,
    redact,
    register_secrets,
    setup_logging,
)
from tests.conftest import FAKE_API_KEY, TEST_EMAIL

REPO = Path(__file__).resolve().parents[2]


def test_shipped_config_is_valid_and_matches_documented_limits() -> None:
    settings = load_settings(REPO / "config.yaml", environ={})
    assert settings.http.ncbi.rate_without_key == 3.0
    assert settings.http.ncbi.rate_with_key == 10.0
    assert settings.http.hosts["pubchem.ncbi.nlm.nih.gov"].rate == 5.0
    assert settings.http.require_contact_email is True
    # Licence-gated and optional sources ship disabled.
    for name in ("drugbank", "ddinter", "gdsc", "pubmed_baseline", "pubtator"):
        assert settings.sources[name].enabled is False
    assert settings.sources["ddinter"].options["terms_acknowledged"] is False


def test_overrides_including_dotted_host_names(tmp_path: Path) -> None:
    raw = apply_overrides(
        {}, ["http.hosts[zenodo.org].rate=0.05", "sample=25", "sources.tdc.enabled=false"]
    )
    assert raw["http"]["hosts"]["zenodo.org"]["rate"] == 0.05
    settings = load_settings(None, ["sample=25", "logging.level=DEBUG"], project_root=tmp_path)
    assert settings.sample == 25 and settings.logging.level == "DEBUG"
    with pytest.raises(ConfigError, match="key=value"):
        apply_overrides({}, ["no-equals-sign"])


def test_unknown_keys_and_secrets_in_yaml_are_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("http:\n  not_a_setting: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid configuration"):
        load_settings(config)
    config.write_text("secrets:\n  ncbi_api_key: abc\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"\.env"):
        load_settings(config)


def test_secrets_come_from_dotenv_and_are_never_serialised(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        f"NCBI_API_KEY={FAKE_API_KEY}\nCONTACT_EMAIL={TEST_EMAIL}\n", encoding="utf-8"
    )
    settings = load_settings(None, project_root=tmp_path, environ={})
    assert settings.secrets.api_key == FAKE_API_KEY
    assert settings.secrets.contact_email == TEST_EMAIL
    dumped = settings.model_dump_json() + repr(settings) + str(settings.secrets)
    assert FAKE_API_KEY not in dumped
    # The real environment wins over the file.
    assert load_secrets(tmp_path, {"NCBI_API_KEY": "from-env-123"}).api_key == "from-env-123"
    with pytest.raises(ValueError, match="CONTACT_EMAIL"):
        load_secrets(tmp_path, {"CONTACT_EMAIL": "not-an-email"})


def test_ncbi_rate_depends_on_api_key(tmp_path: Path) -> None:
    host = "eutils.ncbi.nlm.nih.gov"
    without = load_settings(None, project_root=tmp_path, environ={"CONTACT_EMAIL": TEST_EMAIL})
    with_key = load_settings(
        None, project_root=tmp_path, environ={"CONTACT_EMAIL": TEST_EMAIL, "NCBI_API_KEY": "k" * 36}
    )
    assert PoliteClient._host_limits(without)[host].rate == 3.0
    assert PoliteClient._host_limits(with_key)[host].rate == 10.0
    # A configured value can only lower the limit, never raise it past NCBI's.
    greedy = load_settings(
        None,
        [f"http.hosts[{host}].rate=50"],
        project_root=tmp_path,
        environ={"CONTACT_EMAIL": TEST_EMAIL},
    )
    assert PoliteClient._host_limits(greedy)[host].rate == 3.0


def test_redaction_of_parameters_and_registered_values() -> None:
    register_secrets([FAKE_API_KEY])
    line = redact(f"GET https://x.test/e?db=pubmed&api_key={FAKE_API_KEY}&email=a@b.org&retmax=5")
    assert FAKE_API_KEY not in line and "a@b.org" not in line
    assert "db=pubmed" in line and "retmax=5" in line
    assert FAKE_API_KEY not in redact(f"header value {FAKE_API_KEY} in free text")


def test_json_logs_are_structured_and_never_contain_secrets(tmp_path: Path) -> None:
    logger = setup_logging("INFO", json_logs=True, secrets=[FAKE_API_KEY, TEST_EMAIL])
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    try:
        log_event(
            get_logger("test"),
            logging.INFO,
            "http.response",
            url=f"https://eutils.test/efetch?api_key={FAKE_API_KEY}",
            params={"email": TEST_EMAIL, "nested": [f"key {FAKE_API_KEY}"]},
            status=200,
        )
        try:
            raise RuntimeError(f"failed with key {FAKE_API_KEY}")
        except RuntimeError:
            get_logger("test").exception("boom %s", FAKE_API_KEY)
    finally:
        logger.removeHandler(handler)
    output = stream.getvalue()
    assert FAKE_API_KEY not in output and TEST_EMAIL not in output
    first = json.loads(output.splitlines()[0])
    assert first["event"] == "http.response" and first["status"] == 200
    assert {"ts", "level", "logger"} <= set(first)


def test_no_secret_in_tracked_files() -> None:
    """Nothing that looks like a real NCBI key or a filled-in .env is committed."""
    example = (REPO / ".env.example").read_text(encoding="utf-8")
    for line in example.splitlines():
        if line.startswith(("CONTACT_EMAIL=", "NCBI_API_KEY=")):
            assert line.split("=", 1)[1] == ""
    assert ".env" in (REPO / ".gitignore").read_text(encoding="utf-8").split()
    forbidden = {"secrets", "api_key", "ncbi_api_key", "contact_email", "email", "password"}

    def keys(node: object) -> set[str]:
        if isinstance(node, dict):
            return {str(k) for k in node} | {k for v in node.values() for k in keys(v)}
        return set()

    config = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
    assert not (keys(config) & forbidden)
