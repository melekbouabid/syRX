"""Checks on docker-compose.yml: the Adminer service stays local and pinned."""

import re
from pathlib import Path

import pytest
import yaml

COMPOSE_FILE = Path(__file__).resolve().parents[1] / "docker-compose.yml"


@pytest.fixture(scope="module")
def services() -> dict:
    return yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))["services"]


def test_adminer_service_exists(services: dict) -> None:
    assert "adminer" in services


def test_adminer_port_bound_to_localhost(services: dict) -> None:
    ports = services["adminer"]["ports"]
    assert ports
    for port in ports:
        assert str(port).startswith("127.0.0.1:"), port


def test_adminer_image_pinned(services: dict) -> None:
    image = services["adminer"]["image"]
    name, _, tag = image.partition(":")
    assert name == "adminer"
    assert re.fullmatch(r"\d+\.\d+\.\d+", tag), image


def test_adminer_has_no_password(services: dict) -> None:
    environment = services["adminer"].get("environment", {})
    assert environment.get("ADMINER_DEFAULT_SERVER") == "postgres"
    assert not any("PASSWORD" in key for key in environment)
