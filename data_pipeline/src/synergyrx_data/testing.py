"""Shared test fixtures (pytest plugin). Unit tests never touch the network: HTTP is mocked
with respx.

Used by the pipeline tests (tests/conftest.py) and by the module 2 corpus tests, which run
on this pipeline: `pytest_plugins = ["synergyrx_data.testing"]`. Needs the `dev` extra
(pytest, respx); nothing else in the package imports this module.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import pytest
import respx

from synergyrx_data.config import Secrets, Settings, load_settings
from synergyrx_data.connectors import allowed_hosts
from synergyrx_data.http.client import PoliteClient

# example.org is reserved for documentation (RFC 2606): this is not a real mailbox.
TEST_EMAIL = "pipeline-tests@example.org"
# The shipped configuration, loaded explicitly: the fixtures must not depend on the folder
# pytest is started from (a config.yaml in the working directory would otherwise be used).
SHIPPED_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"  # data_pipeline/config.yaml
FAKE_API_KEY = "0123456789abcdef0123456789abcdef0000"  # pragma: allowlist secret


class FakeClock:
    """A clock that only moves when something sleeps, so timing tests are instant and exact."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Default settings rooted in a temporary project, with fast limits and a test identity."""
    built = load_settings(
        SHIPPED_CONFIG,
        [
            "http.default_limit.rate=1000",
            "http.ncbi.rate_without_key=1000",
            "http.ncbi.rate_with_key=1000",
            "http.safety_factor=1.0",
            "http.retry.max_attempts=4",
            "http.retry.backoff_base=0.5",
            "logging.progress=false",
        ],
        project_root=tmp_path,
        environ={},
    )
    built.secrets = Secrets(contact_email=TEST_EMAIL)
    return built


@pytest.fixture
def mock_http() -> Iterator[respx.MockRouter]:
    """Every request must be mocked explicitly; an unmocked one fails the test."""
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        yield router


ClientFactory = Callable[..., PoliteClient]


@pytest.fixture
async def make_client(settings: Settings, clock: FakeClock) -> AsyncIterator[ClientFactory]:
    """Build clients on the fake clock; they are closed when the test ends."""
    clients: list[PoliteClient] = []

    def factory(custom: Settings | None = None, hosts: set[str] | None = None) -> PoliteClient:
        client = PoliteClient(
            custom or settings,
            allowed_hosts=hosts or allowed_hosts() | {"example.org"},
            sleep=clock.sleep,
            clock=clock,
            rng=lambda: 0.5,
        )
        clients.append(client)
        return client

    yield factory
    for client in clients:
        await client.aclose()


@pytest.fixture
def allow_robots(mock_http: respx.MockRouter) -> None:
    """robots.txt answers 404 everywhere, which means "no restrictions"."""
    mock_http.get(url__regex=r"https://[^/]+/robots\.txt$").respond(404)
