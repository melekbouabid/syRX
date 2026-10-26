"""Shared fixtures. Unit tests never touch the network: HTTP is mocked with respx."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
import respx

from synergyrx_data.config import Secrets, Settings, load_settings
from synergyrx_data.connectors import allowed_hosts
from synergyrx_data.http.client import PoliteClient

# example.org is reserved for documentation (RFC 2606): this is not a real mailbox.
TEST_EMAIL = "pipeline-tests@example.org"
FAKE_API_KEY = "0123456789abcdef0123456789abcdef0000"


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
        None,
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
def mock_http() -> respx.MockRouter:
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
