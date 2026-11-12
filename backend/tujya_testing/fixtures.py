"""Web app fixtures, shared by every test suite that calls the API (pytest plugin).

Each test gets its own application on a fresh SQLite file.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

ADMIN = {"email": "admin@gmail.com", "password": "admin"}
USER = {"email": "user@gmail.com", "password": "user"}


@pytest.fixture
def pharma_url(tmp_path: Path) -> str:
    """Medicine tables of this test: empty until a test asks for `pharma`."""
    return f"sqlite:///{(tmp_path / 'pharma.db').as_posix()}"


@pytest.fixture
def pharma(pharma_url: str, tmp_path: Path) -> str:
    from tujya_testing.pharma_data import load_pharma

    load_pharma(pharma_url, tmp_path / "pharma_inputs")
    return pharma_url


@pytest.fixture
def app(tmp_path: Path, pharma_url: str) -> Iterator[FastAPI]:
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        pharma_database_url=pharma_url,
        models_dir=tmp_path / "models",  # empty: tests that scan write fake models here
        demo_images_dir=tmp_path / "demo_images",
        demo_split_file=tmp_path / "demo_split.csv",
        demo_holdout_file=tmp_path / "demo_holdout.csv",
        demo_gallery_file=tmp_path / "demo_gallery.csv",
        demo_gradcam_dir=tmp_path / "demo_gradcam",
        # No drift reference unless a test writes one (backend/tests/test_drift.py).
        drift_reference_file=tmp_path / "drift_reference.json",
        # The fake RAG corpus is indexed in the pharma database (rag_fakes.build_index);
        # the Qdrant tests pass their own in-memory store to AgentHolder.
        vector_store="sql",
        # No real OCR in the API tests: box tests set a fake engine (tests/test_boxes.py).
        box_ocr_enabled=False,
    )
    application = create_app(settings)
    with TestClient(application):  # runs start-up: creates the tables and the first accounts
        yield application


@pytest.fixture
def browser(app: FastAPI) -> Callable[[], TestClient]:
    """Factory of independent clients, each with its own cookie jar, like separate browsers."""
    return lambda: TestClient(app)


@pytest.fixture
def anonymous(browser: Callable[[], TestClient]) -> TestClient:
    return browser()


def sign_in(client: TestClient, credentials: dict[str, str]) -> TestClient:
    response = client.post("/api/auth/login", json=credentials)
    assert response.status_code == 200, response.text
    return client


@pytest.fixture
def admin(browser: Callable[[], TestClient]) -> TestClient:
    return sign_in(browser(), ADMIN)


@pytest.fixture
def user(browser: Callable[[], TestClient]) -> TestClient:
    return sign_in(browser(), USER)
