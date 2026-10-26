"""Each test gets its own application on a fresh SQLite file and a tiny processed folder."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

ADMIN = {"email": "admin@gmail.com", "password": "admin"}
USER = {"email": "user@gmail.com", "password": "user"}


def write_processed_tables(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    drugs = {
        "drug_id": ["D1", "D2", "D3"],
        "name": ["Erlotinib", "Lapatinib", "Tranylcypromine"],
        "synonyms": [["Tarceva", "erlotinib HCl"], ["Tykerb"], [f"syn{i}" for i in range(9)]],
        "pubchem_cid": pa.array([176870, None, 5530], type=pa.int64()),
        "chembl_id": ["CHEMBL553", None, "CHEMBL313833"],
        "smiles": ["C#C", None, "NC1CC1"],
        "sources": [["drugcomb", "tdc"], ["drugcomb"], ["drugcomb"]],
        "n_measurements": [100, 500, 2],
        "license": ["CC-BY-4.0"] * 3,  # a column the API does not expose
    }
    cell_lines = {
        "cell_line_id": ["ACH-000019", "ACH-000002"],
        "name": ["MCF7", "HL-60"],
        "tissue": ["Breast", "Myeloid"],
        "disease": ["Invasive Breast Carcinoma", "Acute Myeloid Leukemia"],
        "depmap_id": ["ACH-000019", "ACH-000002"],
        "sources": [["drugcomb"], ["drugcomb", "tdc"]],
        "n_measurements": [10, 20],
    }
    pq.write_table(pa.table(drugs), directory / "drugs.parquet")
    pq.write_table(pa.table(cell_lines), directory / "cell_lines.parquet")
    pq.write_table(
        pa.table({"pair_key": ["a", "b", "c", "d"]}), directory / "synergy_pairs.parquet"
    )


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "processed"
    write_processed_tables(directory)
    return directory


@pytest.fixture
def app(tmp_path: Path, processed_dir: Path) -> Iterator[FastAPI]:
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        processed_dir=processed_dir,
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
