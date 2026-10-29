from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from fastapi.testclient import TestClient


def test_summary_counts_rows_and_reports_missing_tables(user: TestClient) -> None:
    assert user.get("/api/public/summary").json() == {
        "drugs": 3,
        "cell_lines": 2,
        "synergy_measurements": 4,
        "abstracts": None,  # not built in this test
    }


def test_table_sizes_are_public_but_rows_are_not(anonymous: TestClient) -> None:
    """The introduction page shows the counts to visitors; nothing else is open."""
    summary = anonymous.get("/api/public/summary")
    assert summary.status_code == 200
    assert summary.json()["drugs"] == 3
    assert set(summary.json()) == {"drugs", "cell_lines", "synergy_measurements", "abstracts"}
    for path in ("/api/catalog/drugs", "/api/catalog/cell-lines"):
        assert anonymous.get(path).status_code == 401, path


def test_drugs_are_listed_most_measured_first(user: TestClient) -> None:
    body = user.get("/api/catalog/drugs").json()
    assert body["total"] == 3
    assert [item["name"] for item in body["items"]] == ["Lapatinib", "Erlotinib", "Tranylcypromine"]
    lapatinib, _, tranylcypromine = body["items"]
    assert lapatinib["pubchem_cid"] is None
    assert len(tranylcypromine["synonyms"]) == 5  # capped
    assert set(lapatinib) == {
        "drug_id",
        "name",
        "synonyms",
        "pubchem_cid",
        "chembl_id",
        "smiles",
        "sources",
        "n_measurements",
    }


def test_drug_search_and_pagination(user: TestClient) -> None:
    def names(**params: object) -> list[str]:
        return [d["name"] for d in user.get("/api/catalog/drugs", params=params).json()["items"]]

    assert names(q="tarceva") == ["Erlotinib"]  # by synonym
    assert names(q="CHEMBL553") == ["Erlotinib"]  # by identifier
    assert names(q="INIB") == ["Lapatinib", "Erlotinib"]
    assert names(q="erlotinib hcl") == ["Erlotinib"]  # every word must match
    assert names(q="nothing-like-this") == []
    assert names(page=2, page_size=2) == ["Tranylcypromine"]
    assert names(page=9) == []


def test_cell_lines(user: TestClient) -> None:
    body = user.get("/api/catalog/cell-lines", params={"q": "leukemia"}).json()
    assert body["total"] == 1
    assert body["items"][0]["name"] == "HL-60"
    assert body["items"][0]["tissue"] == "Myeloid"


def test_a_rebuilt_table_is_picked_up(user: TestClient, processed_dir: Path) -> None:
    assert user.get("/api/catalog/cell-lines").json()["total"] == 2
    path = processed_dir / "cell_lines.parquet"
    table = pq.read_table(path)
    pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), path)
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000_000))
    assert user.get("/api/catalog/cell-lines").json()["total"] == 3
    assert user.get("/api/public/summary").json()["cell_lines"] == 3


def test_missing_or_broken_table_is_a_503_not_a_crash(
    user: TestClient, processed_dir: Path
) -> None:
    (processed_dir / "cell_lines.parquet").unlink()
    (processed_dir / "drugs.parquet").write_bytes(b"half-written")
    for path in ("/api/catalog/cell-lines", "/api/catalog/drugs"):
        response = user.get(path)
        assert response.status_code == 503, path
        assert "synergyrx clean" in response.json()["detail"]
    summary = user.get("/api/public/summary").json()
    assert summary["drugs"] is None and summary["cell_lines"] is None
