"""Read-only access to the tables built by the data pipeline (`synergyrx clean`).

The pipeline may rewrite a table while the API is running, so every file is read in one
short pass, kept in memory, and read again only when its modification time changes.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

DRUG_COLUMNS = [
    "drug_id",
    "name",
    "synonyms",
    "pubchem_cid",
    "chembl_id",
    "smiles",
    "sources",
    "n_measurements",
]
CELL_LINE_COLUMNS = [
    "cell_line_id",
    "name",
    "tissue",
    "disease",
    "depmap_id",
    "sources",
    "n_measurements",
]
MAX_SYNONYMS = 5


class TableUnavailable(Exception):
    """The table has not been built yet, or is being rewritten right now."""


class Catalog:
    def __init__(self, processed_dir: Path) -> None:
        self._dir = processed_dir
        self._lock = threading.Lock()
        self._rows: dict[str, tuple[int, list[dict[str, Any]]]] = {}
        self._counts: dict[str, tuple[int, int]] = {}

    def drugs(self) -> list[dict[str, Any]]:
        return self._table("drugs", DRUG_COLUMNS, search_columns=("name", "synonyms", "chembl_id"))

    def cell_lines(self) -> list[dict[str, Any]]:
        return self._table(
            "cell_lines", CELL_LINE_COLUMNS, search_columns=("name", "tissue", "disease")
        )

    def row_count(self, table: str) -> int | None:
        """Number of rows, read from the file footer only. None when unavailable."""
        path = self._dir / f"{table}.parquet"
        try:
            mtime = path.stat().st_mtime_ns
            cached = self._counts.get(table)
            if cached is None or cached[0] != mtime:
                with path.open("rb") as handle:
                    cached = (mtime, pq.read_metadata(handle).num_rows)
                self._counts[table] = cached
            return cached[1]
        except (OSError, pa.ArrowException):
            return None

    def _table(
        self, table: str, columns: list[str], search_columns: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        path = self._dir / f"{table}.parquet"
        with self._lock:
            try:
                mtime = path.stat().st_mtime_ns
                cached = self._rows.get(table)
                if cached is None or cached[0] != mtime:
                    data = pq.read_table(io.BytesIO(path.read_bytes()), columns=columns)
                    cached = (mtime, _prepare(data.to_pylist(), search_columns))
                    self._rows[table] = cached
            except (OSError, pa.ArrowException) as exc:
                raise TableUnavailable(table) from exc
            return cached[1]


def _prepare(rows: list[dict[str, Any]], search_columns: tuple[str, ...]) -> list[dict[str, Any]]:
    """Most-measured first, with a lower-case `_search` text to match queries against."""
    for row in rows:
        haystack: list[str] = []
        for column in search_columns:
            value = row.get(column)
            values = value if isinstance(value, list) else [value]
            haystack.extend(str(item) for item in values if item)
        row["_search"] = " ".join(haystack).lower()
        if "synonyms" in row:
            row["synonyms"] = list(row["synonyms"] or [])[:MAX_SYNONYMS]
    rows.sort(key=lambda row: (-(row.get("n_measurements") or 0), str(row["name"]).lower()))
    return rows


def search(rows: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Rows whose searchable text contains every word of the query."""
    words = query.lower().split()
    if not words:
        return rows
    return [row for row in rows if all(word in row["_search"] for word in words)]


def public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}
