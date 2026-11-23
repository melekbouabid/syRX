"""Provenance columns carried by every record, and small Parquet helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROVENANCE_COLUMNS = ["source", "source_url", "retrieved_at", "source_version", "license"]


@dataclass(frozen=True)
class Provenance:
    source: str
    source_url: str
    retrieved_at: str
    source_version: str
    license: str

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return `frame` with the five provenance columns appended."""
        columns = pd.DataFrame(
            {
                "source": self.source,
                "source_url": self.source_url,
                "retrieved_at": self.retrieved_at,
                "source_version": self.source_version,
                "license": self.license,
            },
            index=frame.index,
        )
        # One concat rather than five inserts: the expression matrix has 19,000 columns.
        kept = frame.drop(columns=[c for c in PROVENANCE_COLUMNS if c in frame.columns])
        return pd.concat([kept, columns], axis=1)


def write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    """Write Parquet atomically: readers never see a half-written table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False, engine="pyarrow", compression="zstd")
    tmp.replace(path)
    return path


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame | None:
    """Read a Parquet table, or None if it has not been produced yet."""
    if not path.is_file():
        return None
    return pd.read_parquet(path, columns=columns, engine="pyarrow")
