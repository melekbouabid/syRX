"""Schema drift detection between data versions.

Every time a table is written, its column names, types and row count are compared with the
previous snapshot of the same table. A renamed column in a new DrugComb release then shows
up in the quality report instead of silently producing an all-null column downstream.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from synergyrx_data.manifest import utc_now_iso

# Wide matrices (expression) are fingerprinted rather than listed column by column.
WIDE_TABLE_COLUMNS = 500
HISTORY_LENGTH = 20


@dataclass
class DriftFinding:
    table: str
    kind: str  # first_snapshot | column_added | column_removed | dtype_changed | rows_changed
    detail: str
    severity: str = "warning"  # info | warning

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def snapshot(frame: pd.DataFrame, tag: str, *, is_sample: bool = False) -> dict[str, Any]:
    columns = {str(c): str(t) for c, t in frame.dtypes.items()}
    shot: dict[str, Any] = {
        "taken_at": utc_now_iso(),
        "tag": tag,
        "is_sample": is_sample,
        "rows": len(frame),
        "n_columns": len(columns),
    }
    if len(columns) > WIDE_TABLE_COLUMNS:
        digest = hashlib.sha256("\n".join(f"{c}:{t}" for c, t in columns.items()).encode())
        shot["columns_sha256"] = digest.hexdigest()
    else:
        shot["columns"] = columns
    return shot


def compare(
    table: str, previous: dict[str, Any], current: dict[str, Any], row_tolerance: float = 0.2
) -> list[DriftFinding]:
    findings: list[DriftFinding] = []
    if "columns" in previous and "columns" in current:
        before, after = previous["columns"], current["columns"]
        for column in sorted(set(after) - set(before)):
            findings.append(
                DriftFinding(table, "column_added", f"{column} ({after[column]})", "info")
            )
        for column in sorted(set(before) - set(after)):
            findings.append(DriftFinding(table, "column_removed", column))
        for column in sorted(set(before) & set(after)):
            if before[column] != after[column]:
                findings.append(
                    DriftFinding(
                        table, "dtype_changed", f"{column}: {before[column]} -> {after[column]}"
                    )
                )
    elif previous.get("columns_sha256") != current.get("columns_sha256"):
        findings.append(
            DriftFinding(
                table,
                "column_removed"
                if current["n_columns"] < previous["n_columns"]
                else "column_added",
                f"column set changed: {previous['n_columns']} -> {current['n_columns']} columns",
            )
        )
    # Row counts are only comparable between two full (non-sample) builds.
    if not previous.get("is_sample") and not current.get("is_sample") and previous["rows"]:
        change = (current["rows"] - previous["rows"]) / previous["rows"]
        if abs(change) > row_tolerance:
            findings.append(
                DriftFinding(
                    table,
                    "rows_changed",
                    f"{previous['rows']} -> {current['rows']} rows ({change:+.0%})",
                )
            )
    return findings


class DriftTracker:
    """Keeps the snapshot history of each table under `data/state/schemas/`."""

    def __init__(self, directory: Path, *, read_only: bool = False) -> None:
        self.directory = directory
        self.read_only = read_only

    def _path(self, table: str) -> Path:
        return self.directory / f"{table.replace('/', '__')}.json"

    def tables(self) -> list[str]:
        """Every table that has at least one snapshot."""
        if not self.directory.is_dir():
            return []
        return sorted(p.stem.replace("__", "/") for p in self.directory.glob("*.json"))

    def last_change(self, table: str) -> list[DriftFinding]:
        """Differences between the two most recent snapshots of `table`, dated."""
        history = self.history(table)
        if len(history) < 2:
            return []
        previous, current = history[-2], history[-1]
        when = f" [{previous['tag']}, {previous['taken_at'][:10]} -> {current['tag']}, "
        when += f"{current['taken_at'][:10]}]"
        findings = compare(table, previous, current)
        return [DriftFinding(f.table, f.kind, f.detail + when, f.severity) for f in findings]

    def history(self, table: str) -> list[dict[str, Any]]:
        path = self._path(table)
        if not path.is_file():
            return []
        history: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        return history

    def check(
        self, table: str, frame: pd.DataFrame, tag: str, *, is_sample: bool = False
    ) -> list[DriftFinding]:
        """Compare `frame` with the last snapshot of `table`, then record the new one."""
        history = self.history(table)
        current = snapshot(frame, tag, is_sample=is_sample)
        if history:
            findings = compare(table, history[-1], current)
        else:
            findings = [DriftFinding(table, "first_snapshot", f"{len(frame)} rows", "info")]
        unchanged = bool(history) and not findings and history[-1]["rows"] == current["rows"]
        if not self.read_only and not unchanged:
            self.directory.mkdir(parents=True, exist_ok=True)
            history = [*history, current][-HISTORY_LENGTH:]
            self._path(table).write_text(json.dumps(history, indent=1), encoding="utf-8")
        return findings
