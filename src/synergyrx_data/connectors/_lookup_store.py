"""Append-only store of API lookups, kept as sorted JSON Lines in the raw layer.

Lookup connectors (PubChem, ChEMBL, Europe PMC, PubTator) answer one key at a time. The
store remembers every answer, including "not found", so a key is asked only once and a
second run performs no request at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from synergyrx_data.http.cache import atomic_write_bytes


class LookupStore:
    def __init__(self, path: Path, key_fields: tuple[str, ...]) -> None:
        self.path = path
        self.key_fields = key_fields
        self.records: dict[tuple[str, ...], dict[str, Any]] = {}
        self._dirty = False
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        self.records[self._key(record)] = record

    def _key(self, record: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(record[f]) for f in self.key_fields)

    def __contains__(self, key: tuple[str, ...] | str) -> bool:
        return (key if isinstance(key, tuple) else (key,)) in self.records

    def __len__(self) -> int:
        return len(self.records)

    def add(self, record: dict[str, Any]) -> None:
        self.records[self._key(record)] = record
        self._dirty = True

    def save(self) -> bool:
        """Write the store if it changed. Sorted, so equal content gives equal bytes."""
        if not self._dirty:
            return False
        lines = [
            json.dumps(self.records[key], ensure_ascii=False, sort_keys=True)
            for key in sorted(self.records)
        ]
        atomic_write_bytes(self.path, ("\n".join(lines) + "\n").encode("utf-8"))
        self._dirty = False
        return True

    def rows(self) -> list[dict[str, Any]]:
        return [self.records[key] for key in sorted(self.records)]
