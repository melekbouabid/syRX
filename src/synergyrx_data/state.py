"""Connector state that survives between runs (`data/state/connectors.json`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from synergyrx_data.manifest import file_lock, utc_now_iso

OK = "ok"
NEEDS_ATTENTION = "needs_attention"
MANUAL_STEP = "manual_step_required"
DISABLED = "disabled"
NEVER_RUN = "never_run"


class StateStore:
    """Small JSON store. Every update re-reads the file under a lock, so two pipeline
    processes running at the same time do not erase each other's entries."""

    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = path
        self.read_only = read_only
        self._data: dict[str, dict[str, Any]] = self._read()

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.path.is_file():
            return {}
        try:
            data: dict[str, dict[str, Any]] = json.loads(self.path.read_text(encoding="utf-8"))
        except ValueError:
            return {}
        return data

    def get(self, name: str) -> dict[str, Any]:
        return dict(self._data.get(name, {"state": NEVER_RUN}))

    def update(self, name: str, **fields: Any) -> None:
        fields["updated_at"] = utc_now_iso()
        if self.read_only:
            self._data.setdefault(name, {}).update(fields)
            return
        with file_lock(self.path):
            self._data = self._read()
            self._data.setdefault(name, {}).update(fields)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self.path)
