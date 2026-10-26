"""`manifest.json`: what was collected, from where, when, under which licence."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from synergyrx_data import __version__
from synergyrx_data.config import Settings

Layer = Literal["raw", "interim", "processed", "export", "report", "manual"]


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def hash_file(path: Path, chunk_size: int = 1 << 20) -> tuple[str, str]:
    """Return (sha256, md5) of a file. MD5 is only used to check publisher checksums."""
    sha256, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest()


@contextlib.contextmanager
def file_lock(target: Path, *, timeout: float = 30.0, stale_after: float = 120.0) -> Iterator[None]:
    """Hold `<target>.lock` while reading, changing and rewriting `target`.

    Two pipeline processes can run at once (a long download next to a `clean`). Each keeps
    its own copy of the manifest and state in memory, so every save re-reads the file under
    this lock and merges, instead of overwriting what the other process recorded.
    """
    lock = target.with_name(target.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            with contextlib.suppress(OSError):
                if time.time() - lock.stat().st_mtime > stale_after:
                    lock.unlink()  # left behind by a process that died
                    continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"could not lock {target} within {timeout:.0f}s") from None
            time.sleep(0.05)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            lock.unlink()


class ManifestEntry(BaseModel):
    path: str
    source: str
    layer: Layer
    sha256: str
    size_bytes: int
    retrieved_at: str
    url: str | None = None
    version: str | None = None
    license: str | None = None
    terms_url: str | None = None
    is_sample: bool = False
    etag: str | None = None
    last_modified: str | None = None
    rows: int | None = None
    derived_from: list[str] = Field(default_factory=list)
    input_fingerprint: str | None = None


class Manifest:
    """In-memory view of `manifest.json`, keyed by project-relative path."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.path = settings.resolve(settings.manifest_path)
        self._changed: set[str] = set()
        self.entries: dict[str, ManifestEntry] = self._read()

    def _read(self) -> dict[str, ManifestEntry]:
        if not self.path.is_file():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return {k: ManifestEntry(**v) for k, v in raw.get("files", {}).items()}

    def key(self, file: Path) -> str:
        return self._settings.relative(file)

    def get(self, file: Path) -> ManifestEntry | None:
        return self.entries.get(self.key(file))

    def for_source(self, source: str, layer: Layer | None = None) -> list[ManifestEntry]:
        return [
            e
            for e in self.entries.values()
            if e.source == source and (layer is None or e.layer == layer)
        ]

    def is_current(self, file: Path, *, url: str | None, version: str | None) -> bool:
        """True if `file` exists, was recorded for this URL and version, and still hashes equal."""
        entry = self.get(file)
        if entry is None or not file.is_file():
            return False
        if entry.url != url or entry.version != version:
            return False
        return file.stat().st_size == entry.size_bytes and hash_file(file)[0] == entry.sha256

    def record(self, entry: ManifestEntry) -> ManifestEntry:
        if self.entries.get(entry.path) != entry:
            self.entries[entry.path] = entry
            self._changed.add(entry.path)
        return entry

    def record_file(
        self, file: Path, *, source: str, layer: Layer, **fields: object
    ) -> ManifestEntry:
        """Hash `file` and record it. An unchanged file keeps its original `retrieved_at`."""
        sha256, _ = hash_file(file)
        previous = self.get(file)
        if previous is not None and previous.sha256 == sha256:
            fields.setdefault("retrieved_at", previous.retrieved_at)
        fields.setdefault("retrieved_at", utc_now_iso())
        entry = ManifestEntry(
            path=self.key(file),
            source=source,
            layer=layer,
            sha256=sha256,
            size_bytes=file.stat().st_size,
            **fields,
        )
        return self.record(entry)

    def save(self) -> None:
        """Write the entries this process recorded, keeping what other processes added."""
        if not self._changed or self._settings.dry_run:
            return
        with file_lock(self.path):
            merged = self._read()
            merged.update({key: self.entries[key] for key in self._changed})
            payload = {
                "schema_version": 1,
                "generated_by": f"synergyrx-data {__version__}",
                "updated_at": utc_now_iso(),
                "files": {k: merged[k].model_dump() for k in sorted(merged)},
            }
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            tmp.replace(self.path)
        self.entries = merged
        self._changed.clear()
